"""V4 ContextWide: increase learned low-resolution capacity, preserve all v3 weights.

No change to loss, sensor policy, three flow steps, or evaluation support.
Only this module adds trainable parameters relative to model_v3.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN
from model_v3 import AnchorFlowEdge as V3


class ContextBlock(nn.Module):
    def __init__(self, channels, expansion, dilation):
        super().__init__()
        hidden = channels * expansion
        self.expand = ConvBN(channels, hidden)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden, hidden, 5, padding=2*dilation, dilation=dilation,
                      groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.SiLU())
        self.project = ConvBN(hidden, channels, activation=False)

    def forward(self, x):
        return F.silu(x + self.project(self.spatial(self.expand(x))))


def zero_projection(cin, cout):
    head = nn.Conv2d(cin, cout, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


class ContextCapacity(nn.Module):
    """Dense top-down latent pyramid 192@1/16 -> 96@1/8 -> 48@1/4."""
    def __init__(self):
        super().__init__()
        self.in32 = ConvBN(480, 192)
        self.in16 = ConvBN(48+16, 192)
        self.in8 = ConvBN(32+16, 96)
        self.in4 = ConvBN(16+16, 48)
        self.blocks16 = nn.Sequential(*(ContextBlock(192, 4, d) for d in (1, 2, 3)))
        self.blocks8 = nn.Sequential(*(ContextBlock(96, 3, d) for d in (1, 2)))
        self.blocks4 = nn.Sequential(*(ContextBlock(48, 2, d) for d in (1, 2)))
        self.up16 = ConvBN(192, 96)
        self.up8 = ConvBN(96, 48)
        # No BN on 1x1 global descriptor: batch1 is supported during smoke.
        self.global_context = nn.Sequential(nn.Conv2d(192, 48, 1), nn.SiLU(), nn.Conv2d(48, 192, 1))
        self.out16 = zero_projection(192, 96)
        self.out8 = zero_projection(96, 64)
        self.out4 = zero_projection(48, 48)
        self.out2 = zero_projection(48, 8)

    def forward(self, rgb_features, sparse_features):
        f4, f8, f16, f32 = rgb_features
        s4, s8, s16 = sparse_features
        c16 = self.in16(torch.cat((f16, s16), 1))
        c16 = c16 + F.interpolate(self.in32(f32), size=c16.shape[-2:], mode="bilinear", align_corners=False)
        c16 = c16 + self.global_context(c16.mean((-2, -1), keepdim=True))
        c16 = self.blocks16(c16)
        c8 = self.in8(torch.cat((f8, s8), 1))
        c8 = self.blocks8(c8 + F.interpolate(self.up16(c16), size=c8.shape[-2:], mode="bilinear", align_corners=False))
        c4 = self.in4(torch.cat((f4, s4), 1))
        c4 = self.blocks4(c4 + F.interpolate(self.up8(c8), size=c4.shape[-2:], mode="bilinear", align_corners=False))
        return self.out16(c16), self.out8(c8), self.out4(c4), c4


class AnchorFlowEdge(V3):
    def __init__(self, pretrained=False, flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k"):
        super().__init__(pretrained=pretrained, flow_steps=flow_steps, encoder=encoder)
        self.capacity = ContextCapacity()

    def decode(self, features, sparse_features, states):
        add16, add8, add4, capacity = self.capacity(features, sparse_features)
        decoder = self.decoder
        f4, f8, f16 = [block(a, b) for block, a, b in zip(decoder.fuse, features[:3], sparse_features)]
        f16 = f16 + add16
        p8 = decoder.refine8(f8 + F.interpolate(decoder.up16(f16), size=f8.shape[-2:], mode="nearest")) + add8
        p4 = decoder.refine4(f4 + F.interpolate(decoder.up8(p8), size=f4.shape[-2:], mode="nearest")) + add4
        d16 = F.softplus(decoder.head16(f16).float()).clamp(.1, 120)
        z8 = F.interpolate(d16.log(), size=p8.shape[-2:], mode="bilinear", align_corners=False)
        z8 = z8 + .7 * decoder.head8(p8).float().tanh()
        d8 = z8.clamp(math.log(.1), math.log(120)).exp()
        z4 = F.interpolate(d8.log(), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        z4 = z4 + .7 * decoder.head4(p4).float().tanh()
        initial, support = states[0][3], states[0][4]
        gate = support * decoder.prior_gate(p4).float().sigmoid()
        z0 = (1-gate)*z4 + gate*initial.clamp(.1, 120).log()
        return p4, d16, d8, z0.clamp(math.log(.1), math.log(120)), capacity

    def phase_context(self, p4, g2, capacity):
        # Project before resize: extra learned processing stays at 1/4 or lower.
        context = self.context2(p4) + self.capacity.out2(capacity)
        return F.interpolate(context, size=g2.shape[-2:], mode="nearest")
