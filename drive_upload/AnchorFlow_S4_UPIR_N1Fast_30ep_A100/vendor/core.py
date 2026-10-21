"""Only live convolution, sparse and phase primitives; checkpoint keys preserved."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, kernel=1, groups=1, activation=True):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=False) if activation else nn.Identity(),
        )



class LiteBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.body = nn.Sequential(ConvBN(channels, hidden),
                                  ConvBN(hidden, hidden, 3, hidden),
                                  ConvBN(hidden, channels, activation=False))

    def forward(self, x):
        return F.silu(x + self.body(x))



class SparsePyramid(nn.Module):
    """Pool numerator/support together, preserving density at every level."""
    def __init__(self):
        super().__init__()
        self.stems = nn.ModuleList([nn.Sequential(ConvBN(8, 16), LiteBlock(16)) for _ in range(3)])

    def forward(self, sparse, mask, K):
        # Explicit FP32: S^2 and reductions can overflow FP16 at KITTI depths.
        s, m = sparse.float(), mask.float()
        stats = F.avg_pool2d(torch.cat((s * m, m, s.square() * m), 1), 4, 4)
        outputs, states = [], []
        for i, stem in enumerate(self.stems):
            if i:
                stats = F.avg_pool2d(stats, 2, 2)
            numerator, density, second = stats.split(1, 1)
            valid = (density > 0).float()
            mean = numerator / density.clamp_min(1e-6)
            variance = (second / density.clamp_min(1e-6) - mean.square()).clamp_min(0)
            relative_std = variance.sqrt() / mean.clamp_min(0.1)
            local = F.avg_pool2d(stats[:, :2], 7, 1, 3)
            initial = local[:, :1] / local[:, 1:].clamp_min(1e-6)
            initial_valid = (local[:, 1:] > 0).float()
            h, w = mean.shape[-2:]
            scale = 4 * 2 ** i
            u = (torch.arange(w, device=s.device, dtype=torch.float32) + 0.5) * scale - 0.5
            v = (torch.arange(h, device=s.device, dtype=torch.float32) + 0.5) * scale - 0.5
            x = (u.view(1, 1, 1, w) - K[:, 0, 2, None, None, None]) / K[:, 0, 0, None, None, None].clamp_min(1)
            y = (v.view(1, 1, h, 1) - K[:, 1, 2, None, None, None]) / K[:, 1, 1, None, None, None].clamp_min(1)
            x = x.expand_as(mean)
            y = y.expand_as(mean)
            state = torch.cat((mean / 120, valid, density, initial / 120,
                               initial_valid, relative_std.clamp_max(2), x, y), 1)
            outputs.append(stem(state))
            states.append((mean, valid, density, initial, initial_valid, relative_std))
        return outputs, states



class Fusion(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        self.rgb = ConvBN(cin, width)
        self.sparse = ConvBN(16, width)
        self.gate = nn.Conv2d(width * 2, width, 1)
        self.refine = LiteBlock(width)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.4)

    def forward(self, rgb, sparse):
        a, b = self.rgb(rgb), self.sparse(sparse)
        return self.refine(a + self.gate(torch.cat((a, b), 1)).sigmoid() * b)



class PyramidDecoder(nn.Module):
    def __init__(self, rgb_channels):
        super().__init__()
        self.fuse = nn.ModuleList([Fusion(c, w) for c, w in zip(rgb_channels, (48, 64, 96))])
        self.up16 = ConvBN(96, 64)
        self.up8 = ConvBN(64, 48)
        self.refine8, self.refine4 = LiteBlock(64), LiteBlock(48)
        self.head16 = nn.Conv2d(96, 1, 1)
        self.head8, self.head4 = nn.Conv2d(64, 1, 1), nn.Conv2d(48, 1, 1)
        self.prior_gate = nn.Conv2d(48, 1, 1)
        nn.init.normal_(self.head16.weight, std=1e-3)
        nn.init.constant_(self.head16.bias, 20.0)
        for head in (self.head8, self.head4):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.prior_gate.weight)
        nn.init.zeros_(self.prior_gate.bias)

    def forward(self, rgb_features, sparse_features, sparse_state4):
        f4, f8, f16 = [block(a, b) for block, a, b in zip(self.fuse, rgb_features, sparse_features)]
        p8 = self.refine8(f8 + F.interpolate(self.up16(f16), size=f8.shape[-2:], mode="nearest"))
        p4 = self.refine4(f4 + F.interpolate(self.up8(p8), size=f4.shape[-2:], mode="nearest"))
        d16 = F.softplus(self.head16(f16).float()).clamp(0.1, 120)
        z8 = F.interpolate(d16.log(), size=p8.shape[-2:], mode="bilinear", align_corners=False)
        z8 = z8 + 0.7 * self.head8(p8).float().tanh()
        d8 = z8.clamp(math.log(0.1), math.log(120)).exp()
        z4 = F.interpolate(d8.log(), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        z4 = z4 + 0.7 * self.head4(p4).float().tanh()
        initial, support = sparse_state4[3], sparse_state4[4]
        gate = support * self.prior_gate(p4).float().sigmoid()
        z0 = (1 - gate) * z4 + gate * initial.clamp(0.1, 120).log()
        return p4, d16, d8, z0.clamp(math.log(0.1), math.log(120))



class PhaseUpsample(nn.Module):
    """One scalar 3x3 neighbourhood, four phase weights, bounded metric residual."""
    def __init__(self, channels, hidden, residual_base, residual_ratio):
        super().__init__()
        self.trunk = nn.Sequential(ConvBN(channels, hidden), LiteBlock(hidden))
        self.weights = nn.Conv2d(hidden, 36, 1)
        self.residual = nn.Conv2d(hidden, 4, 1)
        self.residual_base, self.residual_ratio = residual_base, residual_ratio
        kernels = torch.eye(9).reshape(9, 1, 3, 3)
        self.register_buffer("neighbours", kernels)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        with torch.no_grad():
            self.weights.bias.reshape(4, 9)[:, 4] = 2
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, depth, features):
        hidden = self.trunk(features)
        b, _, h, w = depth.shape
        # Weights and accumulation in FP32, while all learned convolutions use AMP.
        weights = self.weights(hidden).float().reshape(b, 4, 9, h, w).softmax(2)
        with torch.autocast(device_type=depth.device.type, enabled=False):
            neighbours = F.conv2d(F.pad(depth.float(), (1, 1, 1, 1), mode="replicate"), self.neighbours.float())
        phases = (weights * neighbours[:, None]).sum(2)
        phases = phases + (self.residual_base + self.residual_ratio * phases) * self.residual(hidden).float().tanh()
        return F.pixel_shuffle(phases, 2).clamp(0.1, 120)

