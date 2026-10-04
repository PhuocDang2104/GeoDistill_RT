"""Canonical v3 forward, retained as the matched-budget research control."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from core import AnchorFlowBase as V2, ConvBN, LiteBlock



class Context32(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(ConvBN(channels, 64), LiteBlock(64))
        self.project = nn.Conv2d(64, 48, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, f32, f16):
        context = F.interpolate(self.body(f32), size=f16.shape[-2:], mode="bilinear", align_corners=False)
        return f16 + self.project(context)


class MetricRefine4(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(ConvBN(52, 32), LiteBlock(32),
                                  nn.Conv2d(32, 32, 3, padding=2, dilation=2, groups=32), nn.SiLU())
        self.delta = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, feature, depth, state):
        mean, valid, density, _, _, spread = state
        sensor = torch.cat((depth / 120, valid * (mean - depth) / 20, density, spread.clamp_max(2)), 1)
        raw = self.delta(self.body(torch.cat((feature, sensor), 1))).float()
        delta = (2.0 + .10 * depth) * raw.tanh()
        return (depth + delta).clamp(.1, 120), delta


class PhaseDetailAndTrust(nn.Module):
    """No learned full-resolution feature CNN; full-res scalar fusion is FP32."""
    def __init__(self):
        super().__init__()
        # P2(8), G2(8), raw RGB phases(12), S/M/error phases(12), D phases(4).
        self.body = nn.Sequential(ConvBN(44, 24), LiteBlock(24),
                                  nn.Conv2d(24, 24, 3, padding=2, dilation=2, groups=24),
                                  nn.SiLU(), ConvBN(24, 24))
        self.delta = nn.Conv2d(24, 4, 1)
        self.trust_logits = nn.Conv2d(24, 4, 1)
        self.log_tolerance = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.trust_logits.weight)
        nn.init.constant_(self.trust_logits.bias, math.log(9.0))

    def forward(self, rgb, sparse, mask, depth, p2, g2):
        packed = F.pixel_unshuffle(torch.cat((sparse / 120, mask, mask * (sparse - depth) / 20, depth / 120), 1), 2)
        features = torch.cat((p2, g2, F.pixel_unshuffle(rgb, 2), packed), 1)
        hidden = self.body(features)
        raw = F.pixel_shuffle(self.delta(hidden).float(), 2)
        delta = (.5 + .05 * depth) * raw.tanh()
        refined = (depth + delta).clamp(.1, 120)
        logits = F.pixel_shuffle(self.trust_logits(hidden).float(), 2)
        # Broad consensus prior, not a GT-dependent test-time filter.
        tolerance = (.5 + .02 * refined) * self.log_tolerance.clamp(-1, 2).exp()
        disagreement = (sparse - refined) / tolerance
        confidence_logits = logits - torch.log1p(disagreement.square())
        gate = mask * confidence_logits.sigmoid()
        final = (1 - gate) * refined + gate * sparse
        return refined, final, gate, confidence_logits, delta


class AnchorFlowEdge(V2):
    def __init__(self, pretrained=False, flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k"):
        super().__init__(pretrained=False, flow_steps=flow_steps, encoder=encoder)
        import timm
        # timm's MobileNetV3Features already executes/stores the F32 tail in v2.
        # Returning its output changes no old parameter names/shapes.
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        if tuple(self.encoder.feature_info.reduction()) != (4, 8, 16, 32):
            raise ValueError("Expected F4/F8/F16/F32")
        self.context32 = Context32(self.encoder.feature_info.channels()[-1])
        self.metric4 = MetricRefine4()
        self.detail1 = PhaseDetailAndTrust()

    def decode(self, features, sparse_features, states):
        p4, d16, d8, z0 = self.decoder(features[:3], sparse_features, states[0])
        return p4, d16, d8, z0, None

    def phase_context(self, p4, g2, capacity):
        return F.interpolate(self.context2(p4), size=g2.shape[-2:], mode="nearest")

    def refine_surface(self,p4,d4,state,rgb,K):
        return d4, {"D4_base":d4}

    def refine_half(self,d2,p4,g2,sparse,mask,K,diagnostics):
        return d2, {"D2_base":d2}

    def forward(self, rgb, sparse, mask, K):
        valid = (mask > .5) & (sparse > .1) & (sparse < 120) & torch.isfinite(sparse)
        sparse = torch.where(valid, sparse.float(), torch.zeros_like(sparse, dtype=torch.float32))
        mask = valid.float()
        features = self.encoder((rgb - self.rgb_mean) / self.rgb_std)
        f4, f8, f16, f32 = features
        f16 = self.context32(f32, f16)
        sparse_features, states = self.sparse(sparse, mask, K.float())
        p4, d16, d8, z0, capacity = self.decode((f4, f8, f16, f32), sparse_features, states)
        d4_flow, correction = self.flow(p4, z0, states[0])
        d4, delta4 = self.metric4(p4, d4_flow, states[0])
        d4, surface_diagnostics = self.refine_surface(p4,d4,states[0],rgb,K)
        g2 = self.guidance(F.pixel_unshuffle(rgb, 2))
        g4 = F.pixel_unshuffle(g2, 2)
        d2 = self.up4_2(d4, torch.cat((p4, g4, self._sensor_state(d4, sparse, mask, 4)), 1))
        d2,half_diagnostics = self.refine_half(d2,p4,g2,sparse,mask,K,surface_diagnostics)
        p2 = self.phase_context(p4, g2, capacity)
        d1_base = self.up2_1(d2, torch.cat((p2, g2, self._sensor_state(d2, sparse, mask, 2)), 1))
        d1, final, gate, logits, delta1 = self.detail1(rgb, sparse, mask, d1_base, p2, g2)
        return {"D16": d16, "D8": d8, "D0": z0.exp(), "D4_flow": d4_flow, "D4": d4,
                "D2": d2, "D1_base": d1_base, "D1": d1, "D_full": final,
                "D_hard": torch.where(valid, sparse, d1), "sensor_gate": gate,
                "sensor_logits": logits, "delta4": delta4, "delta1": delta1, "correction": correction,
                **surface_diagnostics,**half_diagnostics}
