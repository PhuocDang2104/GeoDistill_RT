"""Unchanged F32 context and half-resolution detail/sensor-trust primitives."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN, LiteBlock



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


