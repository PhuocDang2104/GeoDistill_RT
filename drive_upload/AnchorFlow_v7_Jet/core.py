"""Shared convolution/sparse/phase/flow primitives; parameter names preserve v3 checkpoints."""
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


def divergence(c, conductance):
    """Symmetric finite differences with zero flux through image boundaries."""
    gx, gy = conductance[:, :1], conductance[:, 1:]
    flux_x = gx[..., :-1] * (c[..., 1:] - c[..., :-1])
    flux_y = gy[..., :-1, :] * (c[..., 1:, :] - c[..., :-1, :])
    return (F.pad(flux_x, (0, 1, 0, 0)) - F.pad(flux_x, (1, 0, 0, 0))
            + F.pad(flux_y, (0, 0, 0, 1)) - F.pad(flux_y, (0, 0, 1, 0)))


class QuarterFlow(nn.Module):
    """Cache context and conductance once; only a 16-channel field is recurrent."""
    def __init__(self, steps=3):
        super().__init__()
        self.steps = int(steps)
        self.context = nn.Sequential(ConvBN(48, 16), LiteBlock(16))
        self.conductance = nn.Conv2d(48, 2, 1)
        self.state = nn.Conv2d(5, 16, 1)
        # Shared step avoids BN running-stat updates that depend on iteration.
        self.update = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16),
                                    nn.SiLU(), nn.Conv2d(16, 16, 1), nn.SiLU())
        self.reaction = nn.Conv2d(16, 1, 1)
        nn.init.normal_(self.reaction.weight, std=1e-3)
        nn.init.zeros_(self.reaction.bias)

    def forward(self, feature, z0, sparse_state):
        mean, mask, density, _, _, spread = sparse_state
        anchor_error = mean.clamp(0.1, 120).log() - z0
        # Mixed foreground/background pooled anchors should not be hard constraints.
        anchor_weight = 0.7 * mask * torch.exp(-4 * spread)
        context = self.context(feature)
        conductance = self.conductance(feature).float().sigmoid()
        c = torch.zeros_like(z0)
        for k in range(self.steps):
            t = torch.full_like(c, k / max(1, self.steps))
            state = torch.cat((c, (anchor_error - c) * mask, mask, density, t), 1)
            reaction = self.reaction(self.update(context + self.state(state))).float().tanh() * 0.75
            c = c + (reaction + 0.2 * divergence(c, conductance)) / self.steps
            c = (1 - anchor_weight) * c + anchor_weight * anchor_error
        return (z0 + c).clamp(math.log(0.1), math.log(120)).exp(), c


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


class AnchorFlowBase(nn.Module):
    def __init__(self, pretrained=True, flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k"):
        super().__init__()
        import timm
        self.encoder_name, self.flow_steps = encoder, int(flow_steps)
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        info = self.encoder.feature_info
        if tuple(info.reduction()) != (4, 8, 16):
            raise ValueError(f"Expected F4/F8/F16, got {info.reduction()}")
        pretrained_cfg = self.encoder.pretrained_cfg
        self.register_buffer("rgb_mean", torch.tensor(pretrained_cfg["mean"]).reshape(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(pretrained_cfg["std"]).reshape(1, 3, 1, 1))
        self.sparse = SparsePyramid()
        self.decoder = PyramidDecoder(info.channels())
        self.flow = QuarterFlow(flow_steps)
        self.guidance = nn.Sequential(ConvBN(12, 8), ConvBN(8, 8, 3, 8))
        self.context2 = ConvBN(48, 8)
        self.up4_2 = PhaseUpsample(48 + 32 + 3, 24, 1.0, 0.05)
        self.up2_1 = PhaseUpsample(8 + 8 + 3, 16, 0.5, 0.02)

    @staticmethod
    def _sensor_state(depth, sparse, mask, scale):
        density = F.avg_pool2d(mask, scale, scale)
        mean = F.avg_pool2d(sparse * mask, scale, scale) / density.clamp_min(1e-6)
        valid = (density > 0).float()
        return torch.cat((depth / 120, valid, (mean - depth) * valid / 120), 1)

    def forward(self, rgb, sparse, mask, K):
        valid = (mask > 0.5) & (sparse > 0.1) & (sparse < 120) & torch.isfinite(sparse)
        sparse = torch.where(valid, sparse.float(), torch.zeros_like(sparse, dtype=torch.float32))
        mask = valid.float()
        rgb_features = self.encoder((rgb - self.rgb_mean) / self.rgb_std)
        sparse_features, states = self.sparse(sparse, mask, K.float())
        p4, d16, d8, z0 = self.decoder(rgb_features, sparse_features, states[0])
        d4, correction = self.flow(p4, z0, states[0])
        g2 = self.guidance(F.pixel_unshuffle(rgb, 2))
        g4 = F.pixel_unshuffle(g2, 2)
        d2 = self.up4_2(d4, torch.cat((p4, g4, self._sensor_state(d4, sparse, mask, 4)), 1))
        p2 = F.interpolate(self.context2(p4), size=g2.shape[-2:], mode="nearest")
        d1 = self.up2_1(d2, torch.cat((p2, g2, self._sensor_state(d2, sparse, mask, 2)), 1))
        final = torch.where(valid, sparse, d1)
        return {"D16": d16, "D8": d8, "D4": d4, "D2": d2, "D1": d1,
                "D_full": final, "D0": z0.exp(), "correction": correction}

    def freeze_encoder_bn(self):
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
