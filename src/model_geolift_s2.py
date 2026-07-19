from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        pad = kernel // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if activate:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class DWPointwise(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            ConvBNAct(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, kernel=1),
        )


class ResidualDWBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel=3, groups=channels),
            ConvBNAct(channels, channels, kernel=1, activate=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.block(x), inplace=True)


def phase_pack(x: torch.Tensor) -> torch.Tensor:
    """Pixel-unshuffle with phase-major channel order: q00, q10, q01, q11."""
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"phase_pack needs even H/W, got {(h, w)}")
    packed = F.pixel_unshuffle(x, 2).reshape(b, c, 4, h // 2, w // 2)
    return packed.permute(0, 2, 1, 3, 4).reshape(b, 4 * c, h // 2, w // 2)


def phase_unpack(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`phase_pack` for phase-major tensors."""
    b, c4, h, w = x.shape
    if c4 % 4:
        raise ValueError(f"phase_unpack needs channels divisible by 4, got {c4}")
    c = c4 // 4
    pixel_shuffle_order = x.reshape(b, 4, c, h, w).permute(0, 2, 1, 3, 4).reshape(b, c4, h, w)
    return F.pixel_shuffle(pixel_shuffle_order, 2)


def _valid_average_pool(value: torch.Tensor, mask: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = F.avg_pool2d(value * mask, scale, stride=scale) * (scale * scale)
    denominator = F.avg_pool2d(mask, scale, stride=scale) * (scale * scale)
    pooled = numerator / denominator.clamp_min(1.0)
    return pooled, (denominator > 0.0).to(value.dtype)


def compact_sparse_prior(
    sparse: torch.Tensor,
    mask: torch.Tensor,
    scale: int = 4,
    radius: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One local normalized sparse propagation pass at 1/scale resolution.

    Empty support stays empty. No global mean fill is permitted because it would
    leak a scene-wide depth shortcut into the learned front-end.
    """
    sparse_q, mask_q = _valid_average_pool(sparse, mask, scale)
    # The v2.1 notation N_7 / AvgPool_7 denotes a 7x7 window.
    kernel = int(radius)
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"Sparse-prior window must be a positive odd integer, got {kernel}")
    padding = kernel // 2
    support = F.avg_pool2d(mask_q, kernel, stride=1, padding=padding) * (kernel * kernel)
    numerator = F.avg_pool2d(sparse_q * mask_q, kernel, stride=1, padding=padding) * (kernel * kernel)
    depth_init = numerator / support.clamp_min(1.0)
    valid_init = (support > 0.0).to(sparse.dtype)
    depth_init = depth_init * valid_init
    density = (support / float(kernel * kernel)).clamp(0.0, 1.0)
    return sparse_q, mask_q, depth_init, valid_init, density


class CompactSparsePrior(nn.Module):
    def __init__(self, scale: int = 4, radius: int = 7) -> None:
        super().__init__()
        self.scale = int(scale)
        self.radius = int(radius)

    def forward(self, sparse: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return compact_sparse_prior(sparse, mask, self.scale, self.radius)


def phase_base_offsets(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    phase_x = torch.tensor((0.0, 1.0, 0.0, 1.0), device=device, dtype=dtype)
    phase_y = torch.tensor((0.0, 0.0, 1.0, 1.0), device=device, dtype=dtype)
    return torch.stack((phase_x / 2.0 - 0.25, phase_y / 2.0 - 0.25), dim=1)


def affine_inverse_depth_transport(
    inverse_depth: torch.Tensor,
    slope_a: torch.Tensor,
    slope_b: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
) -> torch.Tensor:
    return inverse_depth + slope_a * (target_x - source_x) + slope_b * (target_y - source_y)


def _ray_xy(K: torch.Tensor, h: int, w: int, full_hw: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    full_h, full_w = full_hw
    scale_x, scale_y = full_w / float(w), full_h / float(h)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=K.device, dtype=dtype),
        torch.arange(w, device=K.device, dtype=dtype),
        indexing="ij",
    )
    u = (xx + 0.5) * scale_x - 0.5
    v = (yy + 0.5) * scale_y - 0.5
    fx = K[:, 0, 0].to(dtype).view(-1, 1, 1).clamp_min(1e-6)
    fy = K[:, 1, 1].to(dtype).view(-1, 1, 1).clamp_min(1e-6)
    cx = K[:, 0, 2].to(dtype).view(-1, 1, 1)
    cy = K[:, 1, 2].to(dtype).view(-1, 1, 1)
    return torch.stack(((u[None] - cx) / fx, (v[None] - cy) / fy), dim=1)


class RGBQuarterStem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2 = ConvBNAct(3, 12, kernel=3, stride=2)
        self.rep4 = DWPointwise(12, 12, stride=2)
        self.project = ConvBNAct(12, 24, kernel=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.project(self.rep4(self.conv2(rgb)))


class DepthPriorStem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBNAct(5, 16, kernel=1), DWPointwise(16, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualFusion(nn.Module):
    def __init__(self, in_ch: int = 42, out_ch: int = 32) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_ch, out_ch, kernel=1)
        self.mix = ResidualDWBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.reduce(x))


class StageAdaptedMobileViT(nn.Module):
    """MobileViTv2 stage reuse after a learned 1/4-resolution multimodal stem."""

    def __init__(self, model_name: str = "mobilevitv2_0.75") -> None:
        super().__init__()
        import timm  # type: ignore

        aliases = {
            "mobilevitv2_0.75": ("mobilevitv2_075.cvnets_in1k", "mobilevitv2_075"),
            "mobilevitv2_075": ("mobilevitv2_075.cvnets_in1k", "mobilevitv2_075"),
        }
        candidates = aliases.get(model_name, (model_name,))
        base = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                base = timm.create_model(candidate, pretrained=False)
                break
            except Exception as exc:  # pragma: no cover - depends on timm registry
                last_error = exc
        if base is None:
            raise RuntimeError(f"Could not construct {model_name}: {last_error}")
        feature_info = {int(item["reduction"]): item for item in base.feature_info}
        if 2 not in feature_info or 4 not in feature_info or 8 not in feature_info:
            raise RuntimeError(f"Unsupported MobileViTv2 feature layout: {base.feature_info}")

        f4_in = int(feature_info[2]["num_chs"])
        f8_out = int(feature_info[4]["num_chs"])
        f16_out = int(feature_info[8]["num_chs"])
        stage8_index = int(str(feature_info[4]["module"]).split(".")[1])
        stage16_index = int(str(feature_info[8]["module"]).split(".")[1])
        self.adapter4 = ConvBNAct(32, f4_in, kernel=1)
        self.stage8 = base.stages[stage8_index]
        self.stage16 = base.stages[stage16_index]
        self.out_channels = (f4_in, f8_out, f16_out)

    def forward(self, x4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f4 = self.adapter4(x4)
        f8 = self.stage8(f4)
        f16 = self.stage16(f8)
        return f4, f8, f16


class LiteFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], width: int = 24) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.lat4 = nn.Conv2d(c4, width, 1)
        self.lat8 = nn.Conv2d(c8, width, 1)
        self.lat16 = nn.Conv2d(c16, width, 1)
        self.smooth4 = ResidualDWBlock(width)

    def forward(self, f4: torch.Tensor, f8: torch.Tensor, f16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p16 = self.lat16(f16)
        p8 = self.lat8(f8) + F.interpolate(p16, size=f8.shape[-2:], mode="nearest")
        p4 = self.lat4(f4) + F.interpolate(p8, size=f4.shape[-2:], mode="nearest")
        return self.smooth4(p4), p8, p16


class PhasePriorGenerator(nn.Module):
    """Generate the only learned half-resolution guidance tensor G2."""

    def __init__(self, out_ch: int = 12) -> None:
        super().__init__()
        self.phase_project = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            ConvBNAct(16, 16, kernel=3, groups=16),
            ConvBNAct(16, out_ch, kernel=1),
        )

    @staticmethod
    def rgb_edge(rgb: torch.Tensor) -> torch.Tensor:
        dx = F.pad(rgb[:, :, :, 1:] - rgb[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(rgb[:, :, 1:, :] - rgb[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt((dx * dx + dy * dy).sum(dim=1, keepdim=True) + 1e-8).clamp(0.0, 1.0)

    def forward(self, rgb: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor, max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
        log_sparse = torch.log(sparse.clamp_min(1e-3)) * mask
        raw = torch.cat((rgb, mask, log_sparse, self.rgb_edge(rgb)), dim=1)
        q2_phase = self.phase_project(phase_pack(raw))
        return self.mix(q2_phase), q2_phase


@dataclass(frozen=True)
class RayLiftSpec:
    mode: str
    samples: int
    slope_limit: float
    offset_limit: float = 1.5


class RayLiftIDBlock(nn.Module):
    """One inverse-depth transport stage with four explicit child phases."""

    phase_x = (0.0, 1.0, 0.0, 1.0)
    phase_y = (0.0, 0.0, 1.0, 1.0)

    def __init__(self, source_ch: int, guidance_ch_per_phase: int, spec: RayLiftSpec, final: bool = False) -> None:
        super().__init__()
        self.spec = spec
        self.final = bool(final)
        trunk_in = source_ch + 4 * guidance_ch_per_phase + 4
        self.trunk = nn.Sequential(
            ConvBNAct(trunk_in, 24, kernel=1),
            ConvBNAct(24, 24, kernel=3, groups=24),
            ConvBNAct(24, 16, kernel=1),
        )
        geometry_dim = {"cross": 5, "line": 4, "neighbor": 8}[spec.mode]
        self.geometry = nn.Conv2d(16, geometry_dim, 1)
        self.logits = nn.Conv2d(16, 4 * spec.samples, 1)
        self.slopes = nn.Conv2d(16, 8, 1)
        self.eta = nn.Conv2d(16, 4, 1)
        self.gate = nn.Conv2d(16, 4, 1)
        self.conf_calibration = nn.Conv2d(16, 4, 1) if final else None
        self._init_near_bilinear()

    def _init_near_bilinear(self) -> None:
        for layer in (self.geometry, self.logits, self.slopes, self.eta, self.gate):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.eta.bias, math.log(0.02 / 0.98))
        nn.init.constant_(self.gate.bias, math.log(0.05 / 0.95))
        if self.conf_calibration is not None:
            nn.init.zeros_(self.conf_calibration.weight)
            nn.init.constant_(self.conf_calibration.bias, math.log(0.95 / 0.05))

    @staticmethod
    def _sample(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, four, samples, h, w = x.shape
        hs, ws = field.shape[-2:]
        gx = (2.0 * x + 1.0) / float(ws) - 1.0
        gy = (2.0 * y + 1.0) / float(hs) - 1.0
        grid = torch.stack((gx, gy), dim=-1).permute(0, 3, 4, 1, 2, 5).reshape(b, h, w * four * samples, 2)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return sampled[:, 0].reshape(b, h, w, four, samples).permute(0, 3, 4, 1, 2)

    def _offsets(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = geometry.shape
        samples = self.spec.samples
        if self.spec.mode == "cross":
            tx = torch.tanh(geometry[:, 0:1]) * self.spec.offset_limit
            ty = torch.tanh(geometry[:, 1:2]) * self.spec.offset_limit
            theta = math.pi * torch.tanh(geometry[:, 2:3])
            r_parallel = 0.5 + torch.sigmoid(geometry[:, 3:4])
            r_perpendicular = 0.5 + torch.sigmoid(geometry[:, 4:5])
            template = geometry.new_tensor(((0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)))[:samples]
            px = template[:, 0].view(1, 1, samples, 1, 1) * r_parallel[:, :, None]
            py = template[:, 1].view(1, 1, samples, 1, 1) * r_perpendicular[:, :, None]
            cosine, sine = torch.cos(theta)[:, :, None], torch.sin(theta)[:, :, None]
            dx = cosine * px - sine * py + tx[:, :, None]
            dy = sine * px + cosine * py + ty[:, :, None]
            return dx.expand(b, 4, samples, h, w), dy.expand(b, 4, samples, h, w)
        if self.spec.mode == "line":
            tx = torch.tanh(geometry[:, 0:1]) * self.spec.offset_limit
            ty = torch.tanh(geometry[:, 1:2]) * self.spec.offset_limit
            theta = math.pi * torch.tanh(geometry[:, 2:3])
            radius = 0.5 + torch.sigmoid(geometry[:, 3:4])
            positions = geometry.new_tensor((-1.0, 0.0, 1.0))[:samples].view(1, 1, samples, 1, 1)
            dx = tx[:, :, None] + positions * radius[:, :, None] * torch.cos(theta)[:, :, None]
            dy = ty[:, :, None] + positions * radius[:, :, None] * torch.sin(theta)[:, :, None]
            return dx.expand(b, 4, samples, h, w), dy.expand(b, 4, samples, h, w)

        theta = math.pi * torch.tanh(geometry[:, 0::2])
        radius = 0.5 + torch.sigmoid(geometry[:, 1::2])
        dx_neighbor = radius * torch.cos(theta)
        dy_neighbor = radius * torch.sin(theta)
        zeros = torch.zeros_like(dx_neighbor)
        return torch.stack((zeros, dx_neighbor), dim=2), torch.stack((zeros, dy_neighbor), dim=2)

    def forward(
        self,
        depth: torch.Tensor,
        confidence: torch.Tensor,
        source: torch.Tensor,
        guidance_phase: torch.Tensor,
        K: torch.Tensor,
        full_hw: tuple[int, int],
        min_depth: float,
        max_depth: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, _, h, w = depth.shape
        if guidance_phase.shape[-2:] != (h, w):
            raise ValueError(f"guidance must be phase-packed at source resolution {(h, w)}, got {guidance_phase.shape}")
        xy = _ray_xy(K, h, w, full_hw, depth.dtype)
        state = torch.cat((source, guidance_phase, torch.log(depth.clamp_min(min_depth)) / math.log(max_depth), confidence, xy), dim=1)
        trunk = self.trunk(state)
        geometry = self.geometry(trunk)
        dx, dy = self._offsets(geometry)

        yy, xx = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        phase_x = depth.new_tensor(self.phase_x).view(1, 4, 1, 1, 1)
        phase_y = depth.new_tensor(self.phase_y).view(1, 4, 1, 1, 1)
        base_offsets = phase_base_offsets(depth.device, depth.dtype)
        phase_base_x = base_offsets[:, 0].view(1, 4, 1, 1, 1)
        phase_base_y = base_offsets[:, 1].view(1, 4, 1, 1, 1)
        x_parent = xx.view(1, 1, 1, h, w) + phase_base_x + dx
        y_parent = yy.view(1, 1, 1, h, w) + phase_base_y + dy
        inverse = depth.clamp(min_depth, max_depth).reciprocal()
        xi_sample = self._sample(inverse, x_parent, y_parent)

        full_h, full_w = full_hw
        source_scale_x = full_w / float(w)
        source_scale_y = full_h / float(h)
        target_h, target_w = 2 * h, 2 * w
        target_scale_x = full_w / float(target_w)
        target_scale_y = full_h / float(target_h)
        x_source_full = (x_parent + 0.5) * source_scale_x - 0.5
        y_source_full = (y_parent + 0.5) * source_scale_y - 0.5
        x_target = 2.0 * xx.view(1, 1, 1, h, w) + phase_x
        y_target = 2.0 * yy.view(1, 1, 1, h, w) + phase_y
        x_target_full = (x_target + 0.5) * target_scale_x - 0.5
        y_target_full = (y_target + 0.5) * target_scale_y - 0.5

        fx = K[:, 0, 0].view(b, 1, 1, 1, 1).clamp_min(1e-6)
        fy = K[:, 1, 1].view(b, 1, 1, 1, 1).clamp_min(1e-6)
        cx = K[:, 0, 2].view(b, 1, 1, 1, 1)
        cy = K[:, 1, 2].view(b, 1, 1, 1, 1)
        ray_source_x = (x_source_full - cx) / fx
        ray_source_y = (y_source_full - cy) / fy
        ray_target_x = (x_target_full - cx) / fx
        ray_target_y = (y_target_full - cy) / fy

        slope = torch.tanh(self.slopes(trunk)).reshape(b, 4, 2, h, w) * self.spec.slope_limit
        a, bb = slope[:, :, 0:1], slope[:, :, 1:2]
        transported = affine_inverse_depth_transport(
            xi_sample, a, bb, ray_target_x, ray_target_y, ray_source_x, ray_source_y
        )
        transported = transported.clamp(1.0 / max_depth, 1.0 / min_depth)
        eta = torch.sigmoid(self.eta(trunk)).unsqueeze(2)
        candidates = (1.0 - eta) * xi_sample + eta * transported

        logits = self.logits(trunk).reshape(b, 4, self.spec.samples, h, w)
        weights = torch.softmax(logits, dim=2)
        xi_aggregate = (weights * candidates).sum(dim=2)

        xi_bilinear = phase_pack(F.interpolate(inverse, scale_factor=2, mode="bilinear", align_corners=False))
        gate = torch.sigmoid(self.gate(trunk))
        xi_child = ((1.0 - gate) * xi_bilinear + gate * xi_aggregate).clamp(1.0 / max_depth, 1.0 / min_depth)
        depth_child = phase_unpack(xi_child).reciprocal().clamp(min_depth, max_depth)

        confidence_bilinear = phase_pack(F.interpolate(confidence, scale_factor=2, mode="bilinear", align_corners=False))
        dispersion = (weights * (candidates - xi_aggregate.unsqueeze(2)).abs()).sum(dim=2)
        confidence_child = confidence_bilinear * torch.exp(-dispersion / xi_aggregate.detach().abs().clamp_min(1e-4))
        confidence_child = confidence_child.clamp(1e-4, 1.0)
        if self.conf_calibration is not None:
            confidence_child = confidence_child * torch.sigmoid(self.conf_calibration(trunk))
        confidence_child = phase_unpack(confidence_child.clamp(1e-4, 1.0))

        aux = {
            "slope_a": a.squeeze(2),
            "slope_b": bb.squeeze(2),
            "eta": eta.squeeze(2),
            "gate": gate,
            "weights": weights,
        }
        return depth_child, confidence_child, aux


class GeoLiftStudentS2(nn.Module):
    """GeoLift-S2 exactly follows the v2.1 clean-test tensor contract."""

    def __init__(
        self,
        encoder: str = "mobilevitv2_0.75",
        fusion_channels: int = 32,
        fpn_channels: int = 24,
        sparse_scale: int = 4,
        sparse_radius: int = 7,
        min_depth: float = 1e-3,
        max_depth: float = 120.0,
    ) -> None:
        super().__init__()
        if sparse_scale != 4:
            raise ValueError("GeoLift-S2 v2.1 is defined for a 1/4 sparse prior (scale=4)")
        if fusion_channels != 32 or fpn_channels != 24:
            raise ValueError("GeoLift-S2 v2.1 fixes fusion=32 and FPN=24 channels")
        self.sparse_scale = int(sparse_scale)
        self.sparse_radius = int(sparse_radius)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        self.sparse_prior = CompactSparsePrior(self.sparse_scale, self.sparse_radius)
        self.rgb_stem = RGBQuarterStem()
        self.depth_stem = DepthPriorStem()
        self.fusion = ResidualFusion(42, fusion_channels)
        self.encoder = StageAdaptedMobileViT(encoder)
        self.fpn = LiteFPN(self.encoder.out_channels, fpn_channels)
        self.initial_depth = nn.Sequential(DWPointwise(fpn_channels, fpn_channels), nn.Conv2d(fpn_channels, 1, 1))
        self.initial_confidence = nn.Sequential(DWPointwise(fpn_channels, fpn_channels), nn.Conv2d(fpn_channels, 1, 1))

        self.ppg = PhasePriorGenerator(12)
        self.source2 = nn.Sequential(ConvBNAct(14, 12, kernel=1), ResidualDWBlock(12))

        self.lift16_8 = RayLiftIDBlock(fpn_channels, fpn_channels, RayLiftSpec("cross", 5, slope_limit=2.0))
        self.lift8_4 = RayLiftIDBlock(fpn_channels, fpn_channels, RayLiftSpec("line", 3, slope_limit=1.5))
        self.lift4_2 = RayLiftIDBlock(fpn_channels, 12, RayLiftSpec("line", 3, slope_limit=1.0))
        self.lift2_1 = RayLiftIDBlock(12, 4, RayLiftSpec("neighbor", 2, slope_limit=0.5), final=True)
        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.normal_(self.initial_depth[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.initial_depth[-1].bias, math.log(math.expm1(20.0)))
        nn.init.normal_(self.initial_confidence[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.initial_confidence[-1].bias)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GeoLiftStudentS2":
        model_cfg = cfg.get("model", {})
        sparse_cfg = cfg.get("sparse_propagation", {})
        loss_cfg = cfg.get("loss", {})
        student_cfg = cfg.get("student", {})
        return cls(
            encoder=str(model_cfg.get("encoder", "mobilevitv2_0.75")),
            fusion_channels=int(model_cfg.get("fusion_channels", 32)),
            fpn_channels=int(model_cfg.get("fpn_channels", 24)),
            sparse_scale=int(sparse_cfg.get("scale", 4)),
            sparse_radius=int(sparse_cfg.get("radius", 7)),
            min_depth=float(loss_cfg.get("min_depth", 1e-3)),
            max_depth=float(student_cfg.get("max_depth", loss_cfg.get("max_depth", 120.0))),
        )

    def _intrinsics_from_ray(self, ray: torch.Tensor) -> torch.Tensor:
        b, _, h, w = ray.shape
        if w < 2 or h < 2:
            raise ValueError("K is required for degenerate ray maps")
        dx = (ray[:, 0, 0, 1] - ray[:, 0, 0, 0]).clamp_min(1e-7)
        dy = (ray[:, 1, 1, 0] - ray[:, 1, 0, 0]).clamp_min(1e-7)
        fx, fy = dx.reciprocal(), dy.reciprocal()
        cx, cy = -ray[:, 0, 0, 0] * fx, -ray[:, 1, 0, 0] * fy
        K = torch.zeros((b, 3, 3), device=ray.device, dtype=ray.dtype)
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, cx, cy, 1.0
        return K

    def forward(
        self,
        rgb: torch.Tensor,
        sparse: torch.Tensor,
        mask: torch.Tensor,
        ray: torch.Tensor,
        uv: torch.Tensor,
        K: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del uv
        if rgb.shape[-2] % 16 or rgb.shape[-1] % 16:
            raise ValueError(f"GeoLift-S2 needs H/W divisible by 16, got {rgb.shape[-2:]}")
        K = self._intrinsics_from_ray(ray) if K is None else K
        full_hw = rgb.shape[-2:]
        sparse4, mask4, d_init4, valid4, density4 = self.sparse_prior(sparse, mask)
        depth_features = torch.cat(
            (
                sparse4.clamp(0.0, self.max_depth),
                mask4,
                d_init4.clamp(0.0, self.max_depth),
                valid4,
                density4,
            ),
            dim=1,
        )
        xy4 = _ray_xy(K, d_init4.shape[-2], d_init4.shape[-1], full_hw, rgb.dtype)
        fused4 = self.fusion(torch.cat((self.rgb_stem(rgb), self.depth_stem(depth_features), xy4), dim=1))
        f4, f8, f16 = self.encoder(fused4)
        p4, p8, p16 = self.fpn(f4, f8, f16)

        d16 = F.softplus(self.initial_depth(p16)).clamp(self.min_depth, self.max_depth)
        c16 = torch.sigmoid(self.initial_confidence(p16)).clamp(1e-4, 1.0)
        g2, q2_phase = self.ppg(rgb, sparse, mask, self.max_depth)

        g8_phase = phase_pack(p8)
        d8, c8, aux8 = self.lift16_8(
            d16, c16, p16, g8_phase, K, full_hw, self.min_depth, self.max_depth
        )
        g4_phase = phase_pack(p4)
        d4, c4, aux4 = self.lift8_4(d8, c8, p8, g4_phase, K, full_hw, self.min_depth, self.max_depth)

        g2_phase = phase_pack(g2)
        d2, c2, aux2 = self.lift4_2(d4, c4, p4, g2_phase, K, full_hw, self.min_depth, self.max_depth)

        source2 = self.source2(torch.cat((g2, d2 / self.max_depth, c2), dim=1))
        d1, c1, aux1 = self.lift2_1(d2, c2, source2, q2_phase, K, full_hw, self.min_depth, self.max_depth)
        d_pre_anchor = d1
        d_full = ((1.0 - mask) * d1 + mask * sparse.clamp(self.min_depth, self.max_depth)).clamp(
            self.min_depth, self.max_depth
        )
        c_full = ((1.0 - mask) * c1 + mask).clamp(1e-4, 1.0)

        output: dict[str, torch.Tensor] = {
            "D_full": d_full,
            "C_full": c_full,
            "D_pre_anchor": d_pre_anchor,
            "D1": d1,
            "C1": c1,
            "D2": d2,
            "C2": c2,
            "D4": d4,
            "C4": c4,
            "D8": d8,
            "C8": c8,
            "D16": d16,
            "C16": c16,
            "D_1_4": d4,
            "C_1_4": c4,
            "D_c": d4,
            "C": c4,
            "D_init": d_init4,
            "V_init": valid4,
            "rho4": density4,
            "log_var": -torch.log(c1.clamp_min(1e-6)),
        }
        for name, aux in (("8", aux8), ("4", aux4), ("2", aux2), ("1", aux1)):
            output.update({f"{key}_{name}": value for key, value in aux.items()})
        return output
