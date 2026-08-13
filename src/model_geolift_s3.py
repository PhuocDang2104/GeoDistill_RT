from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_geolift_s2 import (
    ConvBNAct,
    DWPointwise,
    RayLiftIDBlock,
    RayLiftSpec,
    ResidualDWBlock,
    compact_sparse_prior,
    phase_pack,
)


def _valid_pool(value: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    support = F.avg_pool2d(valid, 2, stride=2) * 4.0
    numerator = F.avg_pool2d(value * valid, 2, stride=2) * 4.0
    pooled = numerator / support.clamp_min(1.0)
    next_valid = (support > 0.0).to(value.dtype)
    return pooled * next_valid, next_valid


class MobileNetV4RGBEncoder(nn.Module):
    """RGB-only MobileNetV4-Conv features at reductions 4, 8 and 16."""

    def __init__(self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        import timm  # type: ignore

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=bool(pretrained),
                features_only=True,
                out_indices=(1, 2, 3),
            )
        except Exception as exc:  # pragma: no cover - download/registry dependent
            mode = "pretrained" if pretrained else "randomly initialized"
            raise RuntimeError(f"Could not construct {mode} RGB encoder {model_name}: {exc}") from exc
        info = self.backbone.feature_info.get_dicts()
        reductions = tuple(int(item["reduction"]) for item in info)
        if reductions != (4, 8, 16):
            raise RuntimeError(f"GeoLift-S3 requires encoder reductions (4,8,16), got {reductions}")
        self.out_channels = tuple(int(item["num_chs"]) for item in info)
        self.pretrained = bool(pretrained)

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(rgb)
        return features[0], features[1], features[2]


class CompactSparsePyramid(nn.Module):
    """Tiny sparse-only branch; no early RGB-depth input concatenation."""

    def __init__(self, scale: int = 4, radius: int = 7, widths: tuple[int, int, int] = (8, 8, 8)) -> None:
        super().__init__()
        if int(scale) != 4:
            raise ValueError("GeoLift-S3 sparse pyramid starts at 1/4 resolution")
        self.scale = int(scale)
        self.radius = int(radius)
        self.stages = nn.ModuleList(
            nn.Sequential(ConvBNAct(5, width, kernel=1), DWPointwise(width, width)) for width in widths
        )
        self.out_channels = widths

    @staticmethod
    def _state(
        sparse: torch.Tensor,
        mask: torch.Tensor,
        initial: torch.Tensor,
        initial_valid: torch.Tensor,
        density: torch.Tensor,
        max_depth: float,
    ) -> torch.Tensor:
        return torch.cat(
            (
                sparse.clamp(0.0, max_depth) / max_depth,
                mask,
                initial.clamp(0.0, max_depth) / max_depth,
                initial_valid,
                density,
            ),
            dim=1,
        )

    def forward(
        self, sparse: torch.Tensor, mask: torch.Tensor, max_depth: float
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        sparse4, mask4, initial4, valid4, density4 = compact_sparse_prior(
            sparse, mask, self.scale, self.radius
        )
        sparse8, mask8 = _valid_pool(sparse4, mask4)
        initial8, valid8 = _valid_pool(initial4, valid4)
        density8 = F.avg_pool2d(density4, 2, stride=2)
        sparse16, mask16 = _valid_pool(sparse8, mask8)
        initial16, valid16 = _valid_pool(initial8, valid8)
        density16 = F.avg_pool2d(density8, 2, stride=2)
        states = (
            self._state(sparse4, mask4, initial4, valid4, density4, max_depth),
            self._state(sparse8, mask8, initial8, valid8, density8, max_depth),
            self._state(sparse16, mask16, initial16, valid16, density16, max_depth),
        )
        features = tuple(stage(state) for stage, state in zip(self.stages, states))
        aux = {
            "D_init": initial4,
            "V_init": valid4,
            "rho4": density4,
            # Raw metric states are exposed for decoder ablations. They are
            # guidance only; hard anchoring remains a full-resolution operation.
            "S4": sparse4,
            "M4": mask4,
            "S8": sparse8,
            "M8": mask8,
            "rho8": density8,
        }
        return (features[0], features[1], features[2]), aux


class GatedScaleFusion(nn.Module):
    def __init__(self, rgb_ch: int, sparse_ch: int, out_ch: int) -> None:
        super().__init__()
        self.rgb_project = ConvBNAct(rgb_ch, out_ch, kernel=1)
        self.sparse_project = ConvBNAct(sparse_ch, out_ch, kernel=1)
        self.gate = nn.Conv2d(2 * out_ch, out_ch, 1)
        self.mix = ResidualDWBlock(out_ch)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.10 / 0.90))

    def forward(self, rgb: torch.Tensor, sparse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_feature = self.rgb_project(rgb)
        sparse_feature = self.sparse_project(sparse)
        gate = torch.sigmoid(self.gate(torch.cat((rgb_feature, sparse_feature), dim=1)))
        return self.mix(rgb_feature + gate * sparse_feature), gate


class LiteFPN322424(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int]) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.lat16 = nn.Conv2d(c16, 24, 1)
        self.lat8 = nn.Conv2d(c8, 24, 1)
        self.lat4 = nn.Conv2d(c4, 32, 1)
        self.up8_to4 = nn.Conv2d(24, 32, 1)
        self.smooth8 = ResidualDWBlock(24)
        self.smooth4 = ResidualDWBlock(32)

    def forward(self, f4: torch.Tensor, f8: torch.Tensor, f16: torch.Tensor) -> tuple[torch.Tensor, ...]:
        p16 = self.lat16(f16)
        p8 = self.smooth8(self.lat8(f8) + F.interpolate(p16, size=f8.shape[-2:], mode="nearest"))
        p4 = self.smooth4(
            self.lat4(f4) + self.up8_to4(F.interpolate(p8, size=f4.shape[-2:], mode="nearest"))
        )
        return p4, p8, p16


class ResidualMetricLift(nn.Module):
    """Cheap inverse-depth residual upsampling used only below 1/4 resolution."""

    def __init__(self, source_ch: int, target_ch: int, residual_limit: float = 0.02) -> None:
        super().__init__()
        self.residual_limit = float(residual_limit)
        self.trunk = nn.Sequential(
            ConvBNAct(source_ch + target_ch, 24, kernel=1),
            ConvBNAct(24, 24, kernel=3, groups=24),
            ConvBNAct(24, 16, kernel=1),
        )
        self.delta = nn.Conv2d(16, 1, 1)
        self.gate = nn.Conv2d(16, 1, 1)
        nn.init.normal_(self.delta.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.05 / 0.95))

    def forward(
        self,
        depth: torch.Tensor,
        confidence: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        min_depth: float,
        max_depth: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_up = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
        trunk = self.trunk(torch.cat((source_up, target), dim=1))
        xi_base = F.interpolate(depth.reciprocal(), size=target.shape[-2:], mode="bilinear", align_corners=False)
        delta = torch.tanh(self.delta(trunk)) * self.residual_limit
        gate = torch.sigmoid(self.gate(trunk))
        inverse = (xi_base + gate * delta).clamp(1.0 / max_depth, 1.0 / min_depth)
        depth_out = inverse.reciprocal().clamp(min_depth, max_depth)
        confidence_out = F.interpolate(confidence, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return depth_out, confidence_out.clamp(1e-4, 1.0), {"residual_delta": delta, "residual_gate": gate}


class LearnedPhaseGuidance(nn.Module):
    """Learned 1/2-resolution RGB guidance retaining the four full-resolution phases."""

    def __init__(self, channels: int = 12) -> None:
        super().__init__()
        if channels % 4:
            raise ValueError("Phase guidance channels must be divisible by four")
        self.project = nn.Sequential(
            nn.Conv2d(12, channels, 1, groups=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            ConvBNAct(channels, channels, kernel=3, groups=channels),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.project(phase_pack(rgb))


class GeoLiftStudentS3Lite(nn.Module):
    """MobileNetV4 RGB encoder with high-resolution-only Residual RayLift."""

    def __init__(
        self,
        encoder: str = "mobilenetv4_conv_small_050.e3000_r224_in1k",
        encoder_pretrained: bool = True,
        sparse_scale: int = 4,
        sparse_radius: int = 7,
        min_depth: float = 1e-3,
        max_depth: float = 120.0,
    ) -> None:
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.encoder_pretrained = bool(encoder_pretrained)
        self.encoder = MobileNetV4RGBEncoder(encoder, encoder_pretrained)
        self.sparse_pyramid = CompactSparsePyramid(sparse_scale, sparse_radius)
        fusion_widths = (32, 24, 24)
        self.fusion4 = GatedScaleFusion(self.encoder.out_channels[0], 8, fusion_widths[0])
        self.fusion8 = GatedScaleFusion(self.encoder.out_channels[1], 8, fusion_widths[1])
        self.fusion16 = GatedScaleFusion(self.encoder.out_channels[2], 8, fusion_widths[2])
        self.fpn = LiteFPN322424(fusion_widths)
        self.initial_depth = nn.Sequential(DWPointwise(24, 24), nn.Conv2d(24, 1, 1))
        self.lift16_8 = ResidualMetricLift(24, 24)
        self.lift8_4 = ResidualMetricLift(24, 32)
        self.phase_guidance = LearnedPhaseGuidance(12)
        self.source2 = nn.Sequential(ConvBNAct(14, 12, kernel=1), ResidualDWBlock(12))
        self.lift4_2 = RayLiftIDBlock(
            32,
            12,
            RayLiftSpec("line", 3, slope_limit=1.0),
            residual_correction=True,
            residual_limit=0.02,
        )
        self.lift2_1 = RayLiftIDBlock(
            12,
            3,
            RayLiftSpec("neighbor", 2, slope_limit=0.5),
            residual_correction=True,
            residual_limit=0.01,
        )
        nn.init.normal_(self.initial_depth[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.initial_depth[-1].bias, math.log(math.expm1(20.0)))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GeoLiftStudentS3Lite":
        model_cfg = cfg.get("model", {})
        sparse_cfg = cfg.get("sparse_propagation", {})
        loss_cfg = cfg.get("loss", {})
        student_cfg = cfg.get("student", {})
        return cls(
            encoder=str(model_cfg.get("encoder", "mobilenetv4_conv_small_050.e3000_r224_in1k")),
            encoder_pretrained=bool(model_cfg.get("encoder_pretrained", True)),
            sparse_scale=int(sparse_cfg.get("scale", 4)),
            sparse_radius=int(sparse_cfg.get("radius", 7)),
            min_depth=float(loss_cfg.get("min_depth", 1e-3)),
            max_depth=float(student_cfg.get("max_depth", loss_cfg.get("max_depth", 120.0))),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sparse: torch.Tensor,
        mask: torch.Tensor,
        ray: torch.Tensor,
        uv: torch.Tensor,
        K: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del ray, uv
        if K is None:
            raise ValueError("GeoLift-S3 Residual RayLift requires camera intrinsics K")
        if rgb.shape[-2] % 16 or rgb.shape[-1] % 16:
            raise ValueError(f"GeoLift-S3 needs H/W divisible by 16, got {rgb.shape[-2:]}")
        full_hw = rgb.shape[-2:]
        rgb4, rgb8, rgb16 = self.encoder(rgb)
        (sparse4, sparse8, sparse16), sparse_aux = self.sparse_pyramid(sparse, mask, self.max_depth)
        fused4, gate4 = self.fusion4(rgb4, sparse4)
        fused8, gate8 = self.fusion8(rgb8, sparse8)
        fused16, gate16 = self.fusion16(rgb16, sparse16)
        p4, p8, p16 = self.fpn(fused4, fused8, fused16)

        d16 = F.softplus(self.initial_depth(p16)).clamp(self.min_depth, self.max_depth)
        c16 = torch.ones_like(d16)
        d8, c8, aux8 = self.lift16_8(d16, c16, p16, p8, self.min_depth, self.max_depth)
        d4, c4, aux4 = self.lift8_4(d8, c8, p8, p4, self.min_depth, self.max_depth)

        guidance2 = self.phase_guidance(rgb)
        d2, c2, aux2 = self.lift4_2(
            d4, c4, p4, phase_pack(guidance2), K, full_hw, self.min_depth, self.max_depth
        )
        source2 = self.source2(torch.cat((guidance2, d2 / self.max_depth, c2), dim=1))
        d1, c1, aux1 = self.lift2_1(
            d2, c2, source2, guidance2, K, full_hw, self.min_depth, self.max_depth
        )
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
            **sparse_aux,
            "fusion_gate4": gate4,
            "fusion_gate8": gate8,
            "fusion_gate16": gate16,
            "residual_gate_8": aux8["residual_gate"],
            "residual_delta_8": aux8["residual_delta"],
            "residual_gate_4": aux4["residual_gate"],
            "residual_delta_4": aux4["residual_delta"],
        }
        for name, aux in (("2", aux2), ("1", aux1)):
            output[f"ray_gate_{name}"] = aux["gate"]
            output[f"residual_gate_{name}"] = aux["residual_gate"]
            output[f"residual_delta_{name}"] = aux["residual_delta"]
        return output
