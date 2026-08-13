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
    _ray_xy,
    phase_pack,
    phase_unpack,
)
from .model_geolift_s3 import (
    CompactSparsePyramid,
    GatedScaleFusion,
    LearnedPhaseGuidance,
    LiteFPN322424,
    MobileNetV4RGBEncoder,
    ResidualMetricLift,
)


class MobileNetV4RGBEncoderF2(MobileNetV4RGBEncoder):
    """Checkpoint-compatible MobileNetV4 view exposing reductions 2/4/8/16."""

    def __init__(self, model_name: str, pretrained: bool) -> None:
        nn.Module.__init__(self)
        import timm  # type: ignore

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=bool(pretrained),
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception as exc:  # pragma: no cover - registry/download dependent
            mode = "pretrained" if pretrained else "randomly initialized"
            raise RuntimeError(f"Could not construct {mode} RGB encoder {model_name}: {exc}") from exc
        info = self.backbone.feature_info.get_dicts()
        reductions = tuple(int(item["reduction"]) for item in info)
        if reductions != (2, 4, 8, 16):
            raise RuntimeError(f"GeoLift Decoder V2 requires encoder reductions (2,4,8,16), got {reductions}")
        channels = tuple(int(item["num_chs"]) for item in info)
        self.f2_channels = channels[0]
        self.out_channels = channels[1:]
        self.pretrained = bool(pretrained)

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f2, f4, f8, f16 = self.backbone(rgb)
        return f2, f4, f8, f16


class EnhancedMetricLift8to4(ResidualMetricLift):
    """Old 8→4 lift plus no-op direct-depth and sparse-error correction.

    Inheriting ResidualMetricLift preserves all old parameter names, so an S3
    checkpoint restores the original path exactly. New heads start as an exact
    no-op and can be learned during the short decoder fine-tune.
    """

    def __init__(
        self,
        source_ch: int,
        target_ch: int,
        metric_residual_limit: float = 5.0,
        context_ch: int = 8,
    ) -> None:
        super().__init__(source_ch, target_ch)
        self.metric_residual_limit = float(metric_residual_limit)
        # Z8: existing decoder state. Sparse error enters through a separate
        # zero-initialized projection, making checkpoint warm-start exact.
        existing_ch = source_ch + 4 * target_ch + 4  # D8, C8 and ray xy
        self.metric_context = nn.Sequential(
            nn.Conv2d(existing_ch, context_ch, 1, bias=False),
            nn.BatchNorm2d(context_ch),
        )
        self.sparse_metric_project = nn.Conv2d(2, context_ch, 1, bias=False)
        self.metric_delta = nn.Conv2d(context_ch, 4, 1)
        self.metric_gate = nn.Conv2d(context_ch, 4, 1)
        nn.init.normal_(self.sparse_metric_project.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.metric_delta.weight)
        nn.init.zeros_(self.metric_delta.bias)
        nn.init.zeros_(self.metric_gate.weight)
        nn.init.constant_(self.metric_gate.bias, math.log(0.05 / 0.95))

    def forward(
        self,
        depth: torch.Tensor,
        confidence: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        sparse8: torch.Tensor,
        mask8: torch.Tensor,
        K: torch.Tensor,
        full_hw: tuple[int, int],
        min_depth: float,
        max_depth: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        depth_raylift, confidence_out, base_aux = super().forward(
            depth, confidence, source, target, min_depth, max_depth
        )
        if sparse8.shape[-2:] != depth.shape[-2:] or mask8.shape[-2:] != depth.shape[-2:]:
            raise ValueError("S8/M8 must match D8 for sparse metric-error injection")
        sparse_error8 = mask8 * (sparse8 - depth)
        ray8 = _ray_xy(K, depth.shape[-2], depth.shape[-1], full_hw, depth.dtype)
        existing_state = torch.cat(
            (
                source,
                phase_pack(target),
                depth / max_depth,
                confidence,
                ray8,
            ),
            dim=1,
        )
        sparse_state = torch.cat((mask8, sparse_error8 / max_depth), dim=1)
        context = F.silu(self.metric_context(existing_state) + self.sparse_metric_project(sparse_state))
        delta_phase = torch.tanh(self.metric_delta(context)) * self.metric_residual_limit
        gate_phase = torch.sigmoid(self.metric_gate(context))
        metric_delta = phase_unpack(delta_phase)
        metric_gate = phase_unpack(gate_phase)
        depth_out = (depth_raylift + metric_gate * metric_delta).clamp(min_depth, max_depth)
        return depth_out, confidence_out, {
            **base_aux,
            "metric_delta": metric_delta,
            "metric_gate": metric_gate,
            "sparse_error8": sparse_error8,
        }


class GeoLiftDecoderV2(nn.Module):
    """S3 encoder with an enhanced 8→4 metric stage and reused encoder F2."""

    def __init__(
        self,
        encoder: str = "mobilenetv4_conv_small_050.e3000_r224_in1k",
        encoder_pretrained: bool = True,
        sparse_scale: int = 4,
        sparse_radius: int = 7,
        min_depth: float = 1e-3,
        max_depth: float = 120.0,
        metric_residual_limit: float = 5.0,
    ) -> None:
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.encoder_pretrained = bool(encoder_pretrained)
        self.encoder = MobileNetV4RGBEncoderF2(encoder, encoder_pretrained)
        self.sparse_pyramid = CompactSparsePyramid(sparse_scale, sparse_radius)
        fusion_widths = (32, 24, 24)
        self.fusion4 = GatedScaleFusion(self.encoder.out_channels[0], 8, fusion_widths[0])
        self.fusion8 = GatedScaleFusion(self.encoder.out_channels[1], 8, fusion_widths[1])
        self.fusion16 = GatedScaleFusion(self.encoder.out_channels[2], 8, fusion_widths[2])
        self.fpn = LiteFPN322424(fusion_widths)
        self.initial_depth = nn.Sequential(DWPointwise(24, 24), nn.Conv2d(24, 1, 1))
        self.lift16_8 = ResidualMetricLift(24, 24)
        self.lift8_4 = EnhancedMetricLift8to4(24, 32, metric_residual_limit=metric_residual_limit)
        self.phase_guidance = LearnedPhaseGuidance(12)
        self.f2_project = nn.Sequential(
            nn.Conv2d(self.encoder.f2_channels, 12, 1, groups=4, bias=False),
            nn.BatchNorm2d(12),
        )
        self.f2_alpha = nn.Parameter(torch.zeros(()))
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
    def from_config(cls, cfg: dict[str, Any]) -> "GeoLiftDecoderV2":
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
            metric_residual_limit=float(model_cfg.get("metric_residual_limit", 5.0)),
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
            raise ValueError("GeoLift Decoder V2 requires camera intrinsics K")
        if rgb.shape[-2] % 16 or rgb.shape[-1] % 16:
            raise ValueError(f"GeoLift Decoder V2 needs H/W divisible by 16, got {rgb.shape[-2:]}")
        full_hw = rgb.shape[-2:]
        rgb2, rgb4, rgb8, rgb16 = self.encoder(rgb)
        (sparse4, sparse8, sparse16), sparse_aux = self.sparse_pyramid(sparse, mask, self.max_depth)
        fused4, gate4 = self.fusion4(rgb4, sparse4)
        fused8, gate8 = self.fusion8(rgb8, sparse8)
        fused16, gate16 = self.fusion16(rgb16, sparse16)
        p4, p8, p16 = self.fpn(fused4, fused8, fused16)

        d16 = F.softplus(self.initial_depth(p16)).clamp(self.min_depth, self.max_depth)
        c16 = torch.ones_like(d16)
        d8, c8, aux8 = self.lift16_8(d16, c16, p16, p8, self.min_depth, self.max_depth)
        d4, c4, aux4 = self.lift8_4(
            d8,
            c8,
            p8,
            p4,
            sparse_aux["S8"],
            sparse_aux["M8"],
            K,
            full_hw,
            self.min_depth,
            self.max_depth,
        )

        base_guidance2 = self.phase_guidance(rgb)
        f2_guidance = self.f2_project(rgb2)
        f2_alpha = torch.tanh(self.f2_alpha)
        guidance2 = base_guidance2 + f2_alpha * f2_guidance
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
            "metric_gate_4": aux4["metric_gate"],
            "metric_delta_4": aux4["metric_delta"],
            "sparse_error_8": aux4["sparse_error8"],
            "f2_alpha": f2_alpha.reshape(1),
        }
        for name, aux in (("2", aux2), ("1", aux1)):
            output[f"ray_gate_{name}"] = aux["gate"]
            output[f"residual_gate_{name}"] = aux["residual_gate"]
            output[f"residual_delta_{name}"] = aux["residual_delta"]
        return output
