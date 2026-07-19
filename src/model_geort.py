from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_propagation import fast_sparse_propagation


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
        pad = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class EfficientFusion(nn.Sequential):
    """Reduce concatenated modalities, then mix them depthwise-separably."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            ConvBNAct(in_ch, out_ch, kernel=1),
            ConvBNAct(out_ch, out_ch, kernel=3, groups=out_ch),
            ConvBNAct(out_ch, out_ch, kernel=1),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    """Spatial mixing with a depthwise 3x3 followed by channel mixing."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            ConvBNAct(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, kernel=1),
        )


class LightStem(nn.Sequential):
    """Keep the input contract while removing the expensive wide 3x3 layer."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            ConvBNAct(in_ch, out_ch, kernel=3),
            DepthwiseSeparableBlock(out_ch, out_ch),
        )


class TimmMobileViTEncoder(nn.Module):
    def __init__(self, model_name: str, in_ch: int, e4: int, e8: int, e16: int) -> None:
        super().__init__()
        import timm  # type: ignore

        aliases = {
            "mobilevitv2_0.75": ["mobilevitv2_075.cvnets_in1k", "mobilevitv2_075"],
            "mobilevitv2_075": ["mobilevitv2_075.cvnets_in1k", "mobilevitv2_075"],
        }
        names = aliases.get(model_name, [model_name])
        last_error: Exception | None = None
        model = None
        for name in names:
            try:
                model = timm.create_model(name, pretrained=False, in_chans=in_ch)
                break
            except Exception as exc:
                last_error = exc
        if model is None:
            raise RuntimeError(f"Could not create timm MobileViTv2 encoder: {last_error}")
        if not hasattr(model, "stem") or not hasattr(model, "stages") or not isinstance(model.feature_info, list):
            raise RuntimeError(f"Unsupported timm encoder structure for early exit: {type(model).__name__}")

        feature_info = {int(item["reduction"]): item for item in model.feature_info}
        requested = [(4, e4), (8, e8), (16, e16)]
        missing = [reduction for reduction, _ in requested if reduction not in feature_info]
        if missing:
            raise RuntimeError(f"timm model lacks reductions {missing}; available={sorted(feature_info)}")

        self.stem = model.stem
        self.output_stage_indices = []
        self.proj = nn.ModuleList()
        for reduction, out_ch in requested:
            info = feature_info[reduction]
            module_name = str(info["module"])
            if not module_name.startswith("stages."):
                raise RuntimeError(f"Reduction {reduction} is not a stage output: {module_name}")
            self.output_stage_indices.append(int(module_name.split(".")[1]))
            self.proj.append(nn.Conv2d(int(info["num_chs"]), out_ch, kernel_size=1))
        last_stage = max(self.output_stage_indices)
        self.stages = nn.ModuleList(list(model.stages)[: last_stage + 1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        stage_outputs: dict[int, torch.Tensor] = {}
        requested = set(self.output_stage_indices)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in requested:
                stage_outputs[index] = x
        out = [proj(stage_outputs[index]) for index, proj in zip(self.output_stage_indices, self.proj)]
        return out[0], out[1], out[2]


class SparseRayInjection(nn.Module):
    def __init__(self, channels: int, prior_ch: int = 6) -> None:
        super().__init__()
        self.prior = nn.Conv2d(prior_ch, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, feat: torch.Tensor, prior_full: torch.Tensor) -> torch.Tensor:
        prior = F.interpolate(prior_full, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        prior = self.prior(prior)
        gate = torch.sigmoid(self.gate(torch.cat([feat, prior], dim=1)))
        return feat + gate * prior


class GuidedConvexUpsample(nn.Module):
    """Full-resolution guided convex upsampling over a 3x3 local window."""

    def __init__(
        self,
        guide_channels: int = 6,
        hidden: int = 24,
        kernel_size: int = 3,
        max_depth: float = 120.0,
        value_scale: float = 120.0,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.max_depth = float(max_depth)
        self.value_scale = float(value_scale)
        self.weight_head = nn.Sequential(
            ConvBNAct(guide_channels, hidden, kernel=3),
            nn.Conv2d(hidden, self.kernel_size * self.kernel_size, kernel_size=1),
        )

    def forward(self, value_coarse: torch.Tensor, rgb: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, w = rgb.shape[-2:]
        value_up = F.interpolate(value_coarse, size=(h, w), mode="bilinear", align_corners=False)
        log_max = torch.log(torch.tensor(self.max_depth + 1.0, device=sparse.device, dtype=sparse.dtype))
        sparse_norm = torch.log1p(sparse.clamp(0.0, self.max_depth)) / log_max
        value_norm = (value_up / max(self.value_scale, 1e-6)).clamp(0.0, 1.0)
        guide = torch.cat([rgb, sparse_norm, mask, value_norm], dim=1)
        weights = torch.softmax(self.weight_head(guide), dim=1)
        patches = F.unfold(value_up, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.view(value_up.shape[0], 1, self.kernel_size * self.kernel_size, h, w)
        return (patches * weights[:, None]).sum(dim=2)


class AdaptiveSparseAnchor(nn.Module):
    """Learn sensor anchoring from confidence, discrepancy, density and RGB edges.

    The last layer is initialized so the module exactly starts from the former
    fixed correction coefficient. It can then learn to reject inconsistent or
    misaligned sparse samples during training.
    """

    def __init__(self, lambda_min: float = 0.5, lambda_init: float = 0.7, hidden: int = 8) -> None:
        super().__init__()
        if not 0.0 <= lambda_min < 1.0:
            raise ValueError("lambda_min must be in [0, 1)")
        if not lambda_min <= lambda_init < 1.0:
            raise ValueError("lambda_init must be in [lambda_min, 1)")
        self.lambda_min = float(lambda_min)
        self.head = nn.Sequential(
            ConvBNAct(4, hidden, kernel=1),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        normalized_init = (float(lambda_init) - self.lambda_min) / (1.0 - self.lambda_min)
        bias = math.log(normalized_init / (1.0 - normalized_init))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, bias)

    @staticmethod
    def _rgb_edge(rgb: torch.Tensor) -> torch.Tensor:
        gray = rgb.mean(dim=1, keepdim=True)
        dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
        return (dx + dy).clamp(0.0, 1.0)

    def forward(
        self,
        depth: torch.Tensor,
        confidence: torch.Tensor,
        sparse: torch.Tensor,
        mask: torch.Tensor,
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sparse_safe = sparse.clamp_min(1e-3)
        discrepancy = ((sparse_safe - depth).abs() / sparse_safe).clamp(0.0, 2.0) * mask
        density = F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2)
        edge = self._rgb_edge(rgb)
        features = torch.cat([confidence, discrepancy, density, edge], dim=1)
        learned = torch.sigmoid(self.head(features))
        anchor_lambda = mask * (self.lambda_min + (1.0 - self.lambda_min) * learned)
        anchored = depth + anchor_lambda * (sparse_safe - depth)
        sensor_confidence = mask
        confidence_out = (1.0 - mask) * confidence + mask * torch.maximum(confidence, sensor_confidence)
        return anchored, confidence_out, anchor_lambda


@dataclass
class GeoRTConfig:
    encoder: str = "mobilevitv2_0.75"
    fusion_channels: int = 32
    e4_channels: int = 48
    e8_channels: int = 72
    e16_channels: int = 128
    sparse_scale: int = 4
    sparse_k: int = 4
    sparse_mode: str = "local"
    max_depth: float = 120.0
    sparse_anchor_lambda: float = 0.7


class GeoRTStudentS(nn.Module):
    """GeoRT-Student-S.

    Inputs:
      rgb: [B,3,H,W]
      sparse: [B,1,H,W]
      mask: [B,1,H,W]
      ray: [B,3,H,W]
      uv: [B,2,H,W]

    Outputs:
      D_full/C_full: [B,1,H,W] official inference outputs.
      D_1_4/C_1_4: [B,1,H/4,W/4] internal coarse outputs.
      D_c/C are backward-compatible aliases for D_1_4/C_1_4.
    """

    def __init__(
        self,
        encoder: str = "mobilevitv2_0.75",
        fusion_channels: int = 32,
        e4_channels: int = 48,
        e8_channels: int = 72,
        e16_channels: int = 128,
        sparse_scale: int = 4,
        sparse_k: int = 4,
        sparse_mode: str = "local",
        max_depth: float = 120.0,
        sparse_anchor_lambda: float = 0.7,
    ) -> None:
        super().__init__()
        self.sparse_scale = int(sparse_scale)
        self.sparse_k = int(sparse_k)
        self.sparse_mode = str(sparse_mode)
        self.max_depth = float(max_depth)
        self.sparse_anchor_lambda = float(sparse_anchor_lambda)
        self.eps = 1e-3

        self.rgb_stem = LightStem(3, 24)
        self.depth_stem = LightStem(3, 16)
        self.ray_stem = LightStem(5, 12)
        self.fusion = EfficientFusion(24 + 16 + 12, fusion_channels)

        self.encoder = TimmMobileViTEncoder(encoder, fusion_channels, e4_channels, e8_channels, e16_channels)

        self.inject4 = SparseRayInjection(e4_channels)
        self.inject8 = SparseRayInjection(e8_channels)
        self.inject16 = SparseRayInjection(e16_channels)

        fpn_ch = fusion_channels
        self.lat16 = nn.Conv2d(e16_channels, fpn_ch, kernel_size=1)
        self.lat8 = nn.Conv2d(e8_channels, fpn_ch, kernel_size=1)
        self.lat4 = nn.Conv2d(e4_channels, fpn_ch, kernel_size=1)
        self.smooth8 = DepthwiseSeparableBlock(fpn_ch, fpn_ch)
        self.smooth4 = DepthwiseSeparableBlock(fpn_ch, fpn_ch)

        self.depth_head = nn.Sequential(DepthwiseSeparableBlock(fpn_ch, fpn_ch), nn.Conv2d(fpn_ch, 1, kernel_size=1))
        self.conf_head = nn.Sequential(DepthwiseSeparableBlock(fpn_ch, fpn_ch), nn.Conv2d(fpn_ch, 1, kernel_size=1))
        self.depth_up = GuidedConvexUpsample(
            guide_channels=6,
            hidden=24,
            kernel_size=3,
            max_depth=self.max_depth,
            value_scale=self.max_depth,
        )
        self.conf_up = GuidedConvexUpsample(
            guide_channels=6,
            hidden=16,
            kernel_size=3,
            max_depth=self.max_depth,
            value_scale=1.0,
        )
        self.full_residual = nn.Sequential(
            ConvBNAct(7, 16, kernel=3),
            ConvBNAct(16, 16, kernel=3, groups=16),
            ConvBNAct(16, 8, kernel=1),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        self.sparse_anchor = AdaptiveSparseAnchor(
            lambda_min=min(0.5, self.sparse_anchor_lambda),
            lambda_init=self.sparse_anchor_lambda,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "GeoRTStudentS":
        model_cfg = cfg.get("model", {})
        sparse_cfg = cfg.get("sparse_propagation", {})
        student_cfg = cfg.get("student", {})
        return cls(
            encoder=model_cfg.get("encoder", "mobilevitv2_0.75"),
            fusion_channels=int(model_cfg.get("fusion_channels", 32)),
            e4_channels=int(model_cfg.get("e4_channels", 48)),
            e8_channels=int(model_cfg.get("e8_channels", 72)),
            e16_channels=int(model_cfg.get("e16_channels", 128)),
            sparse_scale=int(sparse_cfg.get("scale", 4)),
            sparse_k=int(sparse_cfg.get("k", 4)),
            sparse_mode=str(sparse_cfg.get("mode", "local")),
            max_depth=float(student_cfg.get("max_depth", model_cfg.get("max_depth", 120.0))),
            sparse_anchor_lambda=float(student_cfg.get("sparse_anchor_lambda", 0.7)),
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
        del K  # Kept for a common factory/trainer call signature.
        # D_init: [B,1,H/4,W/4], metric prior from sparse LiDAR only.
        D_init = fast_sparse_propagation(
            rgb,
            sparse,
            mask,
            ray,
            scale=self.sparse_scale,
            k=self.sparse_k,
            mode=self.sparse_mode,
        )
        D_init_full = F.interpolate(D_init, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        log_max = torch.log(torch.tensor(self.max_depth + 1.0, device=rgb.device, dtype=rgb.dtype))
        log_sparse = torch.log1p(sparse.clamp(0.0, self.max_depth)) / log_max * mask
        D_init_feat = torch.log1p(D_init_full.clamp(0.0, self.max_depth)) / log_max
        depth_in = torch.cat([log_sparse, mask, D_init_feat], dim=1)  # [B,3,H,W]
        geom_in = torch.cat([ray, uv], dim=1)  # [B,5,H,W]

        F_rgb = self.rgb_stem(rgb)  # [B,24,H,W]
        F_depth = self.depth_stem(depth_in)  # [B,16,H,W]
        F_ray = self.ray_stem(geom_in)  # [B,12,H,W]
        fused = self.fusion(torch.cat([F_rgb, F_depth, F_ray], dim=1))  # [B,C,H,W]

        E4, E8, E16 = self.encoder(fused)
        prior_full = torch.cat([log_sparse, mask, D_init_feat, ray], dim=1)  # [B,6,H,W]
        E4 = self.inject4(E4, prior_full)
        E8 = self.inject8(E8, prior_full)
        E16 = self.inject16(E16, prior_full)

        P16 = self.lat16(E16)
        P8 = self.lat8(E8) + F.interpolate(P16, size=E8.shape[-2:], mode="nearest")
        P8 = self.smooth8(P8)
        P4 = self.lat4(E4) + F.interpolate(P8, size=E4.shape[-2:], mode="nearest")
        P4 = self.smooth4(P4)

        delta_z_1_4 = self.depth_head(P4).clamp(min=-4.0, max=4.0)
        D_1_4 = (D_init.clamp_min(self.eps) * torch.exp(delta_z_1_4)).clamp(self.eps, self.max_depth)

        s_1_4 = F.softplus(self.conf_head(P4))
        C_1_4 = torch.exp(-s_1_4).clamp(1e-4, 1.0)

        D_up = self.depth_up(D_1_4, rgb, sparse, mask).clamp(self.eps, self.max_depth)
        C_up = self.conf_up(C_1_4, rgb, sparse, mask).clamp(1e-4, 1.0)

        sparse_scaled = sparse.clamp(0.0, self.max_depth) / self.max_depth
        D_up_scaled = D_up / self.max_depth
        refine_in = torch.cat([rgb, sparse_scaled, mask, D_up_scaled, C_up], dim=1)
        delta_z_full = self.full_residual(refine_in).clamp(min=-0.5, max=0.5)
        D_pre_anchor = (D_up * torch.exp(delta_z_full)).clamp(self.eps, self.max_depth)
        D_full, C_full, anchor_lambda = self.sparse_anchor(
            D_pre_anchor,
            C_up.clamp(1e-4, 1.0),
            sparse.clamp(self.eps, self.max_depth),
            mask,
            rgb,
        )
        D_full = D_full.clamp(self.eps, self.max_depth)
        C_full = C_full.clamp(1e-4, 1.0)

        return {
            "D_full": D_full,
            "C_full": C_full,
            "D_1_4": D_1_4,
            "C_1_4": C_1_4,
            "D_init": D_init,
            "D_up": D_up,
            "delta_z_1_4": delta_z_1_4,
            "delta_z_full": delta_z_full,
            "D_pre_anchor": D_pre_anchor,
            "anchor_lambda": anchor_lambda,
            "D_c": D_1_4,
            "C": C_1_4,
            "log_var": -torch.log(C_1_4.clamp_min(1e-6)),
        }
