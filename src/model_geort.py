from __future__ import annotations

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


class InvertedResidual(nn.Module):
    def __init__(self, channels: int, expand_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expand_ratio)
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, kernel=1),
            ConvBNAct(hidden, hidden, kernel=3, groups=hidden),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class LinearAttention2d(nn.Module):
    """Small MobileViTv2-like linear attention block for fallback encoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(channels, 1, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = torch.softmax(self.q(x).reshape(B, 1, H * W), dim=-1)  # [B,1,HW]
        k = self.k(x).reshape(B, C, H * W)
        v = self.v(x).reshape(B, C, H * W)
        context = torch.sum(q * k, dim=-1, keepdim=True)  # [B,C,1]
        out = (F.silu(v) * context).reshape(B, C, H, W)
        return x + self.proj(out)


class LiteMobileViTBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(ConvBNAct(channels, channels, kernel=3, groups=channels), ConvBNAct(channels, channels, kernel=1))
        self.attn = LinearAttention2d(channels)
        self.ffn = InvertedResidual(channels, expand_ratio=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        x = self.attn(x)
        return self.ffn(x)


class FallbackMobileViTEncoder(nn.Module):
    """Fallback encoder returning E4/E8/E16 with requested channels."""

    def __init__(self, in_ch: int, e4: int, e8: int, e16: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_ch, 32, stride=2)  # [B,32,H/2,W/2]
        self.stage4 = nn.Sequential(ConvBNAct(32, e4, stride=2), InvertedResidual(e4), LiteMobileViTBlock(e4))
        self.stage8 = nn.Sequential(ConvBNAct(e4, e8, stride=2), InvertedResidual(e8), LiteMobileViTBlock(e8))
        self.stage16 = nn.Sequential(ConvBNAct(e8, e16, stride=2), InvertedResidual(e16), LiteMobileViTBlock(e16))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        E4 = self.stage4(x)  # [B,e4,H/4,W/4]
        E8 = self.stage8(E4)  # [B,e8,H/8,W/8]
        E16 = self.stage16(E8)  # [B,e16,H/16,W/16]
        return E4, E8, E16


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
                model = timm.create_model(name, pretrained=False, features_only=True, in_chans=in_ch)
                break
            except Exception as exc:
                last_error = exc
        if model is None:
            raise RuntimeError(f"Could not create timm MobileViTv2 encoder: {last_error}")
        self.model = model
        reductions = list(self.model.feature_info.reduction())
        channels = list(self.model.feature_info.channels())
        self.indices = []
        self.proj = nn.ModuleList()
        for reduction, out_ch in [(4, e4), (8, e8), (16, e16)]:
            if reduction not in reductions:
                raise RuntimeError(f"timm model lacks reduction {reduction}; reductions={reductions}")
            idx = reductions.index(reduction)
            self.indices.append(idx)
            self.proj.append(nn.Conv2d(channels[idx], out_ch, kernel_size=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.model(x)
        out = [proj(feats[idx]) for idx, proj in zip(self.indices, self.proj)]
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


@dataclass
class GeoRTConfig:
    encoder: str = "mobilevitv2_0.75"
    fusion_channels: int = 48
    e4_channels: int = 48
    e8_channels: int = 72
    e16_channels: int = 128
    sparse_scale: int = 4
    sparse_k: int = 4


class GeoRTStudentS(nn.Module):
    """GeoRT-Student-S.

    Inputs:
      rgb: [B,3,H,W]
      sparse: [B,1,H,W]
      mask: [B,1,H,W]
      ray: [B,3,H,W]
      uv: [B,2,H,W]

    Outputs:
      D_c: [B,1,H/4,W/4]
      C: [B,1,H/4,W/4]
      log_var: [B,1,H/4,W/4]
    """

    def __init__(
        self,
        encoder: str = "mobilevitv2_0.75",
        fusion_channels: int = 48,
        e4_channels: int = 48,
        e8_channels: int = 72,
        e16_channels: int = 128,
        sparse_scale: int = 4,
        sparse_k: int = 4,
    ) -> None:
        super().__init__()
        self.sparse_scale = int(sparse_scale)
        self.sparse_k = int(sparse_k)
        self.eps = 1e-3

        self.rgb_stem = nn.Sequential(ConvBNAct(3, 24), ConvBNAct(24, 24))
        self.depth_stem = nn.Sequential(ConvBNAct(3, 16), ConvBNAct(16, 16))
        self.ray_stem = nn.Sequential(ConvBNAct(5, 12), ConvBNAct(12, 12))
        self.fusion = nn.Sequential(ConvBNAct(24 + 16 + 12, fusion_channels), ConvBNAct(fusion_channels, fusion_channels))

        try:
            self.encoder = TimmMobileViTEncoder(encoder, fusion_channels, e4_channels, e8_channels, e16_channels)
        except Exception:
            self.encoder = FallbackMobileViTEncoder(fusion_channels, e4_channels, e8_channels, e16_channels)

        self.inject4 = SparseRayInjection(e4_channels)
        self.inject8 = SparseRayInjection(e8_channels)
        self.inject16 = SparseRayInjection(e16_channels)

        fpn_ch = fusion_channels
        self.lat16 = nn.Conv2d(e16_channels, fpn_ch, kernel_size=1)
        self.lat8 = nn.Conv2d(e8_channels, fpn_ch, kernel_size=1)
        self.lat4 = nn.Conv2d(e4_channels, fpn_ch, kernel_size=1)
        self.smooth8 = ConvBNAct(fpn_ch, fpn_ch, kernel=3)
        self.smooth4 = ConvBNAct(fpn_ch, fpn_ch, kernel=3)

        self.depth_head = nn.Sequential(ConvBNAct(fpn_ch, fpn_ch, kernel=3), nn.Conv2d(fpn_ch, 1, kernel_size=1))
        self.conf_head = nn.Sequential(ConvBNAct(fpn_ch, fpn_ch, kernel=3), nn.Conv2d(fpn_ch, 1, kernel_size=1))

    @classmethod
    def from_config(cls, cfg: dict) -> "GeoRTStudentS":
        model_cfg = cfg.get("model", {})
        sparse_cfg = cfg.get("sparse_propagation", {})
        return cls(
            encoder=model_cfg.get("encoder", "mobilevitv2_0.75"),
            fusion_channels=int(model_cfg.get("fusion_channels", 48)),
            e4_channels=int(model_cfg.get("e4_channels", 48)),
            e8_channels=int(model_cfg.get("e8_channels", 72)),
            e16_channels=int(model_cfg.get("e16_channels", 128)),
            sparse_scale=int(sparse_cfg.get("scale", 4)),
            sparse_k=int(sparse_cfg.get("k", 4)),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sparse: torch.Tensor,
        mask: torch.Tensor,
        ray: torch.Tensor,
        uv: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # D_init: [B,1,H/4,W/4], metric prior from sparse LiDAR only.
        D_init = fast_sparse_propagation(rgb, sparse, mask, ray, scale=self.sparse_scale, k=self.sparse_k)
        D_init_full = F.interpolate(D_init, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        log_sparse = torch.log(sparse.clamp_min(self.eps)) * mask
        depth_in = torch.cat([log_sparse, mask, D_init_full], dim=1)  # [B,3,H,W]
        geom_in = torch.cat([ray, uv], dim=1)  # [B,5,H,W]

        F_rgb = self.rgb_stem(rgb)  # [B,24,H,W]
        F_depth = self.depth_stem(depth_in)  # [B,16,H,W]
        F_ray = self.ray_stem(geom_in)  # [B,12,H,W]
        fused = self.fusion(torch.cat([F_rgb, F_depth, F_ray], dim=1))  # [B,C,H,W]

        E4, E8, E16 = self.encoder(fused)
        prior_full = torch.cat([log_sparse, mask, D_init_full, ray], dim=1)  # [B,6,H,W]
        E4 = self.inject4(E4, prior_full)
        E8 = self.inject8(E8, prior_full)
        E16 = self.inject16(E16, prior_full)

        P16 = self.lat16(E16)
        P8 = self.lat8(E8) + F.interpolate(P16, size=E8.shape[-2:], mode="nearest")
        P8 = self.smooth8(P8)
        P4 = self.lat4(E4) + F.interpolate(P8, size=E4.shape[-2:], mode="nearest")
        P4 = self.smooth4(P4)

        log_depth = self.depth_head(P4).clamp(min=-4.0, max=5.5)
        log_var = self.conf_head(P4).clamp(min=-8.0, max=8.0)
        D_c = torch.exp(log_depth)
        C = torch.exp(-log_var)
        return {"D_c": D_c, "C": C, "log_var": log_var}
