from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..utils import ensure_dir, save_npz_atomic


MODEL_NAME_MAP = {
    "metric3d": "metric3d_vit_large",
    "metric3dv2": "metric3d_vit_large",
    "metric3dv2_vit_small": "metric3d_vit_small",
    "metric3dv2_vit_large": "metric3d_vit_large",
    "metric3dv2_vit_giant2": "metric3d_vit_giant2",
    "metric3d_vit_small": "metric3d_vit_small",
    "metric3d_vit_large": "metric3d_vit_large",
    "metric3d_vit_giant2": "metric3d_vit_giant2",
    "metric3d_convnext_tiny": "metric3d_convnext_tiny",
    "metric3d_convnext_large": "metric3d_convnext_large",
}


class Metric3DWrapper:
    """Real Metric3D / Metric3Dv2 inference wrapper.

    Input RGB is expected as uint8 RGB [H,W,3]. Metric3D official
    preprocessing uses ImageNet mean/std in 0-255 RGB space, ratio-preserving
    resize, mean-value padding, and a de-canonical focal-length scale.

    Output key when saving:
      D_m3d: float32 metric depth [H,W]
    """

    def __init__(
        self,
        repo_dir: str | Path,
        weights_dir: str | Path,
        model_name: str = "metric3dv2",
        device: str | torch.device = "cuda",
        input_size: tuple[int, int] | list[int] | None = None,
        canonical_focal: float = 1000.0,
        max_depth: float = 300.0,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.weights_dir = Path(weights_dir).resolve()
        self.device = torch.device(device)
        self.model_name = MODEL_NAME_MAP.get(model_name, model_name)
        self.input_size = tuple(input_size or self._default_input_size(self.model_name))
        self.canonical_focal = float(canonical_focal)
        self.max_depth = float(max_depth)

        hubconf = self.repo_dir / "hubconf.py"
        if not hubconf.exists():
            raise FileNotFoundError(
                f"Metric3D official repo not found at {self.repo_dir}. "
                "Clone https://github.com/YvanYin/Metric3D into third_party/Metric3D."
            )
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        self.model = torch.hub.load(str(self.repo_dir), self.model_name, source="local", pretrain=False)
        ckpt_path = self._find_checkpoint()
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    @staticmethod
    def _default_input_size(model_name: str) -> tuple[int, int]:
        # Official Metric3D hub example: ViT models use (616,1064);
        # ConvNeXt models use (544,1216).
        if "convnext" in model_name:
            return (544, 1216)
        return (616, 1064)

    def _find_checkpoint(self) -> Path:
        candidates = []
        if self.model_name.endswith("vit_small"):
            candidates.append("metric_depth_vit_small_800k.pth")
        elif self.model_name.endswith("vit_large"):
            candidates.append("metric_depth_vit_large_800k.pth")
        elif self.model_name.endswith("vit_giant2"):
            candidates.append("metric_depth_vit_giant2_800k.pth")
        candidates.extend(["*.pth", "*.pt", "*.ckpt"])

        for pattern in candidates:
            matches = sorted(self.weights_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No Metric3D checkpoint found in {self.weights_dir}")

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return {k.replace("module.", "", 1): v for k, v in checkpoint[key].items()}
        if isinstance(checkpoint, dict):
            return {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        raise TypeError("Unsupported Metric3D checkpoint format")

    @torch.no_grad()
    def infer(self, rgb: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return metric depth D_m3d as float32 [H,W]."""
        tensor, pad_info, scaled_K, orig_hw = self._preprocess(rgb, K)
        tensor = tensor.to(self.device, non_blocking=True)
        prediction = self.model.inference({"input": tensor})
        if isinstance(prediction, tuple):
            pred_depth = prediction[0]
        elif isinstance(prediction, dict):
            pred_depth = prediction["prediction"]
        else:
            raise TypeError(f"Unexpected Metric3D inference output: {type(prediction)}")

        pred_depth = pred_depth.squeeze()
        top, bottom, left, right = pad_info
        pred_depth = pred_depth[top : pred_depth.shape[-2] - bottom, left : pred_depth.shape[-1] - right]
        pred_depth = F.interpolate(
            pred_depth[None, None],
            size=orig_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        canonical_to_real_scale = float(scaled_K[0, 0]) / self.canonical_focal
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, min=0.0, max=self.max_depth)
        return pred_depth.detach().cpu().numpy().astype(np.float32)

    def _preprocess(self, rgb: np.ndarray, K: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int, int], np.ndarray, tuple[int, int]]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB [H,W,3], got {rgb.shape}")
        rgb_origin = rgb.astype(np.float32)
        orig_h, orig_w = rgb_origin.shape[:2]
        input_h, input_w = self.input_size
        scale = min(input_h / orig_h, input_w / orig_w)
        resized_w, resized_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(rgb_origin, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        scaled_K = K.astype(np.float32).copy()
        scaled_K[0, 0] *= scale
        scaled_K[0, 2] *= scale
        scaled_K[1, 1] *= scale
        scaled_K[1, 2] *= scale

        pad_h = input_h - resized_h
        pad_w = input_w - resized_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        mean_rgb = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std_rgb = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=mean_rgb.tolist(),
        )
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
        tensor = (tensor - torch.from_numpy(mean_rgb)[:, None, None]) / torch.from_numpy(std_rgb)[:, None, None]
        return tensor.unsqueeze(0), (pad_top, pad_bottom, pad_left, pad_right), scaled_K, (orig_h, orig_w)

    def save(self, path: str | Path, depth: np.ndarray, key: str = "D_m3d") -> None:
        ensure_dir(Path(path).parent)
        save_npz_atomic(path, **{key: depth.astype(np.float32)})
