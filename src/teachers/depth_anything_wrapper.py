from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..utils import ensure_dir, save_npz_atomic


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class DepthAnythingV2Wrapper:
    """Real Depth Anything V2 inference wrapper.

    Input RGB is uint8 [H,W,3]. Official Depth Anything V2 expects BGR images
    in `infer_image`, so this wrapper converts RGB to BGR while keeping the
    official resize/normalization transform.

    Saved keys:
      D_da_raw: float32 relative depth [H,W]
      D_da_aligned: float32 metric-aligned depth [H,W]
      scale, shift: inverse-depth fit parameters where
        1 / D_da_aligned = scale * D_da_raw + shift
    """

    def __init__(
        self,
        repo_dir: str | Path,
        weights_dir: str | Path,
        encoder: str = "vitl",
        device: str | torch.device = "cuda",
        input_size: int = 518,
        allow_transformers_fallback: bool = True,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.weights_dir = Path(weights_dir).resolve()
        self.encoder = encoder
        self.device = torch.device(device)
        self.input_size = int(input_size)
        self.backend = "official"

        dpt_file = self.repo_dir / "depth_anything_v2" / "dpt.py"
        if dpt_file.exists():
            if str(self.repo_dir) not in sys.path:
                sys.path.insert(0, str(self.repo_dir))
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

            self.model = DepthAnythingV2(**MODEL_CONFIGS[self.encoder])
            ckpt_path = self._find_checkpoint()
            self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            self.model.to(self.device).eval()
        elif allow_transformers_fallback:
            self.backend = "transformers"
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore

            hf_name = {
                "vits": "depth-anything/Depth-Anything-V2-Small-hf",
                "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
                "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
            }.get(self.encoder)
            if hf_name is None:
                raise ValueError("Transformers fallback supports vits/vitb/vitl only.")
            self.processor = AutoImageProcessor.from_pretrained(hf_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(hf_name).to(self.device).eval()
        else:
            raise FileNotFoundError(
                f"Depth Anything V2 repo not found at {self.repo_dir}. "
                "Clone https://github.com/DepthAnything/Depth-Anything-V2 into third_party/Depth-Anything-V2."
            )

    def _find_checkpoint(self) -> Path:
        expected = self.weights_dir / f"depth_anything_v2_{self.encoder}.pth"
        if expected.exists():
            return expected
        matches = sorted(self.weights_dir.glob(f"*{self.encoder}*.pth")) + sorted(self.weights_dir.glob("*.pth"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No Depth Anything V2 checkpoint found in {self.weights_dir}")

    @torch.no_grad()
    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """Return relative raw depth D_da_raw as float32 [H,W]."""
        if self.backend == "official":
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            image, (h, w) = self.model.image2tensor(bgr, self.input_size)
            image = image.to(self.device, non_blocking=True)
            depth = self.model(image)
            depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
            return depth.detach().cpu().numpy().astype(np.float32)

        from PIL import Image

        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth
        pred = F.interpolate(pred[:, None], size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0]
        return pred.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def fit_scale_shift(
        raw_depth: np.ndarray,
        sparse_depth: np.ndarray,
        mask: np.ndarray | None = None,
        robust: bool = True,
        min_valid_points: int = 50,
        min_depth: float = 0.1,
        max_depth: float = 120.0,
        huber_delta: float | None = None,
        max_iters: int = 8,
    ) -> tuple[float, float, int]:
        """Fit D_aligned = scale * D_raw + shift on valid sparse pixels."""
        if mask is None:
            mask = sparse_depth > 0
        valid = (
            mask.astype(bool)
            & np.isfinite(raw_depth)
            & np.isfinite(sparse_depth)
            & (sparse_depth >= min_depth)
            & (sparse_depth <= max_depth)
        )
        x = raw_depth[valid].astype(np.float64)
        y = sparse_depth[valid].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size < min_valid_points:
            raise ValueError(f"Not enough valid sparse points for DA alignment: {x.size} < {min_valid_points}")

        A = np.stack([x, np.ones_like(x)], axis=1)
        theta = np.linalg.lstsq(A, y, rcond=None)[0]
        if robust:
            weights = np.ones_like(y)
            for _ in range(max_iters):
                Aw = A * np.sqrt(weights)[:, None]
                yw = y * np.sqrt(weights)
                theta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
                residual = A @ theta - y
                if huber_delta is None:
                    mad = np.median(np.abs(residual - np.median(residual)))
                    delta = max(1.0, 1.4826 * mad)
                else:
                    delta = huber_delta
                abs_r = np.abs(residual)
                weights = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-6))

        scale, shift = float(theta[0]), float(theta[1])
        if not np.isfinite(scale) or not np.isfinite(shift) or abs(scale) < 1e-8:
            raise ValueError(f"Unstable DA scale-shift fit: scale={scale}, shift={shift}")
        return scale, shift, int(x.size)

    @classmethod
    def align_to_sparse(
        cls,
        raw_depth: np.ndarray,
        sparse_depth: np.ndarray,
        mask: np.ndarray | None = None,
        robust: bool = True,
        min_valid_points: int = 50,
        min_depth: float = 0.1,
        max_depth: float = 120.0,
    ) -> tuple[np.ndarray, float, float, int]:
        """Align Depth Anything relative output to metric sparse depth.

        Depth Anything V2 raw output is relative / inverse-depth-like. Fit:

            inv_metric = scale * raw + shift

        then recover metric depth:

            metric = 1 / inv_metric
        """
        if mask is None:
            mask = sparse_depth > 0

        valid = (
            mask.astype(bool)
            & np.isfinite(raw_depth)
            & np.isfinite(sparse_depth)
            & (sparse_depth >= min_depth)
            & (sparse_depth <= max_depth)
        )

        x = raw_depth[valid].astype(np.float64)
        z = sparse_depth[valid].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(z)
        x, z = x[finite], z[finite]
        if x.size < min_valid_points:
            raise ValueError(f"Not enough valid sparse points for DA alignment: {x.size} < {min_valid_points}")

        y = 1.0 / np.clip(z, min_depth, max_depth)
        A = np.stack([x, np.ones_like(x)], axis=1)
        theta = np.linalg.lstsq(A, y, rcond=None)[0]

        if robust:
            weights = np.ones_like(y)
            for _ in range(8):
                Aw = A * np.sqrt(weights)[:, None]
                yw = y * np.sqrt(weights)
                theta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
                residual = A @ theta - y
                mad = np.median(np.abs(residual - np.median(residual)))
                delta = max(1e-3, 1.4826 * mad)
                abs_r = np.abs(residual)
                weights = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-8))

        scale, shift = float(theta[0]), float(theta[1])
        if not np.isfinite(scale) or not np.isfinite(shift):
            raise ValueError(f"Unstable inverse DA scale-shift fit: scale={scale}, shift={shift}")

        inv_aligned = scale * raw_depth.astype(np.float32) + shift
        inv_aligned = np.maximum(inv_aligned, 1.0 / max_depth)
        aligned = 1.0 / inv_aligned
        aligned[~np.isfinite(aligned)] = 0.0
        aligned = np.clip(aligned, 0.0, max_depth).astype(np.float32)
        return aligned, scale, shift, int(x.size)

    def save_raw(self, path: str | Path, raw_depth: np.ndarray, key: str = "D_da_raw") -> None:
        ensure_dir(Path(path).parent)
        save_npz_atomic(path, **{key: raw_depth.astype(np.float32)})

    def save_aligned(
        self,
        path: str | Path,
        aligned_depth: np.ndarray,
        scale: float,
        shift: float,
        key: str = "D_da_aligned",
        alignment_mode: str = "inverse_depth",
    ) -> None:
        ensure_dir(Path(path).parent)
        save_npz_atomic(
            path,
            **{
                key: aligned_depth.astype(np.float32),
                "scale": float(scale),
                "shift": float(shift),
                "alignment_mode": np.array(alignment_mode),
            },
        )
