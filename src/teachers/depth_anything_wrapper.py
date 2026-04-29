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
        """Robust inverse-depth scale-shift fit for Depth Anything V2.

        Fit:
            inv_metric = scale * D_da_raw + shift

        where:
            inv_metric = 1 / sparse_metric_depth

        Returns scale and shift in the ORIGINAL raw_depth domain, so caller can use:
            inv_aligned = scale * raw_depth + shift
            aligned = 1 / inv_aligned
        """
        if raw_depth.shape != sparse_depth.shape:
            raise ValueError(
                f"raw_depth and sparse_depth must have same shape, "
                f"got {raw_depth.shape} vs {sparse_depth.shape}"
            )

        if mask is None:
            mask = sparse_depth > 0

        valid = (
            mask.astype(bool)
            & np.isfinite(raw_depth)
            & np.isfinite(sparse_depth)
            & (sparse_depth >= min_depth)
            & (sparse_depth <= max_depth)
        )

        x_raw = raw_depth[valid].astype(np.float64)
        z = sparse_depth[valid].astype(np.float64)

        finite = np.isfinite(x_raw) & np.isfinite(z)
        x_raw = x_raw[finite]
        z = z[finite]

        if x_raw.size < min_valid_points:
            raise ValueError(
                f"Not enough valid sparse points for DA alignment: "
                f"{x_raw.size} < {min_valid_points}"
            )

        # Target is inverse metric depth.
        y = 1.0 / np.clip(z, min_depth, max_depth)

        # Normalize raw prediction for stable scale-shift estimation.
        x_med = float(np.median(x_raw))
        x_mad = float(np.median(np.abs(x_raw - x_med)))
        x_std = float(np.std(x_raw))
        x_scale = max(1e-6, 1.4826 * x_mad, 0.1 * x_std)

        x = (x_raw - x_med) / x_scale

        def _fit_lstsq(x_fit: np.ndarray, y_fit: np.ndarray) -> tuple[float, float]:
            A = np.stack([x_fit, np.ones_like(x_fit)], axis=1)
            theta = np.linalg.lstsq(A, y_fit, rcond=None)[0]
            return float(theta[0]), float(theta[1])

        # RANSAC first: sparse LiDAR anchors can contain outliers / projected mismatch.
        n = x.size
        rng = np.random.default_rng(0)

        best_inliers = np.ones(n, dtype=bool)
        best_count = -1
        best_error = np.inf

        residual_threshold = 0.015  # inverse-depth residual threshold
        ransac_iters = 256

        if n >= 2:
            for _ in range(ransac_iters):
                ids = rng.choice(n, size=2, replace=False)

                try:
                    s, t = _fit_lstsq(x[ids], y[ids])
                except Exception:
                    continue

                pred = s * x + t
                residual = np.abs(pred - y)

                inliers = residual <= residual_threshold
                count = int(np.count_nonzero(inliers))
                error = float(np.median(residual[inliers])) if count > 0 else np.inf

                if count > best_count or (count == best_count and error < best_error):
                    best_count = count
                    best_error = error
                    best_inliers = inliers

        min_inliers = max(12, min_valid_points // 4)
        if np.count_nonzero(best_inliers) >= min_inliers:
            x_fit = x[best_inliers]
            y_fit = y[best_inliers]
        else:
            x_fit = x
            y_fit = y

        scale_norm, shift_norm = _fit_lstsq(x_fit, y_fit)

        # Huber refinement after RANSAC.
        if robust:
            if huber_delta is None:
                residual0 = scale_norm * x_fit + shift_norm - y_fit
                mad = np.median(np.abs(residual0 - np.median(residual0)))
                delta = max(1e-3, 1.4826 * mad)
            else:
                delta = float(huber_delta)

            weights = np.ones_like(y_fit, dtype=np.float64)

            for _ in range(max_iters):
                A = np.stack([x_fit, np.ones_like(x_fit)], axis=1)
                Aw = A * np.sqrt(weights)[:, None]
                yw = y_fit * np.sqrt(weights)

                theta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
                scale_norm, shift_norm = float(theta[0]), float(theta[1])

                residual = scale_norm * x_fit + shift_norm - y_fit
                abs_r = np.abs(residual)

                weights = np.where(
                    abs_r <= delta,
                    1.0,
                    delta / np.maximum(abs_r, 1e-8),
                )

        # Convert normalized fit back to original raw_depth domain:
        #
        # inv = scale_norm * ((raw - x_med) / x_scale) + shift_norm
        #     = (scale_norm / x_scale) * raw
        #       + (shift_norm - scale_norm * x_med / x_scale)
        scale = scale_norm / x_scale
        shift = shift_norm - scale_norm * x_med / x_scale

        if (
            not np.isfinite(scale)
            or not np.isfinite(shift)
            or abs(scale) < 1e-12
        ):
            raise ValueError(
                f"Unstable inverse DA scale-shift fit: scale={scale}, shift={shift}"
            )

        return float(scale), float(shift), int(x_raw.size)

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

        Keep output format unchanged:
            aligned: metric depth [H,W]
            scale, shift: inverse-depth fit params

        Main formula:
            inv_metric = scale * raw + shift
            metric = 1 / inv_metric

        The function tests both raw and -raw orientations, then picks the one
        with lower sparse-anchor median error. This avoids near/far inversion
        problems from relative-depth outputs.
        """
        if raw_depth.shape != sparse_depth.shape:
            raw_depth = cv2.resize(
                raw_depth.astype(np.float32),
                (sparse_depth.shape[1], sparse_depth.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        if mask is not None and mask.shape != sparse_depth.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (sparse_depth.shape[1], sparse_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if mask is None:
            anchor_mask = sparse_depth > 0
        else:
            anchor_mask = mask.astype(bool)

        anchor_mask = (
            anchor_mask
            & np.isfinite(sparse_depth)
            & (sparse_depth >= min_depth)
            & (sparse_depth <= max_depth)
        )

        if np.count_nonzero(anchor_mask) < min_valid_points:
            raise ValueError(
                f"Not enough valid sparse points for DA alignment: "
                f"{np.count_nonzero(anchor_mask)} < {min_valid_points}"
            )

        min_inv = 1.0 / max_depth
        max_inv = 1.0 / min_depth

        candidates: list[tuple[float, np.ndarray, float, float, int]] = []

        for raw_candidate in (
            raw_depth.astype(np.float32),
            -raw_depth.astype(np.float32),
        ):
            try:
                scale, shift, count = cls.fit_scale_shift(
                    raw_depth=raw_candidate,
                    sparse_depth=sparse_depth,
                    mask=anchor_mask,
                    robust=robust,
                    min_valid_points=min_valid_points,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    huber_delta=None,
                    max_iters=12,
                )

                inv_aligned = scale * raw_candidate.astype(np.float32) + float(shift)

                # Critical: prevent inverse depth from approaching zero.
                # If it approaches zero, depth explodes and Open3D creates a fan
                # spreading from the camera origin.
                inv_aligned = np.clip(inv_aligned, min_inv, max_inv).astype(np.float32)

                aligned = 1.0 / np.maximum(inv_aligned, 1e-8)
                aligned = np.clip(aligned, min_depth, max_depth).astype(np.float32)

                # Measure alignment quality only at sparse anchors.
                residual = np.abs(aligned[anchor_mask] - sparse_depth[anchor_mask])
                score = float(np.median(residual)) if residual.size else np.inf

                candidates.append((score, aligned, scale, shift, count))

            except Exception:
                continue

        if not candidates:
            raise ValueError("No valid Depth Anything alignment candidate.")

        # Pick best raw orientation.
        candidates.sort(key=lambda item: item[0])
        _, aligned, scale, shift, count = candidates[0]

        # Enforce exact sparse anchors to avoid local ray/fan distortion around
        # LiDAR support points.
        aligned[anchor_mask] = sparse_depth[anchor_mask].astype(np.float32)

        aligned[~np.isfinite(aligned)] = 0.0
        aligned = np.clip(aligned, min_depth, max_depth).astype(np.float32)

        return aligned, float(scale), float(shift), int(count)

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
