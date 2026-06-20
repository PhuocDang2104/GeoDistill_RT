from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..utils import ensure_dir, save_npz_atomic


def _evict_foreign_modules(prefixes: tuple[str, ...], repo_dir: Path) -> None:
    repo_text = str(repo_dir)
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and str(Path(module_file).resolve()).startswith(repo_text):
            continue
        del sys.modules[name]


def _prioritize_repo_paths(paths: tuple[Path, ...], repo_dir: Path) -> None:
    target_paths = tuple(path.resolve() for path in paths)
    target_texts = {str(path) for path in target_paths}
    sibling_root = repo_dir.parent.resolve()
    kept: list[str] = []
    for entry in sys.path:
        if not entry:
            kept.append(entry)
            continue
        try:
            entry_path = Path(entry).resolve()
        except OSError:
            kept.append(entry)
            continue
        if entry_path == sibling_root or sibling_root in entry_path.parents:
            if entry_path not in target_paths:
                continue
        if str(entry_path) not in target_texts:
            kept.append(entry)
    sys.path[:] = kept
    for path in reversed(target_paths):
        sys.path.insert(0, str(path))


class DMD3CWrapper:
    """Real DMD3C / BP-Net teacher wrapper.

    DMD3C is a depth-completion teacher, so it consumes RGB, sparse depth and
    camera intrinsics. The official model requires the CUDA extension `BpOps`
    from `third_party/DMD3C/exts`.

    Input:
      RGB image: uint8 RGB [H,W,3]
      sparse: metric sparse depth float [H,W]
      K: camera intrinsics float [3,3]

    Saved key:
      D_dmd3c: float32 metric completed depth [H,W]
    """

    def __init__(
        self,
        repo_dir: str | Path,
        weights_dir: str | Path,
        checkpoint: str | Path | None = None,
        device: str | torch.device = "cuda",
        image_size: tuple[int, int] | list[int] = (352, 1216),
        image_mean: tuple[float, float, float] | list[float] = (90.9950, 96.2278, 94.3213),
        image_std: tuple[float, float, float] | list[float] = (79.2382, 80.5267, 82.1483),
        max_depth: float = 120.0,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.weights_dir = Path(weights_dir).resolve()
        self.device = torch.device(device)
        self.image_size = tuple(int(x) for x in image_size)
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_depth = float(max_depth)

        if not (self.repo_dir / "models" / "BPNet.py").exists():
            raise FileNotFoundError(
                f"DMD3C official repo not found at {self.repo_dir}. "
                "Clone https://github.com/Sharpiless/DMD3C into third_party/DMD3C."
            )
        _prioritize_repo_paths((self.repo_dir, self.repo_dir / "exts"), self.repo_dir)
        _evict_foreign_modules(("models",), self.repo_dir)

        try:
            import BpOps  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "DMD3C requires the compiled CUDA extension BpOps. "
                "Build it with: cd third_party/DMD3C/exts && python setup.py install"
            ) from exc

        from models import Pre_MF_Post  # type: ignore

        self.model = Pre_MF_Post()
        ckpt_path = self._find_checkpoint(checkpoint)
        checkpoint_obj = torch.load(ckpt_path, map_location="cpu")
        state = self._extract_state_dict(checkpoint_obj)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[DMD3C] Missing checkpoint keys: {len(missing)}")
        if unexpected:
            print(f"[DMD3C] Unexpected checkpoint keys: {len(unexpected)}")
        self.model.to(self.device).eval()

    def _find_checkpoint(self, checkpoint: str | Path | None) -> Path:
        candidates: list[Path] = []
        if checkpoint:
            p = Path(checkpoint)
            candidates.extend([p, self.weights_dir / p])
        candidates.extend(sorted(self.weights_dir.glob("*.pth")) + sorted(self.weights_dir.glob("*.pt")) + sorted(self.weights_dir.glob("*.ckpt")))
        for candidate in candidates:
            if candidate.is_absolute() and candidate.exists():
                return candidate.resolve()
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"No DMD3C checkpoint found in {self.weights_dir}")

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("net", "model", "model_state_dict", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return {k.replace("module.", "", 1): v for k, v in checkpoint[key].items()}
            return {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        raise TypeError("Unsupported DMD3C checkpoint format")

    @torch.no_grad()
    def infer(self, rgb: np.ndarray, sparse: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return D_dmd3c as float32 [H,W]."""
        orig_h, orig_w = rgb.shape[:2]
        image, sparse_resized, K_resized = self._preprocess_arrays(rgb, sparse, K)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)[None]).float().to(self.device)
        sparse_tensor = torch.from_numpy(sparse_resized[None, None]).float().to(self.device)
        K_tensor = torch.from_numpy(K_resized[None]).float().to(self.device)
        output = self.model(image_tensor, None, sparse_tensor, K_tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        depth = output.squeeze(0).squeeze(0)
        if depth.shape[-2:] != (orig_h, orig_w):
            depth = F.interpolate(depth[None, None], size=(orig_h, orig_w), mode="bilinear", align_corners=True)[0, 0]
        depth = depth.clamp(min=0.0, max=self.max_depth)
        return depth.detach().cpu().numpy().astype(np.float32)

    def _preprocess_arrays(self, rgb: np.ndarray, sparse: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_h, target_w = self.image_size
        h, w = rgb.shape[:2]
        if (h, w) != (target_h, target_w):
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            sparse = cv2.resize(sparse.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            K = K.astype(np.float32).copy()
            K[0, 0] *= target_w / float(w)
            K[0, 2] *= target_w / float(w)
            K[1, 1] *= target_h / float(h)
            K[1, 2] *= target_h / float(h)
        image = (rgb.astype(np.float32) - self.image_mean) / self.image_std
        sparse = sparse.astype(np.float32)
        sparse[~np.isfinite(sparse)] = 0.0
        return image, sparse, K.astype(np.float32)

    def save(self, path: str | Path, depth: np.ndarray, key: str = "D_dmd3c") -> None:
        ensure_dir(Path(path).parent)
        save_npz_atomic(path, **{key: depth.astype(np.float32)})
