from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from ..utils import ensure_dir, save_npz_atomic


class DSINEWrapper:
    """Real DSINE surface-normal inference wrapper.

    Input RGB is uint8 RGB [H,W,3]. K is a 3x3 camera intrinsic matrix.

    Saved key:
      N_dsine: float32 unit normals [3,H,W]
    """

    def __init__(
        self,
        repo_dir: str | Path,
        weights_dir: str | Path,
        config_file: str | Path | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.weights_dir = Path(weights_dir).resolve()
        self.device = torch.device(device)
        self.config_file = self._resolve_config_file(config_file)

        if not (self.repo_dir / "projects" / "dsine").exists():
            raise FileNotFoundError(
                f"DSINE official repo not found at {self.repo_dir}. "
                "Clone https://github.com/baegwangbin/DSINE into third_party/DSINE."
            )
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        old_argv = sys.argv[:]
        old_cwd = os.getcwd()
        try:
            sys.argv = ["dsine_wrapper.py", str(self.config_file)]
            os.chdir(str(self.repo_dir / "projects" / "dsine"))
            import projects.dsine.config as dsine_config  # type: ignore
            import utils.utils as dsine_utils  # type: ignore

            self.dsine_utils = dsine_utils
            self.args = dsine_config.get_args(test=True)
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)

        self.args.ckpt_path = self._resolve_checkpoint(getattr(self.args, "ckpt_path", None))
        arch = getattr(self.args, "NNET_architecture", "v02")
        if arch == "v00":
            from models.dsine.v00 import DSINE_v00 as DSINE  # type: ignore
        elif arch == "v01":
            from models.dsine.v01 import DSINE_v01 as DSINE  # type: ignore
        elif arch == "v02":
            from models.dsine.v02 import DSINE_v02 as DSINE  # type: ignore
        elif arch == "v02_kappa":
            from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE  # type: ignore
        else:
            raise ValueError(f"Unsupported DSINE architecture: {arch}")

        self.model = DSINE(self.args).to(self.device)
        self.model = self.dsine_utils.load_checkpoint(self.args.ckpt_path, self.model)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _resolve_config_file(self, config_file: str | Path | None) -> Path:
        candidates = []
        if config_file is not None:
            candidates.append(Path(config_file))
        candidates.extend(sorted(self.weights_dir.glob("*.txt")))
        candidates.append(self.repo_dir / "projects" / "dsine" / "experiments" / "exp001_cvpr2024" / "dsine.txt")
        for candidate in candidates:
            if candidate.is_absolute() and candidate.exists():
                return candidate.resolve()
            rel_candidates = [self.weights_dir / candidate, self.repo_dir / candidate]
            for rel in rel_candidates:
                if rel.exists():
                    return rel.resolve()
        raise FileNotFoundError(
            f"No DSINE config file found. Put dsine.txt in {self.weights_dir} or set dsine.config_file."
        )

    def _resolve_checkpoint(self, ckpt_path: Any) -> str:
        candidates = []
        if ckpt_path:
            p = Path(str(ckpt_path))
            candidates.extend([p, self.weights_dir / p, self.repo_dir / p, self.config_file.parent / p])
        candidates.extend(sorted(self.weights_dir.glob("*.pt")) + sorted(self.weights_dir.glob("*.pth")))
        for candidate in candidates:
            if candidate.is_absolute() and candidate.exists():
                return str(candidate.resolve())
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(f"No DSINE checkpoint found in {self.weights_dir}")

    @torch.no_grad()
    def infer(self, rgb: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return N_dsine as float32 unit normals [3,H,W]."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB [H,W,3], got {rgb.shape}")
        orig_h, orig_w = rgb.shape[:2]
        img = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        lrtb = self.dsine_utils.get_padding(orig_h, orig_w)
        left, right, top, bottom = lrtb
        img = F.pad(img, lrtb, mode="constant", value=0.0)
        img = self.normalize(img)

        intrins = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(self.device)
        intrins[:, 0, 2] += left
        intrins[:, 1, 2] += top
        pred_norm = self.model(img, intrins=intrins)[-1]
        pred_norm = pred_norm[:, :, top : top + orig_h, left : left + orig_w]
        pred_norm = F.normalize(pred_norm, dim=1, eps=1e-6)
        return pred_norm[0].detach().cpu().numpy().astype(np.float32)

    def save(self, path: str | Path, normals: np.ndarray, key: str = "N_dsine") -> None:
        ensure_dir(Path(path).parent)
        save_npz_atomic(path, **{key: normals.astype(np.float32)})
