from __future__ import annotations

import json
import logging
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import yaml


LOGGER = logging.getLogger("geort")


DEFAULT_KITTI_K = np.array(
    [[721.5377, 0.0, 609.5593], [0.0, 721.5377, 172.8540], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def repo_root_from(path: str | Path | None = None) -> Path:
    """Find the repo root from a file path or cwd."""
    start = Path(path).resolve() if path is not None else Path.cwd().resolve()
    if start.is_file():
        start = start.parent
    for parent in [start, *start.parents]:
        if (parent / "README.md").exists() and (parent / "src").exists():
            return parent
    return start


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _resolve_config_path(config_path: Path, maybe_path: str | Path) -> Path:
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    root = repo_root_from(config_path)
    candidates = [root / p, config_path.parent / p, Path.cwd() / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (root / p).resolve()


def load_project_config(config_path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load a script config and its path config.

    The default `configs/paths.yaml` is Colab/Drive oriented. If that
    configured root does not exist locally, paths are remapped to this repo
    root while preserving their relative layout.
    """
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)
    paths_file = cfg.get("paths_file", "configs/paths.yaml")
    paths_path = _resolve_config_path(config_path, paths_file)
    paths = load_yaml(paths_path)
    resolved_paths = resolve_runtime_paths(paths, repo_root_from(config_path))
    cfg["_config_path"] = str(config_path)
    cfg["_paths_path"] = str(paths_path)
    return cfg, resolved_paths


def resolve_runtime_paths(paths: Mapping[str, Any], repo_root: Path) -> dict[str, str]:
    configured_root_text = str(paths.get("project_root", repo_root)).replace("\\", "/").rstrip("/")
    configured_root = Path(str(paths.get("project_root", repo_root))).expanduser()
    runtime_root = configured_root if configured_root.exists() else repo_root
    out: dict[str, str] = {}
    out["project_root"] = str(runtime_root)

    for key, value in paths.items():
        if key == "project_root":
            continue
        if not isinstance(value, str):
            out[key] = value
            continue
        value_text = value.replace("\\", "/")
        if configured_root_text and (value_text == configured_root_text or value_text.startswith(configured_root_text + "/")):
            rel_text = value_text[len(configured_root_text) :].lstrip("/")
            out[key] = str((runtime_root / rel_text).resolve())
            continue
        p = Path(value).expanduser()
        if p.is_absolute():
            try:
                rel = p.relative_to(configured_root)
                out[key] = str((runtime_root / rel).resolve())
            except ValueError:
                out[key] = str(p)
        elif key.endswith("_split"):
            out[key] = value
        else:
            out[key] = str((runtime_root / p).resolve())
    return out


def resolve_repo_path(project_root: str | Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (Path(project_root) / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("geort")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def read_split(split_root: str | Path, split_file: str) -> list[str]:
    path = Path(split_root) / split_file
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith("#")]


def safe_sample_id(value: str | Path) -> str:
    text = str(value).strip().replace("\\", "/")
    text = text.strip("/")
    text = re.sub(r"\.[A-Za-z0-9]+$", "", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "__", text)
    text = text.strip("._-")
    if not text:
        raise ValueError(f"Cannot build sample id from {value!r}")
    return text


def read_image_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB image not found or unreadable: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(path: str | Path | None, depth_scale: float = 256.0) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Depth file not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = next((k for k in ("depth", "sparse", "gt", "D", "arr_0") if k in data), None)
        if key is None:
            raise KeyError(f"No depth-like key found in {path}. Keys: {list(data.keys())}")
        arr = data[key].astype(np.float32)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Depth file not readable: {path}")
        arr = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            arr = arr / float(depth_scale)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr[~np.isfinite(arr)] = 0.0
    return arr.astype(np.float32)


def load_intrinsics_from_calib(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        if ":" in line:
            key, values = line.split(":", 1)
            if key.strip() not in {"P2", "P_rect_02", "P_rect_2", "K", "intrinsics"}:
                continue
            nums = [float(x) for x in values.replace(",", " ").split()]
        else:
            nums = [float(x) for x in line.replace(",", " ").split()]
        if len(nums) >= 12:
            p = np.array(nums[:12], dtype=np.float32).reshape(3, 4)
            return np.array([[p[0, 0], 0.0, p[0, 2]], [0.0, p[1, 1], p[1, 2]], [0.0, 0.0, 1.0]], dtype=np.float32)
        if len(nums) == 9:
            return np.array(nums, dtype=np.float32).reshape(3, 3)
        if len(nums) >= 4:
            fx, fy, cx, cy = nums[:4]
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Could not parse intrinsics from {path}")


def scale_intrinsics(K: np.ndarray, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> np.ndarray:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    out = K.astype(np.float32).copy()
    out[0, 0] *= new_w / float(old_w)
    out[0, 2] *= new_w / float(old_w)
    out[1, 1] *= new_h / float(old_h)
    out[1, 2] *= new_h / float(old_h)
    return out


def resize_rgb(rgb: np.ndarray, size_hw: tuple[int, int] | None) -> np.ndarray:
    if size_hw is None:
        return rgb
    h, w = size_hw
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_depth(depth: np.ndarray | None, size_hw: tuple[int, int] | None) -> np.ndarray | None:
    if depth is None or size_hw is None:
        return depth
    h, w = size_hw
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def make_ray_map(K: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return normalized camera rays as float32 [3,H,W]."""
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=0)
    norm = np.linalg.norm(ray, axis=0, keepdims=True).clip(min=1e-8)
    return (ray / norm).astype(np.float32)


def make_uv_map(height: int, width: int) -> np.ndarray:
    """Return normalized image coordinates as float32 [2,H,W]."""
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    return np.stack([u, v], axis=0).astype(np.float32)


def save_npz_atomic(path: str | Path, **arrays: Any) -> None:
    """Atomically save a compressed NPZ file with documented keys."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_npz = tmp_name + ".npz"
    try:
        np.savez_compressed(tmp_npz, **arrays)
        os.replace(tmp_npz, path)
    finally:
        for p in (tmp_name, tmp_npz):
            if os.path.exists(p):
                os.remove(p)


def npz_has_keys(path: str | Path, keys: list[str] | tuple[str, ...]) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return all(k in data for k in keys)
    except Exception:
        return False


def as_save_dtype(array: np.ndarray, save_dtype: str) -> np.ndarray:
    if save_dtype == "float16":
        return array.astype(np.float16)
    if save_dtype == "float32":
        return array.astype(np.float32)
    raise ValueError(f"Unsupported save_dtype: {save_dtype}")


def load_npz_array(path: str | Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} missing key {key}. Keys: {list(data.keys())}")
        return data[key]


def write_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def device_from_config(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)
