from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .dataset import KITTIDepthCompletionDataset
from .utils import ensure_dir, load_project_config, scale_intrinsics


NORMAL_KEYS = ("N_dsine", "normal", "normals")
DEPTH_KEYS = ("D_m3d", "D_da_aligned", "D_dmd3c", "D_full", "D_teacher", "D_da_raw", "D_c", "depth", "arr_0")
CONF_KEYS = ("C_full", "C_teacher", "C", "confidence", "w_m3d", "w_da", "w_dmd3c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a GeoRT teacher/student NPZ output.")
    parser.add_argument("npz_path", type=str, help="Path to a teacher/student .npz file.")
    parser.add_argument("--config", type=str, default="configs/geort_student_s.yaml")
    parser.add_argument("--key", type=str, default="auto", help="NPZ key to visualize, or auto.")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "normal", "pointcloud", "depth2d", "confidence"])
    parser.add_argument("--split", type=str, default="auto", choices=["auto", "train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="visualizations/teacher")
    parser.add_argument("--show", action="store_true", help="Show matplotlib window for 2D images.")
    parser.add_argument("--no-open3d", action="store_true", help="Do not open the Open3D interactive viewer.")
    parser.add_argument("--save-ply", action="store_true", help="Save point cloud as .ply next to PNG previews.")
    parser.add_argument("--stride", type=int, default=2, help="Point-cloud pixel stride for speed.")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=120.0)
    parser.add_argument(
        "--normal-y",
        type=str,
        default="flip",
        choices=["flip", "keep"],
        help="Use flip for OpenGL-style +Y-up normal colors from camera/image-space normals.",
    )
    return parser.parse_args()


def load_npz_payload(path: Path, key: str = "auto") -> tuple[str, np.ndarray, list[str]]:
    with np.load(path) as data:
        keys = list(data.keys())
        if key != "auto":
            if key not in data:
                raise KeyError(f"{path} missing key {key}. Available keys: {keys}")
            return key, data[key], keys
        for candidate in NORMAL_KEYS + DEPTH_KEYS + CONF_KEYS:
            if candidate in data:
                return candidate, data[candidate], keys
        if not keys:
            raise KeyError(f"{path} contains no arrays.")
        return keys[0], data[keys[0]], keys


def infer_mode(key: str, array: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested
    if key in NORMAL_KEYS or (array.ndim == 3 and 3 in (array.shape[0], array.shape[-1])):
        return "normal"
    if key in CONF_KEYS:
        return "confidence"
    return "pointcloud"


def normal_to_chw(normals: np.ndarray) -> np.ndarray:
    if normals.ndim != 3:
        raise ValueError(f"Normal array must be 3D [3,H,W] or [H,W,3], got {normals.shape}")
    if normals.shape[0] == 3:
        chw = normals.astype(np.float32)
    elif normals.shape[-1] == 3:
        chw = normals.transpose(2, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Normal array must have 3 channels, got {normals.shape}")
    norm = np.linalg.norm(chw, axis=0, keepdims=True).clip(min=1e-6)
    return chw / norm


def normal_to_opengl_rgb(normals: np.ndarray, flip_y: bool = True) -> np.ndarray:
    """Convert unit normals to OpenGL-style normal-map RGB.

    R = +X, G = +Y-up, B = +Z. DSINE/camera normals usually use image-space
    +Y downward, so `flip_y=True` gives the conventional OpenGL green channel.
    """
    n = normal_to_chw(normals).copy()
    if flip_y:
        n[1] = -n[1]
    rgb = ((n.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).clip(0, 255)
    return rgb.astype(np.uint8)


def colorize_scalar(array: np.ndarray, min_value: float | None = None, max_value: float | None = None, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(x)
    valid = finite & (x > 0)
    if min_value is None:
        min_value = float(np.percentile(x[valid], 2)) if valid.any() else 0.0
    if max_value is None:
        max_value = float(np.percentile(x[valid], 98)) if valid.any() else 1.0
    denom = max(max_value - min_value, 1e-6)
    u8 = ((x - min_value) / denom * 255.0).clip(0, 255).astype(np.uint8)
    u8[~finite] = 0
    bgr = cv2.applyColorMap(u8, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def infer_split_from_path(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    for split in ("train", "val", "test"):
        if split in parts:
            return split
        if f"{split}_raw" in parts or f"{split}_aligned" in parts or f"{split}_predictions" in parts:
            return split
    return None


def infer_sample_id(path: Path) -> str:
    return path.stem


def find_dataset_sample(sample_id: str, requested_split: str, cfg: dict[str, Any], paths: dict[str, str]) -> dict[str, Any] | None:
    splits = [requested_split] if requested_split != "auto" else []
    inferred = infer_split_from_path(Path(sample_id))
    if inferred and inferred not in splits:
        splits.append(inferred)
    for split in ("train", "val", "test"):
        if split not in splits:
            splits.append(split)

    data_cfg = cfg.get("data", {})
    for split in splits:
        try:
            dataset = KITTIDepthCompletionDataset(
                data_root=paths["data_root"],
                split_root=paths["split_root"],
                split_file=paths[f"{split}_split"],
                split_name=split,
                image_size=None,
                output_scale=int(data_cfg.get("output_scale", 4)),
                depth_scale=float(data_cfg.get("depth_scale", 256.0)),
                teacher_root=paths.get("teacher_root"),
                load_teacher=False,
                return_tensors=False,
            )
        except Exception:
            continue
        for idx, info in enumerate(dataset.samples):
            if info.sample_id == sample_id:
                return dataset.load_sample_np(idx)
    return None


def load_context(npz_path: Path, depth_shape: tuple[int, int], cfg: dict[str, Any], paths: dict[str, str], split: str) -> tuple[np.ndarray, np.ndarray]:
    sample_id = infer_sample_id(npz_path)
    sample = find_dataset_sample(sample_id, split, cfg, paths)
    if sample is None:
        h, w = depth_shape
        K = np.array([[max(h, w), 0.0, (w - 1) * 0.5], [0.0, max(h, w), (h - 1) * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 180
        print(f"[warn] Could not match {sample_id} in dataset splits. Using fallback intrinsics and gray colors.")
        return K, rgb

    rgb = sample["rgb"]
    K = sample["K"]
    if rgb.shape[:2] != depth_shape:
        K = scale_intrinsics(K, rgb.shape[:2], depth_shape)
        rgb = cv2.resize(rgb, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)
    return K.astype(np.float32), rgb.astype(np.uint8)


def depth_to_point_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    rgb: np.ndarray | None,
    stride: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth array must be [H,W], got {depth.shape}")
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride].astype(np.float32)
    z = depth[0:h:stride, 0:w:stride]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    x = (xx - float(K[0, 2])) / float(K[0, 0]) * z
    # Open3D uses a Y-up view more naturally if image-space Y is flipped.
    y = -((yy - float(K[1, 2])) / float(K[1, 1]) * z)
    points = np.stack([x, y, z], axis=-1)[valid]
    if rgb is None:
        colors = colorize_scalar(depth)[0:h:stride, 0:w:stride][valid].astype(np.float32) / 255.0
    else:
        colors = rgb[0:h:stride, 0:w:stride][valid].astype(np.float32) / 255.0
    return points.astype(np.float64), colors.astype(np.float64)


def show_open3d(points: np.ndarray, colors: np.ndarray, title: str, save_ply: Path | None = None, open_viewer: bool = True) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is required for point-cloud visualization. Install with: pip install open3d") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    if save_ply is not None:
        ensure_dir(save_ply.parent)
        o3d.io.write_point_cloud(str(save_ply), pcd)
        print(f"Saved point cloud: {save_ply}")
    if open_viewer:
        o3d.visualization.draw_geometries([pcd, frame], window_name=title)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved image: {path}")


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz_path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    key, array, keys = load_npz_payload(npz_path, args.key)
    mode = infer_mode(key, array, args.mode)
    out_dir = ensure_dir(Path(args.output_dir))
    stem = npz_path.stem
    print(f"File: {npz_path}")
    print(f"Available keys: {keys}")
    print(f"Selected key: {key}")
    print(f"Mode: {mode}")

    if mode == "normal":
        image = normal_to_opengl_rgb(array, flip_y=args.normal_y == "flip")
        out_path = out_dir / f"{stem}_{key}_opengl_normal.png"
        save_rgb(out_path, image)
        if args.show:
            plt.figure(figsize=(12, 4))
            plt.imshow(image)
            plt.title(f"{key} OpenGL normal RGB")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        return

    if array.ndim == 3:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D scalar map for {mode}, got {array.shape}")
    scalar = array.astype(np.float32)
    preview = colorize_scalar(scalar, min_value=args.min_depth if mode != "confidence" else 0.0, max_value=args.max_depth if mode != "confidence" else 1.0)
    preview_path = out_dir / f"{stem}_{key}_{mode}.png"
    save_rgb(preview_path, preview)

    if args.show or mode in {"depth2d", "confidence"}:
        plt.figure(figsize=(12, 4))
        plt.imshow(preview)
        plt.title(f"{key} {mode}")
        plt.axis("off")
        plt.tight_layout()
        if args.show:
            plt.show()
        else:
            plt.close()

    if mode == "pointcloud":
        cfg, paths = load_project_config(args.config)
        K, rgb = load_context(npz_path, scalar.shape, cfg, paths, args.split)
        points, colors = depth_to_point_cloud(scalar, K, rgb, args.stride, args.min_depth, args.max_depth)
        print(f"Point count: {len(points)}")
        if args.no_open3d and not args.save_ply:
            print("Open3D viewer disabled; saved 2D depth preview only.")
            return
        ply_path = out_dir / f"{stem}_{key}.ply" if args.save_ply else None
        show_open3d(points, colors, title=f"{stem}:{key}", save_ply=ply_path, open_viewer=not args.no_open3d)


if __name__ == "__main__":
    main()
