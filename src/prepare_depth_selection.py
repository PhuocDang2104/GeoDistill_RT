from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_dir, safe_sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GeoRT split files for the local KITTI Depth Completion layout.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/kitti_depth_completion",
        help=(
            "Dataset root. Accepts data/kitti_depth_completion, or the nested "
            "depth_selection folder for the small selected split fallback."
        ),
    )
    parser.add_argument("--train_count", type=int, default=800)
    return parser.parse_args()


@dataclass(frozen=True)
class DepthSelectionLayout:
    data_root: Path
    selection_root: Path
    split_root: Path
    full_root: Path | None = None


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _line(sample_id: str, rgb: Path, sparse: Path | None, gt: Path | None, intrinsics: Path, root: Path) -> str:
    sparse_text = _rel(sparse, root) if sparse is not None else "none"
    gt_text = _rel(gt, root) if gt is not None else "none"
    return f"{sample_id} {_rel(rgb, root)} {sparse_text} {gt_text} {_rel(intrinsics, root)}"


def _line_with_k(sample_id: str, rgb: Path, sparse: Path, gt: Path, K: tuple[float, float, float, float], root: Path) -> str:
    fx, fy, cx, cy = K
    return (
        f"{sample_id} {_rel(rgb, root)} {_rel(sparse, root)} {_rel(gt, root)} "
        f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
    )


def _load_kitti_intrinsics(calib_path: Path, camera: str) -> tuple[float, float, float, float]:
    suffix = camera.rsplit("_", 1)[-1]
    keys = (f"P_rect_{suffix}", f"P{int(suffix)}", f"P_rect_{int(suffix)}")
    with open(calib_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if ":" not in raw_line:
                continue
            key, values = raw_line.split(":", 1)
            if key.strip() not in keys:
                continue
            nums = [float(x) for x in values.replace(",", " ").split()]
            if len(nums) < 12:
                raise ValueError(f"Expected 12 projection values in {calib_path} for {key.strip()}")
            return nums[0], nums[5], nums[2], nums[6]
    raise ValueError(f"Could not find projection matrix for {camera} in {calib_path}")


def _resolve_layout(data_root: str | Path) -> DepthSelectionLayout:
    """Resolve either supported local KITTI root shape.

    Supported inputs:
      data/kitti_depth_completion/depth_selection
      data/kitti_depth_completion

    Split files are written next to the selected data root. This keeps split
    paths relative to the same root that the dataset loader receives.
    """
    root = Path(data_root).resolve()

    if (root / "val_selection_cropped").is_dir():
        return DepthSelectionLayout(
            data_root=root,
            selection_root=root,
            split_root=ensure_dir(root / "splits"),
        )

    nested_selection = root / "depth_selection"
    if (nested_selection / "val_selection_cropped").is_dir():
        return DepthSelectionLayout(
            data_root=root,
            selection_root=nested_selection,
            split_root=ensure_dir(root / "splits"),
            full_root=root if (root / "train").is_dir() and (root / "val").is_dir() else None,
        )

    raise FileNotFoundError(
        "Could not find KITTI depth_selection data. Expected either "
        f"{root / 'val_selection_cropped'} or {nested_selection / 'val_selection_cropped'}."
    )


def _build_full_kitti_lines(root: Path, split: str) -> list[str]:
    split_root = root / split
    if not split_root.is_dir():
        return []

    lines: list[str] = []
    for drive_root in sorted(split_root.glob("*_sync")):
        if not drive_root.is_dir():
            continue
        drive = drive_root.name
        date = drive[:10]
        calib = root / date / "calib_cam_to_cam.txt"
        if not calib.exists():
            raise FileNotFoundError(f"Missing calibration for {drive}: {calib}")

        for camera in ("image_02", "image_03"):
            K = _load_kitti_intrinsics(calib, camera)
            gt_dir = drive_root / "proj_depth" / "groundtruth" / camera
            sparse_dir = drive_root / "proj_depth" / "velodyne_raw" / camera
            rgb_dir = root / date / drive / camera / "data"
            for gt in sorted(gt_dir.glob("*.png")):
                sparse = sparse_dir / gt.name
                rgb = rgb_dir / gt.name
                for p in (sparse, rgb):
                    if not p.exists():
                        raise FileNotFoundError(f"Missing paired file for {split}/{drive}/{camera}/{gt.name}: {p}")
                sid = safe_sample_id(f"{drive}_{camera}_{gt.stem}")
                lines.append(_line_with_k(sid, rgb, sparse, gt, K, root))
    return lines


def _build_depth_selection_lines(layout: DepthSelectionLayout, train_count: int) -> tuple[list[str], list[str], list[str]]:
    root = layout.data_root
    val_root = layout.selection_root / "val_selection_cropped"
    test_root = layout.selection_root / "test_depth_completion_anonymous"
    val_images = sorted((val_root / "image").glob("*.png"))
    if not val_images:
        raise FileNotFoundError(f"No validation images found under {val_root / 'image'}")

    val_lines: list[str] = []
    for rgb in val_images:
        sid = safe_sample_id(rgb.stem)
        sparse = val_root / "velodyne_raw" / rgb.name.replace("_sync_image_", "_sync_velodyne_raw_", 1)
        gt = val_root / "groundtruth_depth" / rgb.name.replace("_sync_image_", "_sync_groundtruth_depth_", 1)
        intr = val_root / "intrinsics" / f"{rgb.stem}.txt"
        for p in (sparse, gt, intr):
            if not p.exists():
                raise FileNotFoundError(f"Missing paired file for {rgb.name}: {p}")
        val_lines.append(_line(sid, rgb, sparse, gt, intr, root))

    train_count = min(max(0, int(train_count)), len(val_lines))
    train_lines = val_lines[:train_count]
    holdout_lines = val_lines[train_count:]

    test_lines = _build_test_lines(layout)
    return train_lines, holdout_lines, test_lines


def _build_test_lines(layout: DepthSelectionLayout) -> list[str]:
    root = layout.data_root
    test_root = layout.selection_root / "test_depth_completion_anonymous"
    test_lines: list[str] = []
    test_images = sorted((test_root / "image").glob("*.png"))
    for rgb in test_images:
        sid = safe_sample_id(rgb.stem)
        sparse = test_root / "velodyne_raw" / rgb.name
        intr = test_root / "intrinsics" / f"{rgb.stem}.txt"
        if not sparse.exists():
            raise FileNotFoundError(f"Missing test sparse depth for {rgb.name}: {sparse}")
        if not intr.exists():
            raise FileNotFoundError(f"Missing test intrinsics for {rgb.name}: {intr}")
        test_lines.append(_line(sid, rgb, sparse, None, intr, root))
    return test_lines


def build_depth_selection_splits(data_root: str | Path, train_count: int = 800) -> dict[str, int | str]:
    """Create train/val/test split files for the current local KITTI layout.

    Usable teacher/student samples require RGB, sparse LiDAR, depth target,
    and intrinsics. When full KITTI train/val raw RGB is present, train.txt and
    val.txt are built from:
      {train,val}/<drive>/proj_depth/{velodyne_raw,groundtruth}/{image_02,image_03}
      <date>/<drive>/{image_02,image_03}/data

    test.txt still uses the anonymous KITTI depth_selection test set. If full
    raw RGB is absent, the function falls back to the selected validation split
    policy used by earlier local setups.
    """
    layout = _resolve_layout(data_root)
    root = layout.data_root
    split_dir = layout.split_root

    train_lines: list[str]
    holdout_lines: list[str]
    mode = "depth_selection"

    if layout.full_root is not None:
        train_lines = _build_full_kitti_lines(root, "train")
        holdout_lines = _build_full_kitti_lines(root, "val")
        if train_lines and holdout_lines:
            mode = "full_kitti_depth_completion"
            test_lines = _build_test_lines(layout)
        else:
            train_lines, holdout_lines, test_lines = _build_depth_selection_lines(layout, train_count)
    else:
        train_lines, holdout_lines, test_lines = _build_depth_selection_lines(layout, train_count)

    outputs = {"train.txt": train_lines, "val.txt": holdout_lines, "test.txt": test_lines}
    for name, lines in outputs.items():
        with open(split_dir / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
    return {
        "mode": mode,
        "data_root": str(root),
        "split_root": str(split_dir),
        "train.txt": len(train_lines),
        "val.txt": len(holdout_lines),
        "test.txt": len(test_lines),
    }


def main() -> None:
    args = parse_args()
    counts = build_depth_selection_splits(args.data_root, args.train_count)
    for name, value in counts.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
