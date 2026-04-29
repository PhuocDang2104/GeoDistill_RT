from __future__ import annotations

import argparse
from pathlib import Path

from .utils import ensure_dir, safe_sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GeoRT split files for KITTI depth_selection layout.")
    parser.add_argument("--data_root", type=str, default="data/depth_selection")
    parser.add_argument("--train_count", type=int, default=800)
    return parser.parse_args()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _line(sample_id: str, rgb: Path, sparse: Path | None, gt: Path | None, intrinsics: Path, root: Path) -> str:
    sparse_text = _rel(sparse, root) if sparse is not None else "none"
    gt_text = _rel(gt, root) if gt is not None else "none"
    return f"{sample_id} {_rel(rgb, root)} {sparse_text} {gt_text} {_rel(intrinsics, root)}"


def build_depth_selection_splits(data_root: str | Path, train_count: int = 800) -> dict[str, int]:
    """Create train/val/test split files for KITTI `depth_selection`.

    Expected layout:
      val_selection_cropped/{image,velodyne_raw,groundtruth_depth,intrinsics}
      test_depth_completion_anonymous/{image,velodyne_raw,intrinsics}

    Split policy:
      train.txt: first `train_count` val_selection_cropped samples
      val.txt: remaining val_selection_cropped samples
      test.txt: all test_depth_completion_anonymous samples
    """
    root = Path(data_root).resolve()
    split_dir = ensure_dir(root / "splits")

    val_root = root / "val_selection_cropped"
    test_root = root / "test_depth_completion_anonymous"
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

    outputs = {"train.txt": train_lines, "val.txt": holdout_lines, "test.txt": test_lines}
    for name, lines in outputs.items():
        with open(split_dir / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
    return {name: len(lines) for name, lines in outputs.items()}


def main() -> None:
    args = parse_args()
    counts = build_depth_selection_splits(args.data_root, args.train_count)
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
