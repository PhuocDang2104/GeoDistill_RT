from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .dataset import KITTIDepthCompletionDataset
from .metrics import average_metric_dict, depth_metrics_np
from .utils import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved teacher outputs against available KITTI GT.")
    parser.add_argument("--config", type=str, default="configs/teacher.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def resize_to(arr: np.ndarray, shape_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    if arr.shape == shape_hw:
        return arr.astype(np.float32)
    h, w = shape_hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=interpolation).astype(np.float32)


def load_depth(path: Path, key: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if key not in data:
                return None
            return data[key].astype(np.float32)
    except Exception:
        return None


def add_metrics(
    records: dict[str, list[dict[str, float]]],
    name: str,
    pred: np.ndarray | None,
    gt: np.ndarray,
    gt_mask: np.ndarray,
) -> None:
    if pred is None:
        return
    target_shape = pred.shape
    gt_eval = resize_to(gt, target_shape, cv2.INTER_NEAREST)
    mask_eval = resize_to(gt_mask, target_shape, cv2.INTER_NEAREST) > 0.5
    valid_pred = np.isfinite(pred) & (pred > 1e-3) & (pred < 120.0)
    mask_eval = mask_eval & valid_pred
    if int(mask_eval.sum()) < 1:
        return
    records.setdefault(name, []).append(depth_metrics_np(pred.astype(np.float32), gt_eval.astype(np.float32), mask_eval))


def main() -> None:
    args = parse_args()
    cfg, paths = load_project_config(args.config)
    teacher_root = Path(paths["teacher_root"])
    dataset = KITTIDepthCompletionDataset(
        data_root=paths["data_root"],
        split_root=paths["split_root"],
        split_file=paths[f"{args.split}_split"],
        split_name=args.split,
        image_size=None,
        output_scale=int(cfg.get("output_scale", 4)),
        teacher_root=paths["teacher_root"],
        load_teacher=False,
        return_tensors=False,
    )

    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    records: dict[str, list[dict[str, float]]] = {}
    weight_sums: dict[str, float] = {"w_m3d": 0.0, "w_da": 0.0, "w_dmd3c": 0.0}
    weight_counts: dict[str, int] = {"w_m3d": 0, "w_da": 0, "w_dmd3c": 0}

    for idx in tqdm(range(total), desc=f"eval-teachers:{args.split}"):
        sample = dataset.load_sample_np(idx)
        sid = sample["sample_id"]
        gt = sample["gt"].astype(np.float32)
        gt_mask = sample["gt_mask"].astype(np.float32)
        if int(gt_mask.sum()) < 1:
            continue

        add_metrics(records, "metric3d", load_depth(teacher_root / "metric3d" / args.split / f"{sid}.npz", "D_m3d"), gt, gt_mask)
        add_metrics(
            records,
            "depth_anything_aligned",
            load_depth(teacher_root / "depth_anything" / f"{args.split}_aligned" / f"{sid}.npz", "D_da_aligned"),
            gt,
            gt_mask,
        )
        add_metrics(records, "dmd3c", load_depth(teacher_root / "dmd3c" / args.split / f"{sid}.npz", "D_dmd3c"), gt, gt_mask)

        fused_path = teacher_root / "fused" / args.split / f"{sid}.npz"
        if fused_path.exists():
            with np.load(fused_path) as data:
                if "D_full" in data:
                    add_metrics(records, "fused_full", data["D_full"].astype(np.float32), gt, gt_mask)
                if "D_teacher" in data:
                    add_metrics(records, "fused_teacher", data["D_teacher"].astype(np.float32), gt, gt_mask)
                for key in weight_sums:
                    if key in data:
                        arr = data[key].astype(np.float32)
                        valid = np.isfinite(arr)
                        if valid.any():
                            weight_sums[key] += float(arr[valid].mean())
                            weight_counts[key] += 1

    print("\nTeacher metrics")
    for name, rows in records.items():
        avg = average_metric_dict(rows)
        print(
            f"{name:24s} n={len(rows):4d} "
            f"rmse={avg.get('rmse', float('nan')):.4f} "
            f"mae={avg.get('mae', float('nan')):.4f} "
            f"abs_rel={avg.get('abs_rel', float('nan')):.4f}"
        )

    print("\nMean fusion weights")
    for key, total_weight in weight_sums.items():
        count = max(1, weight_counts[key])
        print(f"{key:8s}: {total_weight / count:.4f} over {weight_counts[key]} files")


if __name__ == "__main__":
    main()
