from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import KITTIDepthCompletionDataset
from .metrics import average_metric_dict, depth_metrics_torch
from .model_factory import build_student
from .sparse_propagation import downsample_depth_with_mask
from .train_student import to_device
from .utils import device_from_config, ensure_dir, load_project_config, save_npz_atomic, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeoRT student inference.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    return parser.parse_args()


def make_loader(cfg: dict[str, Any], paths: dict[str, str], split: str) -> DataLoader:
    data_cfg = cfg["data"]
    dataset = KITTIDepthCompletionDataset(
        data_root=paths["data_root"],
        split_root=paths["split_root"],
        split_file=paths[f"{split}_split"],
        split_name=split,
        image_size=tuple(data_cfg["image_size"]),
        output_scale=int(data_cfg.get("output_scale", 4)),
        depth_scale=float(data_cfg.get("depth_scale", 256.0)),
        teacher_root=paths["teacher_root"],
        load_teacher=False,
        return_tensors=True,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(data_cfg.get("num_workers", 2)), pin_memory=torch.cuda.is_available())


def save_visuals(out_dir: Path, sample_id: str, D_full: np.ndarray, C_full: np.ndarray) -> None:
    depth16 = np.clip(D_full * 256.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(out_dir / f"{sample_id}_D_full_depth16.png"), depth16)

    conf = C_full.astype(np.float32)
    conf = (conf - conf.min()) / max(1e-6, float(conf.max() - conf.min()))
    heat = cv2.applyColorMap((conf * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_dir / f"{sample_id}_C_full.png"), heat)


def _stage_target(batch: dict[str, Any], prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gt = batch["gt"].float()
    mask = ((batch["gt_mask"] > 0.5) & torch.isfinite(gt) & (gt > 1e-3) & (gt < 120.0)).float()
    if prediction.shape[-2:] == gt.shape[-2:]:
        return gt, mask > 0.0
    scale_h = gt.shape[-2] // prediction.shape[-2]
    scale_w = gt.shape[-1] // prediction.shape[-1]
    if scale_h == scale_w and scale_h >= 1:
        target, valid = downsample_depth_with_mask(gt, mask, scale=scale_h)
        return target, valid > 0.0
    weight = torch.nn.functional.interpolate(mask, size=prediction.shape[-2:], mode="area")
    target = torch.nn.functional.interpolate(gt * mask, size=prediction.shape[-2:], mode="area") / weight.clamp_min(1e-6)
    return target, weight > 0.0


@torch.no_grad()
def infer(cfg: dict[str, Any], paths: dict[str, str], checkpoint: str, split: str) -> None:
    device = device_from_config(str(cfg.get("device", "cuda")))
    student_root = Path(paths["student_root"])
    out_dir = ensure_dir(student_root / f"{split}_predictions")
    logger = setup_logger(student_root / "logs" / f"infer_{split}.log")

    loader = make_loader(cfg, paths, split)
    model = build_student(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    metrics_records: list[dict[str, float]] = []
    global_stats = {key: 0.0 for key in ("sq", "abs", "inv_sq", "inv_abs", "abs_rel", "count")}
    stage_stats: dict[str, list[float]] = {}
    benchmark_dir = ensure_dir(out_dir / "benchmark_png")
    save_visual = bool(cfg.get("outputs", {}).get("save_visuals", True))
    for batch in tqdm(loader, desc=f"infer:{split}"):
        batch = to_device(batch, device)
        pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"], batch.get("K"))
        D_full = pred.get("D_full", pred["D_c"])[0, 0].detach().cpu().numpy().astype(np.float32)
        C_full = pred.get("C_full", pred["C"])[0, 0].detach().cpu().numpy().astype(np.float32)
        D_1_4 = pred.get("D_1_4", pred["D_c"])[0, 0].detach().cpu().numpy().astype(np.float32)
        C_1_4 = pred.get("C_1_4", pred["C"])[0, 0].detach().cpu().numpy().astype(np.float32)
        sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], list) else str(batch["sample_id"])
        save_npz_atomic(
            out_dir / f"{sample_id}.npz",
            D_full=D_full,
            C_full=C_full,
            D_1_4=D_1_4,
            C_1_4=C_1_4,
            D_c=D_1_4,
            C=C_1_4,
        )
        cv2.imwrite(str(benchmark_dir / f"{sample_id}.png"), np.clip(D_full * 256.0, 0, 65535).astype(np.uint16))
        if save_visual:
            save_visuals(out_dir, sample_id, D_full, C_full)

        if batch["gt_mask"].sum().item() > 0:
            gt_mask = (batch["gt_mask"] > 0.5) & (batch["gt"] > 1e-3)
            if gt_mask.sum().item() > 0:
                metrics_records.append(depth_metrics_torch(pred.get("D_full", pred["D_c"]), batch["gt"], gt_mask))
                prediction = pred.get("D_full", pred["D_c"]).float().clamp_min(1e-3)
                target = batch["gt"].float().clamp_min(1e-3)
                diff = (prediction - target)[gt_mask]
                inv_diff = (1000.0 / prediction - 1000.0 / target)[gt_mask]
                global_stats["sq"] += float((diff * diff).sum().cpu())
                global_stats["abs"] += float(diff.abs().sum().cpu())
                global_stats["inv_sq"] += float((inv_diff * inv_diff).sum().cpu())
                global_stats["inv_abs"] += float(inv_diff.abs().sum().cpu())
                global_stats["abs_rel"] += float((diff.abs() / target[gt_mask]).sum().cpu())
                global_stats["count"] += int(gt_mask.sum().cpu())
                for stage in ("D_init", "D16", "D8", "D4", "D2", "D1", "D_full"):
                    if stage not in pred:
                        continue
                    stage_target, stage_mask = _stage_target(batch, pred[stage])
                    if stage == "D_init" and "V_init" in pred:
                        stage_mask = stage_mask & (pred["V_init"] > 0.5)
                    stage_diff = (pred[stage].float() - stage_target)[stage_mask]
                    stats = stage_stats.setdefault(stage, [0.0, 0.0])
                    stats[0] += float((stage_diff * stage_diff).sum().cpu())
                    stats[1] += int(stage_mask.sum().cpu())

    if metrics_records:
        metrics = average_metric_dict(metrics_records)
        with open(student_root / "logs" / f"infer_{split}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        logger.info("Inference metrics for %s: %s", split, metrics)
        count = max(1.0, global_stats["count"])
        global_metrics = {
            "rmse": float(np.sqrt(global_stats["sq"] / count)),
            "mae": global_stats["abs"] / count,
            "irmse": float(np.sqrt(global_stats["inv_sq"] / count)),
            "imae": global_stats["inv_abs"] / count,
            "abs_rel": global_stats["abs_rel"] / count,
            "valid_pixels": int(global_stats["count"]),
            "stage_rmse_m": {key: float(np.sqrt(sq / max(1.0, n))) for key, (sq, n) in stage_stats.items()},
            "stage_valid_pixels": {key: int(n) for key, (_, n) in stage_stats.items()},
            "note": "Global pixel aggregation; D_init RMSE is restricted to V_init support, and D_full includes exact sparse anchoring.",
        }
        with open(student_root / "logs" / f"infer_{split}_metrics_global.json", "w", encoding="utf-8") as f:
            json.dump(global_metrics, f, indent=2, sort_keys=True)
        logger.info("Global inference metrics for %s: %s", split, global_metrics)


def main() -> None:
    args = parse_args()
    cfg, paths = load_project_config(args.config)
    infer(cfg, paths, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
