from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import KITTIDepthCompletionDataset
from .metrics import DepthPredictionAudit, GlobalDepthMetricAccumulator, average_metric_dict, depth_metrics_torch
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
    loss_cfg = cfg.get("loss", {})
    min_depth = float(loss_cfg.get("min_depth", 1e-3))
    max_depth = float(loss_cfg.get("max_depth", cfg.get("student", {}).get("max_depth", 120.0)))
    global_accumulator = GlobalDepthMetricAccumulator(min_depth, max_depth)
    audit_cfg = cfg.get("evaluation", {}).get("depth_audit", {})
    audit = None
    if split != "test" and bool(audit_cfg.get("enabled", False)):
        audit = DepthPredictionAudit(
            min_depth=min_depth,
            max_depth=max_depth,
            protocol_min_depth=float(audit_cfg.get("protocol_min_depth", min_depth)),
            protocol_max_depth=float(audit_cfg.get("protocol_max_depth", max_depth)),
            thresholds=tuple(audit_cfg.get("thresholds", [0.01, 0.05, 0.1, 0.5])),
            top_k=int(audit_cfg.get("top_k", 100)),
        )
    stage_stats: dict[str, list[float]] = {}
    forward_seconds: list[float] = []
    timing_warmup = min(10, max(0, len(loader) // 10))
    pipeline_started = time.perf_counter()
    prediction_count = 0
    benchmark_dir = ensure_dir(out_dir / "benchmark_png")
    save_visual = bool(cfg.get("outputs", {}).get("save_visuals", True))
    for batch_index, batch in enumerate(tqdm(loader, desc=f"infer:{split}")):
        batch = to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_started = time.perf_counter()
        pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"], batch.get("K"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - forward_started
        if batch_index >= timing_warmup:
            forward_seconds.append(elapsed)
        prediction_count += 1
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
                global_accumulator.update(pred.get("D_full", pred["D_c"]), batch["gt"], gt_mask)
                if audit is not None:
                    audit.update(pred, batch)
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
        global_metrics = {
            **global_accumulator.compute(),
            "stage_rmse_m": {key: float(np.sqrt(sq / max(1.0, n))) for key, (sq, n) in stage_stats.items()},
            "stage_valid_pixels": {key: int(n) for key, (_, n) in stage_stats.items()},
            "note": "Global pixel aggregation; D_init RMSE is restricted to V_init support, and D_full includes exact sparse anchoring.",
        }
        if audit is not None:
            global_metrics.update(audit.summary_metrics())
            audit_path = student_root / "logs" / f"infer_{split}_depth_audit.json"
            audit_path.write_text(json.dumps(audit.report(), indent=2), encoding="utf-8")
        with open(student_root / "logs" / f"infer_{split}_metrics_global.json", "w", encoding="utf-8") as f:
            json.dump(global_metrics, f, indent=2, sort_keys=True)
        logger.info("Global inference metrics for %s: %s", split, global_metrics)

    pipeline_seconds = time.perf_counter() - pipeline_started
    ordered = sorted(forward_seconds)
    median_seconds = statistics.median(ordered) if ordered else float("nan")
    p95_index = max(0, int(0.95 * len(ordered)) - 1)
    runtime = {
        "split": split,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "fp32",
        "batch_size": 1,
        "samples": prediction_count,
        "timing_warmup_samples": timing_warmup,
        "forward_median_ms": 1000.0 * median_seconds,
        "forward_p95_ms": 1000.0 * ordered[p95_index] if ordered else float("nan"),
        "forward_fps_from_median": 1.0 / median_seconds if median_seconds > 0.0 else 0.0,
        "pipeline_with_output_io_seconds": pipeline_seconds,
        "pipeline_with_output_io_fps": prediction_count / max(pipeline_seconds, 1e-9),
        "timing_scope": "Forward timing excludes data loading and output writes; pipeline timing includes both.",
    }
    with open(student_root / "logs" / f"infer_{split}_runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2, sort_keys=True)
    logger.info("Inference runtime for %s: %s", split, runtime)


def main() -> None:
    args = parse_args()
    cfg, paths = load_project_config(args.config)
    infer(cfg, paths, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
