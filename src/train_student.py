from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import KITTIDepthCompletionDataset
from .losses import geort_loss
from .metrics import average_metric_dict, depth_metrics_by_edge_torch, depth_metrics_by_range_torch, depth_metrics_torch
from .model_geort import GeoRTStudentS
from .utils import device_from_config, ensure_dir, load_project_config, seed_everything, setup_logger, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoRT student.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def make_loader(cfg: dict[str, Any], paths: dict[str, str], split: str, training: bool) -> DataLoader:
    data_cfg = cfg["data"]
    mono_cfg = cfg.get("mono_ssi", {})
    loss_cfg = cfg.get("loss", {})
    load_mono = training and bool(mono_cfg.get("enabled", False))
    load_geometry = training and (float(loss_cfg.get("lambda_ssi", mono_cfg.get("weight", 0.0))) > 0.0 or float(loss_cfg.get("lambda_ord", 0.0)) > 0.0)
    dataset = KITTIDepthCompletionDataset(
        data_root=paths["data_root"],
        split_root=paths["split_root"],
        split_file=paths[f"{split}_split"],
        split_name=split,
        image_size=tuple(data_cfg["image_size"]),
        output_scale=int(data_cfg.get("output_scale", 4)),
        depth_scale=float(data_cfg.get("depth_scale", 256.0)),
        teacher_root=paths["teacher_root"],
        load_teacher=training or split == "val",
        load_geometry=load_geometry,
        load_mono=load_mono,
        mono_key=str(mono_cfg.get("key", "D_da_raw")),
        min_depth=float(loss_cfg.get("min_depth", 1e-3)),
        max_depth=float(loss_cfg.get("max_depth", cfg.get("student", {}).get("max_depth", 120.0))),
        calibrate_metric_teacher=bool(loss_cfg.get("calibrate_metric_teacher", True)),
        metric_conf_min=float(loss_cfg.get("metric_conf_min", 0.05)),
        metric_conf_sparse_decay=float(loss_cfg.get("metric_conf_sparse_decay", 6.0)),
        metric_conf_range_decay=float(loss_cfg.get("metric_conf_range_decay", 0.25)),
        metric_conf_sparse_blend_radius=float(loss_cfg.get("metric_conf_sparse_blend_radius", 48.0)),
        return_tensors=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]) if training else 1,
        shuffle=training,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=int(data_cfg.get("num_workers", 2)) > 0,
    )


def _has_any_npz(root: Path | None, candidates: list[Path]) -> bool:
    if root is None:
        return False
    return any(path.exists() for path in candidates)


def _teacher_coverage(dataset: KITTIDepthCompletionDataset) -> dict[str, float]:
    root = dataset.teacher_root
    total = len(dataset.samples)
    if root is None or total == 0:
        return {"metric": 0.0, "geometry": 0.0}

    metric_count = 0
    geometry_count = 0
    for sample in dataset.samples:
        sid = sample.sample_id
        metric_paths = [
            root / "metric_coarse" / dataset.split_name / f"{sid}.npz",
            root / "dmd3c" / dataset.split_name / f"{sid}.npz",
            root / "fused" / dataset.split_name / f"{sid}.npz",
        ]
        geometry_paths = [
            root / "geometry_fused" / dataset.split_name / f"{sid}.npz",
            root / "geometry_raw" / "depth_anything_v2" / dataset.split_name / f"{sid}.npz",
            root / "geometry_raw" / "depth_anything" / dataset.split_name / f"{sid}.npz",
            root / "depth_anything" / f"{dataset.split_name}_raw" / f"{sid}.npz",
            root / "depth_anything" / dataset.split_name / f"{sid}.npz",
            root / "depth_anything" / f"{dataset.split_name}_aligned" / f"{sid}.npz",
            root / "metric3d" / dataset.split_name / f"{sid}.npz",
            root / "dmd3c" / dataset.split_name / f"{sid}.npz",
        ]
        metric_count += int(_has_any_npz(root, metric_paths))
        geometry_count += int(_has_any_npz(root, geometry_paths))
    return {"metric": metric_count / total, "geometry": geometry_count / total}


def preflight_teacher_coverage(cfg: dict[str, Any], dataset: KITTIDepthCompletionDataset, logger: Any) -> None:
    checks = cfg.get("teacher_checks", {})
    if not bool(checks.get("enabled", True)):
        return
    loss_cfg = cfg.get("loss", {})
    schedule_cfg = cfg.get("schedule", {})
    epochs = int(cfg.get("train", {}).get("epochs", 30))
    coverage = _teacher_coverage(dataset)
    logger.info(
        "teacher_coverage split=%s metric=%.3f geometry=%.3f root=%s",
        dataset.split_name,
        coverage["metric"],
        coverage["geometry"],
        dataset.teacher_root,
    )

    teacher_scheduled = int(schedule_cfg.get("add_teacher_epoch", 5)) < epochs and float(loss_cfg.get("lambda_cm", 0.0)) > 0.0
    geometry_scheduled = (
        int(schedule_cfg.get("add_geometry_epoch", 10)) < epochs
        and (float(loss_cfg.get("lambda_ssi", 0.0)) > 0.0 or float(loss_cfg.get("lambda_ord", 0.0)) > 0.0)
    )
    min_metric = float(checks.get("min_metric_coverage", 0.95))
    min_geometry = float(checks.get("min_geometry_coverage", 0.95))

    if bool(checks.get("require_metric", True)) and teacher_scheduled and coverage["metric"] < min_metric:
        raise RuntimeError(
            f"Metric teacher coverage for split={dataset.split_name} is {coverage['metric']:.1%}, below {min_metric:.1%}. "
            "Generate DMD3C/metric_coarse teacher files before student training, e.g. "
            "`python -m src.teachers.generate_teachers --config configs/teacher.yaml --split train --run_dmd3c --run_fusion`."
        )
    if bool(checks.get("require_geometry", True)) and geometry_scheduled and coverage["geometry"] < min_geometry:
        raise RuntimeError(
            f"Geometry teacher coverage for split={dataset.split_name} is {coverage['geometry']:.1%}, below {min_geometry:.1%}. "
            "Generate geometry/Depth Anything teacher files before student training, e.g. "
            "`python -m src.teachers.generate_teachers --config configs/teacher.yaml --split train --run_depth_anything --run_fusion`."
        )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_rmse: float,
    cfg: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_rmse": best_rmse,
            "config": cfg,
        },
        path,
    )


def append_csv(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


@torch.no_grad()
def validate(model: GeoRTStudentS, loader: DataLoader, device: torch.device, cfg: dict[str, Any]) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float]] = []
    loss_cfg = cfg.get("loss", {})
    min_depth = float(loss_cfg.get("min_depth", 1e-3))
    max_depth = float(loss_cfg.get("max_depth", cfg.get("student", {}).get("max_depth", 120.0)))
    range_bins = loss_cfg.get("range_bins", [0.0, 20.0, 40.0, 60.0, 80.0, 120.0])
    for batch in tqdm(loader, desc="val", leave=False):
        batch = to_device(batch, device)
        pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"])
        D_eval = pred.get("D_full", pred["D_c"])
        gt_mask = (batch["gt_mask"] > 0.5) & (batch["gt"] > min_depth) & (batch["gt"] < max_depth)
        if gt_mask.sum().item() < 1:
            continue
        metrics = depth_metrics_torch(D_eval, batch["gt"], gt_mask, min_depth=min_depth, max_depth=max_depth)
        metrics.update(depth_metrics_by_range_torch(D_eval, batch["gt"], gt_mask, bins=range_bins, min_depth=min_depth, max_depth=max_depth))
        metrics.update(depth_metrics_by_edge_torch(D_eval, batch["gt"], gt_mask, batch["rgb"], min_depth=min_depth, max_depth=max_depth))
        records.append(metrics)
    return average_metric_dict(records) if records else {"rmse": float("inf"), "mae": float("inf"), "abs_rel": float("inf")}


def train(cfg: dict[str, Any], paths: dict[str, str]) -> None:
    seed_everything(int(cfg.get("seed", 42)))
    device = device_from_config(str(cfg.get("device", "cuda")))
    student_root = Path(paths["student_root"])
    logger = setup_logger(student_root / "logs" / "train_student.log")

    train_loader = make_loader(cfg, paths, "train", training=True)
    val_loader = make_loader(cfg, paths, "val", training=False)
    preflight_teacher_coverage(cfg, train_loader.dataset, logger)
    model = GeoRTStudentS.from_config(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
    )
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    ckpt_dir = student_root / "checkpoints"
    log_csv = student_root / "logs" / "train_log.csv"
    log_jsonl = student_root / "logs" / "train_log.jsonl"
    best_rmse = float("inf")
    start_epoch = 0

    resume = cfg["train"].get("resume")
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_rmse = float(ckpt.get("best_rmse", best_rmse))
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    epochs = int(cfg["train"].get("epochs", 30))
    mono_cfg = cfg.get("mono_ssi", {})
    mono_enabled = bool(mono_cfg.get("enabled", False))
    mono_start_epoch = int(mono_cfg.get("start_epoch", 5))
    warned_missing_mono = False
    for epoch in range(start_epoch, epochs):
        model.train()
        running: dict[str, float] = {}
        count = 0
        pbar = tqdm(train_loader, desc=f"train:{epoch}")
        for batch in pbar:
            batch = to_device(batch, device)
            geometry_available = "C_G" in batch and float(batch["C_G"].sum().detach().cpu()) >= 1.0
            if (
                mono_enabled
                and epoch >= mono_start_epoch
                and not warned_missing_mono
                and not geometry_available
                and (
                    "D_da_raw" not in batch
                    or "da_raw_valid" not in batch
                    or float(batch["da_raw_valid"].sum().detach().cpu()) < 1.0
                )
            ):
                logger.warning("mono_ssi is enabled but no Depth Anything raw map was found for this batch; SSI loss will be skipped.")
                warned_missing_mono = True
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"])
                loss, items = geort_loss(pred, batch, cfg["loss"], cfg["schedule"], epoch, mono_cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            count += 1
            for key, value in items.items():
                running[key] = running.get(key, 0.0) + float(value)
            avg_loss = running["loss"] / count
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        train_items = {f"train_{k}": v / max(1, count) for k, v in running.items()}
        val_items = validate(model, val_loader, device, cfg)
        rmse = float(val_items.get("rmse", float("inf")))
        is_best = rmse < best_rmse
        if is_best:
            best_rmse = rmse

        record: dict[str, Any] = {"epoch": epoch, **train_items, **{f"val_{k}": v for k, v in val_items.items()}, "best_rmse": best_rmse}
        append_csv(log_csv, record)
        write_jsonl(log_jsonl, record)
        logger.info("epoch=%d train_loss=%.6f val_rmse=%.6f best=%.6f", epoch, train_items.get("train_loss", 0.0), rmse, best_rmse)

        save_every = int(cfg.get("outputs", {}).get("save_every", 1))
        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scaler, epoch, best_rmse, cfg)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scaler, epoch, best_rmse, cfg)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scaler, epoch, best_rmse, cfg)


def main() -> None:
    args = parse_args()
    cfg, paths = load_project_config(args.config)
    train(cfg, paths)


if __name__ == "__main__":
    main()
