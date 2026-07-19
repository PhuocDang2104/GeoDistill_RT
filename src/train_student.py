from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .dataset import KITTIDepthCompletionDataset
from .losses import geort_loss
from .metrics import average_metric_dict, depth_metrics_by_edge_torch, depth_metrics_by_range_torch, depth_metrics_torch
from .model_factory import build_student
from .utils import device_from_config, ensure_dir, load_project_config, seed_everything, setup_logger, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoRT student.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def to_device(batch: dict[str, Any], device: torch.device, channels_last: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if channels_last and value.ndim == 4 and value.is_floating_point():
                out[key] = value.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def make_loader(cfg: dict[str, Any], paths: dict[str, str], split: str, training: bool, distributed: bool = False) -> DataLoader:
    data_cfg = cfg["data"]
    mono_cfg = cfg.get("mono_ssi", {})
    loss_cfg = cfg.get("loss", {})
    load_mono = training and bool(mono_cfg.get("enabled", False))
    load_geometry = training and (
        float(loss_cfg.get("lambda_G", loss_cfg.get("lambda_ssi", mono_cfg.get("weight", 0.0)))) > 0.0
        or float(loss_cfg.get("lambda_ord", 0.0)) > 0.0
    )
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
        geometry_fallback=bool(loss_cfg.get("geometry_fallback", True)),
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
    sampler = DistributedSampler(dataset, shuffle=training) if distributed else None
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]) if training else 1,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=int(data_cfg.get("num_workers", 2)) > 0,
    )


def _has_any_npz(root: Path | None, candidates: list[Path]) -> bool:
    if root is None:
        return False
    return any(path.exists() for path in candidates)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while True:
        if hasattr(model, "_orig_mod"):
            model = getattr(model, "_orig_mod")
            continue
        if isinstance(model, DistributedDataParallel):
            model = model.module
            continue
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            continue
        return model


def _use_data_parallel(train_cfg: dict[str, Any], device: torch.device) -> bool:
    value = train_cfg.get("data_parallel", "auto")
    if isinstance(value, str):
        enabled = value.lower() in {"1", "true", "yes", "on", "auto"}
    else:
        enabled = bool(value)
    return enabled and device.type == "cuda" and torch.cuda.device_count() > 1


def _distributed_requested(train_cfg: dict[str, Any]) -> bool:
    value = train_cfg.get("distributed", "auto")
    if isinstance(value, str):
        enabled = value.lower() in {"1", "true", "yes", "on", "auto"}
    else:
        enabled = bool(value)
    return enabled and int(os.environ.get("WORLD_SIZE", "1")) > 1


def _init_distributed(train_cfg: dict[str, Any]) -> tuple[bool, int, int, int]:
    if not _distributed_requested(train_cfg):
        return False, 0, 0, 1
    backend = str(train_cfg.get("distributed_backend", "gloo")).lower()
    if not dist.is_available():
        raise RuntimeError("Distributed training was requested but torch.distributed is unavailable.")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, local_rank, rank, world_size


def _amp_dtype(cfg: dict[str, Any], device: torch.device, logger: Any) -> torch.dtype:
    requested = str(cfg.get("train", {}).get("amp_dtype", "float16")).lower()
    if requested in {"bf16", "bfloat16"}:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            logger.warning("amp_dtype=bf16 requested but CUDA device does not support bf16; falling back to float16.")
            return torch.float16
        return torch.bfloat16
    if requested in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported train.amp_dtype: {requested}")


def _make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_optimizer(model: torch.nn.Module, cfg: dict[str, Any], device: torch.device, logger: Any) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    kwargs: dict[str, Any] = {
        "lr": float(train_cfg.get("lr", 1e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 1e-5)),
    }
    use_fused = bool(train_cfg.get("fused_adamw", True)) and device.type == "cuda"
    if use_fused:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        if "fused" in kwargs:
            logger.warning("Fused AdamW is unavailable in this PyTorch build; using standard AdamW.")
            kwargs.pop("fused", None)
            return torch.optim.AdamW(model.parameters(), **kwargs)
        raise


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = cfg.get("train", {})
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    if name in {"", "none", "constant"}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported train.scheduler: {name}")

    total_steps = max(1, int(steps_per_epoch) * int(epochs))
    warmup_steps_cfg = train_cfg.get("warmup_steps")
    if warmup_steps_cfg is not None:
        warmup_steps = int(warmup_steps_cfg)
    else:
        warmup_steps = int(float(train_cfg.get("warmup_epochs", 1.0)) * max(1, int(steps_per_epoch)))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_lr_ratio, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _teacher_coverage(dataset: KITTIDepthCompletionDataset) -> dict[str, float]:
    root = dataset.teacher_root
    total = len(dataset.samples)
    if root is None or total == 0:
        return {"metric": 0.0, "geometry": 0.0, "geometry_fused": 0.0, "geometry_main": 0.0, "geometry_dmd_only": 0.0}

    metric_count = 0
    geometry_count = 0
    geometry_fused_count = 0
    geometry_main_count = 0
    geometry_dmd_only_count = 0
    for sample in dataset.samples:
        sid = sample.sample_id
        metric_paths = [
            root / "metric_coarse" / dataset.split_name / f"{sid}.npz",
            root / "dmd3c" / dataset.split_name / f"{sid}.npz",
            root / "fused" / dataset.split_name / f"{sid}.npz",
        ]
        geometry_fused_path = root / "geometry_fused" / dataset.split_name / f"{sid}.npz"
        geometry_paths = [
            geometry_fused_path,
            root / "geometry_raw" / "depth_anything_v2" / dataset.split_name / f"{sid}.npz",
            root / "geometry_raw" / "depth_anything" / dataset.split_name / f"{sid}.npz",
            root / "depth_anything" / f"{dataset.split_name}_raw" / f"{sid}.npz",
            root / "depth_anything" / dataset.split_name / f"{sid}.npz",
            root / "depth_anything" / f"{dataset.split_name}_aligned" / f"{sid}.npz",
            root / "metric3d" / dataset.split_name / f"{sid}.npz",
        ]
        dmd_geometry_paths = [
            root / "dmd3c" / dataset.split_name / f"{sid}.npz",
        ]
        metric_count += int(_has_any_npz(root, metric_paths))
        geometry_fused_count += int(geometry_fused_path.exists())
        has_main_geometry = _has_any_npz(root, geometry_paths)
        has_dmd_geometry = _has_any_npz(root, dmd_geometry_paths)
        geometry_main_count += int(has_main_geometry)
        geometry_dmd_only_count += int((not has_main_geometry) and has_dmd_geometry)
        geometry_count += int(has_main_geometry or has_dmd_geometry)
    return {
        "metric": metric_count / total,
        "geometry": geometry_count / total,
        "geometry_fused": geometry_fused_count / total,
        "geometry_main": geometry_main_count / total,
        "geometry_dmd_only": geometry_dmd_only_count / total,
    }


def preflight_teacher_coverage(cfg: dict[str, Any], dataset: KITTIDepthCompletionDataset, logger: Any) -> None:
    checks = cfg.get("teacher_checks", {})
    if not bool(checks.get("enabled", True)):
        return
    loss_cfg = cfg.get("loss", {})
    schedule_cfg = cfg.get("schedule", {})
    epochs = int(cfg.get("train", {}).get("epochs", 30))
    coverage = _teacher_coverage(dataset)
    logger.info(
        "teacher_coverage split=%s metric=%.3f geometry=%.3f geometry_fused=%.3f geometry_main=%.3f geometry_dmd_only=%.3f root=%s",
        dataset.split_name,
        coverage["metric"],
        coverage["geometry"],
        coverage["geometry_fused"],
        coverage["geometry_main"],
        coverage["geometry_dmd_only"],
        dataset.teacher_root,
    )

    teacher_scheduled = int(schedule_cfg.get("add_teacher_epoch", 5)) < epochs and float(loss_cfg.get("lambda_cm", 0.0)) > 0.0
    geometry_scheduled = (
        int(schedule_cfg.get("add_geometry_epoch", 10)) < epochs
        and (
            float(loss_cfg.get("lambda_G", loss_cfg.get("lambda_ssi", 0.0))) > 0.0
            or float(loss_cfg.get("lambda_ord", 0.0)) > 0.0
        )
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
    if bool(checks.get("require_geometry_fused", False)) and geometry_scheduled and coverage["geometry_fused"] < min_geometry:
        raise RuntimeError(
            f"Fused geometry coverage for split={dataset.split_name} is {coverage['geometry_fused']:.1%}, below {min_geometry:.1%}. "
            "Expected canonical geometry_fused/<split>/<sample_id>.npz files containing R_G and C_G; DA/DMD fallback is not accepted."
        )
    if geometry_scheduled and coverage["geometry_main"] < min_geometry and coverage["geometry"] >= min_geometry:
        logger.warning(
            "Main geometry teacher coverage for split=%s is %.1f%%, below %.1f%%; %.1f%% of samples rely on DMD3C-only geometry fallback.",
            dataset.split_name,
            100.0 * coverage["geometry_main"],
            100.0 * min_geometry,
            100.0 * coverage["geometry_dmd_only"],
        )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_rmse: float,
    cfg: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float]] = []
    loss_cfg = cfg.get("loss", {})
    min_depth = float(loss_cfg.get("min_depth", 1e-3))
    max_depth = float(loss_cfg.get("max_depth", cfg.get("student", {}).get("max_depth", 120.0)))
    range_bins = loss_cfg.get("range_bins", [0.0, 20.0, 40.0, 60.0, 80.0, 120.0])
    for batch in tqdm(loader, desc="val", leave=False):
        batch = to_device(batch, device, channels_last=channels_last)
        pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"], batch.get("K"))
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
    train_cfg = cfg.get("train", {})
    distributed, local_rank, rank, world_size = _init_distributed(train_cfg)
    is_main = rank == 0
    deterministic = bool(train_cfg.get("deterministic", True))
    seed_everything(int(cfg.get("seed", 42)) + rank, deterministic=deterministic)
    if distributed and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = device_from_config(str(cfg.get("device", "cuda")))
    student_root = Path(paths["student_root"])
    log_name = "train_student.log" if is_main else f"train_student_rank{rank}.log"
    logger = setup_logger(student_root / "logs" / log_name)
    if distributed:
        logger.info("Using distributed training backend=%s rank=%d/%d local_rank=%d.", dist.get_backend(), rank, world_size, local_rank)
    matmul_precision = str(train_cfg.get("float32_matmul_precision", "high"))
    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        logger.warning("Could not set float32 matmul precision to %s: %s", matmul_precision, exc)
    if device.type == "cuda" and not deterministic:
        torch.backends.cudnn.benchmark = True

    train_loader = make_loader(cfg, paths, "train", training=True, distributed=distributed)
    val_loader = make_loader(cfg, paths, "val", training=False) if is_main else None
    preflight_teacher_coverage(cfg, train_loader.dataset, logger)
    model = build_student(cfg).to(device)
    channels_last = bool(train_cfg.get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    ckpt_dir = student_root / "checkpoints"
    log_csv = student_root / "logs" / "train_log.csv"
    log_jsonl = student_root / "logs" / "train_log.jsonl"
    best_rmse = float("inf")
    start_epoch = 0

    resume = train_cfg.get("resume")
    ckpt: dict[str, Any] | None = None
    if resume:
        ckpt = torch.load(resume, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_rmse = float(ckpt.get("best_rmse", best_rmse))
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    if bool(train_cfg.get("compile", False)):
        if not hasattr(torch, "compile"):
            logger.warning("train.compile=true but torch.compile is unavailable; continuing eager.")
        else:
            compile_mode = str(train_cfg.get("compile_mode", "reduce-overhead"))
            try:
                model = torch.compile(model, mode=compile_mode, fullgraph=False)
                logger.info("Compiled student model with torch.compile(mode=%s).", compile_mode)
            except Exception as exc:
                logger.warning("torch.compile failed; continuing eager. Error: %s", exc)

    if distributed:
        logger.info("Wrapping model with DistributedDataParallel.")
        ddp_kwargs = {"find_unused_parameters": bool(train_cfg.get("find_unused_parameters", True))}
        if device.type == "cuda":
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
        else:
            model = DistributedDataParallel(model, **ddp_kwargs)
    elif _use_data_parallel(train_cfg, device):
        logger.info("Using DataParallel across %d CUDA devices.", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    optimizer = _make_optimizer(model, cfg, device, logger)
    epochs = int(train_cfg.get("epochs", 30))
    scheduler = _make_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader), epochs=epochs)
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    autocast_dtype = _amp_dtype(cfg, device, logger)
    scaler = _make_grad_scaler(device, enabled=amp_enabled and autocast_dtype == torch.float16)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))

    if ckpt is not None:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

    mono_cfg = cfg.get("mono_ssi", {})
    mono_enabled = bool(mono_cfg.get("enabled", False))
    mono_start_epoch = int(mono_cfg.get("start_epoch", 5))
    warned_missing_mono = False
    for epoch in range(start_epoch, epochs):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        running: dict[str, float] = {}
        count = 0
        pbar = tqdm(train_loader, desc=f"train:{epoch}", disable=not is_main)
        for batch in pbar:
            batch = to_device(batch, device, channels_last=channels_last)
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
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=autocast_dtype):
                pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"], batch.get("K"))
                loss, items = geort_loss(pred, batch, cfg["loss"], cfg["schedule"], epoch, mono_cfg)
            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            count += 1
            for key, value in items.items():
                running[key] = running.get(key, 0.0) + float(value)
            avg_loss = running["loss"] / count
            if is_main:
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        train_items = {f"train_{k}": v / max(1, count) for k, v in running.items()}
        if distributed:
            for key, value in train_items.items():
                reduced = torch.tensor(float(value), device=device)
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                train_items[key] = float(reduced.item()) / float(world_size)
        if distributed:
            dist.barrier()
        if is_main:
            assert val_loader is not None
            val_items = validate(_unwrap_model(model), val_loader, device, cfg, channels_last=channels_last)
            rmse = float(val_items.get("rmse", float("inf")))
            is_best = rmse < best_rmse
            if is_best:
                best_rmse = rmse

            record: dict[str, Any] = {"epoch": epoch, **train_items, **{f"val_{k}": v for k, v in val_items.items()}, "best_rmse": best_rmse}
            append_csv(log_csv, record)
            write_jsonl(log_jsonl, record)
            logger.info("epoch=%d train_loss=%.6f val_rmse=%.6f best=%.6f", epoch, train_items.get("train_loss", 0.0), rmse, best_rmse)

            save_every = int(cfg.get("outputs", {}).get("save_every", 1))
            save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scaler, scheduler, epoch, best_rmse, cfg)
            if save_every > 0 and (epoch + 1) % save_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scaler, scheduler, epoch, best_rmse, cfg)
            if is_best:
                save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scaler, scheduler, epoch, best_rmse, cfg)
            backup_root_value = cfg.get("outputs", {}).get("backup_root")
            if backup_root_value:
                backup_root = Path(str(backup_root_value))
                backup_ckpt = ensure_dir(backup_root / "checkpoints")
                backup_logs = ensure_dir(backup_root / "logs")
                shutil.copy2(ckpt_dir / "last.pth", backup_ckpt / "last.pth")
                if (ckpt_dir / "best.pth").exists():
                    shutil.copy2(ckpt_dir / "best.pth", backup_ckpt / "best.pth")
                for log_path in (log_csv, log_jsonl, student_root / "logs" / log_name):
                    if log_path.exists():
                        shutil.copy2(log_path, backup_logs / log_path.name)
                logger.info("Backed up epoch %d checkpoint/logs to %s", epoch, backup_root)
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    cfg, paths = load_project_config(args.config)
    train(cfg, paths)


if __name__ == "__main__":
    main()
