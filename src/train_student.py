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
from .metrics import average_metric_dict, depth_metrics_torch
from .model_geort import GeoRTStudentS
from .sparse_propagation import downsample_depth_with_mask
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
def validate(model: GeoRTStudentS, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="val", leave=False):
        batch = to_device(batch, device)
        pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"])
        D_c = pred["D_c"]
        scale = max(1, batch["gt"].shape[-2] // D_c.shape[-2])
        gt_ds, gt_mask_ds = downsample_depth_with_mask(batch["gt"], batch["gt_mask"], scale=scale)
        if gt_mask_ds.sum().item() < 1:
            continue
        metrics = depth_metrics_torch(D_c, gt_ds, gt_mask_ds > 0.5)
        records.append(metrics)
    return average_metric_dict(records) if records else {"rmse": float("inf"), "mae": float("inf"), "abs_rel": float("inf")}


def train(cfg: dict[str, Any], paths: dict[str, str]) -> None:
    seed_everything(int(cfg.get("seed", 42)))
    device = device_from_config(str(cfg.get("device", "cuda")))
    student_root = Path(paths["student_root"])
    logger = setup_logger(student_root / "logs" / "train_student.log")

    train_loader = make_loader(cfg, paths, "train", training=True)
    val_loader = make_loader(cfg, paths, "val", training=False)
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
    for epoch in range(start_epoch, epochs):
        model.train()
        running: dict[str, float] = {}
        count = 0
        pbar = tqdm(train_loader, desc=f"train:{epoch}")
        for batch in pbar:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"])
                loss, items = geort_loss(pred, batch, cfg["loss"], cfg["schedule"], epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            count += 1
            for key, value in items.items():
                running[key] = running.get(key, 0.0) + float(value)
            avg_loss = running["loss"] / count
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        train_items = {f"train_{k}": v / max(1, count) for k, v in running.items()}
        val_items = validate(model, val_loader, device)
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
