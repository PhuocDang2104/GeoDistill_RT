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
from .model_geort import GeoRTStudentS
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


@torch.no_grad()
def infer(cfg: dict[str, Any], paths: dict[str, str], checkpoint: str, split: str) -> None:
    device = device_from_config(str(cfg.get("device", "cuda")))
    student_root = Path(paths["student_root"])
    out_dir = ensure_dir(student_root / f"{split}_predictions")
    logger = setup_logger(student_root / "logs" / f"infer_{split}.log")

    loader = make_loader(cfg, paths, split)
    model = GeoRTStudentS.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    metrics_records: list[dict[str, float]] = []
    save_visual = bool(cfg.get("outputs", {}).get("save_visuals", True))
    for batch in tqdm(loader, desc=f"infer:{split}"):
        batch = to_device(batch, device)
        pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["ray"], batch["uv"])
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
        if save_visual:
            save_visuals(out_dir, sample_id, D_full, C_full)

        if batch["gt_mask"].sum().item() > 0:
            gt_mask = (batch["gt_mask"] > 0.5) & (batch["gt"] > 1e-3)
            if gt_mask.sum().item() > 0:
                metrics_records.append(depth_metrics_torch(pred.get("D_full", pred["D_c"]), batch["gt"], gt_mask))

    if metrics_records:
        metrics = average_metric_dict(metrics_records)
        with open(student_root / "logs" / f"infer_{split}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        logger.info("Inference metrics for %s: %s", split, metrics)


def main() -> None:
    args = parse_args()
    cfg, paths = load_project_config(args.config)
    infer(cfg, paths, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
