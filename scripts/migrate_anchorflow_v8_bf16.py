"""Explicit FP16 -> BF16 fork; never mutate the original run or checkpoint.

Upload this standalone script to /content and invoke with the frozen code/config.
The original trainer still enforces its full resume protocol in the new fork.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

import torch


def migrate_payload(payload, old_protocol, new_protocol):
    if payload["protocol"] != old_protocol:
        raise RuntimeError("Original checkpoint/config/source/data protocol mismatch; refusing migration")
    for name, tensor in payload["model"].items():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Checkpoint model state is already nonfinite: {name}")
    for state in payload["optimizer"]["state"].values():
        for name, tensor in state.items():
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                raise RuntimeError(f"Checkpoint optimizer state is nonfinite: {name}")
    result = dict(payload)
    result["protocol"] = new_protocol
    # Disabled GradScaler in BF16 expects an empty state. Preserve the old one
    # in provenance rather than loading FP16 scaling into this new experiment.
    result["scaler"] = {}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    sys.path.insert(0, str(args.code.resolve()))
    run = importlib.import_module("run")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg["amp"] != "fp16":
        raise RuntimeError("Pass the ORIGINAL FP16 resolved config, not an already edited config")
    old_protocol = run.protocol(cfg)
    variant = "metric_kd"
    old_dir = Path(cfg["drive_runs"]) / cfg["run_name"] / variant
    new_cfg = dict(cfg, amp="bf16", run_name=cfg["run_name"] + "_BF16_Recovery")
    new_protocol = run.protocol(new_cfg)
    new_dir = Path(new_cfg["drive_runs"]) / new_cfg["run_name"] / variant
    provenance_file = new_dir / "precision_migration.json"
    new_config = args.code / "resolved_config_bf16_recovery.json"
    if new_dir.exists():
        if not provenance_file.is_file():
            raise RuntimeError(f"Destination exists without completed migration; inspect it: {new_dir}")
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        if provenance["old_protocol"] != old_protocol or provenance["new_protocol"] != new_protocol:
            raise RuntimeError("Existing recovery fork does not match this config/source/data")
        checkpoint = torch.load(new_dir / "last.pth", map_location="cpu", weights_only=False)
        migrate_payload(checkpoint, new_protocol, new_protocol)  # validate finite state; do not rewrite
        print("Recovery fork already exists: preserving its latest checkpoint", flush=True)
    else:
        original = torch.load(old_dir / "last.pth", map_location="cpu", weights_only=False)
        migrated = migrate_payload(original, old_protocol, new_protocol)
        best = torch.load(old_dir / "best.pth", map_location="cpu", weights_only=False)
        migrated_best = migrate_payload(best, old_protocol, new_protocol)
        new_dir.mkdir(parents=True, exist_ok=False)
        # Stage checkpoints locally, then atomic-copy to Drive.
        for name, payload in (("last.pth", migrated), ("best.pth", migrated_best)):
            staging = args.code / ("migration_" + name)
            torch.save(payload, staging)
            run.copy_atomic(staging, new_dir / name)
            staging.unlink()
        for name in ("train_log.csv", "train_log.jsonl", "train.log", "initial_val_metrics.json", "best_val_metrics.json"):
            if (old_dir / name).is_file():
                run.copy_atomic(old_dir / name, new_dir / name)
        if (old_dir / "source_bundle").is_dir():
            shutil.copytree(old_dir / "source_bundle", new_dir / "source_bundle")
        provenance = {
            "reason": "FP16 encoder forward nonfinite; identical batch/state BF16 and FP32 finite",
            "original_run": str(old_dir), "new_run": str(new_dir),
            "next_epoch": original["epoch"] + 1,
            "old_protocol": old_protocol, "new_protocol": new_protocol,
            "old_scaler_state": original["scaler"],
            "original_last_sha256": hashlib.sha256((old_dir / "last.pth").read_bytes()).hexdigest(),
            "preserved": ["model", "optimizer", "epoch", "global_step", "RNG", "early_stopping", "best_rmse"],
            "comparison_note": "Inherited best and earlier log rows are FP16; later rows BF16. This is a recovery continuation, NOT a pure BF16 fresh run or matched precision ablation.",
        }
        run.write_json(provenance_file, provenance)
    run.write_json(new_config, new_cfg)
    print("BF16 config:", new_config, flush=True)
    print("BF16 output:", new_dir, flush=True)
    print("Original FP16 run remains untouched:", old_dir, flush=True)
    if args.train:
        objective = run.objective

        def finite_objective(*values, **options):
            loss, stats = objective(*values, **options)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("BF16 recovery objective nonfinite BEFORE backward; no bad batch skipped")
            return loss, stats

        run.objective = finite_objective
        print("Recovery observer: finite loss checked every batch before backward", flush=True)
        # Preserve an explicit pointer alongside the trainer's canonical manifest.
        run.write_json(new_dir / "recovery_status.json", {
            "precision": "bf16", "migration": str(provenance_file),
            "config": new_cfg, "finite_loss_guard": "before backward every batch",
        })
        run.train(new_cfg, variant)


if __name__ == "__main__":
    main()
