"""Commands: prepare, smoke, train, evaluate, test, profile, export."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import KITTIDataset, SIZE, digest, prepare, write_json
from losses import objective, pooled
from metrics import Metrics
from boundaries import BoundaryMetrics, BOUNDARY_PROTOCOL
from model import AnchorFlowEdge, Deploy
from model_v8 import AnchorFlowEdge as V8Control


def source_hash():
    sha = hashlib.sha256()
    for name in ("model.py", "model_v8.py", "support.py", "boundaries.py", "core.py", "losses.py", "losses_v8.py", "loss_helpers.py", "data.py", "relative_data.py", "metrics.py", "run.py"):
        sha.update((Path(__file__).parent / name).read_bytes())
    return sha.hexdigest()


def protocol(config):
    ignored = {"drive_data", "drive_runs", "work", "run_name", "workers",
               "log_every", "profile_warmup", "profile_runs", "compile", "init_checkpoint"}
    contract = json.loads((Path(config["work"]) / "data_contract.json").read_text(encoding="utf-8"))
    return {"config": {k: v for k, v in config.items() if k not in ignored},
            "data": contract, "source_sha256": source_hash()}


def recipe_hash(config):
    ignored = {"model_name", "architecture", "drive_data", "drive_runs", "work", "run_name",
               "init_checkpoint", "workers", "log_every", "profile_warmup", "profile_runs"}
    value = {"config": {k: v for k, v in config.items() if k not in ignored},
             "data": json.loads((Path(config["work"]) / "data_contract.json").read_text()),
             "shared_sources": {name: digest(Path(__file__).with_name(name)) for name in
                                ("losses.py", "loss_helpers.py", "boundaries.py", "data.py", "metrics.py", "run.py")}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_setup(config, require_gpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_gpu and device.type != "cuda":
        raise RuntimeError("Select a GPU runtime in Colab before training")
    if device.type == "cuda" and config["amp"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 student training requires supported GPU (e.g. L4/A100/Blackwell); do not silently fallback to FP16")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def autocast(config, device):
    if device.type == "cuda" and config["amp"] != "fp32":
        dtype = torch.bfloat16 if config["amp"] == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def make_model(config, device, pretrained):
    control = config["model_name"] == "v8_control"
    builder = V8Control if control else AnchorFlowEdge
    model = builder(pretrained=pretrained, flow_steps=config["flow_steps"], encoder=config["encoder"],
                    model_name="v8_dynamics" if control else config["model_name"]).to(device)
    if config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    return model


def move_batch(batch, config, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
            if value.ndim == 4 and config["channels_last"]:
                value = value.contiguous(memory_format=torch.channels_last)
        result[key] = value
    return result


def data_loader(config, split, generator=None, batch_size=None):
    dataset = KITTIDataset(config, split, teacher=split == "train" and config["teacher_enabled"])
    workers = int(config["workers"])
    options = {"prefetch_factor": 2, "persistent_workers": True} if workers else {}
    return DataLoader(dataset, batch_size=batch_size or (config["batch_size"] if split == "train" else 1),
                      shuffle=split == "train", num_workers=workers, pin_memory=True,
                      drop_last=False, generator=generator, **options)


def augment(batch, config):
    if not config["horizontal_flip"]:
        return batch
    flip = torch.rand(batch["rgb"].shape[0], device=batch["rgb"].device) < 0.5
    result = {}
    for name, tensor in batch.items():
        if torch.is_tensor(tensor) and tensor.ndim == 4:
            result[name] = torch.where(flip[:, None, None, None], tensor.flip(-1), tensor)
        else:
            result[name] = tensor
    K = batch["K"].clone()
    K[:, 0, 2] = torch.where(flip, batch["rgb"].shape[-1] - 1 - K[:, 0, 2], K[:, 0, 2])
    result["K"] = K
    return result


def input_with_holdout(batch, rate):
    holdout = batch["mask"] * (torch.rand_like(batch["mask"]) < rate)
    mask = batch["mask"] * (1 - holdout)
    # Keeping original S in loss is intentional; model sanitizes it with input M.
    return mask, holdout


def directories(config, variant):
    local = Path(config["work"]) / "runs" / config["run_name"] / variant
    drive = Path(config["drive_runs"]) / config["run_name"] / variant
    local.mkdir(parents=True, exist_ok=True)
    drive.mkdir(parents=True, exist_ok=True)
    return local, drive


def copy_atomic(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, partial)
    partial.replace(destination)


def restore_rng(checkpoint, generator):
    random.setstate(checkpoint["rng_python"])
    np.random.set_state(checkpoint["rng_numpy"])
    torch.set_rng_state(checkpoint["rng_torch"])
    generator.set_state(checkpoint["rng_loader"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])


def save_checkpoint(model, optimizer, scaler, epoch, best, step, generator, contract):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_rmse": best, "global_step": step, "protocol": contract,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(), "rng_loader": generator.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def load_trained(config, variant, device):
    _, drive = directories(config, variant)
    checkpoint = torch.load(drive / "best.pth", map_location="cpu", weights_only=False)
    if checkpoint["protocol"] != protocol(config):
        raise RuntimeError("Checkpoint/protocol mismatch: use the resolved config and unchanged source bundle")
    model = make_model(config, device, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


@torch.inference_mode()
def validate(model, loader, config, device):
    score, pre_anchor, hard_anchor = Metrics(device), Metrics(device), Metrics(device)
    boundary, boundary_pre = BoundaryMetrics(device), BoundaryMetrics(device)
    stages = ("D16", "D8", "D0", "D4_step1", "D4_step2", "D4_step3", "D4", "D2_base", "D2_query", "D2", "D1_base", "D1", "D_full", "D_hard")
    minima = torch.full((len(stages),), float("inf"), device=device)
    low_counts = torch.zeros(len(stages), device=device, dtype=torch.int64)
    stage_sums = torch.zeros(len(stages), 3, dtype=torch.float64, device=device)
    sensor_sums = torch.zeros(5, dtype=torch.float64, device=device)
    query_names = ("query_uncertainty_mean", "query_gate_mean", "query_center_weight_mean", "query_entropy_mean")
    query_sums = torch.zeros(len(query_names)+1, dtype=torch.float64, device=device)
    for batch in loader:
        batch = move_batch(batch, config, device)
        with autocast(config, device):
            output = model(batch["rgb"], batch["sparse"], batch["mask"], batch["K"])
        if query_names[0] in output:
            count = batch["rgb"].shape[0]
            query_sums[:-1] += torch.stack([output[name].double() for name in query_names])*count
            query_sums[-1] += count
        score.update(output["D_full"], batch["gt"], batch["gt_mask"], batch["rgb"])
        pre_anchor.update(output["D1"], batch["gt"], batch["gt_mask"], batch["rgb"])
        hard_anchor.update(output["D_hard"], batch["gt"], batch["gt_mask"], batch["rgb"])
        boundary.update(output["D_full"],batch["gt"],batch["gt_mask"])
        boundary_pre.update(output["D1"],batch["gt"],batch["gt_mask"])
        overlap = batch["mask"] * batch["gt_mask"]
        sensor_sums += torch.stack((((batch["sparse"]-batch["gt"]).square()*overlap).sum(dtype=torch.float64),
                                    overlap.sum(), batch["gt_mask"].sum(), output["sensor_gate"].sum(dtype=torch.float64),
                                    batch["mask"].sum()))
        minima = torch.minimum(minima, torch.stack([output[name].float().min() for name in stages]))
        low_counts += torch.stack([(output[name] < 0.5).sum() for name in stages])
        targets = {}
        for i, name in enumerate(stages):
            depth = output[name].float()
            shape = depth.shape[-2:]
            if shape not in targets:
                targets[shape] = pooled(batch["gt"], batch["gt_mask"], shape)
            target, support = targets[shape]
            valid = support > 0
            error = depth - target
            stage_sums[i] += torch.stack(((error.square() * valid).sum(dtype=torch.float64),
                                         (error.abs() * valid).sum(dtype=torch.float64), valid.sum()))
    native = {name: {"rmse_m": (sse / n) ** .5 if n else None, "mae_m": sae / n if n else None, "pixels": int(n)}
              for name, (sse, sae, n) in zip(stages, stage_sums.cpu().tolist())}
    sensor = sensor_sums.cpu().tolist()
    return {"final": score.report(), "pre_anchor": pre_anchor.report(), "legacy_hard_anchor": hard_anchor.report(),
            "hard_anchor_oracle_floor_m": (sensor[0] / sensor[2]) ** .5 if sensor[2] else None,
            "sensor_gate_mean_on_observed": sensor[3] / sensor[4] if sensor[4] else None,
            "output_policy": "learned confidence fusion (no GT in forward); legacy hard anchor logged separately",
            "stage_native_gt_metrics": native,
            "stage_metric_protocol": "Native stage vs valid-area-mean GT. D0 and D4_step1/2/3 share the quarter-grid target. Different scales do NOT share identical targets; dynamics does NOT guarantee monotone RMSE per step.",
            "stage_min_m": dict(zip(stages, minima.cpu().tolist())),
            "stage_count_below_0_5": dict(zip(stages, low_counts.cpu().tolist())),
            "gt_boundary":boundary.report(),"gt_boundary_pre_anchor":boundary_pre.report(),
            "query_diagnostics": {name: float(value/query_sums[-1].clamp_min(1))
                                  for name,value in zip(query_names,query_sums[:-1])}
                                 if config["model_name"] == "v9_consensus" else {},
            "query_diagnostics_protocol": "Sample-weighted full-frame means, not GT-masked or calibrated uncertainty"}


def learning_rate(step, total, warmup, minimum):
    if step < warmup:
        return max(0.01, (step + 1) / max(1, warmup))
    phase = (step - warmup) / max(1, total - warmup - 1)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * min(1, phase)))


def early_stop_update(state, rmse, epoch, config):
    """Patience tracks cumulative meaningful gains; checkpoint best remains raw minimum.

    min_epochs is a stopping floor, not a reset of pre-floor observations.
    Initial untrained validation (epoch -1) is never counted.
    """
    state = dict(state)
    if state["monitor_best_rmse"] is None or rmse < state["monitor_best_rmse"]-config["early_stop_min_delta_m"]:
        state.update(monitor_best_rmse=rmse, bad_epochs=0)
    else:
        state["bad_epochs"] += 1
    state["stopped"] = bool(config["early_stopping"] and epoch+1 >= config["early_stop_min_epochs"]
                            and state["bad_epochs"] >= config["early_stop_patience"])
    return state


def train(config, variant):
    device = device_setup(config, require_gpu=True)
    seed_everything(config["seed"])
    local, drive = directories(config, variant)
    contract = protocol(config)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader, val_loader = data_loader(config, "train", generator), data_loader(config, "val")
    resume_file = drive / "last.pth"
    # Avoid re-downloading pretrained weights when restoring the complete student.
    model = make_model(config, device, pretrained=config["encoder_pretrained"] and not resume_file.is_file())
    if not resume_file.is_file():
        # Never load a V5/V6/V7 student. Only ImageNet encoder initialization.
        seed_everything(config["seed"])
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    groups = [{"params": list(model.encoder.parameters()), "lr": config["learning_rate"] * config["encoder_lr_ratio"],
               "lr_scale": config["encoder_lr_ratio"]},
              {"params": [p for p in model.parameters() if id(p) not in encoder_ids],
               "lr": config["learning_rate"], "lr_scale": 1.0}]
    optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                 fused=config["fused_adamw"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] == "fp16")
    start, best, global_step = 0, float("inf"), 0
    stopping = {"monitor_best_rmse": None, "bad_epochs": 0, "stopped": False}
    if resume_file.is_file():
        checkpoint = torch.load(resume_file, map_location="cpu", weights_only=False)
        if checkpoint["protocol"] != contract:
            raise RuntimeError("Resume protocol changed; use a different run_name for a new experiment")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, global_step = checkpoint["epoch"] + 1, checkpoint["best_rmse"], checkpoint["global_step"]
        stopping = checkpoint["early_stopping"]
        restore_rng(checkpoint, generator)
        for name in ("train_log.csv", "train_log.jsonl", "train.log"):
            if (drive / name).is_file():
                shutil.copy2(drive / name, local / name)
        # Handle an interruption after log append but before checkpoint sync.
        csv_path = local / "train_log.csv"
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames, rows = reader.fieldnames, [r for r in reader if int(r["epoch"]) < start]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        jsonl_path = local / "train_log.jsonl"
        if jsonl_path.is_file():
            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records if r["epoch"] < start), encoding="utf-8")
        print(f"Resume {variant}: next epoch={start}, best={best:.4f} m", flush=True)
        if stopping["stopped"]:
            # Recover status too if Colab died after checkpoint sync but before report sync.
            write_json(drive / "training_status.json", {
                "status": "early_stopped", "epochs_completed": start, "max_epochs": config["epochs"],
                "best_rmse_m": best, "early_stopping": stopping, "patience": config["early_stop_patience"],
                "min_delta_m": config["early_stop_min_delta_m"], "min_epochs": config["early_stop_min_epochs"]})
            print("Run already early-stopped; preserving best checkpoint. Use a new run name/recipe to train again.", flush=True)
            return
    if start == 0 and not resume_file.is_file():
        model.eval()
        initial = validate(model, val_loader, config, device)
        best = initial["final"]["all"]["rmse_m"]
        if best is None or not math.isfinite(best):
            raise RuntimeError("Initial validation is nonfinite; no checkpoint saved")
        write_json(local / "initial_val_metrics.json", {"epoch": -1, **initial})
        payload = save_checkpoint(model, optimizer, scaler, -1, best, 0, generator, contract)
        payload["early_stopping"] = stopping
        torch.save(payload, local / "best.pth")
        copy_atomic(local / "best.pth", drive / "best.pth")
        copy_atomic(local / "initial_val_metrics.json", drive / "initial_val_metrics.json")
        copy_atomic(local / "initial_val_metrics.json", drive / "best_val_metrics.json")
        print(f"INITIAL (not newly trained) soft={best:.4f}; hard={initial['legacy_hard_anchor']['all']['rmse_m']:.4f}; pre={initial['pre_anchor']['all']['rmse_m']:.4f}", flush=True)
    write_json(local / "resolved_config.json", config)
    write_json(local / "run_manifest.json", {"protocol": contract,
               "parameters": sum(p.numel() for p in model.parameters()),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "teacher_at_inference": False, "pretrained_normalization": model.encoder.pretrained_cfg["mean"],
               "student_checkpoint_loaded": resume_file.is_file(), "encoder_pretrained": config["encoder_pretrained"],
               "output_policy": "learned sensor reliability fusion", "extra_epochs": config["epochs"],
               "note": "Fresh student + ImageNet RGB only. V9 consensus readout uses metric and relative training targets; no teacher inference. V8 historical run is not a matched precision/initialization ablation; optional v8_control trains fresh BF16 on the same split.",
               "relative_teacher_used": config.get("relative_enabled", False),
               "boundary_protocol":BOUNDARY_PROTOCOL,
               "recipe_sha256": recipe_hash(config), "model_name": config["model_name"]})
    for name in ("resolved_config.json", "run_manifest.json"):
        copy_atomic(local / name, drive / name)
    forward_model = torch.compile(model) if config["compile"] else model
    accumulation = int(config["accumulation"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * config["epochs"]
    warmup_updates = int(updates_per_epoch * config["warmup_epochs"])
    for epoch in range(start, config["epochs"]):
        model.train()
        if config["freeze_encoder_bn"]:
            model.freeze_encoder_bn()
        optimizer.zero_grad(set_to_none=True)
        totals, keys, seen = None, None, 0
        started = time.perf_counter()
        for index, batch in enumerate(train_loader):
            batch = augment(move_batch(batch, config, device), config)
            mask, holdout = input_with_holdout(batch, config["holdout_rate"])
            window = min(accumulation, len(train_loader) - (index // accumulation) * accumulation)
            with autocast(config, device):
                output = forward_model(batch["rgb"], batch["sparse"], mask, batch["K"])
            loss, stats = objective(output, batch, holdout, epoch + index / len(train_loader), config)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss BEFORE backward at epoch={epoch} batch={index+1}; no batch skipping or NaN masking")
            scaler.scale(loss / window).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                factor = learning_rate(global_step, total_updates, warmup_updates, config["min_lr_ratio"])
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * group["lr_scale"] * factor
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            keys = list(stats)
            batch_size = batch["rgb"].shape[0]
            vector = torch.stack([stats[key].float() for key in keys]) * batch_size
            totals = vector if totals is None else totals + vector
            seen += batch_size
            if (index + 1) % config["log_every"] == 0:
                values = (totals / seen).cpu().tolist()  # one synchronization for all logged terms
                snapshot = dict(zip(keys, values))
                if not math.isfinite(snapshot["total"]):
                    raise RuntimeError(f"Nonfinite training loss at epoch={epoch} batch={index+1}")
                message = (f"{variant} epoch={epoch} batch={index+1}/{len(train_loader)} "
                           f"loss={snapshot['total']:.4f} KD={snapshot['weighted_metric_kd']:.4f}")
                print(message, flush=True)
                with (local / "train.log").open("a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
        train_seconds = time.perf_counter() - started
        means = dict(zip(keys, (totals / seen).cpu().tolist()))
        if not all(math.isfinite(v) for v in means.values()):
            raise RuntimeError("Nonfinite loss statistics; checkpoint not saved")
        model.eval()
        validation = validate(model, val_loader, config, device)
        score = validation["final"]["all"]
        rmse = score["rmse_m"]
        if rmse is None or not math.isfinite(rmse):
            raise RuntimeError("Validation returned no valid finite RMSE")
        improved = rmse < best
        best = min(best, rmse)
        stopping = early_stop_update(stopping, rmse, epoch, config)
        surface_norm = sum(float(p.detach().float().norm()) for name,p in model.named_parameters()
                           if name.startswith("dynamics.") and name.endswith("weight"))
        new_norm = sum(float(p.detach().float().norm()) for name,p in model.named_parameters()
                       if name.startswith("dynamics.reaction") and name.endswith("weight"))
        row = {"epoch": epoch, "cumulative_epoch_index": epoch,
               "dynamics_parameter_norm": surface_norm,
               "reaction_parameter_norm": new_norm,
               "train_samples": seen, "val_samples": len(val_loader.dataset),
               "global_step": global_step, "lr_decoder": optimizer.param_groups[1]["lr"],
               "lr_encoder": optimizer.param_groups[0]["lr"],
               "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - started,
               **{f"loss_{k}": v for k, v in means.items()},
               "val_rmse_m": rmse, "val_mae_m": score["mae_m"], "val_irmse_km_inv": score["irmse_km_inv"],
               "early_stop_bad_epochs": stopping["bad_epochs"],
               "early_stop_monitor_best_rmse_m": stopping["monitor_best_rmse"],
               "early_stop_triggered": stopping["stopped"],
               "val_pre_anchor_rmse_m": validation["pre_anchor"]["all"]["rmse_m"],
               "val_legacy_hard_rmse_m": validation["legacy_hard_anchor"]["all"]["rmse_m"],
               "val_native_D0_rmse_m": validation["stage_native_gt_metrics"]["D0"]["rmse_m"],
               "val_native_D4_rmse_m": validation["stage_native_gt_metrics"]["D4"]["rmse_m"],
               **{f"val_native_D4_step{k}_rmse_m":validation["stage_native_gt_metrics"][f"D4_step{k}"]["rmse_m"] for k in (1,2,3)},
               "val_native_D2_base_rmse_m":validation["stage_native_gt_metrics"]["D2_base"]["rmse_m"],
               "val_native_D2_query_rmse_m":validation["stage_native_gt_metrics"]["D2_query"]["rmse_m"],
               "val_native_D2_rmse_m":validation["stage_native_gt_metrics"]["D2"]["rmse_m"],
               **{f"val_tail_{t}m_sse_fraction": validation["final"]["all"]["error_tail"][str(t)]["sse_fraction"] for t in (1,2,5,10,20)},
               **{f"val_tail_{t}m_sse_m2": validation["final"]["all"]["error_tail"][str(t)]["sse_m2"] for t in (1,2,5,10,20)},
               **{f"val_tail_{t}m_pixel_fraction": validation["final"]["all"]["error_tail"][str(t)]["pixel_fraction"] for t in (1,2,5,10,20)},
               **{f"val_rmse_{name}_m": validation["final"][name]["rmse_m"]
                  for name in ("0-20", "20-40", "40-60", "60-80", "80-120", "edge")},
               **{f"val_gt_boundary_{r}px_rmse_m":validation["gt_boundary"]["bands"][str(r)]["rmse_m"]
                  for r in (1,2,3,5,10)}}
        with (local / "train_log.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        with (local / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": row, "validation": validation}) + "\n")
        payload = save_checkpoint(model, optimizer, scaler, epoch, best, global_step, generator, contract)
        payload["early_stopping"] = stopping
        torch.save(payload, local / "last.pth.partial")
        (local / "last.pth.partial").replace(local / "last.pth")
        if improved:
            shutil.copy2(local / "last.pth", local / "best.pth")
            write_json(local / "best_val_metrics.json", {"epoch": epoch, **validation})
            # Sync best before last so a resumed run cannot lose its best checkpoint.
            copy_atomic(local / "best.pth", drive / "best.pth")
            copy_atomic(local / "best_val_metrics.json", drive / "best_val_metrics.json")
        copy_atomic(local / "last.pth", drive / "last.pth")
        write_json(local / "training_status.json", {
            "status": "early_stopped" if stopping["stopped"] else ("max_epochs_completed" if epoch+1==config["epochs"] else "running"),
            "epochs_completed": epoch+1, "max_epochs": config["epochs"], "best_rmse_m": best,
            "early_stopping": stopping, "patience": config["early_stop_patience"],
            "min_delta_m": config["early_stop_min_delta_m"], "min_epochs": config["early_stop_min_epochs"]})
        copy_atomic(local / "training_status.json", drive / "training_status.json")
        for name in ("train_log.csv", "train_log.jsonl", "train.log"):
            if (local / name).is_file():
                copy_atomic(local / name, drive / name)
        print(f"{variant} epoch={epoch} val_rmse={rmse:.4f} m iRMSE={score['irmse_km_inv']:.3f} best={best:.4f}; backed up to {drive}", flush=True)
        if stopping["stopped"]:
            print(f"EARLY STOP: {epoch+1} epochs completed; no meaningful gain for {stopping['bad_epochs']} epochs. Evaluation/test use best.pth, not last.pth.", flush=True)
            break


def smoke(config):
    device = device_setup(config)
    seed_everything(config["seed"])
    model = make_model(config, device, pretrained=config["encoder_pretrained"])
    model.train()
    if config["freeze_encoder_bn"]:
        model.freeze_encoder_bn()
    batch = move_batch(next(iter(data_loader(config, "train", batch_size=1))), config, device)
    mask, holdout = input_with_holdout(batch, config["holdout_rate"])
    with autocast(config, device):
        output = model(batch["rgb"], batch["sparse"], mask, batch["K"])
    loss, stats = objective(output, batch, holdout, 0, config)
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss is nonfinite")
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Smoke gradients are nonfinite")
    if not torch.equal(output["D_hard"][mask.bool()], batch["sparse"][mask.bool()]):
        raise RuntimeError("Legacy hard-anchor diagnostic contract failed")
    if not torch.isfinite(output["D_full"]).all() or not ((output["sensor_gate"] >= 0) & (output["sensor_gate"] <= 1)).all():
        raise RuntimeError("Sensor trust/soft output contract failed")
    if config["teacher_enabled"] and float(stats["kd_coverage"]) <= 0:
        raise RuntimeError("KD has zero coverage")
    if config.get("relative_enabled", False) and float(stats["relative_pair_coverage"]) <= 0:
        raise RuntimeError("Relative KD has zero unobserved pair coverage; inspect teacher confidence/masks")
    reaction_grad = model.dynamics.reaction.weight.grad
    if reaction_grad is None or not reaction_grad.abs().sum() > 0:
        raise RuntimeError("Dynamics reaction receives no learning signal")
    if any(not float(stats[f"dynamics_state_change_{k}"]) > 0 for k in (1,2,3)):
        raise RuntimeError("Jet state did not evolve across all three steps")
    write_json(Path(config["work"]) / "smoke_report.json", {
        "loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters()),
        "kd_coverage": float(stats["kd_coverage"]), "device": str(device),
        "relative_pair_coverage": float(stats["relative_pair_coverage"]),
        "teacher_at_inference": False, "student_checkpoint_loaded": False,
        "imagenet_encoder_only": config["encoder_pretrained"],
        "reaction_gradient_l1":float(reaction_grad.abs().sum()),
        "dynamics":{key:float(value) for key,value in stats.items() if key.startswith("dynamics_")},
        "passed": True})
    print("REAL-DATA SMOKE PASS", flush=True)


def evaluate(config, variant):
    device = device_setup(config)
    model, checkpoint = load_trained(config, variant, device)
    report = validate(model, data_loader(config, "val"), config, device)
    _, drive = directories(config, variant)
    report.update({"checkpoint_epoch": checkpoint["epoch"], "samples": 400, "teacher_at_inference": False})
    write_json(drive / "val_metrics.json", report)
    print(json.dumps(report["final"], indent=2), flush=True)


@torch.inference_mode()
def test(config, variant):
    device = device_setup(config)
    model, checkpoint = load_trained(config, variant, device)
    local, drive = directories(config, variant)
    loader = data_loader(config, "test")
    zip_path = local / "kitti_test_predictions.zip"
    started = time.perf_counter()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as output:
        for i, batch in enumerate(loader, 1):
            batch = move_batch(batch, config, device)
            with autocast(config, device):
                pred = model(batch["rgb"], batch["sparse"], batch["mask"], batch["K"])["D_full"]
            depth = pred[0, 0].float().cpu().numpy()
            if depth.shape != SIZE or not np.isfinite(depth).all() or np.min(depth) <= 0:
                raise RuntimeError(f"Invalid test prediction: {batch['sid'][0]}")
            ok, png = cv2.imencode(".png", np.clip(np.rint(depth * 256), 1, 65535).astype(np.uint16))
            if not ok:
                raise RuntimeError("PNG encode failed")
            output.writestr(f"{batch['sid'][0]}.png", png.tobytes())
            if i % 100 == 0:
                print(f"Test {i}/1000", flush=True)
    with zipfile.ZipFile(zip_path) as check:
        if len(check.namelist()) != 1000 or len(set(check.namelist())) != 1000:
            raise RuntimeError("Test ZIP must have 1000 unique PNG files")
    copy_atomic(zip_path, drive / zip_path.name)
    write_json(drive / "test_report.json", {"samples": 1000, "checkpoint_epoch": checkpoint["epoch"],
               "teacher_used": False, "anonymous_no_gt": True,
               "wall_seconds_including_data_io_and_png": time.perf_counter() - started,
               "shape": list(SIZE), "encoding": "uint16 round(depth_m * 256)"})
    print("Test ZIP:", drive / zip_path.name, flush=True)


def sample_inputs(device, height=352, width=1216, channels_last=True):
    rgb = torch.rand(1, 3, height, width, device=device)
    mask = (torch.rand(1, 1, height, width, device=device) < 0.05).float()
    sparse = (3 + torch.rand_like(mask) * 77) * mask
    K = torch.tensor([[[0.6 * width, 0, width / 2], [0, 0.6 * width, height / 2], [0, 0, 1]]],
                     dtype=torch.float32, device=device)
    inputs = (rgb, sparse, mask, K)
    return tuple(t.contiguous(memory_format=torch.channels_last) if channels_last and t.ndim == 4 else t for t in inputs)


def count_operations(model, inputs):
    macs, handles = {}, []
    def hook(name):
        def record(module, args, output):
            group = name.split(".")[0]
            if isinstance(module, nn.Conv2d):
                operations = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            else:
                operations = output.numel() * module.in_features
            macs[group] = macs.get(group, 0) + operations
        return record
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        model(*inputs)
    for handle in handles:
        handle.remove()
    params = {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()}
    return {"conv_linear_macs": macs, "parameters": params,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "total_conv_linear_macs": sum(macs.values()),
            "mac_exclusions": ["functional 1->9 fixed neighbourhood convolutions", "fixed directional shifts and plane transport", "pooling", "interpolation",
                               "softmax", "finite differences", "elementwise operations", "memory traffic"]}


@torch.inference_mode()
def profile(config, variant, untrained=False):
    device = device_setup(config)
    model = make_model(config, device, pretrained=False).eval() if untrained else load_trained(config, variant, device)[0]
    seed_everything(config["seed"])  # reproducible synthetic input, independent of constructor RNG
    inputs = sample_inputs(device, channels_last=config["channels_last"])
    report = count_operations(model, inputs)
    measured = torch.compile(Deploy(model)) if config["compile"] else Deploy(model)
    for _ in range(config["profile_warmup"]):
        with autocast(config, device):
            measured(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(config["profile_runs"]):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            started = time.perf_counter()
        with autocast(config, device):
            measured(*inputs)
        if device.type == "cuda":
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end))
        else:
            latencies.append((time.perf_counter() - started) * 1000)
    # Separate eager component pass: instrumentation is excluded from total timing.
    timed_modules = dict(model.named_children())
    components = {name: [] for name in timed_modules}
    starts, hooks = {}, []
    def pre(name):
        def begin(module, args):
            if device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                starts[name] = event
            else:
                starts[name] = time.perf_counter()
        return begin
    def post(name):
        def finish(module, args, output):
            if device.type == "cuda":
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                components[name].append((starts[name], end))
            else:
                components[name].append((time.perf_counter() - starts[name]) * 1000)
        return finish
    for name, module in timed_modules.items():
        hooks.extend((module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))))
    for _ in range(min(20, config["profile_runs"])):
        with autocast(config, device):
            model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for handle in hooks:
        handle.remove()
    report["eager_component_median_ms"] = {
        name: float(np.median([a.elapsed_time(b) for a, b in times] if device.type == "cuda" else times))
        for name, times in components.items() if times}
    report.update({"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                   "torch": torch.__version__, "batch": 1, "shape": [352, 1216],
                   "precision": config["amp"] if device.type == "cuda" else "fp32",
                   "cpu_threads": torch.get_num_threads(),
                   "channels_last": config["channels_last"], "compiled_total": config["compile"],
                   "untrained_weights": untrained, "median_ms": float(np.median(latencies)),
                   "p95_ms": float(np.percentile(latencies, 95)), "runs": len(latencies),
                   "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None,
                   "inference_includes": "RGB encoder + sparse pyramid + F32 context + decoder + three shared jet Euler steps + phase query/detail + learned sensor fusion",
                   "phase_query": "five-jet inverse consensus + disagreement gate" if config["model_name"] == "v9_consensus" else "original V8 center-jet readout",
                   "excludes": "disk I/O and host-to-device transfer; there is no teacher/prior model",
                   "component_timings_are_separate_eager_instrumented_pass": True})
    _, drive = directories(config, variant)
    write_json(drive / "profile.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


@torch.inference_mode()
def export(config, variant, untrained=False):
    import onnx
    import onnxruntime as ort
    device = torch.device("cpu")
    model = make_model(config, device, pretrained=False).eval() if untrained else load_trained(config, variant, device)[0]
    model = model.to(memory_format=torch.contiguous_format)
    inputs = sample_inputs(device, channels_last=False)
    local, drive = directories(config, variant)
    path = local / "anchorflow_edge_fp32.onnx"
    deploy = Deploy(model).eval()
    torch.onnx.export(deploy, inputs, str(path), input_names=["rgb", "sparse", "mask", "K"],
                      output_names=["depth_m"], opset_version=17, do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    errors = []
    for use_anchors in (True, False):
        rgb, sparse, mask, K = inputs
        if not use_anchors:
            sparse, mask = torch.zeros_like(sparse), torch.zeros_like(mask)
        sample = (rgb, sparse, mask, K)
        expected = deploy(*sample).numpy()
        actual = session.run(None, {name: x.numpy() for name, x in zip(("rgb", "sparse", "mask", "K"), sample)})[0]
        max_error = float(np.max(np.abs(actual - expected)))
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=0.01, rtol=1e-4):
            raise RuntimeError(f"ONNX parity failed: max abs error={max_error} m")
        errors.append(max_error)
    copy_atomic(path, drive / path.name)
    report = {"opset": 17, "static_batch": 1, "shape": [352, 1216], "precision": "fp32",
              "max_abs_errors_m": errors, "untrained_weights": untrained,
              "onnxruntime_cpu_parity": True, "teacher_inputs": False,
              "inputs": ["rgb", "sparse", "mask", "K"], "outputs": ["depth_m"],
              "operators": sorted({node.op_type for node in graph.graph.node}),
              "tensorrt_engine_built": False, "target_device_latency_measured": False}
    write_json(drive / "export_report.json", report)
    print("ONNX checker + CPU runtime parity PASS:", drive / path.name, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "train", "evaluate", "test", "profile", "export"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--variant", choices=("dual_teacher", "metric_kd", "gt_only"), default="dual_teacher")
    parser.add_argument("--untrained", action="store_true", help="Structural profile/export only; never an accuracy benchmark")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["teacher_enabled"] = args.variant in ("metric_kd", "dual_teacher")
    config["relative_enabled"] = args.variant == "dual_teacher"
    if config["model_name"] == "v8_control" and (config["relative_enabled"] or config.get("inverse_weight", 0) != 0):
        raise ValueError("Matched V8 control requires metric_kd, relative disabled and inverse_weight=0")
    if config["flow_steps"] != 3 or config["accumulation"] < 1 or config["epochs"] < 1:
        raise ValueError("Unsupported flow_steps/accumulation/epochs")
    if config["amp"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("amp must be fp32, fp16 or bf16")
    if config["early_stop_patience"] < 1 or config["early_stop_min_epochs"] < 1 or config["early_stop_min_delta_m"] < 0:
        raise ValueError("Invalid early-stop patience/min_epochs/min_delta")
    if Path(config["run_name"]).name != config["run_name"] or config["run_name"] in ("", ".", ".."):
        raise ValueError("run_name must be a single directory name")
    if config.get("init_checkpoint") or config.get("init_source"):
        raise ValueError("V8 fresh workflow does not accept old student initialization")
    if args.command == "profile":
        profile(config, args.variant, args.untrained)
    elif args.command == "export":
        globals()[args.command](config, args.variant, args.untrained)
    elif args.command in ("prepare", "smoke"):
        globals()[args.command](config)
    else:
        globals()[args.command](config, args.variant)


if __name__ == "__main__":
    main()
