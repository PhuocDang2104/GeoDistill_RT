"""CLI: prepare / smoke / train / benchmark / compare. Sequential A100 ablations."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import experiment as E
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import prepare, KITTIDataset, write_json, digest
from metrics import Metrics
from boundaries import BoundaryMetrics
from loss_helpers import pooled
from selection import update_policy_best
from objective import loss_fn


def setup(cfg, require_gpu=True):
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("A100 GPU runtime required")
    if require_gpu and ("A100" not in torch.cuda.get_device_name(0)
            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise RuntimeError("This registered runtime comparison requires native-BF16 A100")
    torch.backends.cuda.matmul.allow_tf32 = cfg["tf32"]
    torch.backends.cudnn.allow_tf32 = cfg["tf32"]
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    return torch.device("cuda" if require_gpu else "cpu")


def worker_init(_):
    torch.set_num_threads(1)
    cv2.setNumThreads(0)


def loader(cfg, split, generator=None):
    workers = cfg["workers"]
    kwargs = dict(prefetch_factor=cfg["prefetch_factor"], persistent_workers=True,
                  worker_init_fn=worker_init) if workers else {}
    return DataLoader(KITTIDataset(cfg, split, teacher=split == "train"),
        batch_size=cfg["batch_size"] if split == "train" else 1,
        shuffle=split == "train", generator=generator, num_workers=workers,
        pin_memory=True, drop_last=False, **kwargs)


def move(batch, device, cfg):
    return {k: (v.to(device, non_blocking=True, memory_format=torch.channels_last)
                if v.ndim == 4 and cfg["channels_last"] else v.to(device, non_blocking=True))
            if torch.is_tensor(v) else v for k, v in batch.items()}


def augment(batch, cfg):
    # RNG independent of model construction/capture; saved and restored on resume.
    flip = torch.rand(batch["rgb"].shape[0], device=batch["rgb"].device) < .5
    if cfg["horizontal_flip"]:
        original = batch["K"]
        batch = {k: torch.where(flip[:, None, None, None], v.flip(-1), v)
                 if torch.is_tensor(v) and v.ndim == 4 else v for k, v in batch.items()}
        batch["K"] = original.clone()
        batch["K"][:, 0, 2] = torch.where(flip, batch["rgb"].shape[-1]-1-original[:, 0, 2],
                                        original[:, 0, 2])
    holdout = batch["mask"]*(torch.rand_like(batch["mask"]) < cfg["holdout_rate"])
    inputs = dict(batch, mask=batch["mask"]*(1-holdout))
    return batch, inputs, holdout


def run_dir(cfg, variant):
    return Path(cfg["drive_runs"]) / cfg["run_name"] / variant


def contract(cfg, variant):
    config = {k: v for k, v in cfg.items() if k not in
              ("drive_data", "drive_runs", "work", "workers", "prefetch_factor")}
    return {"variant": variant, "config": config, "source_sha256": E.source_hash(),
            "data": json.loads((Path(cfg["work"])/"data_contract.json").read_text())}


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+".partial")
    torch.save(payload, partial)
    partial.replace(path)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


@torch.no_grad()
def validate(model, batches, cfg, device, runner=None):
    metrics, boundary = Metrics(device), BoundaryMetrics(device)
    stage_names = ("D16", "D8", "D0", "D4_step1", "D4", "D2_pre_fine", "D2", "D1", "D_full")
    sums = torch.zeros(len(stage_names), 3, dtype=torch.float64, device=device)
    jpir_sums=torch.zeros(4,dtype=torch.float64,device=device)
    jpir_max=torch.zeros(2,dtype=torch.float64,device=device)
    model.eval()
    model.set_diagnostics(True)
    for batch in batches:
        batch = move(batch, device, cfg)
        with E.amp(device):
            prediction = model(*(batch[key] for key in E.INPUTS)) if runner is None else None
        final = prediction["D_full"] if runner is None else runner(batch)
        metrics.update(final, batch["gt"], batch["gt_mask"], batch["rgb"])
        boundary.update(final, batch["gt"], batch["gt_mask"])
        if prediction is not None:
            raw_u=prediction['query_uncertainty'].float()
            normalized=raw_u.clamp_min(0)/(raw_u.clamp_min(0)+cfg['upir_uncertainty_scale'])
            jpir_sums+=torch.stack((raw_u.sum(dtype=torch.float64),
                normalized.sum(dtype=torch.float64),raw_u.new_full((),raw_u.numel(),dtype=torch.float64),
                normalized.sum(dtype=torch.float64)*model.detail1.pir.body[0].weight[:,40:].abs().mean()))
            jpir_max=torch.maximum(jpir_max,torch.stack((raw_u.max(),normalized.max())).double())
            for i, name in enumerate(stage_names):
                p = prediction[name]
                target, support = pooled(batch["gt"], batch["gt_mask"], p.shape[-2:])
                valid = support > 0
                sums[i] += torch.stack((((p-target).square()*valid).sum(dtype=torch.float64),
                    ((p-target).abs()*valid).sum(dtype=torch.float64), valid.sum()))
    report = metrics.report()
    if report["all"]["pixels"] != 25424992:
        raise RuntimeError("Validation GT support changed; expected exactly 25424992")
    native = {name: {"rmse_m": (se/n)**.5, "mae_m": ae/n, "pixels": int(n)}
              for name, (se, ae, n) in zip(stage_names, sums.cpu().tolist()) if n}
    raw_sum,u_sum,n,shift_sum=jpir_sums.cpu().tolist()
    raw_max,u_max=jpir_max.cpu().tolist()
    return {"final": report, "gt_boundary": boundary.report(),
            "native_stage_metrics": native,
            "upir_diagnostics":{"raw_uncertainty_mean":raw_sum/n if n else 0,
                "raw_uncertainty_max":raw_max,"normalized_uncertainty_mean":u_sum/n if n else 0,
                "normalized_uncertainty_max":u_max,"abs_input_projection_contribution_mean":shift_sum/n if n else 0}}


def learning_rate(step, total, warmup, minimum):
    if step < warmup:
        return max(.01, (step+1)/warmup)
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1,total-warmup-1))))


def load_checkpoint(cfg, variant, device):
    path = run_dir(cfg, variant)/"best_policy.pth"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["contract"] != contract(cfg, variant):
        raise RuntimeError("Checkpoint source/config/data protocol changed")
    model = E.make_model(cfg, variant, device)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval(), ckpt, path


def train(cfg, variant):
    from train_graph import TrainingForward
    device = setup(cfg)
    E.assert_config(cfg)
    root = run_dir(cfg, variant)
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader, val_loader = loader(cfg, "train", generator), loader(cfg, "val")
    if len(train_loader.dataset) % cfg["batch_size"]:
        raise RuntimeError("CUDA graph requires full fixed-size batches; do not drop samples")
    resume = root/"last.pth"
    model = E.make_model(cfg, variant, device, pretrained=not resume.is_file())
    initial_hash = E.common_weight_hash(model)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr_scale": cfg["encoder_lr_ratio"]},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr_scale": 1.}
    ], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], fused=True)
    state = {"epoch": -1, "global_step": 0, "best_rmse": float("inf"),
             "best_inverse": float("inf"), "selection": {}, "rows": [],
             "initial_shared_sha256": initial_hash}
    if resume.is_file():
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        if ckpt["contract"] != contract(cfg, variant):
            raise RuntimeError("Strict resume mismatch: new source/recipe needs new RUN_TAG")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        restore_rng(ckpt["rng"], generator)
        print(f"Resume {variant}: next epoch={state['epoch']+1}", flush=True)
    else:
        # Reset data/augmentation RNG after construction, identical across A/B/C.
        E.seed_all(cfg["seed"])
    write_json(root/"resolved_config.json", cfg)
    graph = None
    start = state["epoch"]+1
    for epoch in range(start, cfg["epochs"]):
        # A fresh fixed epoch seed also reproduces order after Colab worker restart.
        # Do not depend on persistent-worker iterator/base-seed consumption history.
        generator.manual_seed(cfg["seed"]+epoch)
        if cfg["workers"]:
            old_iterator=getattr(train_loader,"_iterator",None)
            if old_iterator is not None:old_iterator._shutdown_workers()
        train_loader=loader(cfg,"train",generator)
        model.train()
        E.freeze_encoder_bn(model)
        totals, seen = {}, 0
        torch.cuda.synchronize()
        begun = time.perf_counter()
        for index, cpu in enumerate(train_loader):
            batch, inputs, holdout = augment(move(cpu, device, cfg), cfg)
            progress = epoch+index/len(train_loader)
            if graph is None:
                # Loss sees original mask/sensor; forward sees held-out input mask.
                # Capture gate must use exactly that same division.
                graph = TrainingForward(model, inputs, cfg, holdout, progress, original_batch=batch)
                write_json(root/"training_graph_report.json", graph.report)
            optimizer.zero_grad(set_to_none=True)
            prediction = graph(inputs)
            loss, stats = loss_fn(prediction, batch, holdout, progress, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss {variant} epoch={epoch} batch={index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            factor = learning_rate(state["global_step"], cfg["epochs"]*len(train_loader),
                                   cfg["warmup_epochs"]*len(train_loader), cfg["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = cfg["learning_rate"]*group["lr_scale"]*factor
            optimizer.step()
            state["global_step"] += 1
            size = batch["rgb"].shape[0]
            for name, value in stats.items():
                totals[name] = totals.get(name, 0) + value.detach()*size
            seen += size
            if (index+1) % cfg["log_every"] == 0:
                print(f"{variant} epoch={epoch} batch={index+1}/{len(train_loader)} loss={float(loss):.4f}", flush=True)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter()-begun
        if seen != 1600:
            raise RuntimeError(f"Train coverage {seen}/1600")
        validation = validate(model, val_loader, cfg, device)
        score = validation["final"]["all"]
        rmse, inverse = score["rmse_m"], score["irmse_km_inv"]
        if not math.isfinite(rmse+inverse):
            raise RuntimeError("Nonfinite validation, checkpoint not saved")
        selection, policy_gain = update_policy_best(state["selection"], rmse, inverse, epoch, cfg)
        rmse_gain, inverse_gain = rmse < state["best_rmse"], inverse < state["best_inverse"]
        state.update(epoch=epoch, best_rmse=min(rmse,state["best_rmse"]),
                     best_inverse=min(inverse,state["best_inverse"]), selection=selection)
        row = {"epoch":epoch, "updates":state["global_step"], "train_samples":seen,
               "upir_input_weight_abs_mean":float(model.detail1.pir.body[0].weight[:,40:].detach().abs().mean()),
               **{f"val_upir_{k}":v for k,v in validation['upir_diagnostics'].items()},
               "train_seconds":training_seconds, "epoch_seconds":time.perf_counter()-begun,
               **{f"val_stage_{k}_rmse_m":v['rmse_m'] for k,v in validation['native_stage_metrics'].items()},
               "val_rmse_m":rmse, "val_irmse_km_inv":inverse, "val_mae_m":score["mae_m"],
               "val_boundary3_rmse_m":validation["gt_boundary"]["bands"]["3"]["rmse_m"],
               **{f"loss_{k}":float(v/seen) for k,v in totals.items()}}
        state["rows"].append(row)
        payload = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "state":state,
                   "contract":contract(cfg,variant), "rng":rng_state(generator)}
        for label, save in (("best_policy",policy_gain),("best_rmse",rmse_gain),("best_inverse",inverse_gain)):
            if save:
                atomic_save(payload,root/f"{label}.pth")
                write_json(root/f"{label}_val_metrics.json",dict(epoch=epoch,**validation))
        atomic_save(payload,resume)
        with (root/"train_log.csv").open("w",newline="",encoding="utf-8") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(row))
            writer.writeheader()
            writer.writerows(state["rows"])
        write_json(root/"training_status.json", {
            "completed_epochs":epoch+1,"requested_epochs":cfg["epochs"],
            "status":"complete" if epoch+1==cfg["epochs"] else "running","selection":selection,
            "initial_shared_sha256":state["initial_shared_sha256"],
            "parameters":sum(p.numel() for p in model.parameters())})
        print(f"{variant} [{epoch+1}/{cfg['epochs']}] RMSE={rmse:.6f} iRMSE={inverse:.6f} U_weight={row['upir_input_weight_abs_mean']:.6f}",flush=True)


def smoke(cfg, variant):
    from train_graph import TrainingForward
    device = setup(cfg)
    model = E.make_model(cfg,variant,device,pretrained=False).train()
    initial_hash=E.common_weight_hash(model)
    E.seed_all(cfg["seed"])
    E.freeze_encoder_bn(model)
    cpu = next(iter(loader(cfg,"train")))
    batch,inputs,holdout = augment(move(cpu,device,cfg),cfg)
    graph = TrainingForward(model,inputs,cfg,holdout,1.,original_batch=batch)
    output = graph(inputs)
    loss,_ = loss_fn(output,batch,holdout,1.,cfg)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                      for p in model.parameters()):
        raise RuntimeError("GPU smoke failed")
    root=run_dir(cfg,variant);root.mkdir(parents=True,exist_ok=True)
    write_json(root/"smoke_report.json",{"passed":True,"loss":float(loss.detach()),
        "upir":{"input_channels":41,"added_parameters":16,
            "new_weight_abs_max":float(model.detail1.pir.body[0].weight[:,40:].detach().abs().max())},
        "coarse":model.dynamics.last_solver_report,
        "fine":{"method":"none","nfe":0},"training_graph":graph.report,
        "shared_initial_sha256":initial_hash})
    write_json(root/'architecture.json',E.architecture(model))
    # Test parser/build/capture BEFORE spending 30 epochs on an unexportable arm.
    del output,loss,graph
    model.zero_grad(set_to_none=True)
    model.eval();model.set_diagnostics(False)
    gc.collect();torch.cuda.empty_cache()
    from deployment import typed_deployment,export_onnx,build_engine,TRTRunner,parity
    export=typed_deployment(model).eval()
    single={k:batch[k][:1].contiguous() for k in E.INPUTS}
    with torch.no_grad():
        export_onnx(export,single,root/"smoke.onnx",cfg["benchmark"]["export_opset"])
        build=build_engine(root/"smoke.onnx",root/"smoke.plan",
                           cfg["benchmark"]["trt_workspace_gib"],cfg["tf32"])
        runner=TRTRunner(root/"smoke.plan",single,capture=True)
        check=parity(runner(single),export(*(single[k] for k in E.INPUTS)),cfg["benchmark"])
    write_json(root/"deployment_smoke.json",dict(build=build,parity=check,untrained=True))
    if not check["passed"]:raise RuntimeError("Pre-training TensorRT graph smoke parity failed")


def compare(cfg):
    rows=[]
    shared=[]
    for variant in E.VARIANTS:
        root=run_dir(cfg,variant)
        report=json.loads((root/"trt_benchmark.json").read_text())
        if not report["qualification_passed"]:
            raise RuntimeError(f"Unqualified TensorRT result: {variant}")
        shared.append(report["initial_shared_sha256"])
        row={"variant":variant,"checkpoint_epoch":report["checkpoint_epoch"],
             "rmse_m":report["trt_graph"]["all"]["rmse_m"],
             "irmse_km_inv":report["trt_graph"]["all"]["irmse_km_inv"],
             "mae_m":report["trt_graph"]["all"]["mae_m"],
             "boundary3_rmse_m":report["trt_graph_boundary"]["bands"]["3"]["rmse_m"],
             "median_ms":report["timing"]["trt_graph"]["synchronized_wall"]["median_ms"],
             "p95_ms":report["timing"]["trt_graph"]["synchronized_wall"]["p95_ms"],
             "parameters":report["parameters"]}
        rows.append(row)
    if len(set(shared))!=1:
        raise RuntimeError("Actual pretrained common initialization differed across trained arms")
    for row in rows:
        row.update(rmse_improvement_pct=100*(1-row["rmse_m"]/rows[0]["rmse_m"]),
                   speedup_vs_A=rows[0]["median_ms"]/row["median_ms"])
    root=Path(cfg["drive_runs"])/cfg["run_name"]
    with (root/"comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/"comparison.json",rows)
    preview_dir=root/"comparison_previews"
    preview_dir.mkdir(exist_ok=True)
    for index in range(0,1000,100):
        pair=cv2.imread(str(run_dir(cfg,E.VARIANTS[0])/"test_preview"/f"{index:04d}_rgb_depth.png"))
        panels=[pair[:,:1216]]
        labels=["RGB"]
        for variant in E.VARIANTS:
            panels.append(cv2.imread(str(run_dir(cfg,variant)/"test_preview"/f"{index:04d}_depth.png")))
            labels.append(variant)
        # Vertically stacked full-resolution rows preserve pixel readability.
        named=[]
        for label,panel in zip(labels,panels):
            title=np.full((32,panel.shape[1],3),245,np.uint8)
            cv2.putText(title,label,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1,cv2.LINE_AA)
            named.append(np.concatenate((title,panel),axis=0))
        cv2.imwrite(str(preview_dir/f"{index:04d}_RGB_A_B_C.png"),np.concatenate(named,axis=0))
    print(json.dumps(rows,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","smoke","train","benchmark","compare"))
    parser.add_argument("--config",required=True)
    parser.add_argument("--variant",choices=E.VARIANTS,default=E.VARIANTS[0])
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    E.assert_config(cfg)
    if args.action=="prepare":prepare(cfg)
    elif args.action=="benchmark":
        from trt_benchmark import benchmark
        try:benchmark(cfg,args.variant)
        except Exception as error:
            import traceback
            write_json(run_dir(cfg,args.variant)/"benchmark_failure.json",
                {"qualification_passed":False,"error":str(error),"traceback":traceback.format_exc(),
                 "torch":torch.__version__,"cuda":torch.version.cuda,
                 "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
            raise
    elif args.action=="compare":
        from compare_history import compare_history
        compare_history(cfg)
    else:globals()[args.action](cfg,args.variant)


if __name__=="__main__":
    main()
