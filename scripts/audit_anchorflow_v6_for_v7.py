"""Restore exact student artifacts; causal inference-only ablation on 400 GT images."""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root/"drive_upload/AnchorFlow_v7_Jet"
    sys.path.insert(0, str(folder))
    from model import AnchorFlowEdge, load_parent_state
    from data import KITTIDataset, write_json
    from metrics import Metrics
    archive = root/"results/metric_kd-20261002T124131Z-1-001.zip"
    restore = {"best.pth": "init_v6_best.pth", "train_log.csv": "v6_train_log.csv",
               "val_metrics.json": "parent_val_metrics.json", "initial_val_metrics.json": "v6_initial_val_metrics.json",
               "resolved_config.json": "v6_resolved_config.json", "run_manifest.json": "v6_run_manifest.json"}
    with zipfile.ZipFile(archive) as z:
        for member, name in restore.items():
            target = folder/name
            raw = z.read("metric_kd/"+member)
            if target.exists() and target.read_bytes()!=raw:
                raise RuntimeError(f"Refusing to overwrite a different artifact: {target}")
            if not target.exists():
                with target.open("wb") as handle:
                    handle.write(raw)
    payload = torch.load(folder/"init_v6_best.pth", map_location="cpu", weights_only=False)
    torch.set_num_threads(2)
    full = AnchorFlowEdge(model_name="v6_connection").eval()
    reduced = AnchorFlowEdge(model_name="v6_reduced").eval()
    new = AnchorFlowEdge().eval()
    full.load_state_dict(payload["model"], strict=True)
    reduced.load_state_dict(payload["model"], strict=True)
    migration = load_parent_state(new, payload["model"])
    ds = object.__new__(KITTIDataset)
    ds.root = root/"data/teacher_subset_2000/kitti_bundle"
    ds.split, ds.teacher = "val", False
    ds.rows = [r.split() for r in (ds.root/"splits/val_400.txt").read_text().splitlines()]
    scores = {key: Metrics("cpu") for key in ("full_v6", "reduced_v6")}
    noop, delta = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(min(args.samples, len(ds))):
            row = ds[i]
            inputs = tuple(row[k].unsqueeze(0) for k in ("rgb", "sparse", "mask", "K"))
            old, control = full(*inputs), reduced(*inputs)
            if i in (0, 199, 399):
                candidate = new(*inputs)
                error = float((candidate["D_full"]-control["D_full"]).abs().max())
                torch.testing.assert_close(candidate["D_full"], control["D_full"], atol=1e-5, rtol=1e-6)
                noop.append({"index": i, "max_abs_m": error})
            delta.append(float((old["D_full"]-control["D_full"]).abs().mean()))
            for key, prediction in (("full_v6", old), ("reduced_v6", control)):
                scores[key].update(prediction["D_full"], row["gt"].unsqueeze(0), row["gt_mask"].unsqueeze(0), row["rgb"].unsqueeze(0))
            if (i+1)%25==0:
                print(f"Causal ablation {i+1}/{args.samples}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    report = {"samples": min(args.samples, len(ds)), "split": "existing val_400", "device": "CPU", "precision": "fp32",
              "mode": "inference-only removal, NOT independently trained ablation", "checkpoint_epoch": payload["epoch"],
              "checkpoint_best_rmse_m": payload["best_rmse"], "parent_protocol": payload["protocol"],
              "checkpoint_sha256": hashlib.sha256((folder/"init_v6_best.pth").read_bytes()).hexdigest(),
              "migration": migration, "v7_zero_head_vs_reduced_v6": noop,
              "mean_pixel_change_m": sum(delta)/len(delta), "elapsed_seconds": time.perf_counter()-started,
              "metrics": {key: value.report() for key,value in scores.items()}}
    write_json(folder/"v6_causal_audit.json", report)
    print(json.dumps({k:v["all"] for k,v in report["metrics"].items()}, indent=2), flush=True)


if __name__=="__main__":
    main()
