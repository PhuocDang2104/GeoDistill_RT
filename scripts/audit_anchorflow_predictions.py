"""Read-only 400-val parent audit: signed bias, tail SSE, same-mask v3/v4 no-op probe."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "drive_upload/AnchorFlow_Research"
    sys.path.insert(0, str(folder))
    from data import KITTIDataset, write_json
    from model import AnchorFlowEdge, load_parent_state
    torch.set_num_threads(2)
    torch.manual_seed(42)
    payload = torch.load(folder / "init_v3_best.pth", map_location="cpu", weights_only=False)
    parent = AnchorFlowEdge(model_name="v3").eval().to(memory_format=torch.channels_last)
    child = AnchorFlowEdge(model_name="v4").eval().to(memory_format=torch.channels_last)
    load_parent_state(parent, payload["model"])
    migration = load_parent_state(child, payload["model"])
    ds = object.__new__(KITTIDataset)
    ds.root = root / "data/teacher_subset_2000/kitti_bundle"
    ds.split, ds.teacher = "val", False
    ds.rows = [r.split() for r in (ds.root / "splits/val_400.txt").read_text().splitlines()]
    stats = np.zeros((5, 4), np.float64)  # count, SSE, SAE, signed sum
    tails = {v: [0, 0.] for v in (1, 2, 5, 10, 20)}
    rows, all_errors, max_noop = [], [], 0.
    with torch.inference_mode():
        for i, batch in enumerate(DataLoader(ds, batch_size=1, num_workers=0), 1):
            inputs = tuple(batch[k].contiguous(memory_format=torch.channels_last) if batch[k].ndim==4 else batch[k]
                           for k in ("rgb", "sparse", "mask", "K"))
            pred = parent(*inputs)["D_full"]
            if i in (1, 200, 400):
                candidate = child(*inputs)["D_full"]
                max_noop = max(max_noop, float((candidate-pred).abs().max()))
                torch.testing.assert_close(candidate, pred, atol=1e-5, rtol=1e-6)
            valid = batch["gt_mask"].bool()
            error = (pred-batch["gt"])[valid].numpy().astype(np.float64)
            gt = batch["gt"][valid].numpy()
            squared = error**2
            all_errors.append(error.astype(np.float32))
            for j,(low,high) in enumerate(zip((0,20,40,60,80),(20,40,60,80,120))):
                mask=(gt>=low)&(gt<high)
                stats[j] += (mask.sum(), squared[mask].sum(), np.abs(error[mask]).sum(), error[mask].sum())
            for threshold in tails:
                mask=np.abs(error)>threshold
                tails[threshold][0] += int(mask.sum())
                tails[threshold][1] += float(squared[mask].sum())
            rows.append({"sid":batch["sid"][0],"pixels":len(error),"sse":float(squared.sum()),"rmse_m":float(np.sqrt(squared.mean()))})
            if i%50==0:
                print(f"Parent prediction audit {i}/400",flush=True)
    count,sse,sae,signed=stats.sum(axis=0)
    errors=np.concatenate(all_errors)
    report={"samples":400,"device":"CPU FP32","model":"trained v3 parent; no training/threshold fitting",
            "valid_pixels":int(count),"rmse_m":float(np.sqrt(sse/count)),"mae_m":float(sae/count),
            "signed_bias_m":float(signed/count),"migration":migration,"v4_D_full_noop_max_abs_error_m":max_noop,
            "ranges":{f"{low}-{high}":{"pixels":int(n),"rmse_m":float(np.sqrt(ss/n)),"mae_m":float(ab/n),"signed_bias_m":float(b/n),"sse_share":float(ss/sse)}
                      for (low,high),(n,ss,ab,b) in zip(zip((0,20,40,60,80),(20,40,60,80,120)),stats)},
            "tails":{str(t):{"pixels":n,"pixel_share":n/count,"sse_share":ss/sse} for t,(n,ss) in tails.items()},
            "abs_error_quantiles_m":dict(zip(("p50","p90","p95","p99","p99_9"),np.quantile(np.abs(errors),(.5,.9,.95,.99,.999)).tolist())),
            "worst_20_samples_by_sse":sorted(rows,key=lambda r:r["sse"],reverse=True)[:20]}
    write_json(root/"results/anchorflow_v4_audit/parent_prediction_audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="worst_20_samples_by_sse"},indent=2),flush=True)


if __name__ == "__main__":
    main()
