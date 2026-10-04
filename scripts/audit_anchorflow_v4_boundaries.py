"""Read-only trained-v4 audit on all 400 validation samples; no fitting."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"drive_upload/AnchorFlow_Research"))
    sys.path.append(str(root/"drive_upload/AnchorFlow_v5_Piecewise"))
    from model_v4 import AnchorFlowEdge
    from data import KITTIDataset,write_json,digest
    from boundaries import BoundaryMetrics,boundary_mask,band_mask,quarter_targets
    torch.set_num_threads(2)
    checkpoint = root/"results/anchorflow_v4_completed_audit/best.pth"
    payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
    model = AnchorFlowEdge().eval().to(memory_format=torch.channels_last)
    model.load_state_dict(payload["model"],strict=True)
    dataset = object.__new__(KITTIDataset)
    dataset.root = root/"data/teacher_subset_2000/kitti_bundle"
    dataset.split,dataset.teacher = "val",False
    dataset.rows = [r.split() for r in (dataset.root/"splits/val_400.txt").read_text().splitlines()]
    bands = BoundaryMetrics("cpu")
    ranges = np.zeros((5,4),np.float64)
    tails = {t:[0,0.,0.] for t in (1,2,5,10,20)}
    quarter = np.zeros(3,np.float64)
    rows = []
    with torch.inference_mode():
        for i,batch in enumerate(DataLoader(dataset,batch_size=1,num_workers=0),1):
            inputs = tuple(batch[k].contiguous(memory_format=torch.channels_last) if batch[k].ndim==4 else batch[k]
                           for k in ("rgb","sparse","mask","K"))
            prediction = model(*inputs)["D_full"]
            gt,valid = batch["gt"],batch["gt_mask"].bool()
            bands.update(prediction,gt,valid)
            band = band_mask(boundary_mask(gt,valid),3)[valid].numpy()
            error = (prediction-gt)[valid].numpy().astype(np.float64)
            truth = gt[valid].numpy()
            for j,(lo,hi) in enumerate(zip((0,20,40,60,80),(20,40,60,80,120))):
                m = (truth>=lo)&(truth<hi)
                ranges[j] += (m.sum(),(error[m]**2).sum(),np.abs(error[m]).sum(),error[m].sum())
            for t in tails:
                m = np.abs(error)>t
                tails[t] += np.array((m.sum(),(error[m]**2).sum(),(error[m&band]**2).sum()))
            label,support = quarter_targets(gt,valid)
            quarter += (float(support.sum()),float((label*support).sum()),support.numel())
            rows.append(dict(sid=batch["sid"][0],sse=float((error**2).sum()),pixels=len(error)))
            if i%25==0:
                print(f"Trained V4 GT-boundary audit {i}/400",flush=True)
    n,sse,sae,signed = ranges.sum(0)
    report = dict(samples=400,device="CPU FP32",checkpoint_epoch=payload["epoch"],checkpoint_sha256=digest(checkpoint),
                  valid_pixels=int(n),rmse_m=float((sse/n)**.5),mae_m=float(sae/n),signed_bias_m=float(signed/n),
                  gt_boundary=bands.report(),
                  tails={str(t):dict(pixels=int(count),pixel_fraction=count/n,sse_fraction_global=ss/sse,
                                    tail_sse_fraction_in_3px_gt_band=bs/ss if ss else 0) for t,(count,ss,bs) in tails.items()},
                  ranges={f"{lo}-{hi}":dict(pixels=int(c),rmse_m=float((ss/c)**.5),mae_m=ab/c,signed_bias_m=si/c,
                                               sse_fraction_global=ss/sse) for (lo,hi),(c,ss,ab,si) in zip(zip((0,20,40,60,80),(20,40,60,80,120)),ranges)},
                  quarter_label_support_fraction=quarter[0]/quarter[2],quarter_positive_fraction=quarter[1]/quarter[0],
                  worst_20_samples=sorted(rows,key=lambda r:r["sse"],reverse=True)[:20],
                  note="No training or threshold fitting. Fixed GT-discontinuity mask; sparse/accumulated KITTI GT is not perfect occlusion ground truth.")
    write_json(root/"results/anchorflow_v5_audit/v4_gt_boundary_audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("worst_20_samples","ranges")},indent=2),flush=True)


if __name__=="__main__":
    main()
