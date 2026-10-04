"""Read-only trained-checkpoint audit; minmod is an explicit experimental probe."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnx
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"drive_upload/AnchorFlow_v9_Consensus"
AUDIT=ROOT/"results/anchorflow_v9_completed_audit"
sys.path.insert(0,str(FOLDER))
import run
import model_v8
from model import AnchorFlowEdge
from data import KITTIDataset,write_json
from metrics import Metrics


def minmod_derivatives(x):
    east,west,south,north=model_v8.adjacent(x).unbind(1)
    a,b=east-x,x-west; c,d=south-x,x-north
    gx=.5*(a.sign()+b.sign())*torch.minimum(a.abs(),b.abs())
    gy=.5*(c.sign()+d.sign())*torch.minimum(c.abs(),d.abs())
    gx=torch.cat((a[...,:1],gx[...,1:-1],b[...,-1:]),-1)
    gy=torch.cat((c[...,:1,:],gy[...,1:-1,:],d[...,-1:,:]),-2)
    return gx,gy


def model_and_dataset():
    model=AnchorFlowEdge().eval()
    payload=torch.load(AUDIT/"best.pth",map_location="cpu",weights_only=False)
    model.load_state_dict(payload["model"],strict=True)
    ds=object.__new__(KITTIDataset); ds.root=ROOT/"data/teacher_subset_2000/kitti_bundle"
    ds.split,ds.teacher="val",False
    ds.rows=[r.split() for r in (ds.root/"splits/val_400.txt").read_text().splitlines()]
    return model,ds


def batched(ds,i):
    return {k:(v[None] if torch.is_tensor(v) else v) for k,v in ds[i].items()}


def export_probe(model):
    keys=("D0","D4_step1","D4_step2","D4","D2_base","D2_query","D2","D1_base","D1","D_full")
    class Stages(nn.Module):
        def __init__(self): super().__init__(); self.model=model
        def forward(self,rgb,sparse,mask,K):
            out=self.model(rgb,sparse,mask,K)
            return tuple(out[k] for k in keys)
    deploy=Stages().eval()
    torch.manual_seed(42)
    inputs=run.sample_inputs(torch.device("cpu"),channels_last=False)
    results={}
    original=model_v8.derivatives
    for label,derivative in (("original",original),("minmod_probe",minmod_derivatives)):
        model_v8.derivatives=derivative
        path=AUDIT/("stage_probe_"+label+".onnx")
        torch.onnx.export(deploy,inputs,str(path),input_names=["rgb","sparse","mask","K"],
                          output_names=list(keys),opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        options=ort.SessionOptions(); options.intra_op_num_threads=2; options.log_severity_level=3
        session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
        cases={}
        for empty in (False,True):
            rgb,s,m,K=inputs; sample=(rgb,s*0,m*0,K) if empty else inputs
            with torch.no_grad(): expected=deploy(*sample)
            actual=session.run(None,{k:v.numpy() for k,v in zip(("rgb","sparse","mask","K"),sample)})
            rows={k:{"max_abs_m":float(np.max(np.abs(a-e.numpy()))),
                     "mean_abs_m":float(np.mean(np.abs(a-e.numpy()))),
                     "allclose_atol_0_01_rtol_1e_4":bool(np.allclose(a,e.numpy(),atol=.01,rtol=1e-4))}
                  for k,a,e in zip(keys,actual,expected)}
            cases["empty_sparse" if empty else "observed_sparse"]=rows
        results[label]=cases
        del session; gc.collect()
    model_v8.derivatives=original
    write_json(AUDIT/"trained_onnx_derivative_probe.json",results)
    print("EXPORT PROBE",json.dumps(results,indent=2),flush=True)


@torch.inference_mode()
def pixel_audit(model,ds):
    model=model.to(memory_format=torch.channels_last)
    score=Metrics(torch.device("cpu"))
    groups={}
    top=[]
    for i in range(400):
        batch=batched(ds,i)
        out=model(*(batch[k] for k in ("rgb","sparse","mask","K")))
        gt=batch["gt"]; valid=batch["gt_mask"].bool(); error=out["D_full"]-gt
        score.update(out["D_full"],gt,batch["gt_mask"],batch["rgb"])
        uncertainty=F.interpolate(out["query_uncertainty"],size=gt.shape[-2:],mode="nearest")
        delta2=F.interpolate(out["phase2_delta"].abs()/(1+.05*out["D2_base"]),size=gt.shape[-2:],mode="nearest")
        delta1=out["delta1"].abs()/(.5+.05*out["D1_base"])
        sparse_near=F.max_pool2d(batch["mask"],17,1,8)>.5
        for label,mask in (("all",valid),("tail_gt5",valid&(error.abs()>5)),
                           ("tail_gt20",valid&(error.abs()>20)),
                           ("near_0_20",valid&(gt<20)),("far_40_80",valid&(gt>=40)&(gt<80))):
            n=int(mask.sum())
            if not n: continue
            entry=groups.setdefault(label,{"pixels":0,"sse":0.,"overprediction":0,"sensor_observed":0,
                                           "sparse_within_8px":0,"U_sum":0.,"U_gt_01":0,
                                           "D2_delta_gt_09":0,"D1_delta_gt_09":0,"abs_delta1_sum":0.})
            entry["pixels"]+=n; entry["sse"]+=float(error[mask].square().double().sum())
            entry["overprediction"]+=int(((error>0)&mask).sum())
            entry["sensor_observed"]+=int((batch["mask"].bool()&mask).sum())
            entry["sparse_within_8px"]+=int((sparse_near&mask).sum())
            entry["U_sum"]+=float(uncertainty[mask].double().sum())
            entry["U_gt_01"]+=int(((uncertainty>.1)&mask).sum())
            entry["D2_delta_gt_09"]+=int(((delta2>.9)&mask).sum())
            entry["D1_delta_gt_09"]+=int(((delta1>.9)&mask).sum())
            entry["abs_delta1_sum"]+=float(out["delta1"][mask].abs().double().sum())
        se=error.square().masked_fill(~valid,-1).flatten()
        values,locations=torch.topk(se,10)
        for value,location in zip(values.tolist(),locations.tolist()):
            y,x=divmod(location,1216)
            top.append({"sample_id":batch["sid"],"u":x,"v":y,"sq_error_m2":value,
                        "gt_m":float(gt[0,0,y,x]),"pred_m":float(out["D_full"][0,0,y,x]),
                        "D1_base_m":float(out["D1_base"][0,0,y,x]),"D1_m":float(out["D1"][0,0,y,x]),
                        "sparse_m":float(batch["sparse"][0,0,y,x]),"U":float(uncertainty[0,0,y,x]),
                        "delta1_saturation_ratio":float(delta1[0,0,y,x])})
        top=sorted(top,key=lambda x:x["sq_error_m2"],reverse=True)[:100]
        if (i+1)%50==0: print(f"V9 actual checkpoint FP32 audit {i+1}/400",flush=True)
    for entry in groups.values():
        n=entry["pixels"]
        entry["rmse_m"]=(entry["sse"]/n)**.5
        for name in ("overprediction","sensor_observed","sparse_within_8px","U_gt_01","D2_delta_gt_09","D1_delta_gt_09"):
            entry[name+"_fraction"]=entry[name]/n
        entry["U_mean"]=entry["U_sum"]/n; entry["abs_delta1_mean_m"]=entry["abs_delta1_sum"]/n
    report={"precision":"CPU FP32; not uploaded GPU BF16 evaluation","groups":groups,
            "final":score.report(),"top_100":top,"label_artifacts_not_classified":True}
    write_json(AUDIT/"trained_pixel_audit_fp32.json",report)
    print("PIXEL AUDIT",json.dumps(groups,indent=2),flush=True)


def main():
    torch.set_num_threads(4)
    model,ds=model_and_dataset()
    print("MODEL STRICT LOAD",run.source_hash(),flush=True)
    export_probe(model)
    pixel_audit(model,ds)


if __name__=="__main__": main()
