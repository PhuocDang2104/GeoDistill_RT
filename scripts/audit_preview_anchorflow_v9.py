"""Audit completed V9 and render ten actual matched RGB/V8/V9 test predictions."""
import gc
import hashlib
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"results/anchorflow_v9_completed_audit"
PREVIEW=ROOT/"results/v9_depth_preview"
V9=ROOT/"results/dual_teacher-20261004T040538Z-1-001.zip"
V8=ROOT/"results/metric_kd-20261003T091559Z-1-001.zip"


def selected_predictions(path,prefix,ids):
    with zipfile.ZipFile(path) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read(prefix+"/kitti_test_predictions.zip"))) as inner:
            assert len(inner.namelist())==len(set(inner.namelist()))==1000
            return {sid:inner.read(sid+".png") for sid in ids}


def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    for directory in (PREVIEW,PREVIEW/"pairs",PREVIEW/"depth_color",PREVIEW/"depth_metric_uint16"):
        directory.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(V9) as source:
        for member in source.namelist():
            p=Path(member)
            if p.parent.as_posix()=="dual_teacher" and (p.suffix in (".json",".csv",".jsonl",".log") or p.name=="best.pth"):
                (AUDIT/p.name).write_bytes(source.read(member))
        new=json.loads(source.read("dual_teacher/val_metrics.json"))
        newprofile=json.loads(source.read("dual_teacher/profile.json"))
    old=json.loads((ROOT/"results/anchorflow_v8_completed_audit/val_metrics.json").read_text())
    oldprofile=json.loads((ROOT/"results/anchorflow_v8_completed_audit/profile.json").read_text())
    assert old["final"]["all"]["pixels"]==new["final"]["all"]["pixels"]==25424992
    rows=[]
    def metric(name,a,b): rows.append({"metric":name,"V8":a,"V9":b,"delta_V9_minus_V8":b-a,"change_percent":100*(b/a-1) if a else None})
    for name in ("rmse_m","mae_m","irmse_km_inv","imae_km_inv","abs_rel","delta1"):
        metric(name,old["final"]["all"][name],new["final"]["all"][name])
    for band in ("0-20","20-40","40-60","60-80","80-120","edge","non_edge"):
        metric("rmse_"+band,old["final"][band]["rmse_m"],new["final"][band]["rmse_m"])
    for t in ("2","5","10","20"):
        for kind in ("sse_m2","pixel_fraction"):
            metric(f"tail>{t}_{kind}",old["final"]["all"]["error_tail"][t][kind],new["final"]["all"]["error_tail"][t][kind])
    metric("boundary3_rmse",old["gt_boundary"]["bands"]["3"]["rmse_m"],new["gt_boundary"]["bands"]["3"]["rmse_m"])
    for key in ("total_parameters","total_conv_linear_macs","median_ms","p95_ms","peak_cuda_allocated_mib"):
        metric(key,oldprofile[key],newprofile[key])
    comparison=pd.DataFrame(rows); comparison.to_csv(AUDIT/"v8_vs_v9_full_comparison.csv",index=False)
    stages=[]
    for key,a in old["stage_native_gt_metrics"].items():
        b=new["stage_native_gt_metrics"][key]
        assert a["pixels"]==b["pixels"]
        stages.append({"stage":key,"V8_RMSE_m":a["rmse_m"],"V9_RMSE_m":b["rmse_m"],"delta_m":b["rmse_m"]-a["rmse_m"],"pixels":a["pixels"]})
    pd.DataFrame(stages).to_csv(AUDIT/"stage_comparison.csv",index=False)
    log=pd.read_csv(AUDIT/"train_log.csv"); best=log.loc[log.val_rmse_m.idxmin()]
    budget=best[[c for c in log if c.startswith("loss_weighted_")]].rename(lambda s:s.removeprefix("loss_weighted_"))
    budget.to_frame("weighted_value").assign(fraction=budget/best.loss_total).to_csv(AUDIT/"best_loss_budget.csv")
    ids=[f"{i:010d}" for i in np.linspace(0,999,10,dtype=int)]
    raw={"V8":selected_predictions(V8,"metric_kd",ids)}
    gc.collect()
    raw["V9"]=selected_predictions(V9,"dual_teacher",ids)
    samples=[]; records=[]
    for sid in ids:
        rgb=cv2.cvtColor(cv2.imread(str(ROOT/"data/depth_selection/test_depth_completion_anonymous/image"/(sid+".png"))),cv2.COLOR_BGR2RGB)
        depths={}
        for version in ("V8","V9"):
            png=raw[version][sid]
            encoded=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_UNCHANGED)
            assert encoded.dtype==np.uint16 and encoded.shape==(352,1216)
            depths[version]=encoded.astype(np.float32)/256
            assert np.isfinite(depths[version]).all()
            if version=="V9":
                (PREVIEW/"depth_metric_uint16"/(sid+".png")).write_bytes(png)
                plt.imsave(PREVIEW/"depth_color"/(sid+".png"),depths[version],cmap="turbo_r",vmin=0,vmax=80)
        samples.append((sid,rgb,depths))
        records.append({"id":sid,"v8_sha256":hashlib.sha256(raw["V8"][sid]).hexdigest(),
                        "v9_sha256":hashlib.sha256(raw["V9"][sid]).hexdigest(),
                        "v9_min_m":float(depths["V9"].min()),"v9_max_m":float(depths["V9"].max())})
    norm=Normalize(0,80,clip=True)
    for sid,rgb,depths in samples:
        fig,axes=plt.subplots(1,3,figsize=(24,3.8),layout="constrained")
        axes[0].imshow(rgb); axes[0].set_title("RGB | "+sid)
        for col,version in enumerate(("V8","V9"),1):
            im=axes[col].imshow(depths[version],cmap="turbo_r",norm=norm,interpolation="nearest")
            axes[col].set_title(version+" final metric depth | "+sid)
        for ax in axes: ax.axis("off")
        fig.colorbar(im,ax=axes[1:].tolist(),fraction=.018,extend="max",label="Metres (red near / blue far)")
        fig.savefig(PREVIEW/"pairs"/(sid+"_rgb_v8_v9.png"),dpi=160,facecolor="white"); plt.close(fig)
    for page in range(2):
        fig,axes=plt.subplots(5,3,figsize=(24,12),layout="constrained")
        fig.suptitle("Actual KITTI anonymous test | RGB / V8 / V9 | page "+str(page+1)+"/2\nShared 0-80 m scale (>80 display-saturated); no public test GT, no per-image normalization",fontsize=16)
        for row,(sid,rgb,depths) in enumerate(samples[page*5:page*5+5]):
            axes[row,0].imshow(rgb); axes[row,0].set_title("RGB | "+sid)
            for col,version in enumerate(("V8","V9"),1):
                im=axes[row,col].imshow(depths[version],cmap="turbo_r",norm=norm,interpolation="nearest")
                axes[row,col].set_title(version+" | "+sid)
            for ax in axes[row]: ax.axis("off")
        fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),fraction=.012,extend="max",label="Depth (m)")
        fig.savefig(PREVIEW/(f"rgb_v8_v9_sheet_{page+1}.png"),dpi=150,facecolor="white"); plt.close(fig)
    manifest={"source_v9":str(V9),"v9_checkpoint_epoch":new["checkpoint_epoch"],"samples":records,
              "selection":"ten evenly spaced IDs, no error-based cherry-picking","display_scale_m":[0,80],
              "prediction_encoding":"Actual uint16 PNG /256 = metres","test_rmse":None,
              "historical_v8_not_matched_training_initialization":True}
    (PREVIEW/"preview_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    summary={"V8_RMSE_m":old["final"]["all"]["rmse_m"],"V9_RMSE_m":new["final"]["all"]["rmse_m"],
             "V9_best_epoch":new["checkpoint_epoch"],"epochs":len(log),"validation_pixels":new["final"]["all"]["pixels"],
             "SSE_reduction_needed_for_0_9":1-(.9/new["final"]["all"]["rmse_m"])**2,
             "V9_query":new["query_diagnostics"],"best_train_rmse_proxy":float(best.loss_rmse),
             "best_loss_total":float(best.loss_total),"V9_export_failed":True}
    (AUDIT/"audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(comparison.to_string(index=False)); print(json.dumps(summary,indent=2)); print("PREVIEW",PREVIEW)


if __name__=="__main__": main()
