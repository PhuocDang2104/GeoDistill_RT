"""Generate the V7 Colab artifact from the audited TAR2000 workflow."""
from pathlib import Path
import nbformat


def main():
    root=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(root/"drive_upload/AnchorFlow_v6_Connection/AnchorFlow_v6_TAR2000_15ep.ipynb",as_version=4)
    def cell(i,s):
        notebook.cells[i].source=s.strip()+"\n"
    cell(0,"""
# AnchorFlow V7 — Projective Jet + Phase Query · FT15

Upload nguyên folder `AnchorFlow_v7_Jet` vào MyDrive, chọn GPU, Run all.
Đã bundle **best V6 thật, epoch14**. Không cần teacher weights/HF_TOKEN.

- Cùng1.600train/400val/1.000anonymous test; metric teacher chỉ train.
- Giữ trained context/phase CNN; thay vector correction bằng residual inverse-depth2-jet.
- Zero heads khớp **reduced V6**, không phải full V6. Đo full V6 trước train.
- Một run15epoch bổ sung; <0,8m là mục tiêu chưa được chứng minh.
- Squared-excess GT tail loss; không loại outlier/conflict GT khỏi evaluation.
""")
    cell(1,"""
## 1. Mount + config

Default `v7_jet`, parent bundled. Controls optional `v6_reduced` (same recipe),
`v7_no_transport` (T=0); không tự chạy nhiều run. Đổi batch/LR/loss cần RUN_TAG mới.
""")
    cell(2,"""
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
BUNDLE=Path("/content/drive/MyDrive/AnchorFlow_v7_Jet")
DRIVE_DATA=Path("/content/drive/MyDrive/GeoLift_Data")
DRIVE_RUNS=Path("/content/drive/MyDrive/GeoLift_RT_Runs")
WORK=Path("/content/anchorflow_v7_work")
CODE=Path("/content/anchorflow_v7_code")
MODEL_NAME="v7_jet"  # optional: "v6_reduced", "v7_no_transport"
INIT_SOURCE="bundled_v6"  # optional: "drive_v6"
PARENT_CHECKPOINT=DRIVE_RUNS/"AnchorFlow_v6_Connection_FromV5_FT15/metric_kd/best.pth"
assert MODEL_NAME in ("v7_jet","v6_reduced","v7_no_transport")
assert INIT_SOURCE in ("bundled_v6","drive_v6")
RUN_NAMES={"v7_jet":"AnchorFlow_v7_Jet","v6_reduced":"AnchorFlow_v6_Reduced_V7RecipeControl",
           "v7_no_transport":"AnchorFlow_v7_NoTransportControl"}
INIT_TAG="_FromV6_FT15"
RUN_TAG=""  # e.g. "_B2" when BATCH_SIZE=2; never change frozen run
BOUNDARY_WEIGHT=0.05
BARRIER_WEIGHT=0.01
TAIL_WEIGHT=0.1
TAIL_THRESHOLD_M=2.0
RUN_NAME=RUN_NAMES[MODEL_NAME]+INIT_TAG+RUN_TAG
EPOCHS=15
BATCH_SIZE=4
WORKERS=2
RUN_EXPORT=True
assert BUNDLE.is_dir(),f"Upload folder to {BUNDLE}"
assert EPOCHS==15
RUN_ROOT=DRIVE_RUNS/RUN_NAME
RUN_DIR=RUN_ROOT/"metric_kd"
print("Input:",DRIVE_DATA/"teacher_subset_2000")
print("Outputs:",RUN_DIR)
print("Colab SSD free GiB:",round(shutil.disk_usage("/content").free/2**30,1))
""")
    cell(5,"""
## 3. Resolve full V6 parent + snapshot

Epoch/SHA/cumulative lấy từ checkpoint thật. Default cumulative=59; run mới60–74.
LR1e-4: encoder0,1×; old decoder1×; **chỉ518 new parameters**2×.
Warm-up1epoch/cosine, fresh optimizer. encoder_pretrained=false vì đã nạp student đầy đủ.
""")
    cell(6,"""
cfg=json.loads((CODE/"config.json").read_text(encoding="utf-8"))
source_parent=PARENT_CHECKPOINT if INIT_SOURCE=="drive_v6" else CODE/"init_v6_best.pth"
assert source_parent.is_file(),f"Thiếu full V6 checkpoint: {source_parent}; không tự fallback."
local_parent=CODE/"init_parent.pth"
if source_parent.resolve()!=local_parent.resolve():
    shutil.copy2(source_parent,local_parent)
parent_sha=hashlib.sha256(local_parent.read_bytes()).hexdigest()
payload=torch.load(local_parent,map_location="cpu",weights_only=False)
parent_cfg=payload["protocol"]["config"]
expected_arch="AnchorFlow-v6-RayConnection"
assert parent_cfg["architecture"]==expected_arch,parent_cfg["architecture"]
assert all(k in payload["model"] for k in ("surface.amplitude","connection4.field.weight","phase2.delta.weight"))
cumulative=int(parent_cfg["parent_cumulative_epoch"])+1+int(payload["epoch"])
if INIT_SOURCE=="bundled_v6":
    assert parent_sha==manifest["parent_v6_sha256"]
    assert payload["epoch"]==14 and cumulative==59
cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(DRIVE_RUNS),work=str(WORK),run_name=RUN_NAME,
    model_name=MODEL_NAME,epochs=EPOCHS,batch_size=BATCH_SIZE,workers=WORKERS,
    init_source=INIT_SOURCE,init_checkpoint=str(local_parent),init_sha256=parent_sha,
    init_architecture=expected_arch,parent_epoch=int(payload["epoch"]),parent_cumulative_epoch=cumulative,
    boundary_weight=BOUNDARY_WEIGHT,barrier_weight=BARRIER_WEIGHT,robust_mse_weight=0.0,
    tail_weight=TAIL_WEIGHT,tail_threshold_m=TAIL_THRESHOLD_M)
cfg["architecture"]={"v7_jet":"AnchorFlow-v7-ProjectiveJet","v6_reduced":"AnchorFlow-v6-ReducedControl",
                     "v7_no_transport":"AnchorFlow-v7-NoTransportControl"}[MODEL_NAME]
assert cfg["teacher_enabled"] and not cfg["encoder_pretrained"] and cfg["flow_steps"]==3
print("Full V6 parent RMSE:",payload["best_rmse"],"epoch:",payload["epoch"],"cumulative:",cumulative)
del payload
CONFIG=CODE/"resolved_config.json"
CONFIG.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
SUBSET=DRIVE_DATA/"teacher_subset_2000"
for name in ("selected_2000_ids.json","kitti_trainval_2000.tar",cfg["metric_tar"]):
    assert (SUBSET/name).is_file(),f"Thiếu {SUBSET/name}"
RUN_ROOT.mkdir(parents=True,exist_ok=True)
snapshot=RUN_ROOT/"source_bundle"
snapshot.mkdir(parents=True,exist_ok=True)
for name in [*manifest["files"],"init_parent.pth","resolved_config.json"]:
    if name.endswith(".ipynb"):
        continue
    source,target=CODE/name,snapshot/name
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(source.read_bytes()).hexdigest(),f"Run có source/config/parent khác: {name}; dùng RUN_TAG mới"
    else:
        shutil.copy2(source,target)
shutil.copy2(BUNDLE/"bundle_manifest.json",snapshot/"bundle_manifest.json")
print(CONFIG.read_text())
def command(action,extra=()):
    subprocess.run([sys.executable,"-u",str(CODE/"run.py"),action,"--config",str(CONFIG),
                    "--variant","metric_kd",*extra],cwd=CODE,check=True)
""")
    cell(11,"""
## 6. Real-data smoke + full V6 reference400val

Kiểm518newparams và chỉ2vector-head tensors bị bỏ; zero/no-op **reduced V6**;
AMP loss/gradients finite, KD coverage. Đo full V6 cùng GPU/precision trước train.
""")
    cell(12,"""
command("smoke")
smoke=json.loads((WORK/"smoke_report.json").read_text())
assert smoke["passed"]
expected_drops=["connection4.field.bias","connection4.field.weight"] if MODEL_NAME.startswith("v7_") else []
assert smoke["migration"]["old_keys_dropped"]==expected_drops
assert smoke["migration"]["parent_parameters_loaded"]==(612701 if MODEL_NAME.startswith("v7_") else 612896)
shutil.copy2(WORK/"smoke_report.json",RUN_ROOT/"smoke_report.json")
print(json.dumps(smoke,indent=2))
command("parent_evaluate")
torch.cuda.empty_cache()
""")
    cell(15,"""
## 8. Best metrics và acceptance

Cùng25.424.992validGT pixels. So sánh full V6 cùng runtime, reduced initialization và V7.
Native stage dùng valid-area-mean GT; khác scale không phải cùng target.
""")
    cell(16,"""
command("evaluate")
import pandas as pd
parent=json.loads((RUN_DIR/"parent_same_runtime_val_metrics.json").read_text())
initial=json.loads((RUN_DIR/"initial_val_metrics.json").read_text())
best=json.loads((RUN_DIR/"val_metrics.json").read_text())
run_manifest=json.loads((RUN_DIR/"run_manifest.json").read_text())
entries=[("full V6 same runtime",parent,"final"),(MODEL_NAME+" zero-head initial",initial,"final"),
         (MODEL_NAME+" best",best,"final"),(MODEL_NAME+" pre-fusion",best,"pre_anchor"),
         (MODEL_NAME+" hard diagnostic",best,"legacy_hard_anchor")]
for other_name in RUN_NAMES:
    if other_name==MODEL_NAME:
        continue
    other_dir=DRIVE_RUNS/(RUN_NAMES[other_name]+INIT_TAG+RUN_TAG)/"metric_kd"
    if (other_dir/"val_metrics.json").is_file() and (other_dir/"run_manifest.json").is_file():
        other=json.loads((other_dir/"run_manifest.json").read_text())
        assert other["recipe_sha256"]==run_manifest["recipe_sha256"],"Different parent/data/loss/budget"
        entries.append((other_name+" matched recipe FT15",json.loads((other_dir/"val_metrics.json").read_text()),"final"))
rows=[]
for label,report,policy in entries:
    values=report[policy]; score=values["all"]
    assert score["pixels"]==25424992,"GT support changed"
    rows.append({"model/output":label,"RMSE m":score["rmse_m"],"MAE m":score["mae_m"],
                 "iRMSE km^-1":score["irmse_km_inv"],"edge RMSE m":values["edge"]["rmse_m"],
                 **{f"RMSE {k} m":values[k]["rmse_m"] for k in ("0-20","20-40","40-60","60-80","80-120")}})
comparison=pd.DataFrame(rows)
comparison.to_csv(RUN_DIR/"comparison_accuracy.csv",index=False)
display(comparison)
print("Full V6 improvement m:",parent["final"]["all"]["rmse_m"]-best["final"]["all"]["rmse_m"])
print("Best new epoch:",best["checkpoint_epoch"],"(-1 = initialization retained)")
print("Target<0.8:",best["final"]["all"]["rmse_m"]<.8)
display(pd.DataFrame(best["stage_native_gt_metrics"]).T)
display(pd.DataFrame(best["final"]["all"]["error_tail"]).T)
boundary_rows=[]
for name,report in (("full V6",parent),("reduced initial",initial),(MODEL_NAME+" best",best)):
    for kind in ("bands","rings"):
        for radius,values in report["gt_boundary"][kind].items():
            boundary_rows.append({"model":name,"kind":kind,"radius_px":int(radius),
                                 **{k:v for k,v in values.items() if k!="bad_pixel_rates"}})
boundary_table=pd.DataFrame(boundary_rows)
boundary_table.to_csv(RUN_DIR/"gt_boundary_metrics.csv",index=False)
display(boundary_table)
fig,ax=plt.subplots(figsize=(7,4))
for name,rows in boundary_table[boundary_table["kind"]=="bands"].groupby("model"):
    ax.plot(rows["radius_px"],rows["rmse_m"],marker="o",label=name)
ax.set(xlabel="Distance to observed GT discontinuity (px)",ylabel="Cumulative band RMSE (m)")
ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(RUN_DIR/"gt_boundary_curve.png",dpi=160)
plt.show()
""")
    s=notebook.cells[18].source.replace('"loss_connection_abs_delta_mean"','"loss_jet_abs_delta4_mean","loss_jet_phase_abs_delta_mean"')
    s=s.replace('"loss_connection_tangent_mean","loss_connection_normal_mean"','"loss_jet_gradient_abs","loss_jet_hessian_abs"')
    s+='\n# Absolute SSE, not just its changing fraction.\ndisplay(log[["epoch","loss_weighted_tail","loss_tail_active_fraction","val_tail_5m_sse_m2","val_tail_20m_sse_m2","loss_jet_neighbour_mass"]])\n'
    cell(18,s)
    cell(21,"""
## 11. Full V6 và V7 cùng GPU

Median/P95 batch1FP16, peak VRAM, component timing. Conv MAC không đếm shift/translation/memory.
Không suy edge latency từ params hoặc thời gian trainA100.
""")
    cell(24,"""
if RUN_EXPORT:
    command("export")
    export_report=json.loads((RUN_DIR/"export_report.json").read_text())
    assert export_report["onnxruntime_cpu_parity"] and not export_report["untrained_weights"]
    print(json.dumps(export_report,indent=2))
current_rmse=best["final"]["all"]["rmse_m"]
parent_rmse=parent["final"]["all"]["rmse_m"]
latency_ratio=new_profile["median_ms"]/old_profile["median_ms"]
summary={"architecture":cfg["architecture"],"model_name":MODEL_NAME,"recipe_sha256":run_manifest["recipe_sha256"],
         "parent_sha256":parent_sha,"parent_cumulative_epoch":cfg["parent_cumulative_epoch"],
         "extra_epochs_completed":len(log),"best_new_epoch":best["checkpoint_epoch"],
         "full_v6_same_runtime_rmse_m":parent_rmse,"zero_head_reduced_initial_rmse_m":initial["final"]["all"]["rmse_m"],
         "best_rmse_m":current_rmse,"goal_0_8_reached":current_rmse<.8,"improves_full_v6":current_rmse<parent_rmse,
         "latency_ratio_to_full_v6":latency_ratio,"provisional_accept":current_rmse<parent_rmse and latency_ratio<=1.10,
         "params":new_profile["total_parameters"],"median_ms":new_profile["median_ms"],"gpu":new_profile["device"],
         "teacher_train_only":True,"anonymous_test_has_public_gt":False,"target_device_latency_measured":False,
         "comparison_scope":"V7 architecture + squared-excess tail; optional reduced V6 same recipe isolates architecture"}
(RUN_DIR/"experiment_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
print("ALL OUTPUTS:",RUN_DIR)
""")
    cell(25,"""
## Output

```text
MyDrive/GeoLift_RT_Runs/AnchorFlow_v7_Jet_FromV6_FT15/
├── source_bundle/   # frozen source + actual V6 student + config
├── data_contract.json / smoke_report.json
└── metric_kd/
    ├── best.pth / last.pth / train_log.csv / train_log.jsonl / train.log
    ├── parent_same_runtime_val_metrics.json / initial_val_metrics.json
    ├── val_metrics.json / comparison_accuracy.csv / gt_boundary_metrics.csv
    ├── loss_budget.csv / training_diagnostics.png / experiment_summary.json
    ├── profile.json / parent_profile_same_device.json / component_profile.csv
    ├── kitti_test_predictions.zip / test_report.json
    └── anchorflow_edge_fp32.onnx / export_report.json
```

Không geometry TAR/DSINE/teacher weights. Extract/cache trên SSD `/content`; Drive giữ input và backup.
Không overwrite V6. RMSE<0,8 là mục tiêu nghiên cứu, không phải kết quả đã chạy.
""")
    for c in notebook.cells:
        if c.cell_type=="code":
            c.execution_count=None
            c.outputs=[]
    nbformat.validate(notebook)
    output=root/"drive_upload/AnchorFlow_v7_Jet/AnchorFlow_v7_Jet_TAR2000_FT15.ipynb"
    nbformat.write(notebook,output)
    print(output)


if __name__=="__main__":
    main()
