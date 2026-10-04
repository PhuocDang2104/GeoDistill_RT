"""Build a self-contained, fresh-epoch-zero Colab workflow (no student parent)."""
from pathlib import Path
import ast
import nbformat as nb


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"drive_upload/AnchorFlow_v8_Dynamics"
    cells=[]
    def md(s): cells.append(nb.v4.new_markdown_cell(s.strip()))
    def code(s):
        ast.parse(s.strip())
        cells.append(nb.v4.new_code_cell(s.strip()))
    md("""
# AnchorFlow V8 — Feedback Jet Dynamics · Fresh30 + Early Stop

Upload nguyên folder `AnchorFlow_v8_Dynamics` vào MyDrive. Chọn GPU → Run all.
Student mới từ epoch 0; **chỉ RGB encoder ImageNet pretrained**, không lấy checkpoint V5/V6/V7.
Ba Euler steps dùng chung CNN, đọc lại jet state và sparse error ở từng bước.

Giữ 1.600 train / 400 val / 1.000 anonymous test. Metric teacher chỉ train.
V7 tốt nhất 1,0061 m sau nhiều run nối tiếp; <0,8 m là mục tiêu nghiên cứu, không bảo đảm.
Tối đa 30 epoch fresh, early stop không sớm hơn 15 epoch; không tương đương 75 epoch tích lũy của V7.
""")
    md("""
## 1. Mount và đường dẫn

Default tối đa 30 epoch. Early stop: patience 7, min delta 0,001 m, minimum 15 epochs.
Optional `v8_frozen_feedback` là control cùng số tham số/compute,
nhưng CNN luôn đọc state ban đầu. Không tự chạy nhiều experiment.
Thay batch/epochs/recipe/source cần `RUN_TAG` mới; không ghi đè run cũ.
""")
    code('''
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
BUNDLE=Path("/content/drive/MyDrive/AnchorFlow_v8_Dynamics")
DRIVE_DATA=Path("/content/drive/MyDrive/GeoLift_Data")
DRIVE_RUNS=Path("/content/drive/MyDrive/GeoLift_RT_Runs")
WORK=Path("/content/anchorflow_v8_work")
CODE=Path("/content/anchorflow_v8_code")
MODEL_NAME="v8_dynamics"  # optional: "v8_frozen_feedback"
RUN_NAMES={"v8_dynamics":"AnchorFlow_v8_Dynamics_Fresh30_ES",
           "v8_frozen_feedback":"AnchorFlow_v8_FrozenFeedback_Fresh30_ES"}
RUN_TAG=""  # e.g. "_B2" if changing batch to 2
EPOCHS=30
EARLY_STOP_PATIENCE=7
EARLY_STOP_MIN_DELTA_M=0.001
EARLY_STOP_MIN_EPOCHS=15
BATCH_SIZE=4
WORKERS=2
RUN_EXPORT=True
assert MODEL_NAME in RUN_NAMES
assert BUNDLE.is_dir(),f"Upload folder to {BUNDLE}"
RUN_NAME=RUN_NAMES[MODEL_NAME]+RUN_TAG
RUN_ROOT=DRIVE_RUNS/RUN_NAME
RUN_DIR=RUN_ROOT/"metric_kd"
print("Input:",DRIVE_DATA/"teacher_subset_2000")
print("Outputs:",RUN_DIR)
print("Colab SSD free GiB:",round(shutil.disk_usage("/content").free/2**30,1))
assert shutil.disk_usage("/content").free/2**30>12,"Need at least 12 GiB free; 20+ GiB recommended"
''')
    md("""
## 2. Verify bundle → copy code nhỏ vào SSD → install

Không copy repo/weights/data từ Drive. Notebook không cần git pull.
ImageNet weights tải qua timm/Hugging Face; cảnh báo thiếu HF_TOKEN không phải lỗi.
Không tự fallback random encoder nếu download thất bại.
""")
    code('''
manifest=json.loads((BUNDLE/"bundle_manifest.json").read_text(encoding="utf-8"))
assert not manifest["contains_parent_student_checkpoint"]
CODE.mkdir(parents=True,exist_ok=True)
for name,expected in manifest["files"].items():
    # Colab mutates this UI artifact; verify the immutable JSON snapshot instead.
    if name.endswith(".ipynb"):
        continue
    source=BUNDLE/name
    assert source.is_file(),f"Missing bundle file: {source}"
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,f"Bundle changed: {name}"
    if source.resolve()!=(CODE/name).resolve():
        shutil.copy2(source,CODE/name)
shutil.copy2(BUNDLE/"bundle_manifest.json",CODE/"bundle_manifest.json")
os.chdir(CODE)
subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(CODE/"requirements.txt")],check=True)
import torch
assert torch.cuda.is_available(),"Colab: Runtime > Change runtime type > GPU"
print(torch.__version__,torch.cuda.get_device_name(0))
tests=subprocess.run([sys.executable,"-u","-m","unittest","discover","-s",str(CODE),"-v"],
                     cwd=CODE,capture_output=True,text=True,encoding="utf-8",errors="replace")
test_log=tests.stdout+tests.stderr
print(test_log,flush=True)
(CODE/"unit_test_output.txt").write_text(test_log,encoding="utf-8")
if tests.returncode:
    raise RuntimeError(f"Bundle tests failed (exit {tests.returncode}). Full traceback printed above; log: {CODE/'unit_test_output.txt'}")
''')
    md("""
## 3. Resolve fresh config + freeze source

LR decoder/dynamics 3e-4; encoder 0,5×. Warm-up 1 epoch → cosine.
Tất cả module học từ epoch 0. Chỉ squared tail loss ramp trong 2 epoch đầu để tránh
magnitude lớn khi depth chưa học. Intermediate jet steps có GT supervision.
Run cùng tên đã có `last.pth` sẽ resume đúng V8; không bao giờ dùng checkpoint V7.
""")
    code('''
cfg=json.loads((CODE/"config.json").read_text(encoding="utf-8"))
cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(DRIVE_RUNS),work=str(WORK),
           run_name=RUN_NAME,model_name=MODEL_NAME,epochs=EPOCHS,batch_size=BATCH_SIZE,workers=WORKERS,
           early_stopping=True,early_stop_patience=EARLY_STOP_PATIENCE,
           early_stop_min_delta_m=EARLY_STOP_MIN_DELTA_M,early_stop_min_epochs=EARLY_STOP_MIN_EPOCHS)
cfg["architecture"]="AnchorFlow-v8-FeedbackJetDynamics" if MODEL_NAME=="v8_dynamics" else "AnchorFlow-v8-FrozenFeedbackControl"
assert cfg["teacher_enabled"] and cfg["encoder_pretrained"] and cfg["flow_steps"]==3
assert not cfg.get("init_checkpoint")
CONFIG=CODE/"resolved_config.json"
CONFIG.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
SUBSET=DRIVE_DATA/"teacher_subset_2000"
for name in ("selected_2000_ids.json","kitti_trainval_2000.tar",cfg["metric_tar"]):
    assert (SUBSET/name).is_file(),f"Missing {SUBSET/name}"
RUN_ROOT.mkdir(parents=True,exist_ok=True)
snapshot=RUN_ROOT/"source_bundle"
snapshot.mkdir(parents=True,exist_ok=True)
for name in [*manifest["files"],"resolved_config.json"]:
    if name.endswith(".ipynb"): continue
    source,target=CODE/name,snapshot/name
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(source.read_bytes()).hexdigest(),f"Frozen run differs: {name}; use a new RUN_TAG"
    else:
        shutil.copy2(source,target)
shutil.copy2(BUNDLE/"bundle_manifest.json",snapshot/"bundle_manifest.json")
print(CONFIG.read_text())
def command(action,extra=()):
    subprocess.run([sys.executable,"-u",str(CODE/"run.py"),action,"--config",str(CONFIG),
                    "--variant","metric_kd",*extra],cwd=CODE,check=True)
''')
    md("""
## 4. Data gate — không bỏ qua

Đọc TAR trực tiếp từ Drive → extract RGB/sparse/GT/K và cache **chỉ 1.600 train teachers**
vào `/content`. Kiểm tra paired IDs, raw-drive disjoint, confidence và KD coverage.
Test ưu tiên `GeoLift_Data/test_1000/kitti_test_1000.tar`; thiếu thì tải official KITTI
và lưu TAR test vào Drive một lần. Không dùng geometry TAR, DSINE hay teacher weights.
""")
    code('''
command("prepare")
contract=json.loads((WORK/"data_contract.json").read_text())
assert (contract["train"],contract["val"],contract["test"])==(1600,400,1000)
assert not contract["validation_teacher_used"]
shutil.copy2(WORK/"data_contract.json",RUN_ROOT/"data_contract.json")
print(json.dumps(contract,indent=2))
''')
    md("## 5. Inspect RGB / sparse / GT / metric teacher trước train")
    code('''
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,str(CODE))
from data import KITTIDataset
fig,axes=plt.subplots(2,4,figsize=(20,7))
for row,split in enumerate(("train","val")):
    sample=KITTIDataset(cfg,split,teacher=split=="train")[0]
    axes[row,0].imshow(sample["rgb"].permute(1,2,0)); axes[row,0].set_title(split+" RGB")
    for col,key in enumerate(("sparse","gt","teacher"),1):
        if key in sample:
            d=sample[key][0].numpy()
            axes[row,col].imshow(np.ma.masked_where(d<=0,d),vmin=0,vmax=80,cmap="turbo")
            axes[row,col].set_title(key+" (m)")
        else: axes[row,col].set_title("Val: no teacher loaded")
    print(split,sample["sid"],"keys",list(sample))
for a in axes.flat: a.axis("off")
plt.tight_layout(); plt.show()
fig.savefig(RUN_ROOT/"data_preview.png",dpi=120)
''')
    md("""
## 6. Real-data AMP smoke

Kiểm tra finite loss/gradients, metric KD có support, shared reaction nhận gradient,
và jet thay đổi ở cả ba bước. Smoke không cập nhật model train; trainer tạo student fresh riêng.
""")
    code('''
command("smoke")
smoke=json.loads((WORK/"smoke_report.json").read_text())
assert smoke["passed"] and not smoke["student_checkpoint_loaded"]
assert smoke["imagenet_encoder_only"]
shutil.copy2(WORK/"smoke_report.json",RUN_ROOT/"smoke_report.json")
print(json.dumps(smoke,indent=2))
torch.cuda.empty_cache()
''')
    md("""
## 7. Train tối đa epoch 0…29, có early stop

Mỗi epoch backup `last.pth`, `best.pth` khi cải thiện, và log vào `RUN_DIR`.
`initial_val_metrics.json` là epoch −1 chưa train; không lẫn với epoch 0.
Đứt Colab: chạy lại từ đầu notebook, giữ config/run name để resume.
Theo dõi global val RMSE; dừng khi 7 validation liên tiếp không giảm >0,001 m,
nhưng phải hoàn tất ít nhất 15 epoch. Counter được lưu/resume trong checkpoint.
Best checkpoint vẫn lưu mọi raw RMSE tốt hơn, không bị min_delta loại bỏ.
""")
    code('''
command("train")
print("Logs/checkpoints:",RUN_DIR)
print(json.loads((RUN_DIR/"training_status.json").read_text()))
''')
    md("## 8. Best checkpoint: global / range / boundary / tail / từng dynamics step")
    code('''
command("evaluate")
import pandas as pd
best=json.loads((RUN_DIR/"val_metrics.json").read_text())
initial=json.loads((RUN_DIR/"initial_val_metrics.json").read_text())
rows=[]
for label,report,policy in (("V8 initial",initial,"final"),("V8 best",best,"final"),
                            ("V8 pre-fusion",best,"pre_anchor"),("V8 hard diagnostic",best,"legacy_hard_anchor")):
    score=report[policy]
    rows.append({"output":label,"RMSE m":score["all"]["rmse_m"],"MAE m":score["all"]["mae_m"],
                 "iRMSE km^-1":score["all"]["irmse_km_inv"],"pixels":score["all"]["pixels"],
                 **{f"RMSE {k}":score[k]["rmse_m"] for k in ("0-20","20-40","40-60","60-80","80-120","edge")}})
comparison=pd.DataFrame(rows)
comparison.to_csv(RUN_DIR/"comparison_accuracy.csv",index=False)
display(comparison)
assert best["final"]["all"]["pixels"]==25424992,"GT support/protocol changed"
display(pd.DataFrame(best["stage_native_gt_metrics"]).T)
display(pd.DataFrame(best["final"]["all"]["error_tail"]).T)
display(pd.DataFrame(best["gt_boundary"]["bands"]).T)
print("Best trained epoch:",best["checkpoint_epoch"],"Target <0.8:",best["final"]["all"]["rmse_m"]<.8)
v7=pd.read_csv(CODE/"v7_train_log.csv")
print("Historical V7 best:",v7.val_rmse_m.min(),"NOT matched training budget/initialization")
''')
    md("## 9. Learning curves + loss budget + state evolution")
    code('''
log=pd.read_csv(RUN_DIR/"train_log.csv")
display(log[["epoch","val_rmse_m","val_mae_m","train_seconds","epoch_seconds"]])
fig,axes=plt.subplots(1,3,figsize=(18,4))
axes[0].plot(log.epoch,log.val_rmse_m,label="V8 global"); axes[0].set_ylabel("RMSE m")
for k in (1,2,3):
    axes[1].plot(log.epoch,log[f"val_native_D4_step{k}_rmse_m"],label=f"D4 step{k}")
    axes[2].plot(log.epoch,log[f"loss_dynamics_state_change_{k}"],label=f"normalized state change{k}")
for a in axes: a.set_xlabel("fresh epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.show(); fig.savefig(RUN_DIR/"dynamics_learning.png",dpi=150)
weighted=log[[c for c in log if c.startswith("loss_weighted_")]].copy()
weighted.insert(0,"epoch",log.epoch)
weighted.to_csv(RUN_DIR/"loss_budget.csv",index=False)
display(weighted.tail())
dynamics=log[["epoch"]+[c for c in log if "dynamics_" in c or "native_D4_step" in c]]
dynamics.to_csv(RUN_DIR/"dynamics_evolution.csv",index=False)
display(dynamics.tail())
''')
    md("""
## 10. 1.000 anonymous test PNG + inference efficiency + ONNX

Test không có public GT, không thể tính RMSE nội bộ. ZIP dùng để submit KITTI.
Profile batch 1, cùng input shape/AMP: median/P95, VRAM và component timing;
Conv/Linear MAC không tính tensor shifts/transport/memory. ONNX kiểm tra CPU parity,
không đồng nghĩa đã validate TensorRT hoặc latency trên thiết bị edge.
""")
    code('''
command("test")
command("profile")
profile=json.loads((RUN_DIR/"profile.json").read_text())
display(pd.DataFrame({"parameters":profile["parameters"],"ConvLinear MAC":profile["conv_linear_macs"],
                      "eager median ms":profile["eager_component_median_ms"]}))
print("Median/P95 ms:",profile["median_ms"],profile["p95_ms"])
if RUN_EXPORT:
    command("export")
    print(json.loads((RUN_DIR/"export_report.json").read_text()))
print("DONE:",RUN_DIR)
print("Test:",RUN_DIR/"kitti_test_predictions.zip")
''')
    md("""
## Đọc kết quả đúng

- Student fresh tối đa30 có thể chưa hội tụ. Không xem thua V7 đã fine-tune nhiều lần là chứng minh dynamics sai.
- Dynamics không cam kết RMSE giảm đơn điệu từng step. Kiểm tra `D4_step1/2/3`, forcing/transport,
  far-range, GT boundary và tail SSE để đánh giá lợi ích.
- Để isolate feedback: chạy notebook lần khác với `MODEL_NAME="v8_frozen_feedback"`, cùng seed/budget/early-stop policy/recipe; báo cả số epoch thực tế.
- Early stop không khóa notebook: evaluate/test/profile/export tiếp tục dùng best.pth. Run đã early-stop không tự train tiếp khi resume.
- Muốn đổi budget/early-stop policy: đổi config và RUN_TAG **trước** run mới; không sửa config checkpoint đang resume.
- Các module train từ epoch đầu; không teacher inference, không GT input, không adaptive early stopping.
""")
    notebook=nb.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python"},"colab":{"name":"AnchorFlow_v8_Dynamics_TAR2000_Fresh30_EarlyStop.ipynb"},"accelerator":"GPU"})
    nb.validate(notebook)
    nb.write(notebook,folder/"AnchorFlow_v8_Dynamics_TAR2000_Fresh30_EarlyStop.ipynb")
    # The tests copy/read this immutable, checksummed canonical snapshot. They
    # never need the user-opened notebook, which Colab modifies while executing.
    (folder/"notebook_contract.json").write_text(nb.writes(notebook),encoding="utf-8")
    print("Built",len(cells),"cells")


if __name__=="__main__": main()
