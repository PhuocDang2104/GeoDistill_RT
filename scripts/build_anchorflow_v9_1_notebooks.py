"""Two clean V9.1 notebooks; reuse relative recipe and model-only V9 parent."""
import ast
import json
from pathlib import Path
import textwrap
import nbformat as nb
from build_anchorflow_v9_notebooks import COMMAND

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"drive_upload/AnchorFlow_v9_1_MetricRefine"

def md(s): return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s): return nb.v4.new_code_cell(textwrap.dedent(s).strip())

MOUNT='''
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
BUNDLE=Path("/content/drive/MyDrive/AnchorFlow_v9_1_MetricRefine")
DRIVE_DATA=Path("/content/drive/MyDrive/GeoLift_Data")
DRIVE_RUNS=Path("/content/drive/MyDrive/GeoLift_RT_Runs")
assert BUNDLE.is_dir(), f"Upload extracted folder to {BUNDLE}"
assert (DRIVE_DATA/"teacher_subset_2000/selected_2000_ids.json").is_file()
'''
STAGE='''
manifest=json.loads((BUNDLE/"bundle_manifest.json").read_text(encoding="utf-8"))
assert not manifest["contains_data"] and not manifest["contains_teacher_weights"]
assert manifest["contains_student_checkpoint"] and manifest["parent_checkpoint_model_only"]
CODE.mkdir(parents=True,exist_ok=True)
for name,sha in manifest["files"].items():
    # Notebook UI is mutable; source and immutable _contract.json remain strict.
    if name.endswith(".ipynb"): continue
    source=BUNDLE/name
    assert source.is_file(), f"Missing {source}"
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha, f"Bundle changed: {name}"
    destination=CODE/name
    if source.resolve()!=destination.resolve(): shutil.copy2(source,destination)
shutil.copy2(BUNDLE/"bundle_manifest.json",CODE/"bundle_manifest.json")
os.chdir(CODE)
'''

def write(name,cells):
    book=nb.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"accelerator":"GPU","colab":{"name":name}})
    nb.validate(book)
    for cell in book.cells:
        if cell.cell_type=="code": ast.parse(cell.source)
    nb.write(book,FOLDER/name)
    (FOLDER/(Path(name).stem+"_contract.json")).write_text(nb.writes(book),encoding="utf-8")

def main():
    # Identical teacher code/revisions/recipe: existing per-image resume and final TAR stay valid.
    old=nb.read(ROOT/"drive_upload/AnchorFlow_v9_Consensus/01_Generate_Relative_Teacher_DA3MONO_TAR2000.ipynb",as_version=4)
    cells=[]
    for cell in old.cells:
        text=cell.source.replace("AnchorFlow_v9_Consensus","AnchorFlow_v9_1_MetricRefine")
        text=text.replace("anchorflow_v9_teacher_code","anchorflow_v9_1_teacher_code")
        if cell.cell_type=="code" and "manifest=json.loads" in text:
            end=text.index('subprocess.run([sys.executable,"-m","pip"')
            text=textwrap.dedent(STAGE)+text[end:]
            text+='\nassert torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported(including_emulation=False), "Optional teacher generation requires native BF16 GPU. Existing V9 relative TAR: skip notebook01 and use notebook02 (T4 FP32 supported)."'
        cells.append(code(text) if cell.cell_type=="code" else md(text))
    cells.insert(1,md('''
    > **Tùy chọn.** Bạn đã train V9 nên thường **bỏ qua notebook 01**, chạy thẳng notebook 02.
    > Tái sử dụng `relative_teacher_2000_DA3MONO_LARGE.tar` đã có; không cần generate lại.
    > Nếu generation trước bị ngắt, giữ nguyên recipe/LONG_SIDE và chạy từ đầu notebook để resume.
    '''))
    write("01_Optional_Generate_Relative_Teacher.ipynb",cells)
    cells=[md('''
    # 02 — AnchorFlow V9.1: Sparse-Innovation Metric Readout

    Default: **fine-tune thêm tối đa 15 epoch** từ V9 best epoch23 (model-only checkpoint kèm bundle).
    Optimizer/scheduler mới; không tiếp tục optimizer cũ. New metric head zero-init nhưng minmod
    initialization thay đổi geometry, nên phải đo lại initial validation. Dùng trực tiếp data hiện có.
    `MODE="fresh"`: ImageNet RGB encoder, student epoch0, tối đa30 epoch; không load V9.
    AMP_REQUEST="auto": native BF16 GPU → BF16; T4/V100 → FP32 được ghi rõ trước khi freeze config.
    Không FP16 fallback. Teacher TAR dùng lại nguyên vẹn. Mục tiêu <0.9 m là hypothesis, không bảo đảm.
    '''),code(MOUNT),code('''
    CODE=Path("/content/anchorflow_v9_1_code")
    WORK=Path("/content/anchorflow_v9_1_work")
    MODE="finetune"  # "fresh" = student epoch0 + ImageNet encoder only
    RUN_TAG="_precision_v2"  # isolated from older failed smoke/source snapshots
    AMP_REQUEST="auto"      # "auto", "bf16" (native only), or "fp32"
    BATCH_SIZE=4
    WORKERS=2
    RUN_EXPORT=True
    assert MODE in ("finetune","fresh")
    EPOCHS=15 if MODE=="finetune" else 30
    VARIANT="dual_teacher"
    RUN_NAME=("AnchorFlow_v9_1_MetricRefine_FromV9Best15_ES" if MODE=="finetune"
              else "AnchorFlow_v9_1_MetricRefine_Fresh30_ES")+RUN_TAG
    assert shutil.disk_usage("/content").free/2**30>25,"Need >25 GiB local SSD free"
    '''),md("## Verify bundle → stage code and small parent → install → contract tests"),code(textwrap.dedent(STAGE)+textwrap.dedent('''
    subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(CODE/"requirements.txt")],check=True)
    import torch
    assert torch.cuda.is_available(),"Select a GPU runtime"
    sys.path.insert(0,str(CODE))
    from run import select_training_precision, native_bf16_supported
    ACTUAL_AMP=select_training_precision(AMP_REQUEST)
    RUN_NAME+="_"+ACTUAL_AMP
    print("Precision resolved BEFORE config freeze:",AMP_REQUEST,"->",ACTUAL_AMP,
          "native BF16:",native_bf16_supported(),"compute capability:",torch.cuda.get_device_capability(0))
    print(torch.__version__,torch.cuda.get_device_name(0))
    result=subprocess.run([sys.executable,"-u","-m","unittest","discover","-s",str(CODE),"-v"],
                          cwd=CODE,capture_output=True,text=True,encoding="utf-8",errors="replace")
    print(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError("Unit tests failed; read full traceback above")
    ''')),md('''
    ## Resolve config and freeze source

    Fine-tune LR decoder6e-5 / encoder1.5e-5 / new head1.2e-4; early-stop patience5, min8.
    Fresh LR3e-4 / encoder1.5e-4 / new head6e-4; patience7, min15. No teacher in inference.
    Same run resumes only identical source/config/data; use a new RUN_TAG after any recipe change.
    '''),code(textwrap.dedent('''
    cfg=json.loads((CODE/"config.json").read_text())
    cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(DRIVE_RUNS),work=str(WORK),
               run_name=RUN_NAME,epochs=EPOCHS,batch_size=BATCH_SIZE,workers=WORKERS,
               amp=ACTUAL_AMP,amp_request=AMP_REQUEST,precision_policy="native_bf16_else_explicit_fp32_v2")
    # Explicit backend settings, not an automatic precision/architecture fallback.
    cfg.update(cudnn_enabled=True,cudnn_benchmark=False,cudnn_deterministic=False)
    if MODE=="fresh":
        cfg.update(init_checkpoint=None,init_checkpoint_sha256=None,
                   initialization="fresh_student_imagenet_rgb_only",learning_rate=3e-4,
                   encoder_lr_ratio=.5,early_stop_patience=7,early_stop_min_epochs=15)
    CONFIG=CODE/"resolved_config.json"
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    RUN_ROOT=DRIVE_RUNS/RUN_NAME
    RUN_DIR=RUN_ROOT/VARIANT
    SUBSET=DRIVE_DATA/"teacher_subset_2000"
    for name in ("selected_2000_ids.json","kitti_trainval_2000.tar",cfg["metric_tar"],cfg["relative_tar"]):
        assert (SUBSET/name).is_file(),f"Missing {SUBSET/name}; relative teacher from completed V9 is reused"
    snapshot=RUN_ROOT/"source_bundle"
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in [*manifest["files"],"resolved_config.json"]:
        if name.endswith(".ipynb"): continue
        src,dest=CODE/name,snapshot/name
        if dest.exists(): assert src.read_bytes()==dest.read_bytes(),f"Frozen run differs: {name}; use new RUN_TAG"
        elif src.resolve()!=dest.resolve(): shutil.copy2(src,dest)
    shutil.copy2(BUNDLE/"bundle_manifest.json",snapshot/"bundle_manifest.json")
    print("Mode:",MODE,"Outputs:",RUN_DIR)
    print(json.dumps(cfg,indent=2))
    ''')+COMMAND.replace('    if exit_code:',
        '    diagnostic=Path(current["work"])/"gpu_failure_report.json"\n'
        '    if diagnostic.is_file(): shutil.copy2(diagnostic,target/(action+"_gpu_failure_report.json"))\n'
        '    if exit_code:')),md('''
    ## Data gate + preview

    Same1.600train/400val/1.000anonymous test. Teacher targets are train-only.
    Fine-tune additionally checks subset + metric/relative content against the parent checkpoint.
    Test reuses `GeoLift_Data/test_1000/kitti_test_1000.tar`; missing test is downloaded officially.
    '''),code('''
    command("prepare")
    contract=json.loads((WORK/"data_contract.json").read_text())
    assert (contract["train"],contract["val"],contract["test"])==(1600,400,1000)
    assert not contract["validation_teacher_used"]
    assert contract["relative_teacher"]["cached_train"]==1600 and contract["relative_teacher"]["cached_val"]==0
    shutil.copy2(WORK/"data_contract.json",RUN_ROOT/"data_contract.json")
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(CODE))
    from data import KITTIDataset
    sample=KITTIDataset(cfg,"train",teacher=True)[0]
    val=KITTIDataset(cfg,"val",teacher=False)[0]
    assert not {"teacher","relative","confidence","relative_confidence"}.intersection(val)
    fig,axes=plt.subplots(1,6,figsize=(24,4))
    axes[0].imshow(sample["rgb"].permute(1,2,0)); axes[0].set_title("RGB")
    for ax,key in zip(axes[1:],("sparse","gt","teacher","relative","relative_confidence")):
        lo,hi=(-2,2) if key=="relative" else ((0,1) if key=="relative_confidence" else (0,80))
        ax.imshow(sample[key][0],cmap="turbo",vmin=lo,vmax=hi); ax.set_title(key)
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show(); fig.savefig(RUN_ROOT/"data_preview.png",dpi=120)
    del sample,val
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Parent notebook GPU allocation MiB:",torch.cuda.memory_allocated()/2**20)
    command("smoke")
    print(json.loads((WORK/"smoke_report.json").read_text()))
    torch.cuda.empty_cache()
    '''),md('''
    ## Train — resume safely, synchronize each epoch

    Initial validation is saved **before** any new update. New head zero-init; minmod is not a
    no-op change. Best may remain initial checkpoint epoch−1 if all updates regress.
    Native BF16 + FP32 geometry/loss, or all-FP32 on T4. Fail before backward/optimizer on nonfinite.
    Metric/GT losses run from epoch0; robust tail ramps2epochs, relative gradient ramps3.
    Default budget: prior full V9 run30 + up to15 additional; best parent state came from24 epochs.
    '''),code('''
    command("train")
    print(json.loads((RUN_DIR/"training_status.json").read_text()))
    print("Logs/checkpoints:",RUN_DIR)
    '''),md("## Evaluate best — accuracy, ranges, GT-boundary, tails, native stages"),code('''
    command("evaluate")
    import pandas as pd
    best=json.loads((RUN_DIR/"val_metrics.json").read_text())
    initial=json.loads((RUN_DIR/"initial_val_metrics.json").read_text())
    assert best["final"]["all"]["pixels"]==25424992,"GT protocol changed"
    display(pd.DataFrame(best["final"]).T)
    display(pd.DataFrame(best["stage_native_gt_metrics"]).T)
    display(pd.DataFrame(best["gt_boundary"]["bands"]).T)
    display(pd.DataFrame(best["final"]["all"]["error_tail"]).T)
    print("Query/innovation:",best["query_diagnostics"])
    print("Best local epoch:",best["checkpoint_epoch"],"newly trained:",best["checkpoint_epoch"]>=0)
    print("Goal <0.9:",best["final"]["all"]["rmse_m"]<.9)
    '''),md("## Loss / dynamics / innovation learning curves"),code('''
    log=pd.read_csv(RUN_DIR/"train_log.csv")
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    axes[0].plot(log.epoch,log.val_rmse_m); axes[0].axhline(.9,ls="--",color="red"); axes[0].set_title("Global validation RMSE m")
    for name in ("D2_base","D2_query","D2"):
        axes[1].plot(log.epoch,log[f"val_native_{name}_rmse_m"],label=name)
    for name in ("innovation_abs_mean_m","innovation_head_raw_abs","metric_head_saturation_fraction"):
        axes[2].plot(log.epoch,log["loss_"+name],label=name)
    for ax in axes[1:]: ax.legend()
    for ax in axes: ax.grid(alpha=.3); ax.set_xlabel("local epoch")
    plt.tight_layout(); plt.show(); fig.savefig(RUN_DIR/"v9_1_learning_curves.png",dpi=150)
    log[["epoch"]+[c for c in log if c.startswith("loss_weighted_")]].to_csv(RUN_DIR/"loss_budget.csv",index=False)
    display(log.tail())
    '''),md('''
    ## Anonymous test + trained profile + strict ONNX

    Test has no public GT. ONNX validates random seeds, empty/observed sparse, plus a real KITTI
    sample; tolerance is unchanged. CPU ORT parity is not a TensorRT/edge performance guarantee.
    Export failure does not delete predictions or accuracy reports; inspect full console traceback.
    '''),code('''
    command("test")
    command("profile")
    profile=json.loads((RUN_DIR/"profile.json").read_text())
    print("Params/MAC/median/P95:",profile["total_parameters"],profile["total_conv_linear_macs"],profile["median_ms"],profile["p95_ms"])
    if RUN_EXPORT: command("export")
    '''),md('''
    ## Compare measured V8 / V9 / V9.1 initial / V9.1 best

    Historical comparisons **are not equal-budget ablations**. V8 changed precision after5epochs;
    V9 was fresh30BF16; default V9.1 uses V9 weights + additional training + modified loss.
    Gain versus V9.1 initial measures this refinement run, not an isolated architecture effect.
    '''),code('''
    def record(label,report):
        score=report["final"]
        return {"run":label,"RMSE m":score["all"]["rmse_m"],"MAE m":score["all"]["mae_m"],
                "iRMSE km^-1":score["all"]["irmse_km_inv"],
                **{f"RMSE {k}":score[k]["rmse_m"] for k in ("0-20","20-40","40-60","60-80","80-120","edge")},
                "boundary3 RMSE":report["gt_boundary"]["bands"]["3"]["rmse_m"],
                "tail>5 pixels":score["all"]["error_tail"]["5"]["pixels"],
                "tail>5 absolute SSE":score["all"]["error_tail"]["5"]["sse_m2"]}
    rows=[record("V8 historical",json.loads((CODE/"v8_val_metrics.json").read_text())),
          record("V9 historical parent",json.loads((CODE/"v9_val_metrics.json").read_text())),
          record("V9.1 initial (not newly trained)",initial),record("V9.1 best "+MODE,best)]
    comparison=pd.DataFrame(rows); display(comparison)
    print("Current precision:",cfg["amp"],"GPU:",torch.cuda.get_device_name(0),
          "; FP32 T4 vs historical BF16 Blackwell is NOT matched precision/runtime.")
    comparison.to_csv(RUN_DIR/"v9_1_vs_v8_v9_accuracy.csv",index=False)
    efficiency=[]
    for label,p in (("V8 historical",json.loads((CODE/"v8_profile.json").read_text())),
                    ("V9 historical",json.loads((CODE/"v9_profile.json").read_text())),("V9.1",profile)):
        same=all(p[k]==profile[k] for k in ("device","torch","precision","batch","shape","channels_last","compiled_total"))
        same=same and p.get("runtime_backend",{}).get("cudnn_benchmark")==profile.get("runtime_backend",{}).get("cudnn_benchmark")
        efficiency.append({"run":label,"params":p["total_parameters"],"ConvLinear MAC":p["total_conv_linear_macs"],
                           "median ms":p["median_ms"],"P95 ms":p["p95_ms"],"runtime_settings_match":same})
    ef=pd.DataFrame(efficiency); display(ef); ef.to_csv(RUN_DIR/"v9_1_vs_v8_v9_efficiency.csv",index=False)
    print("DONE:",RUN_DIR,"test ZIP:",RUN_DIR/"kitti_test_predictions.zip")
    print("Target <0.9 m is not guaranteed. Compare absolute tail SSE and near/far metrics, not just tail share.")
    ''')]
    write("02_Train_V9_1_Refine15_or_Fresh30.ipynb",cells)
    print("Built two notebooks + immutable source snapshots")

if __name__=="__main__": main()
