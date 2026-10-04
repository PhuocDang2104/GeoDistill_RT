"""Two clean Colab notebooks: pinned relative-cache generation and fresh V9 train."""
from pathlib import Path
import textwrap
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"drive_upload/AnchorFlow_v9_Consensus"


def markdown(text): return nb.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text): return nb.v4.new_code_cell(textwrap.dedent(text).strip())


MOUNT=code('''
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
BUNDLE=Path("/content/drive/MyDrive/AnchorFlow_v9_Consensus")
DRIVE_DATA=Path("/content/drive/MyDrive/GeoLift_Data")
DRIVE_RUNS=Path("/content/drive/MyDrive/GeoLift_RT_Runs")
assert BUNDLE.is_dir(), f"Upload the folder to {BUNDLE}"
assert (DRIVE_DATA/"teacher_subset_2000/selected_2000_ids.json").is_file()
''')

STAGING='''
manifest=json.loads((BUNDLE/"bundle_manifest.json").read_text(encoding="utf-8"))
assert not manifest["contains_student_checkpoint"] and not manifest["contains_data"]
CODE.mkdir(parents=True,exist_ok=True)
for name,sha in manifest["files"].items():
    # Colab changes notebook metadata/outputs; never hash the UI artifact.
    if name.endswith(".ipynb"): continue
    source=BUNDLE/name
    assert source.is_file(), f"Missing file: {source}"
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha, f"Bundle changed: {name}"
    destination=CODE/name
    if source.resolve()!=destination.resolve(): shutil.copy2(source,destination)
shutil.copy2(BUNDLE/"bundle_manifest.json",CODE/"bundle_manifest.json")
os.chdir(CODE)
'''

COMMAND='''
def command(action, extra=()):
    # Derive paths from this exact config; no stale RUN_DIR after switching experiments.
    current=json.loads(CONFIG.read_text(encoding="utf-8"))
    target=Path(current["drive_runs"])/current["run_name"]/VARIANT
    target.mkdir(parents=True,exist_ok=True)
    logfile=CODE/(action+"_console.log")
    args=[sys.executable,"-u",str(CODE/"run.py"),action,"--config",str(CONFIG),"--variant",VARIANT,*extra]
    with logfile.open("w",encoding="utf-8") as log:
        proc=subprocess.Popen(args,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding="utf-8",errors="replace")
        for line in proc.stdout:
            print(line,end="",flush=True); log.write(line); log.flush()
        exit_code=proc.wait()
    shutil.copy2(logfile,target/logfile.name)
    if exit_code:
        raise RuntimeError(f"{action} failed: full traceback above; Drive console: {target/logfile.name}")
'''


def write(name,cells):
    book=nb.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"accelerator":"GPU","colab":{"name":name}})
    nb.validate(book)
    nb.write(book,FOLDER/name)
    (FOLDER/(Path(name).stem+"_contract.json")).write_text(nb.writes(book),encoding="utf-8")


def main():
    cells=[markdown('''
    # 01 — Generate DA3MONO-LARGE relative teacher, selected 2.000 RGB

    Upload **AnchorFlow_v9_Consensus** lên MyDrive; không upload repo/data/weights lần nữa.
    Input dùng thẳng GeoLift_Data/teacher_subset_2000/{selected_2000_ids.json,kitti_trainval_2000.tar}.
    Teacher chỉ thấy RGB, từng ảnh độc lập, không GT/sparse/K/metric teacher.
    Generate cả 2.000; 400 val chỉ audit. Student train chỉ load 1.600 train targets.
    Cần GPU; default long-side1232, flip TTA, full-frame resize không crop. 12+ GiB SSD trống.
    '''),MOUNT,code('''
    CODE=Path("/content/anchorflow_v9_teacher_code")
    TEACHER_WORK=Path("/content/anchorflow_v9_teacher_work")
    LONG_SIDE=1232
    assert shutil.disk_usage("/content").free/2**30>12,"Need >=12 GiB local SSD free"
    print("Teacher output:",DRIVE_DATA/"teacher_subset_2000/relative_teacher_2000_DA3MONO_LARGE.tar")
    '''),markdown('''
    ## Verify and install minimal dependencies

    Không pip-install torch/torchvision hoặc toàn bộ DA3 extras. Adapter dùng official
    network và strict checkpoint load; native PyTorch SDPA. Không silent random teacher fallback.
    '''),code(STAGING+textwrap.dedent('''
    subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(CODE/"teacher_requirements.txt")],check=True)
    import torch
    assert torch.cuda.is_available(),"Select GPU runtime"
    print(torch.__version__,torch.cuda.get_device_name(0))
    ''')),markdown('''
    ## Generate + per-image Drive resume

    Pinned model revision và pinned official Git revision ghi trong recipe. Mỗi ảnh
    lưu .npz vào Drive staging: mất runtime có thể chạy lại để tiếp tục. Không đổi LONG_SIDE
    giữa run; đổi recipe phải dùng staging mới. Không tự xóa cache/data cũ.
    R_T là normalized relative inverse depth (lớn = gần), C_T là flip-agreement confidence,
    không phải depth mét và không phải calibrated uncertainty.
    '''),code('''
    proc=subprocess.Popen([sys.executable,"-u",str(CODE/"generate_relative.py"),
        "--drive-data",str(DRIVE_DATA),"--work",str(TEACHER_WORK),"--long-side",str(LONG_SIDE)],
        cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
    for line in proc.stdout: print(line,end="",flush=True)
    result=proc.wait()
    if result: raise RuntimeError("Teacher generation failed; read the real traceback above, do not train")
    report_path=DRIVE_DATA/"teacher_subset_2000/relative_teacher_2000_report.json"
    report=json.loads(report_path.read_text())
    assert report["records"]==2000 and report["student_training_records"]==1600
    print(json.dumps(report,indent=2))
    '''),markdown('''
    ## Inspect one train and one val-audit sample

    Relative colors không có đơn vị mét. Archive final giữ 2.000 records + checksums;
    intermediate .npz vẫn giữ để resume, nên Drive chứa cả staging và TAR.
    '''),code('''
    import cv2, numpy as np, matplotlib.pyplot as plt
    selected=json.loads((DRIVE_DATA/"teacher_subset_2000/selected_2000_ids.json").read_text())
    fig,axes=plt.subplots(2,3,figsize=(18,6))
    for row,split in enumerate(("train","val")):
        sid=selected[split+"_ids"][0]
        rows=[x.split() for x in (TEACHER_WORK/"kitti/splits"/("train_1600.txt" if split=="train" else "val_400.txt")).read_text().splitlines()]
        rgb=cv2.cvtColor(cv2.imread(str(TEACHER_WORK/"kitti"/rows[0][1])),cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(1216,352),interpolation=cv2.INTER_LINEAR)
        with np.load(DRIVE_DATA/"teacher_subset_2000/relative_DA3MONO_LARGE_generation"/(sid+".npz"),allow_pickle=False) as payload:
            r,c=payload["R_T"],payload["C_T"]
        axes[row,0].imshow(rgb); axes[row,0].set_title(split+" RGB")
        axes[row,1].imshow(r,cmap="turbo",vmin=-2,vmax=2); axes[row,1].set_title("Relative inverse depth | near higher")
        axes[row,2].imshow(c,vmin=0,vmax=1,cmap="viridis"); axes[row,2].set_title("TTA confidence | heuristic")
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout(); plt.show()
    fig.savefig(DRIVE_DATA/"teacher_subset_2000/relative_teacher_preview.png",dpi=150)
    print("Now open notebook 02 in a clean GPU runtime")
    ''')]
    write("01_Generate_Relative_Teacher_DA3MONO_TAR2000.ipynb",cells)
    cells=[markdown('''
    # 02 — AnchorFlow V9: Multi-Jet Consensus + metric / relative teachers

    Fresh student epoch0, ImageNet RGB only; không load checkpoint V8. Max30 + early-stop.
    Run notebook01 trước để tạo relative TAR. GPU cần BF16 (L4/A100/Blackwell); T4 không
    supported cho recipe này, không fallback FP16 vì V8 từng overflow FP16.
    Giữ split1600/400/test1000, image352x1216 và metric protocol V8.
    '''),MOUNT,code('''
    CODE=Path("/content/anchorflow_v9_code")
    WORK=Path("/content/anchorflow_v9_work")
    EXPERIMENT="v9"  # "v8_control" = optional matched fresh/BF16 metric-only baseline
    RUN_TAG=""
    EPOCHS=30
    BATCH_SIZE=4
    WORKERS=2
    RUN_EXPORT=True
    assert EXPERIMENT in ("v9","v8_control")
    VARIANT="dual_teacher" if EXPERIMENT=="v9" else "metric_kd"
    RUN_NAME=("AnchorFlow_v9_Consensus_DualTeacher_Fresh30_ES" if EXPERIMENT=="v9" else "AnchorFlow_v8_Control_Metric_Fresh30_ES")+RUN_TAG
    assert shutil.disk_usage("/content").free/2**30>20,"Need >20 GiB SSD free; 25+ recommended for two caches"
    '''),markdown("## Verify immutable bundle → copy small code only → tests"),code(STAGING+textwrap.dedent('''
    subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(CODE/"requirements.txt")],check=True)
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"Use BF16-capable GPU"
    print(torch.__version__,torch.cuda.get_device_name(0))
    result=subprocess.run([sys.executable,"-u","-m","unittest","discover","-s",str(CODE),"-v"],
                          cwd=CODE,capture_output=True,text=True,encoding="utf-8",errors="replace")
    print(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError("Unit tests failed; full traceback above")
    ''')),markdown('''
    ## Resolve isolated config and freeze recipe

    V9 dùng hai teacher. Optional V8 control tắt relative/inverse mới, giữ đúng objective
    V8 gốc; architecture cũng là V8 nguyên bản. Run names độc lập, không merge checkpoint.
    '''),code(textwrap.dedent('''
    cfg=json.loads((CODE/"config.json").read_text())
    cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(DRIVE_RUNS),work=str(WORK),
               run_name=RUN_NAME,epochs=EPOCHS,batch_size=BATCH_SIZE,workers=WORKERS)
    if EXPERIMENT=="v8_control":
        cfg.update(model_name="v8_control",architecture="AnchorFlow-v8-MatchedFreshControl",relative_enabled=False,inverse_weight=0)
    CONFIG=CODE/"resolved_config.json"
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    RUN_ROOT=Path(cfg["drive_runs"])/cfg["run_name"]
    RUN_DIR=RUN_ROOT/VARIANT
    SUBSET=DRIVE_DATA/"teacher_subset_2000"
    needed=["selected_2000_ids.json","kitti_trainval_2000.tar",cfg["metric_tar"]]
    if cfg["relative_enabled"]: needed.append(cfg["relative_tar"])
    for name in needed: assert (SUBSET/name).is_file(),f"Missing {SUBSET/name}"
    snapshot=RUN_ROOT/"source_bundle"
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in [*manifest["files"],"resolved_config.json"]:
        if name.endswith(".ipynb"): continue
        src,dest=CODE/name,snapshot/name
        if dest.exists(): assert src.read_bytes()==dest.read_bytes(),f"Frozen run differs: {name}; use new RUN_TAG"
        elif src.resolve()!=dest.resolve(): shutil.copy2(src,dest)
    shutil.copy2(BUNDLE/"bundle_manifest.json",snapshot/"bundle_manifest.json")
    print("Outputs:",RUN_DIR)
    print(json.dumps(cfg,indent=2))
    ''')+COMMAND),markdown('''
    ## Data gate: paired IDs + both teacher content checksums

    Metric cache và relative cache chỉ chứa1600 train. Relative val targets chỉ được audit
    để kiểm tra TAR completeness, không loader val/test, không GT mask từ teacher.
    Test dùng thẳng GeoLift_Data/test_1000/kitti_test_1000.tar; thiếu thì official KITTI download.
    '''),code('''
    command("prepare")
    contract=json.loads((WORK/"data_contract.json").read_text())
    assert (contract["train"],contract["val"],contract["test"])==(1600,400,1000)
    assert not contract["validation_teacher_used"]
    if EXPERIMENT=="v9":
        assert contract["relative_teacher"]["cached_train"]==1600
        assert contract["relative_teacher"]["cached_val"]==0
    shutil.copy2(WORK/"data_contract.json",RUN_ROOT/"data_contract.json")
    print("DATA GATE PASSED",contract["manifest_sha256"])
    '''),markdown("## Inspect actual inputs + AMP smoke (fresh smoke model, not trained checkpoint)"),code('''
    import numpy as np, matplotlib.pyplot as plt
    sys.path.insert(0,str(CODE))
    from data import KITTIDataset
    sample=KITTIDataset(cfg,"train",teacher=True)[0]
    val_sample=KITTIDataset(cfg,"val",teacher=False)[0]
    assert not {"teacher","relative","confidence","relative_confidence"}.intersection(val_sample)
    fig,axes=plt.subplots(1,6,figsize=(24,4))
    axes[0].imshow(sample["rgb"].permute(1,2,0)); axes[0].set_title("RGB")
    for col,key in enumerate(("sparse","gt","teacher","relative","relative_confidence"),1):
        if key in sample:
            array=sample[key][0].numpy()
            scale=(-2,2) if key=="relative" else ((0,1) if key=="relative_confidence" else (0,80))
            axes[col].imshow(array,cmap="turbo",vmin=scale[0],vmax=scale[1]); axes[col].set_title(key)
        else: axes[col].set_title("disabled for V8 control")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show(); fig.savefig(RUN_ROOT/"data_preview.png",dpi=120)
    del sample,val_sample
    command("smoke")
    print(json.loads((WORK/"smoke_report.json").read_text()))
    torch.cuda.empty_cache()
    '''),markdown('''
    ## Train epoch0…29, early-stop and backup each epoch

    Resume chỉ cùng source/config/data. BF16 từ đầu; không NaN masking/batch skipping.
    Finite loss kiểm tra trước backward và gradient clipping fail rõ nếu nonfinite.
    Loss V8 giữ nguyên + inverse .01 + relative .05 ramp0→1 trong3 epoch.
    '''),code('''
    command("train")
    print(json.loads((RUN_DIR/"training_status.json").read_text()))
    print("Drive outputs:",RUN_DIR)
    '''),markdown("## Evaluate best: global / range / boundary / tail / query diagnostics"),code('''
    command("evaluate")
    import pandas as pd
    best=json.loads((RUN_DIR/"val_metrics.json").read_text())
    assert best["final"]["all"]["pixels"]==25424992,"Validation GT protocol changed"
    display(pd.DataFrame(best["final"]).T)
    display(pd.DataFrame(best["stage_native_gt_metrics"]).T)
    display(pd.DataFrame(best["gt_boundary"]["bands"]).T)
    display(pd.DataFrame(best["final"]["all"]["error_tail"]).T)
    print("Consensus diagnostics:",best["query_diagnostics"])
    print("Best epoch:",best["checkpoint_epoch"])
    '''),markdown("## Learning curves, loss budget, consensus and dynamics logs"),code('''
    log=pd.read_csv(RUN_DIR/"train_log.csv")
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    axes[0].plot(log.epoch,log.val_rmse_m); axes[0].set_title("Global validation RMSE m")
    for stage in ("D2_base","D2_query","D2"):
        axes[1].plot(log.epoch,log[f"val_native_{stage}_rmse_m"],label=stage)
    axes[1].legend()
    for key in ("loss_query_uncertainty_mean","loss_relative_gradient","loss_relative_ordinal"):
        axes[2].plot(log.epoch,log[key],label=key)
    axes[2].legend()
    for ax in axes: ax.grid(alpha=.3); ax.set_xlabel("epoch")
    plt.tight_layout(); plt.show(); fig.savefig(RUN_DIR/"v9_learning_curves.png",dpi=150)
    weighted=log[["epoch"]+[c for c in log if c.startswith("loss_weighted_")]]
    weighted.to_csv(RUN_DIR/"loss_budget.csv",index=False)
    display(log.tail())
    '''),markdown('''
    ## Anonymous test + trained BF16 profile + optional ONNX

    Anonymous test không có GT. Profile không tính I/O/H2D; MAC bỏ qua multi-jet transport/memory.
    ONNX CPU parity không bảo đảm TensorRT/edge latency.
    '''),code('''
    command("test")
    command("profile")
    profile=json.loads((RUN_DIR/"profile.json").read_text())
    print("Params/MAC/median/P95:",profile["total_parameters"],profile["total_conv_linear_macs"],profile["median_ms"],profile["p95_ms"])
    if RUN_EXPORT: command("export")
    '''),markdown('''
    ## Compare V8 / V9

    Historical V8 best: 0.99755 m nhưng epoch0–4 FP16, khác initialization. Không claim causal gain.
    Để có matched baseline, chạy cùng notebook lần khác EXPERIMENT="v8_control".
    Model + two-teacher + inverse là một bundle experiment; chưa isolate từng contribution.
    '''),code('''
    historical=json.loads((CODE/"v8_val_metrics.json").read_text())
    rows=[]
    def record(label,report):
        score=report["final"]
        return {"run":label,"RMSE m":score["all"]["rmse_m"],"MAE m":score["all"]["mae_m"],
                "iRMSE km^-1":score["all"]["irmse_km_inv"],
                **{f"RMSE {k}":score[k]["rmse_m"] for k in ("0-20","20-40","40-60","60-80","edge")},
                "boundary3 RMSE":report["gt_boundary"]["bands"]["3"]["rmse_m"],
                "tail>5 pixel fraction":score["all"]["error_tail"]["5"]["pixel_fraction"],
                "tail>5 SSE share":score["all"]["error_tail"]["5"]["sse_fraction"],
                "tail>5 absolute SSE":score["all"]["error_tail"]["5"]["sse_m2"]}
    rows=[record("V8 historical (not matched)",historical),record(EXPERIMENT+" fresh",best)]
    control=DRIVE_RUNS/("AnchorFlow_v8_Control_Metric_Fresh30_ES"+RUN_TAG)/"metric_kd"
    if EXPERIMENT=="v9" and (control/"val_metrics.json").is_file():
        control_cfg=json.loads((control/"resolved_config.json").read_text())
        intentionally_different={"architecture","model_name","run_name","relative_enabled","inverse_weight"}
        assert {k:v for k,v in cfg.items() if k not in intentionally_different}=={k:v for k,v in control_cfg.items() if k not in intentionally_different},"Control recipe differs; compare shared training settings before claiming matched"
        control_contract=json.loads((control/"run_manifest.json").read_text())["protocol"]["data"]
        assert control_contract["manifest_sha256"]==contract["manifest_sha256"]
        assert control_contract["teacher_report"]==contract["teacher_report"]
        rows.append(record("V8 fresh/BF16 control",json.loads((control/"val_metrics.json").read_text())))
    comparison=pd.DataFrame(rows)
    display(comparison); comparison.to_csv(RUN_DIR/"v9_vs_v8_accuracy.csv",index=False)
    efficiency=[{"run":EXPERIMENT,"device":profile["device"],"precision":profile["precision"],
                 "parameters":profile["total_parameters"],"ConvLinear MAC":profile["total_conv_linear_macs"],
                 "median_ms":profile["median_ms"],"p95_ms":profile["p95_ms"]}]
    if EXPERIMENT=="v9" and (control/"profile.json").is_file():
        cp=json.loads((control/"profile.json").read_text())
        same=all(cp[k]==profile[k] for k in ("device","torch","precision","batch","shape","channels_last","compiled_total"))
        efficiency.append({"run":"V8 control","device":cp["device"],"precision":cp["precision"],
                           "parameters":cp["total_parameters"],"ConvLinear MAC":cp["total_conv_linear_macs"],
                           "median_ms":cp["median_ms"],"p95_ms":cp["p95_ms"],"runtime_comparable":same})
    ef=pd.DataFrame(efficiency); display(ef); ef.to_csv(RUN_DIR/"v9_vs_v8_efficiency.csv",index=False)
    print("DONE:",RUN_DIR,"test ZIP:",RUN_DIR/"kitti_test_predictions.zip")
    print("Hypothesis targets, not guarantees: RMSE<0.90; iRMSE<3.1; D2 query<1.27. Compare absolute tail SSE, not only its share.")
    ''')]
    write("02_Train_V9_DualTeacher_Fresh30_CompareV8.ipynb",cells)
    print("Built exactly two clean notebooks")


if __name__=="__main__": main()
