"""Seal V9.1 upload folder after proof checks; frozen V9 stays untouched."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"drive_upload/AnchorFlow_v9_1_MetricRefine"
OLD=ROOT/"drive_upload/AnchorFlow_v9_Consensus"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    names=[
        "model.py","model_v9_reference.py","model_v8.py","core.py","support.py","data.py","relative_data.py",
        "metrics.py","boundaries.py","losses.py","losses_v8.py","loss_helpers.py","run.py","generate_relative.py",
        "config.json","requirements.txt","teacher_requirements.txt","test_contracts.py","colab_t4_fp32_fix.py","parent_v9_best_weights.pth",
        "README.md","ARCHITECTURE.md","ANALYSIS_V9.md","VERIFICATION.md","local_verification.json",
        "v8_val_metrics.json","v8_profile.json","v9_val_metrics.json","v9_profile.json","v9_train_log.csv",
        "01_Optional_Generate_Relative_Teacher.ipynb","01_Optional_Generate_Relative_Teacher_contract.json",
        "02_Train_V9_1_Refine15_or_Fresh30.ipynb","02_Train_V9_1_Refine15_or_Fresh30_contract.json"]
    cfg=json.loads((FOLDER/"config.json").read_text())
    assert cfg["model_name"]=="v9_metric_refine" and cfg["amp"]=="bf16"
    assert cfg["teacher_enabled"] and cfg["relative_enabled"] and cfg["limited_jet_init"]
    assert cfg["epochs"]==15 and cfg["flow_steps"]==3 and cfg["early_stopping"]
    assert cfg["initialization"]=="v9_best_model_weights_only_new_optimizer"
    assert cfg["init_checkpoint"]=="parent_v9_best_weights.pth"
    assert sha(FOLDER/cfg["init_checkpoint"])==cfg["init_checkpoint_sha256"]
    assert (cfg["early_stop_patience"],cfg["early_stop_min_delta_m"],cfg["early_stop_min_epochs"])==(5,.001,8)
    proof=json.loads((FOLDER/"local_verification.json").read_text())
    assert proof["tests_returncode"]==0 and proof["tests"]==31
    assert cfg["cudnn_enabled"] and not cfg["cudnn_benchmark"]
    assert proof["colab_copy_without_ipynb_tests_returncode"]==0
    assert proof["real_kitti_bf16_gradients_finite"] and proof["parent_loaded"]
    assert proof["real_kitti_fp32_gradients_finite"]
    assert proof["trained_parent_opened_new_head_onnx"]["onnxruntime_cpu_parity"]
    assert len(proof["trained_parent_opened_new_head_onnx"]["all_stage_cases"])==11
    assert proof["complexity_v9_1"]["total_parameters"]==581868
    assert proof["complexity_v9_1"]["total_conv_linear_macs"]==3313997632
    assert proof["initialization_only_cpu_fp32_validation"]["all"]["pixels"]==25424992
    assert not proof["gpu_latency_measured"] and not proof["new_training_accuracy_measured"]
    for name,expected in proof["verified_files"].items():
        assert sha(FOLDER/name)==expected,f"Stale verification: {name}"
    prior=json.loads((OLD/"bundle_manifest.json").read_text())
    for name,expected in prior["files"].items():
        if not name.endswith(".ipynb"): assert sha(OLD/name)==expected,f"Frozen V9 changed: {name}"
    assert (FOLDER/"model_v9_reference.py").read_bytes()==(OLD/"model.py").read_bytes()
    for name in ("generate_relative.py","data.py","relative_data.py","model_v8.py","core.py","losses_v8.py"):
        assert (FOLDER/name).read_bytes()==(OLD/name).read_bytes(),name
    assert len(list(FOLDER.glob("*.ipynb")))==2
    for name in names:
        p=FOLDER/name; assert p.is_file(),p
        if p.suffix==".py": ast.parse(p.read_text(encoding="utf-8"))
        if p.suffix==".md":
            body=p.read_text(encoding="utf-8")
            assert body.count("~~~")%2==0 and body.count(chr(96)*3)%2==0,name
            assert sum(x.strip()=="$$" for x in body.splitlines())%2==0,name
            assert "\\operatorname" not in body and "\\left{" not in body,name
        if p.suffix==".ipynb":
            book=nbformat.read(p,as_version=4); nbformat.validate(book)
            snapshot=FOLDER/(p.stem+"_contract.json")
            assert json.loads(p.read_text(encoding="utf-8"))==json.loads(snapshot.read_text(encoding="utf-8"))
            for cell in book.cells:
                if cell.cell_type=="code":
                    ast.parse(cell.source); assert cell.execution_count is None and not cell.outputs
    digest=hashlib.sha256()
    for name in ("model.py","model_v9_reference.py","model_v8.py","support.py","boundaries.py","core.py",
                 "losses.py","losses_v8.py","loss_helpers.py","data.py","relative_data.py","metrics.py","run.py"):
        digest.update((FOLDER/name).read_bytes())
    assert digest.hexdigest()==proof["source_sha256"]
    manifest={"name":FOLDER.name,"default_model":"v9_metric_refine","default_epochs":15,
              "default_mode":"finetune","default_variant":"dual_teacher","notebooks":2,
              "notebook_precision":"native_bf16_else_explicit_fp32","raw_config_precision":"bf16",
              "fresh_mode_epochs":30,"contains_data":False,"contains_teacher_weights":False,
              "contains_student_checkpoint":True,"parent_checkpoint_model_only":True,
              "parent_checkpoint_epoch":23,"parent_run_completed_epochs":30,
              "parent_checkpoint_sha256":cfg["init_checkpoint_sha256"],
              "source_sha256":digest.hexdigest(),
              "runtime_notebook_checksum_policy":"Skip mutable .ipynb UI; verify immutable _contract.json, sources and model-only parent",
              "files":{name:sha(FOLDER/name) for name in names}}
    (FOLDER/"bundle_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    output=FOLDER.with_suffix(".zip")
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in names+["bundle_manifest.json"]: archive.write(FOLDER/name,FOLDER.name+"/"+name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name,expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(FOLDER.name+"/"+name)).hexdigest()==expected
        assert not any("__pycache__" in name for name in archive.namelist())
    print(json.dumps({"folder":str(FOLDER),"zip":str(output),"files":len(names)+1,
                      "bytes":output.stat().st_size,"sha256":sha(output),"notebooks":2},indent=2))

if __name__=="__main__": main()
