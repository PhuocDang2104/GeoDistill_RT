"""Seal exactly two notebooks and data-free code after verified structural gates."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"drive_upload/AnchorFlow_v9_Consensus"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names=["model.py","model_v8.py","core.py","support.py","data.py","relative_data.py",
           "metrics.py","boundaries.py","losses.py","losses_v8.py","loss_helpers.py","run.py",
           "generate_relative.py","config.json","requirements.txt","teacher_requirements.txt","test_contracts.py",
           "README.md","ARCHITECTURE.md","ANALYSIS_V8.md","VERIFICATION.md","local_verification.json",
           "v8_val_metrics.json","v8_profile.json","v8_training_status.json",
           "01_Generate_Relative_Teacher_DA3MONO_TAR2000.ipynb",
           "01_Generate_Relative_Teacher_DA3MONO_TAR2000_contract.json",
           "02_Train_V9_DualTeacher_Fresh30_CompareV8.ipynb",
           "02_Train_V9_DualTeacher_Fresh30_CompareV8_contract.json"]
    cfg=json.loads((FOLDER/"config.json").read_text())
    assert cfg["model_name"]=="v9_consensus" and cfg["amp"]=="bf16"
    assert cfg["teacher_enabled"] and cfg["relative_enabled"] and cfg["encoder_pretrained"]
    assert cfg["initialization"]=="fresh_student_imagenet_rgb_only" and not cfg.get("init_checkpoint")
    assert cfg["epochs"]==30 and cfg["flow_steps"]==3 and cfg["early_stopping"]
    assert (cfg["early_stop_patience"],cfg["early_stop_min_delta_m"],cfg["early_stop_min_epochs"])==(7,.001,15)
    proof=json.loads((FOLDER/"local_verification.json").read_text())
    assert proof["tests_returncode"]==0 and proof["tests"]>=17
    assert proof["colab_copy_without_ipynb_tests_returncode"]==0
    assert proof["real_kitti_bf16_gradients_finite"] and proof["onnx_opened_consensus"]["onnxruntime_cpu_parity"]
    assert proof["complexity_v9"]["total_parameters"]==578568
    assert proof["complexity_v9"]["total_conv_linear_macs"]==3227535168
    assert proof["complexity_v8_control"]["total_parameters"]==578064
    assert proof["teacher_adapter"]["official_tensor_keys_and_shapes_matched"]==406
    assert not proof["gpu_latency_measured"] and not proof["new_training_accuracy_measured"]
    assert sha(FOLDER/"model_v8.py")==sha(ROOT/"drive_upload/AnchorFlow_v8_Dynamics/model.py")
    assert sha(FOLDER/"losses_v8.py")==sha(ROOT/"drive_upload/AnchorFlow_v8_Dynamics/losses.py")
    for name,expected in proof["verified_files"].items():
        assert sha(FOLDER/name)==expected,f"Stale verification: {name}"
    assert len(list(FOLDER.glob("*.ipynb")))==2
    for name in names:
        p=FOLDER/name
        assert p.is_file(),p
        assert p.suffix not in (".pth",".pt",".tar",".onnx",".npz",".npy")
        if p.suffix==".py": ast.parse(p.read_text(encoding="utf-8"))
        if p.suffix==".md":
            body=p.read_text(encoding="utf-8")
            assert body.count("```")%2==0,name
            assert sum(x.strip()=="$$" for x in body.splitlines())%2==0,name
            assert "\\operatorname" not in body and "\\left{" not in body,name
        if p.suffix==".ipynb":
            notebook=nbformat.read(p,as_version=4); nbformat.validate(notebook)
            snapshot=FOLDER/(p.stem+"_contract.json")
            assert json.loads(snapshot.read_text(encoding="utf-8"))==json.loads(p.read_text(encoding="utf-8"))
            for cell in notebook.cells:
                if cell.cell_type=="code":
                    assert cell.execution_count is None and not cell.outputs
                    ast.parse(cell.source)
    source_names=("model.py","model_v8.py","support.py","boundaries.py","core.py","losses.py","losses_v8.py",
                  "loss_helpers.py","data.py","relative_data.py","metrics.py","run.py")
    source=hashlib.sha256()
    for name in source_names: source.update((FOLDER/name).read_bytes())
    assert source.hexdigest()==proof["source_sha256"]
    manifest={"name":FOLDER.name,"default_model":"v9_consensus","default_epochs":30,
              "default_variant":"dual_teacher","notebooks":2,"precision":"bf16",
              "early_stopping":{"patience":7,"min_delta_m":.001,"min_epochs":15},
              "relative_teacher":cfg["relative_model_id"],"relative_revision":cfg["relative_model_revision"],
              "contains_data":False,"contains_teacher_weights":False,"contains_student_checkpoint":False,
              "runtime_notebook_checksum_policy":"Skip mutable .ipynb UI; verify immutable _contract.json and all source files",
              "source_sha256":source.hexdigest(),"files":{name:sha(FOLDER/name) for name in names}}
    (FOLDER/"bundle_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    output=FOLDER.with_suffix(".zip")
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in names+["bundle_manifest.json"]:
            archive.write(FOLDER/name,FOLDER.name+"/"+name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name,expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(FOLDER.name+"/"+name)).hexdigest()==expected
    print(json.dumps({"folder":str(FOLDER),"zip":str(output),"files":len(names)+1,
                      "bytes":output.stat().st_size,"sha256":sha(output),"notebooks":2},indent=2))


if __name__=="__main__": main()
