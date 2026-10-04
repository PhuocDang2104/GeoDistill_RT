"""Seal a data-free, parent-checkpoint-free V8 bundle with verified source hashes."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
import nbformat


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"drive_upload/AnchorFlow_v8_Dynamics"
    names=["model.py","core.py","support.py","data.py","metrics.py","boundaries.py","losses.py",
           "loss_helpers.py","run.py","config.json","requirements.txt","test_contracts.py","test_notebook.py",
           "README.md","ARCHITECTURE.md","ANALYSIS_V7.md","VERIFICATION.md","v7_train_log.csv","notebook_contract.json",
           "local_verification.json","AnchorFlow_v8_Dynamics_TAR2000_Fresh30_EarlyStop.ipynb"]
    cfg=json.loads((folder/"config.json").read_text())
    assert cfg["encoder_pretrained"] and cfg["teacher_enabled"] and cfg["flow_steps"]==3
    assert cfg["epochs"]==30 and cfg["model_name"]=="v8_dynamics" and cfg["early_stopping"]
    assert (cfg["early_stop_patience"],cfg["early_stop_min_delta_m"],cfg["early_stop_min_epochs"])==(7,.001,15)
    assert not cfg.get("init_checkpoint")
    assert not any((folder/name).suffix in (".pth",".pt",".tar",".onnx") for name in names)
    proof=json.loads((folder/"local_verification.json").read_text())
    source=hashlib.sha256()
    for name in ("model.py","support.py","boundaries.py","core.py","losses.py","loss_helpers.py","data.py","metrics.py","run.py"):
        source.update((folder/name).read_bytes())
    assert proof["source_sha256"]==source.hexdigest(),"Stale verification; rerun verify_anchorflow_v8.py"
    assert proof["tests_returncode"]==0 and proof["tests"]>=22
    assert proof["colab_copy_without_ipynb_tests_returncode"]==0
    assert proof["real_kitti_bf16_gradients_finite"] and proof["onnx_opened_dynamics"]["onnxruntime_cpu_parity"]
    assert proof["complexity"]["total_parameters"]==578064
    assert proof["complexity"]["total_conv_linear_macs"]==3214694208
    assert sha(root/"train_log_v7.csv")==sha(folder/"v7_train_log.csv")
    for name in names:
        p=folder/name
        assert p.is_file(),p
        if p.suffix==".py": ast.parse(p.read_text(encoding="utf-8"))
        if p.suffix==".md":
            body=p.read_text(encoding="utf-8")
            assert body.count("```")%2==0,name
            assert sum(line.strip()=="$$" for line in body.splitlines())%2==0,name
            assert "\\operatorname" not in body and "\\left{" not in body,name
    notebook=nbformat.read(folder/names[-1],as_version=4)
    nbformat.validate(notebook)
    assert json.loads((folder/"notebook_contract.json").read_text(encoding="utf-8"))==json.loads((folder/names[-1]).read_text(encoding="utf-8"))
    for c in notebook.cells:
        if c.cell_type=="code":
            assert c.execution_count is None and not c.outputs
            ast.parse(c.source)
    manifest={"name":folder.name,"default_model":"v8_dynamics","default_epochs":30,
              "early_stopping":{"patience":7,"min_delta_m":.001,"min_epochs":15},
              "default_variants":["metric_kd"],"available_models":["v8_dynamics","v8_frozen_feedback"],
              "contains_data":False,"contains_teacher_weights":False,"contains_parent_student_checkpoint":False,
              "initialization":"fresh_student_imagenet_rgb_only","source_sha256":source.hexdigest(),
              "files":{name:sha(folder/name) for name in names}}
    (folder/"bundle_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    output=folder.with_suffix(".zip")
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in names+["bundle_manifest.json"]:
            z.write(folder/name,f"{folder.name}/{name}")
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name,expected in manifest["files"].items():
            assert hashlib.sha256(z.read(f"{folder.name}/{name}")).hexdigest()==expected
    print(json.dumps({"folder":str(folder),"zip":str(output),"files":len(names)+1,
                      "bytes":output.stat().st_size,"sha256":sha(output),"cells":len(notebook.cells)},indent=2))


if __name__=="__main__": main()
