"""Whitelist, validate and seal the standalone V7 Drive artifact."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
import nbformat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"drive_upload/AnchorFlow_v7_Jet"
    names=["model.py","model_v3.py","model_v5.py","model_v6.py","model_v7.py","core.py","data.py","metrics.py",
           "boundaries.py","losses.py","loss_helpers.py","run.py","config.json","requirements.txt",
           "test_contracts.py","test_surface.py","test_jet.py","test_notebook.py",
           "README.md","ARCHITECTURE.md","ANALYSIS_V6.md","VERIFICATION.md","init_v6_best.pth",
           "v6_train_log.csv","parent_val_metrics.json","v6_initial_val_metrics.json","v6_resolved_config.json",
           "v6_run_manifest.json","v6_causal_audit.json","local_verification.json","AnchorFlow_v7_Jet_TAR2000_FT15.ipynb"]
    cfg=json.loads((folder/"config.json").read_text())
    assert cfg["epochs"]==15 and cfg["batch_size"]==4 and cfg["model_name"]=="v7_jet"
    assert cfg["teacher_enabled"] and cfg["tail_weight"]==.1 and cfg["tail_threshold_m"]==2.
    assert cfg["robust_mse_weight"]==0 and cfg["init_source"]=="bundled_v6"
    parent_sha="7ea785092601da3f64aebc1350c7d327aa4f6ce44591988d3bccc67cccd2f809"
    assert sha(folder/"init_v6_best.pth")==parent_sha
    proof=json.loads((folder/"local_verification.json").read_text())
    assert proof["tests_returncode"]==0 and proof["tests"]>=30
    source=hashlib.sha256()
    for name in ("model.py","model_v3.py","model_v5.py","model_v6.py","model_v7.py","boundaries.py","core.py","losses.py","loss_helpers.py","data.py","metrics.py","run.py"):
        source.update((folder/name).read_bytes())
    assert proof["source_sha256"]==source.hexdigest(),"Verification stale: rerun verify_anchorflow_v7.py"
    assert proof["full_size_noop_vs_reduced_v6_max_abs_m"]==[0.,0.,0.]
    assert proof["complexity"]["total_parameters"]==613219
    assert proof["complexity"]["total_conv_linear_macs"]==4032235328
    assert proof["onnx_nonzero_jet_parity"]["onnxruntime_cpu_parity"]
    audit=json.loads((folder/"v6_causal_audit.json").read_text())
    assert audit["samples"]==400 and audit["metrics"]["full_v6"]["all"]["pixels"]==25424992
    assert audit["checkpoint_sha256"]==parent_sha
    for name in names:
        path=folder/name
        assert path.is_file(),path
        if path.suffix==".py":
            ast.parse(path.read_text(encoding="utf-8"),filename=name)
        if path.suffix==".md":
            body=path.read_text(encoding="utf-8")
            assert body.count("```")%2==0,name
            assert sum(line.strip()=="$$" for line in body.splitlines())%2==0,name
            assert "\\operatorname" not in body,name
    n=nbformat.read(folder/names[-1],as_version=4)
    nbformat.validate(n)
    settings={}
    for i,c in enumerate(n.cells):
        if c.cell_type!="code":
            continue
        assert c.execution_count is None and not c.outputs
        tree=ast.parse(c.source,filename=f"cell{i}")
        for item in tree.body:
            if isinstance(item,ast.Assign):
                try:
                    value=ast.literal_eval(item.value)
                except (ValueError,TypeError):
                    continue
                for target in item.targets:
                    if isinstance(target,ast.Name):
                        settings[target.id]=value
    ast.parse("\n".join(c.source for c in n.cells if c.cell_type=="code"))
    assert settings["EPOCHS"]==15 and settings["BATCH_SIZE"]==4
    assert settings["MODEL_NAME"]=="v7_jet" and settings["INIT_SOURCE"]=="bundled_v6"
    assert settings["TAIL_WEIGHT"]==cfg["tail_weight"] and settings["TAIL_THRESHOLD_M"]==cfg["tail_threshold_m"]
    assert settings["RUN_NAMES"]["v7_jet"]+settings["INIT_TAG"]==cfg["run_name"]
    assert len(set(settings["RUN_NAMES"].values()))==3
    manifest={"name":folder.name,"default_epochs":15,"default_model":"v7_jet","default_variants":["metric_kd"],
              "available_models":["v7_jet","v6_reduced","v7_no_transport"],"contains_data":False,
              "contains_teacher_weights":False,"contains_parent_student_checkpoint":True,
              "parent_v6_sha256":parent_sha,"parent_v6_epoch":14,"parent_cumulative_epoch":59,
              "source_sha256":source.hexdigest(),"files":{name:sha(folder/name) for name in names}}
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
                      "bytes":output.stat().st_size,"sha256":sha(output),"notebook_cells":len(n.cells)},indent=2))


if __name__=="__main__":
    main()
