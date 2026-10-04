"""Validate + package a whitelist of standalone v5 files, never datasets/teacher weights."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
import nbformat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/"drive_upload/AnchorFlow_v5_Piecewise"
    names = ["model.py","model_v3.py","model_v5.py","core.py","boundaries.py","data.py","losses.py",
             "loss_helpers.py","metrics.py","run.py","config.json","requirements.txt","test_contracts.py",
             "test_surface.py","README.md","ARCHITECTURE.md","ANALYSIS_V4.md","VERIFICATION.md",
             "init_v3_best.pth","parent_val_metrics.json","v4_train_log.csv","v4_val_metrics.json",
             "v4_profile.json","v3_profile_same_device.json","v4_gt_boundary_audit.json",
             "local_verification.json","AnchorFlow_v5_TAR2000_15ep.ipynb"]
    cfg = json.loads((folder/"config.json").read_text(encoding="utf-8"))
    assert cfg["epochs"]==15 and cfg["model_name"]=="v5_piecewise"
    assert cfg["teacher_enabled"] and cfg["boundary_weight"]==.05 and cfg["barrier_weight"]==.01
    assert sha(folder/"init_v3_best.pth")==cfg["init_sha256"]
    proof = json.loads((folder/"local_verification.json").read_text(encoding="utf-8"))
    assert proof["tests_returncode"]==0 and proof["tests"]==21
    source = hashlib.sha256()
    for name in ("model.py","model_v3.py","model_v5.py","boundaries.py","core.py","losses.py","loss_helpers.py","data.py","metrics.py","run.py"):
        source.update((folder/name).read_bytes())
    assert proof["source_sha256"]==source.hexdigest(),"Verification is stale: re-run scripts/verify_anchorflow_v5.py"
    assert proof["complexity"]["total_parameters"]==579993
    assert proof["full_size_noop_max_abs_m"]==[0.,0.,0.]
    audit = json.loads((folder/"v4_gt_boundary_audit.json").read_text())
    assert audit["samples"]==400 and audit["valid_pixels"]==25424992 and audit["checkpoint_epoch"]==7
    for name in names:
        path = folder/name
        assert path.is_file(),path
        if path.suffix==".py":
            ast.parse(path.read_text(encoding="utf-8"),filename=name)
        if path.suffix==".md":
            body = path.read_text(encoding="utf-8")
            assert body.count("```")%2==0,name
            assert sum(line.strip()=="$$" for line in body.splitlines())%2==0,name
            assert "\\operatorname" not in body,name
    notebook = nbformat.read(folder/names[-1],as_version=4)
    nbformat.validate(notebook)
    settings = {}
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type!="code":
            continue
        assert cell.execution_count is None and not cell.outputs
        tree = ast.parse(cell.source,filename=f"cell{i}")
        for item in tree.body:
            if isinstance(item,ast.Assign):
                try:
                    value = ast.literal_eval(item.value)
                except (ValueError,TypeError):
                    continue
                for target in item.targets:
                    if isinstance(target,ast.Name):
                        settings[target.id] = value
    ast.parse("\n".join(c.source for c in notebook.cells if c.cell_type=="code"))
    assert settings["MODEL_NAME"]=="v5_piecewise" and settings["EPOCHS"]==15
    assert settings["BOUNDARY_WEIGHT"]==cfg["boundary_weight"]
    assert settings["BARRIER_WEIGHT"]==cfg["barrier_weight"]
    assert settings["RUN_NAMES"]["v5_piecewise"]==cfg["run_name"]
    assert len(set(settings["RUN_NAMES"].values()))==3
    manifest = {"name":folder.name,"default_epochs":15,"default_model":"v5_piecewise",
                "available_models":["v3","v5_transport","v5_piecewise"],"default_variants":["metric_kd"],
                "contains_data":False,"contains_teacher_weights":False,"contains_parent_student_checkpoint":True,
                "parent_sha256":cfg["init_sha256"],"source_sha256":source.hexdigest(),
                "files":{name:sha(folder/name) for name in names}}
    (folder/"bundle_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    output = folder.with_suffix(".zip")
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in names+["bundle_manifest.json"]:
            archive.write(folder/name,f"{folder.name}/{name}")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name,digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(f"{folder.name}/{name}")).hexdigest()==digest
    print(json.dumps(dict(folder=str(folder),zip=str(output),files=len(names)+1,bytes=output.stat().st_size,
                          sha256=sha(output),notebook_cells=len(notebook.cells),epochs=15,default_model="v5_piecewise"),indent=2))


if __name__=="__main__":
    main()
