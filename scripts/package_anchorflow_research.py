"""Small standalone Drive bundle: shared pipeline and interchangeable v3/v4 models."""
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
    folder=root/"drive_upload/AnchorFlow_Research"
    names=["model.py","model_v3.py","model_v4.py","core.py","data.py","losses.py","loss_helpers.py",
           "metrics.py","run.py","config.json","requirements.txt","test_contracts.py","test_models.py",
           "README.md","ARCHITECTURE.md","ANALYSIS_V3.md","VERIFICATION.md","init_v3_best.pth",
           "parent_train_log.csv","parent_val_metrics.json","parent_profile.json","parent_prediction_audit.json",
           "v4_complexity.json","AnchorFlow_Research_TAR2000_15ep.ipynb"]
    config=json.loads((folder/"config.json").read_text())
    assert config["epochs"]==15 and config["model_name"]=="v4" and config["parent_cumulative_epoch"]==29
    assert config["teacher_enabled"] and config["new_lr_ratio"]==1
    assert sha(folder/"init_v3_best.pth")==config["init_sha256"]
    for name in names:
        path=folder/name
        assert path.is_file(),path
        if path.suffix==".py":
            ast.parse(path.read_text(encoding="utf-8"),filename=name)
        if path.suffix==".md":
            body=path.read_text(encoding="utf-8")
            assert body.count("```")%2==0,name
            assert sum(line.strip()=="$$" for line in body.splitlines())%2==0,name
    nb=nbformat.read(folder/names[-1],as_version=4)
    nbformat.validate(nb)
    settings={}
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=="code":
            assert not cell.outputs and cell.execution_count is None
            tree=ast.parse(cell.source,filename=f"cell{i}")
            for item in tree.body:
                if isinstance(item,ast.Assign):
                    try:
                        value=ast.literal_eval(item.value)
                    except (ValueError,TypeError):
                        continue
                    for target in item.targets:
                        if isinstance(target,ast.Name):
                            settings[target.id]=value
    assert settings["MODEL_NAME"]=="v4" and settings["EPOCHS"]==15
    assert settings["RUN_NAMES"]["v4"]==config["run_name"]
    assert settings["RUN_NAMES"]["v3"]!=settings["RUN_NAMES"]["v4"]
    ast.parse("\n".join(c.source for c in nb.cells if c.cell_type=="code"))
    manifest={"name":folder.name,"default_epochs":15,"default_model":"v4","available_models":["v3","v4"],
              "default_variants":["metric_kd"],"contains_data":False,"contains_teacher_weights":False,
              "contains_parent_student_checkpoint":True,"parent_sha256":config["init_sha256"],
              "files":{name:sha(folder/name) for name in names}}
    (folder/"bundle_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    output=folder.with_suffix(".zip")
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in names+["bundle_manifest.json"]:
            z.write(folder/name,f"{folder.name}/{name}")
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name,digest in manifest["files"].items():
            assert hashlib.sha256(z.read(f"{folder.name}/{name}")).hexdigest()==digest
    print(json.dumps({"folder":str(folder),"zip":str(output),"bytes":output.stat().st_size,
                      "sha256":sha(output),"files":len(names)+1,"notebook_cells":len(nb.cells),
                      "default_model":"v4","epochs":15},indent=2))


if __name__=="__main__":
    main()
