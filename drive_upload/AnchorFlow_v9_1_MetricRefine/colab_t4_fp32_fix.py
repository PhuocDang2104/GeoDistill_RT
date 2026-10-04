"""Explicit FP32 repair for an ALREADY staged V9.1 notebook; no source edits.

Upload this single file to Drive, then exec its text in the existing notebook.
The full rebuilt notebook instead resolves auto precision before config freeze.
"""
from pathlib import Path
import hashlib
import json
import gc
import torch


def repair_current_notebook(namespace):
    code=Path(namespace["CODE"]); config_path=Path(namespace["CONFIG"])
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    # Idempotent isolated run, never relabel a BF16 checkpoint as FP32/resume.
    cfg.update(amp="fp32",amp_request="fp32",precision_policy="explicit_t4_fp32_repair")
    if not cfg["run_name"].endswith("_t4_fp32"): cfg["run_name"]+="_t4_fp32"
    root=Path(cfg["drive_runs"])/cfg["run_name"]
    snapshot=root/"source_bundle"
    manifest=json.loads((code/"bundle_manifest.json").read_text(encoding="utf-8"))
    if not (Path(cfg["work"])/"data_contract.json").is_file():
        raise RuntimeError("Run the existing data gate first; this repair does not regenerate/extract data")
    resolved=json.dumps(cfg,indent=2).encode("utf-8")
    contents={}
    for name,expected in manifest["files"].items():
        if name.endswith(".ipynb"): continue
        raw=(code/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise RuntimeError(f"Source bundle changed: {name}; no checksum bypass")
        contents[name]=raw
    contents[config_path.name]=resolved
    contents["bundle_manifest.json"]=(code/"bundle_manifest.json").read_bytes()
    # Validate all existing frozen files before writing anything.
    for name,raw in contents.items():
        dest=snapshot/name
        if dest.exists() and dest.read_bytes()!=raw:
            raise RuntimeError(f"FP32 run already has a different recipe: {name}; choose a new run tag")
    snapshot.mkdir(parents=True,exist_ok=True)
    for name,raw in contents.items():
        dest=snapshot/name
        if not dest.exists(): dest.write_bytes(raw)
    config_path.write_bytes(resolved)
    namespace.update(cfg=cfg,RUN_NAME=cfg["run_name"],RUN_ROOT=root,
                     RUN_DIR=root/namespace.get("VARIANT","dual_teacher"))
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("EXPLICIT FP32 RUN:",namespace["RUN_DIR"])
    print("No model/source/loss/teacher changes. New run; data cache reused. FP16 NOT enabled.")
    return cfg


if "CODE" in globals() and "CONFIG" in globals() and callable(globals().get("command")):
    repair_current_notebook(globals())
    command("smoke")
    print("Smoke passed. Continue the existing train cell; do not re-run the old config cell.")
