"""Isolated V9.1 scaffold, with trusted parent model weights but no parent optimizer."""
from pathlib import Path
import hashlib
import json
import shutil
import torch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"drive_upload/AnchorFlow_v9_Consensus"
NEW=ROOT/"drive_upload/AnchorFlow_v9_1_MetricRefine"


def main():
    NEW.mkdir(parents=True,exist_ok=True)
    unchanged=("core.py","support.py","model_v8.py","losses_v8.py","loss_helpers.py","boundaries.py",
               "metrics.py","data.py","relative_data.py","generate_relative.py","requirements.txt","teacher_requirements.txt")
    for name in unchanged:
        if not (NEW/name).exists(): shutil.copy2(OLD/name,NEW/name)
    for source,target in (("model.py","model_v9_reference.py"),("run.py","run.py"),
                          ("config.json","config.json"),("test_contracts.py","test_contracts.py")):
        if not (NEW/target).exists(): shutil.copy2(OLD/source,NEW/target)
    for source,target in ((OLD/"v8_val_metrics.json","v8_val_metrics.json"),(OLD/"v8_profile.json","v8_profile.json"),
                          (ROOT/"results/anchorflow_v9_completed_audit/val_metrics.json","v9_val_metrics.json"),
                          (ROOT/"results/anchorflow_v9_completed_audit/profile.json","v9_profile.json"),
                          (ROOT/"results/anchorflow_v9_completed_audit/train_log.csv","v9_train_log.csv")):
        if not (NEW/target).exists(): shutil.copy2(source,NEW/target)
    checkpoint=ROOT/"results/anchorflow_v9_completed_audit/best.pth"
    parent=NEW/"parent_v9_best_weights.pth"
    if not parent.exists():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        # The model and audited provenance only; optimizer/scaler/RNG are NOT transferred.
        torch.save({"model":saved["model"],"epoch":saved["epoch"],"best_rmse":saved["best_rmse"],
                    "protocol":saved["protocol"],"source_checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "parent_completed_epochs":30,"contains_parent_optimizer":False},parent)
    print(NEW)


if __name__=="__main__": main()
