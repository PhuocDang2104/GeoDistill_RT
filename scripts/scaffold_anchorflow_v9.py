"""Reuse frozen V8 primitives without modifying its package; scaffold once."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "drive_upload/AnchorFlow_v8_Dynamics"
TARGET = ROOT / "drive_upload/AnchorFlow_v9_Consensus"

if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in ("core.py", "support.py", "boundaries.py", "metrics.py", "loss_helpers.py", "data.py", "run.py", "requirements.txt"):
        if not (TARGET / name).exists():
            shutil.copy2(SOURCE / name, TARGET / name)
    for src, dest in (("model.py", "model_v8.py"), ("losses.py", "losses_v8.py")):
        if not (TARGET / dest).exists():
            shutil.copy2(SOURCE / src, TARGET / dest)
    for name in ("val_metrics.json", "profile.json", "training_status.json"):
        dest = TARGET / ("v8_" + name)
        if not dest.exists():
            shutil.copy2(ROOT / "results/anchorflow_v8_completed_audit" / name, dest)
