"""Copy unchanged, audited data/evaluation primitives; run once for V8 scaffold."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "drive_upload/AnchorFlow_v7_Jet"
TARGET = ROOT / "drive_upload/AnchorFlow_v8_Dynamics"

if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in ("core.py", "data.py", "metrics.py", "boundaries.py", "loss_helpers.py",
                 "losses.py", "run.py", "requirements.txt"):
        destination = TARGET / name
        if not destination.exists():
            shutil.copy2(SOURCE / name, destination)
    helpers = (SOURCE / "model_v3.py").read_text(encoding="utf-8")
    helpers = helpers[:helpers.index("class MetricRefine4")] + helpers[
        helpers.index("class PhaseDetailAndTrust"):helpers.index("class AnchorFlowEdge(V2)")]
    helpers = helpers.replace("from core import AnchorFlowBase as V2, ConvBN, LiteBlock",
                              "from core import ConvBN, LiteBlock")
    helpers = helpers.replace('"""Canonical v3 forward, retained as the matched-budget research control."""',
                              '"""Unchanged F32 context and half-resolution detail/sensor-trust primitives."""')
    if not (TARGET / "support.py").exists():
        (TARGET / "support.py").write_text(helpers, encoding="utf-8")
    if not (TARGET / "v7_train_log.csv").exists():
        shutil.copy2(ROOT / "train_log_v7.csv", TARGET / "v7_train_log.csv")
