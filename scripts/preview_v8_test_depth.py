"""Audit uploaded V8 reports and render real metric predictions, not synthetic images."""
import io
import json
import hashlib
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "results/metric_kd-20261003T091559Z-1-001.zip"
    output = root / "results/v8_depth_preview"
    audit = root / "results/anchorflow_v8_completed_audit"
    rgb_root = root / "data/depth_selection/test_depth_completion_anonymous/image"
    for folder in (output, audit, output / "pairs", output / "depth_color", output / "depth_metric_uint16"):
        folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as outer:
        # Extract reports only, never checkpoints, binaries or arbitrary archive paths.
        for member in outer.namelist():
            path = Path(member)
            if path.parent.as_posix() == "metric_kd" and path.suffix in (".json", ".csv", ".jsonl", ".log"):
                (audit / path.name).write_bytes(outer.read(member))
        metrics = json.loads(outer.read("metric_kd/val_metrics.json"))
        test_report = json.loads(outer.read("metric_kd/test_report.json"))
        samples, records = [], []
        with zipfile.ZipFile(io.BytesIO(outer.read("metric_kd/kitti_test_predictions.zip"))) as predictions:
            names = predictions.namelist()
            assert len(names) == len(set(names)) == 1000
            for index in np.linspace(0, 999, 10, dtype=int):
                sid = f"{index:010d}"
                raw = predictions.read(sid + ".png")
                encoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                assert encoded.dtype == np.uint16 and encoded.shape == (352, 1216)
                depth = encoded.astype(np.float32) / 256
                rgb = cv2.cvtColor(cv2.imread(str(rgb_root / (sid + ".png"))), cv2.COLOR_BGR2RGB)
                assert rgb.shape == (352, 1216, 3) and np.isfinite(depth).all()
                (output / "depth_metric_uint16" / (sid + ".png")).write_bytes(raw)
                samples.append((sid, rgb, depth))
                records.append({"id": sid, "sha256": hashlib.sha256(raw).hexdigest(),
                                "min_m": float(depth.min()), "median_m": float(np.median(depth)),
                                "max_m": float(depth.max()), "display_saturated_above_80_fraction": float((depth > 80).mean())})
    norm = Normalize(0, 80, clip=True)
    for sid, rgb, depth in samples:
        plt.imsave(output / "depth_color" / (sid + ".png"), depth, cmap="turbo_r", vmin=0, vmax=80)
        fig, axes = plt.subplots(1, 2, figsize=(18, 3.3), layout="constrained")
        axes[0].imshow(rgb)
        axes[0].set_title(f"RGB | KITTI {sid}")
        im = axes[1].imshow(depth, cmap="turbo_r", norm=norm, interpolation="nearest")
        axes[1].set_title("V8 final metric depth | red near / blue far")
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im, ax=axes[1], fraction=.025, extend="max", label="Depth (m)")
        fig.savefig(output / "pairs" / (sid + "_rgb_depth.png"), dpi=180)
        plt.close(fig)
    for page in range(2):
        fig, axes = plt.subplots(5, 2, figsize=(18, 12.5), layout="constrained")
        fig.suptitle(f"V8 actual KITTI test predictions | {page + 1}/2\nRGB left / depth right | shared 0-80 m colors; >80 m display-saturated; no public test GT")
        for row, (sid, rgb, depth) in enumerate(samples[page * 5:page * 5 + 5]):
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(sid)
            im = axes[row, 1].imshow(depth, cmap="turbo_r", norm=norm, interpolation="nearest")
            axes[row, 1].set_title("V8 metric depth (m)")
            for ax in axes[row]:
                ax.axis("off")
        fig.colorbar(im, ax=axes[:, 1].tolist(), fraction=.015, extend="max", label="Depth (m)")
        fig.savefig(output / f"v8_rgb_depth_sheet_{page + 1}.png", dpi=150)
        plt.close(fig)
    report = {"source": str(source), "checkpoint_epoch": test_report.get("checkpoint_epoch"),
              "validation_rmse_m": metrics["final"]["all"]["rmse_m"], "test_rmse_m": None,
              "samples": records, "display_scale_m": [0, 80],
              "selection": "10 evenly spaced IDs; not cherry-picked by error",
              "encoding": "Original uint16 PNG / 256 = metres",
              "meaning": "Final student depth, not teacher or intermediate coarse depth. No smoothing, per-image normalization or AI generation."}
    (output / "preview_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log = pd.read_csv(audit / "train_log.csv")
    print("OUTPUT", output)
    print("LOG", len(log), "epochs", log.epoch.min(), log.epoch.max())
    print("BEST LOG", log.loc[log.val_rmse_m.idxmin()].to_json())
    for name in ("val_metrics.json", "profile.json", "training_status.json", "export_report.json"):
        print(name, (audit / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
