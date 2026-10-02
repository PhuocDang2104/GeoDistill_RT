"""Render ten actual V6 KITTI predictions; shared metric colors, no smoothing."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "results/metric_kd-20261002T124131Z-1-001.zip"
    rgb_root = root / "data/depth_selection/test_depth_completion_anonymous/image"
    output = root / "results/v6_depth_preview"
    for name in ("pairs", "depth_color", "rgb", "depth_metric_uint16"):
        (output/name).mkdir(parents=True,exist_ok=True)
    samples,records = [],[]
    ids = [f"{i:010d}" for i in np.linspace(0,999,10,dtype=int)]
    with zipfile.ZipFile(source) as outer:
        test_report = json.loads(outer.read("metric_kd/test_report.json"))
        metrics = json.loads(outer.read("metric_kd/val_metrics.json"))
        with zipfile.ZipFile(io.BytesIO(outer.read("metric_kd/kitti_test_predictions.zip"))) as predictions:
            assert len(predictions.namelist())==len(set(predictions.namelist()))==1000
            for sid in ids:
                raw = predictions.read(sid+".png")
                encoded = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
                assert encoded.dtype==np.uint16 and encoded.shape==(352,1216)
                depth = encoded.astype(np.float32)/256
                rgb_path = rgb_root/(sid+".png")
                rgb_raw = rgb_path.read_bytes()
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(rgb_raw,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                assert rgb.shape==(352,1216,3) and np.isfinite(depth).all() and (depth>0).all()
                (output/"rgb"/(sid+".png")).write_bytes(rgb_raw)
                (output/"depth_metric_uint16"/(sid+".png")).write_bytes(raw)
                samples.append((sid,rgb,depth))
                records.append(dict(sid=sid,raw_prediction_sha256=hashlib.sha256(raw).hexdigest(),
                    min_m=float(depth.min()),median_m=float(np.median(depth)),max_m=float(depth.max())))
    norm = Normalize(0,80,clip=True)
    for sid,rgb,depth in samples:
        plt.imsave(output/"depth_color"/(sid+".png"),depth,cmap="turbo_r",vmin=0,vmax=80)
        fig,axes = plt.subplots(1,2,figsize=(18,3.2),layout="constrained")
        axes[0].imshow(rgb); axes[0].set_title(f"KITTI test {sid} | RGB")
        im = axes[1].imshow(depth,cmap="turbo_r",norm=norm,interpolation="nearest")
        axes[1].set_title("V6 predicted metric depth | red near / blue far")
        for ax in axes: ax.axis("off")
        fig.colorbar(im,ax=axes[1],fraction=.025,extend="max",label="Depth (m)")
        fig.savefig(output/"pairs"/f"{sid}_rgb_depth.png",dpi=180,facecolor="white")
        plt.close(fig)
    for page in range(2):
        fig,axes = plt.subplots(5,2,figsize=(18,12.5),layout="constrained")
        fig.suptitle(f"V6 actual test predictions | page {page+1}/2 | RGB left / depth right\n"
            "Shared colors: 0-80 m, red near / blue far; >80 m display-saturated; no public test GT",fontsize=14)
        for row,(sid,rgb,depth) in enumerate(samples[page*5:page*5+5]):
            axes[row,0].imshow(rgb); axes[row,0].set_title(f"RGB | {sid}")
            im = axes[row,1].imshow(depth,cmap="turbo_r",norm=norm,interpolation="nearest")
            axes[row,1].set_title(f"V6 metric depth | {sid}")
            for ax in axes[row]: ax.axis("off")
        fig.colorbar(im,ax=axes[:,1].tolist(),fraction=.015,extend="max",label="Depth (m)")
        fig.savefig(output/f"v6_rgb_depth_sheet_{page+1}.png",dpi=150,facecolor="white")
        plt.close(fig)
    report = dict(source_archive=str(source),prediction_member="metric_kd/kitti_test_predictions.zip",
        checkpoint_epoch=test_report.get("checkpoint_epoch"),samples=records,
        selection="10 evenly spaced IDs; no error-based selection",display_scale_m=[0,80],
        encoding="uint16 PNG / 256 = metres",validation_rmse_m=metrics["final"]["all"]["rmse_m"],test_rmse_m=None,
        interpretation="Final model predictions, NOT intermediate coarse D4 or teacher depth; no rerun, smoothing or per-image normalization")
    (output/"preview_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(output),pairs=10,checkpoint_epoch=report["checkpoint_epoch"]),indent=2))


if __name__=="__main__":
    main()
