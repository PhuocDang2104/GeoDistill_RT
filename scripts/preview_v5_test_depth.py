"""Six measured V5 test outputs paired with their actual KITTI RGB, no AI images."""
import hashlib
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
    archive_path = root / "results/kitti_test_predictions.zip"
    rgb_root = root / "data/depth_selection/test_depth_completion_anonymous/image"
    output = root / "results/v5_depth_preview"
    (output / "pairs").mkdir(parents=True, exist_ok=True)
    samples, records = [], []
    ids = [f"{i:010d}" for i in (0,199,399,599,799,999)]
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == 1000
        for sid in ids:
            raw = archive.read(sid + ".png")
            encoded = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
            assert encoded.dtype == np.uint16 and encoded.shape == (352,1216)
            depth = encoded.astype(np.float32)/256
            rgb = cv2.cvtColor(cv2.imread(str(rgb_root/(sid+".png"))),cv2.COLOR_BGR2RGB)
            assert rgb.shape == (352,1216,3) and np.isfinite(depth).all() and (depth>0).all()
            samples.append((sid,rgb,depth))
            records.append(dict(sid=sid,raw_prediction_sha256=hashlib.sha256(raw).hexdigest(),
                min_m=float(depth.min()),median_m=float(np.median(depth)),max_m=float(depth.max())))
    norm = Normalize(0,80,clip=True)
    for sid,rgb,depth in samples:
        fig,axes = plt.subplots(1,2,figsize=(16,3),layout="constrained")
        axes[0].imshow(rgb); axes[0].set_title(f"KITTI test {sid} | RGB")
        im = axes[1].imshow(depth,cmap="turbo_r",norm=norm,interpolation="nearest")
        axes[1].set_title("V5 metric depth | red near / blue far")
        for ax in axes: ax.axis("off")
        fig.colorbar(im,ax=axes[1],fraction=.025,extend="max",label="Depth (m)")
        fig.savefig(output/"pairs"/f"{sid}_rgb_depth.png",dpi=180,facecolor="white")
        plt.close(fig)
    fig,axes = plt.subplots(6,2,figsize=(16,15),layout="constrained")
    fig.suptitle("Actual V5 test predictions | RGB left / metric depth right\n"
        "Shared scale 0-80 m; >80 m display-saturated | Test RMSE unknown: no public GT",fontsize=15)
    for row,(sid,rgb,depth) in enumerate(samples):
        axes[row,0].imshow(rgb); axes[row,0].set_title(f"RGB | {sid}")
        im = axes[row,1].imshow(depth,cmap="turbo_r",norm=norm,interpolation="nearest")
        axes[row,1].set_title(f"V5 depth | {sid}")
        for ax in axes[row]: ax.axis("off")
    fig.colorbar(im,ax=axes[:,1].tolist(),fraction=.015,extend="max",label="Depth (m)")
    fig.savefig(output/"v5_rgb_depth_contact_sheet.png",dpi=150,facecolor="white")
    plt.close(fig)
    report = dict(source_archive=str(archive_path),samples=records,selection="Six evenly spaced IDs; same as previous V4 preview",
        interpretation="Archived predictions, not rerun inference; no per-image normalization or smoothing",
        encoding="uint16 / 256 = metres",display_scale=[0,80],validation_rmse_m=1.02340515639136,
        test_rmse_m=None,warning="Validation RMSE is NOT the test score; sky and other unobserved regions are not certified.")
    (output/"preview_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
