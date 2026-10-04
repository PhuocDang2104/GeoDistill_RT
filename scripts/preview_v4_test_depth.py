"""Render actual archived V4 anonymous-test predictions, paired with official RGB.

No model rerun, normalization per image, smoothing, or editing prediction values.
The 0..80m color clipping is DISPLAY ONLY; original uint16 PNGs remain untouched.
"""
import io
import json
import shutil
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
    archive_path = root/"results/metric_kd-20261002T091341Z-1-001.zip"
    rgb_root = root/"data/depth_selection/test_depth_completion_anonymous/image"
    output = root/"results/v4_depth_preview"
    for name in ("pairs","depth_metric_uint16","rgb"):
        (output/name).mkdir(parents=True,exist_ok=True)
    # Evenly spaced IDs, selected before viewing results, not cherry-picked by error.
    ids = [f"{n:010d}" for n in (0,199,399,599,799,999)]
    norm = Normalize(vmin=0,vmax=80,clip=True)
    cmap = "turbo_r"
    samples,report = [],[]
    with zipfile.ZipFile(archive_path) as outer:
        nested = io.BytesIO(outer.read("metric_kd/kitti_test_predictions.zip"))
        with zipfile.ZipFile(nested) as predictions:
            assert len(predictions.namelist())==1000
            for sid in ids:
                raw = predictions.read(sid+".png")
                encoded = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
                assert encoded.dtype==np.uint16 and encoded.shape==(352,1216)
                depth = encoded.astype(np.float32)/256
                assert np.isfinite(depth).all() and (depth>0).all()
                rgb_path = rgb_root/(sid+".png")
                rgb = cv2.cvtColor(cv2.imread(str(rgb_path)),cv2.COLOR_BGR2RGB)
                assert rgb.shape==(352,1216,3)
                # Exact bytes of the submitted prediction, not a newly normalized PNG.
                (output/"depth_metric_uint16"/(sid+".png")).write_bytes(raw)
                shutil.copy2(rgb_path,output/"rgb"/rgb_path.name)
                samples.append((sid,rgb,depth))
                report.append(dict(sample_id=sid,min_m=float(depth.min()),median_m=float(np.median(depth)),
                                   p99_m=float(np.quantile(depth,.99)),max_m=float(depth.max()),
                                   fraction_above_display_80m=float((depth>80).mean())))
    plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans"})
    for sid,rgb,depth in samples:
        fig,axes = plt.subplots(1,2,figsize=(16,3.0),layout="constrained")
        axes[0].imshow(rgb); axes[0].set_title(f"KITTI anonymous test {sid} | RGB")
        image = axes[1].imshow(depth,cmap=cmap,norm=norm,interpolation="nearest")
        axes[1].set_title("V4 metric depth | red = near, blue = far")
        for ax in axes:
            ax.axis("off")
        bar = fig.colorbar(image,ax=axes[1],fraction=.025,pad=.015,extend="max")
        bar.set_label("Depth (m)")
        fig.savefig(output/"pairs"/f"{sid}_rgb_depth.png",dpi=180,facecolor="white")
        plt.close(fig)
    fig,axes = plt.subplots(6,2,figsize=(16,15),layout="constrained")
    fig.suptitle("Actual V4 test predictions | RGB (left) vs metric depth (right)\n"
                 "Shared display scale: 0-80 m; >80 m saturated blue | Test RMSE: unknown (no GT)",fontsize=15)
    for row,(sid,rgb,depth) in enumerate(samples):
        axes[row,0].imshow(rgb)
        axes[row,0].set_title(f"RGB | test {sid}",fontsize=10)
        image = axes[row,1].imshow(depth,cmap=cmap,norm=norm,interpolation="nearest")
        axes[row,1].set_title(f"V4 depth | test {sid}",fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
    colorbar = fig.colorbar(image,ax=axes[:,1].tolist(),fraction=.015,pad=.015,extend="max")
    colorbar.set_label("Depth (m): red = near / blue = far")
    fig.savefig(output/"v4_rgb_depth_contact_sheet.png",dpi=170,facecolor="white")
    plt.close(fig)
    manifest = dict(source_archive=str(archive_path),checkpoint_epoch=7,
                    predictions="Actual archived test submission, not new inference or generated illustration",
                    ids=ids,selection="Six evenly spaced IDs; no error-based selection",
                    encoding="uint16 PNG; metric depth_m = pixel_value / 256",
                    display=dict(min_m=0,max_m=80,colormap=cmap,above_80m="saturated; raw depths unchanged"),
                    validation_rmse_m=1.0300762691495098,test_rmse_m=None,
                    note="Anonymous test has no public GT. Validation RMSE is not a per-image test score. Sky/unobserved regions may be unreliable.",
                    samples=report)
    (output/"preview_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(output),samples=6,contact_sheet=str(output/"v4_rgb_depth_contact_sheet.png")),indent=2))


if __name__=="__main__":
    main()
