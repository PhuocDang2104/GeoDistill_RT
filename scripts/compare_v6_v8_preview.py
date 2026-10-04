"""Five matched KITTI RGB/V6/V8 rows with identical metric color scales."""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/v8_depth_preview/v6_vs_v8_5_rgb_depth.png"
    ids = ["0000000000", "0000000222", "0000000444", "0000000777", "0000000999"]
    fig, axes = plt.subplots(5, 3, figsize=(24, 12), layout="constrained")
    fig.suptitle("KITTI test | matched RGB / V6 / V8\nShared metric colors: 0-80 m; red near, blue/purple far; >80 m display-saturated. No public test GT.", fontsize=16)
    norm = Normalize(0, 80, clip=True)
    for row, sid in enumerate(ids):
        rgb_path = root / "data/depth_selection/test_depth_completion_anonymous/image" / (sid + ".png")
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"RGB | {sid}")
        for col, version in ((1, "v6"), (2, "v8")):
            path = root / f"results/{version}_depth_preview/depth_metric_uint16" / (sid + ".png")
            encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert encoded is not None and encoded.dtype == np.uint16 and encoded.shape == rgb.shape[:2]
            im = axes[row, col].imshow(encoded.astype(np.float32) / 256, cmap="turbo_r", norm=norm, interpolation="nearest")
            axes[row, col].set_title(f"{version.upper()} | {sid}")
        for ax in axes[row]:
            ax.axis("off")
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), fraction=.012, extend="max", label="Depth (m)")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
