# Visualize Teacher Outputs

Use `src.visualize_teacher` to inspect any saved `.npz` teacher or student output.

## Install Dependency

Open3D is required for interactive depth point-cloud viewing:

```bash
pip install -r requirements.txt
```

If you only want 2D PNG previews, Open3D is not used.

## DSINE Normal Visualization

Input:

```text
teacher_outputs/dsine/{split}/{sample_id}.npz
key: N_dsine float32 [3,H,W]
```

Run on Windows:

```powershell
python -m src.visualize_teacher "C:\Users\ADMIN\Desktop\GeoDistill_RT\teacher_outputs\dsine\val\2011_10_03_drive_0047_sync_image_0000000791_image_03.npz"
```

Output:

```text
visualizations/teacher/2011_10_03_drive_0047_sync_image_0000000791_image_03_N_dsine_opengl_normal.png
```

Normal RGB uses OpenGL-style encoding:

```text
R = +X
G = +Y up
B = +Z
```

Because camera/image-space normals often use +Y downward, the script flips Y by default. To keep the raw Y channel:

```powershell
python -m src.visualize_teacher "...\sample.npz" --normal-y keep
```

## Metric3D Depth Point Cloud

Input:

```text
teacher_outputs/metric3d/{split}/{sample_id}.npz
key: D_m3d float32 [H,W]
```

Run:

```powershell
python -m src.visualize_teacher "C:\Users\ADMIN\Desktop\GeoDistill_RT\teacher_outputs\metric3d\val\2011_10_03_drive_0047_sync_image_0000000791_image_03.npz"
```

The script will:

1. Load `D_m3d`.
2. Find the sample in `data/depth_selection/splits`.
3. Load the matching RGB image and intrinsics.
4. Save a 2D depth colormap PNG.
5. Open an interactive Open3D point-cloud window.

Useful options:

```powershell
python -m src.visualize_teacher "...\sample.npz" --stride 1
python -m src.visualize_teacher "...\sample.npz" --max-depth 80
python -m src.visualize_teacher "...\sample.npz" --save-ply
python -m src.visualize_teacher "...\sample.npz" --no-open3d --save-ply
```

`--stride 1` shows all valid pixels but is heavier. `--stride 2` is the default.

## Other Supported Keys

The script auto-detects these keys:

```text
N_dsine
D_m3d
D_da_raw
D_da_aligned
D_dmd3c
D_teacher
C_teacher
w_m3d
w_da
w_dmd3c
D_c
C
```

Override key or mode manually:

```powershell
python -m src.visualize_teacher "...\sample.npz" --key D_teacher --mode pointcloud
python -m src.visualize_teacher "...\sample.npz" --key C_teacher --mode confidence
```

## Colab Note

Interactive Open3D windows usually require a local desktop display. On Colab, save PNG/PLY instead:

```bash
python -m src.visualize_teacher "$PROJECT_DIR/teacher_outputs/metric3d/val/sample.npz" --no-open3d --save-ply
```
