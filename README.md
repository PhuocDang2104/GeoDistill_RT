# GeoDistill-RT / GeoRT

Conflict-Aware Geometry Distillation for Real-Time Sparse Depth Completion.

## Project Goal

GeoRT trains a small sparse depth completion student. Teacher models run offline to generate pseudo labels; the deployed student uses only RGB, sparse LiDAR, validity mask, and camera intrinsics.

At inference time there is no Metric3D, Depth Anything V2, DSINE, DMD3C, normal head, or heavy teacher decoder in the student path.

## Method Summary

1. Metric3D / Metric3Dv2 predicts metric dense depth.
2. Depth Anything V2 predicts relative dense depth and is aligned to sparse LiDAR with robust scale-shift fitting.
3. DSINE predicts surface normals.
4. DMD3C can optionally predict a fine depth-completion teacher from RGB + sparse LiDAR.
5. Teacher depth candidates are fused per pixel using sparse-depth consistency and DSINE normal consistency.
6. The fused teacher target and confidence are saved at 1/4 resolution.
7. GeoRT-Student-S learns coarse metric depth and confidence from GT, sparse LiDAR, and fused teacher supervision.

## Repo Structure

```text
GeoRT/
├── README.md
├── requirements.txt
├── configs/
├── data/depth_selection/
├── third_party/
│   ├── Metric3D/
│   ├── Depth-Anything-V2/
│   ├── DSINE/
│   └── DMD3C/
├── weights/
│   ├── metric3d/
│   ├── depth_anything_v2/
│   ├── dsine/
│   └── dmd3c/
├── teacher_outputs/
├── student_outputs/
├── notebooks/
└── src/
```

## Dataset Layout

This repo is now wired to KITTI `depth_selection`:

```text
data/depth_selection/
├── val_selection_cropped/
│   ├── image/
│   ├── velodyne_raw/
│   ├── groundtruth_depth/
│   └── intrinsics/
├── test_depth_completion_anonymous/
│   ├── image/
│   ├── velodyne_raw/
│   └── intrinsics/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Create or refresh split files:

```bash
python -m src.prepare_depth_selection --data_root data/depth_selection --train_count 800
```

Current policy: first 800 `val_selection_cropped` samples become `train`, remaining 200 become `val`, all `test_depth_completion_anonymous` samples become `test`.

KITTI 16-bit depth PNG files are decoded as meters with scale `256.0`.

## Colab Project Path

The expected Drive project root is:

```python
PROJECT_DIR = "/content/drive/MyDrive/DEPTH-FUSION | Workspace/monocular_sparse_fusion/GeoRT"
```

In shell cells, always quote this path because it contains `|`:

```bash
PROJECT_DIR="/content/drive/MyDrive/DEPTH-FUSION | Workspace/monocular_sparse_fusion/GeoRT"
cd "$PROJECT_DIR"
```

## Teacher Setup

Clone official repositories:

```bash
rm -f third_party/Metric3D/.gitkeep
rm -f third_party/Depth-Anything-V2/.gitkeep
rm -f third_party/DSINE/.gitkeep
rm -f third_party/DMD3C/.gitkeep

git clone https://github.com/YvanYin/Metric3D.git third_party/Metric3D
git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
git clone https://github.com/baegwangbin/DSINE.git third_party/DSINE
git clone https://github.com/Sharpiless/DMD3C.git third_party/DMD3C
```

Install base dependencies:

```bash
pip install -r requirements.txt
```

Install teacher-specific dependencies when present:

```bash
pip install -r third_party/Metric3D/requirements_v2.txt
pip install -r third_party/Depth-Anything-V2/requirements.txt
pip install -r third_party/DSINE/requirements.txt
```

Build DMD3C / BP-Net CUDA extension on a CUDA machine:

```bash
cd third_party/DMD3C/exts
python setup.py install
cd ../../..
```

Teacher wrappers:

```text
src/teachers/metric3d_wrapper.py
src/teachers/depth_anything_wrapper.py
src/teachers/dsine_wrapper.py
src/teachers/dmd3c_wrapper.py
```

## Teacher Weights Placement

Place weights here:

```text
weights/metric3d/
weights/depth_anything_v2/
weights/dsine/
weights/dmd3c/
```

Examples:

```text
weights/metric3d/metric_depth_vit_large_800k.pth
weights/depth_anything_v2/depth_anything_v2_vitl.pth
weights/dsine/dsine.pt
weights/dsine/dsine.txt
weights/dmd3c/dmd3c_distillation_depth_anything_v2.pth
```

## Generate Pseudo Labels

```bash
python -m src.teachers.generate_teachers \
  --config configs/teacher.yaml \
  --split val \
  --run_metric3d \
  --run_depth_anything \
  --run_dsine \
  --run_dmd3c \
  --run_fusion
```

Teacher generation is restartable. Existing `.npz` files are skipped when `skip_existing: true`.

## Train Student

```bash
python -m src.train_student --config configs/geort_student_s.yaml
```

Checkpoints and logs are saved under `student_outputs/`.

## Run Inference

```bash
python -m src.infer_student \
  --config configs/geort_student_s.yaml \
  --checkpoint student_outputs/checkpoints/best.pth \
  --split test
```

## Output Formats

Metric3D:

```text
teacher_outputs/metric3d/{split}/{sample_id}.npz
keys: D_m3d float32 [H,W]
```

Depth Anything V2:

```text
teacher_outputs/depth_anything/{split}_raw/{sample_id}.npz
keys: D_da_raw float32 [H,W]

teacher_outputs/depth_anything/{split}_aligned/{sample_id}.npz
keys: D_da_aligned float32 [H,W], scale float, shift float
```

DSINE:

```text
teacher_outputs/dsine/{split}/{sample_id}.npz
keys: N_dsine float32 [3,H,W]
```

DMD3C:

```text
teacher_outputs/dmd3c/{split}/{sample_id}.npz
keys: D_dmd3c float32 [H,W]
```

Fused teacher:

```text
teacher_outputs/fused/{split}/{sample_id}.npz
keys: D_teacher [H/4,W/4], C_teacher [H/4,W/4], w_m3d optional, w_da optional, w_dmd3c optional
```

Student inference:

```text
student_outputs/{split}_predictions/{sample_id}.npz
keys: D_c float32 [H/4,W/4], C float32 [H/4,W/4]
```

## Troubleshooting

- If splits are missing, run `python -m src.prepare_depth_selection --data_root data/depth_selection --train_count 800`.
- If a teacher wrapper fails to import, verify the matching official repo exists under `third_party/`.
- If a teacher fails to load weights, verify the expected checkpoint exists under `weights/`.
- If Metric3D depth scale looks wrong, verify the KITTI intrinsics file for that sample.
- If DSINE output is invalid, verify `weights/dsine/dsine.txt` points to a valid checkpoint.
- If DMD3C import fails on `BpOps`, build `third_party/DMD3C/exts` with `python setup.py install` on CUDA.
