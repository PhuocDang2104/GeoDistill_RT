# AnchorFlow V6 — team development

Canonical code: [V6 folder](../drive_upload/AnchorFlow_v6_Connection/README.md).

- [Architecture, math and objective](../drive_upload/AnchorFlow_v6_Connection/ARCHITECTURE.md).
- [V5 audit and design decisions](../drive_upload/AnchorFlow_v6_Connection/ANALYSIS_V5.md).
- [Colab notebook, 15 epochs](../drive_upload/AnchorFlow_v6_Connection/AnchorFlow_v6_TAR2000_15ep.ipynb).
- [Verification scope](../drive_upload/AnchorFlow_v6_Connection/VERIFICATION.md).

V6 retains V5 and adds ray-projected connection-aware metric correction at D4 and phase metric correction at D2. Teacher metric targets are training-only. CPU/ONNX checks are not trained accuracy or edge-device latency benchmarks. RMSE<0.7 is a research target, not a guarantee. Existing GeoLift S2/S3 implementations remain unchanged.

## Git versus Drive artifacts

Git contains source, notebook, documentation and small audit CSV/JSON. It does not contain datasets, student/teacher checkpoints, ONNX exports, results ZIPs or preview images. `bundle_manifest.json` describes the COMPLETE Drive package, not every file available in a Git clone.

For the full notebook/tests/packager, restore `init_v3_best.pth` from the team's Drive artifact into the V6 folder. SHA256: `1c767a871cb57336ccc49460f84f87afb81b30d95f7e746f34e7e908280a3d07`. This is the fallback/test artifact; default training loads the actual V5 best from Drive using `PARENT_CHECKPOINT`. Never commit weights. Alternatively use the complete Drive ZIP prepared locally.

Data: `MyDrive/GeoLift_Data/teacher_subset_2000/`. Default output: `MyDrive/GeoLift_RT_Runs/AnchorFlow_v6_Connection_FromV5_FT15/`.

## Code-only geometry/model tests

From the V6 folder, with PyTorch and requirements installed:

```bash
python -m unittest test_connection -v
```

Full tests require the V3 artifact above. The local real-data verifier additionally needs KITTI at its documented local path. Packaging checks notebook syntax and source/proof hashes; full training and actual teacher coverage are checked in Colab.

Keep data/split/evaluation fixed when developing. Change the run name when changing source, parent or recipe; do not edit source midway through resume. Use matched V5 and ambient-vector controls before attributing gains to architecture or connection transport.
