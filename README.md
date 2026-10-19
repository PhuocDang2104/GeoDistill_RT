# GeoDistill-RT / AnchorFlow

**Current deployment baseline: V11-derived S4-U-PIR NoRelative (`G_s4_upir_norel`).**
Coarse **4 NFE**, no fine NODE,583,880 parameters. `best_policy.pth` epoch27 from
fresh30, seed42,1.600train/400val; metric teacher ON, relative loss OFF.

| A100 · TensorRT FP32 + CUDA Graph | Measured |
|---|---:|
| RMSE / iRMSE |0.990151m /3.172074km⁻¹|
| MAE / boundary3 RMSE |0.258984m /1.685668m|
| Wall median / P95 |3.827 /3.845ms|
| FPS |261.3|

Full400 /25,424,992 valid pixels. B1×352×1216,100 changing timed images,
D2D included, decoding/preprocessing/H2D excluded. **Not end-to-end or Orin latency.**
G improves RMSE versus no-fine D by0.274%, but changes both U-conditioning and
relative loss; separate architectural gain is not established. Not all old
targets pass; MAE/boundary winners differ. No leaderboard/universal accuracy claim.

## Start here

- [Baseline architecture, equations, evidence and limits](docs/AnchorFlow_V11_Baseline.md).
- [Frozen Drive training package + notebook](drive_upload/AnchorFlow_S4_UPIR_NoRelative_30ep_A100/README.md).
- [Orin NX / AGX GPU inference package](production/AnchorFlow_S4_UPIR_Orin/README.md):
  export ONNX/assets on Colab, build engine and benchmark on target; no training.
- [Completed run evidence and attribution analysis](benchmarks/s4_upir_norel_2026_10_19/README.md).
- [Historical V11.3 accuracy](docs/AnchorFlow_V11_3.md),
  [old Fixed6 RK3+fine2 contract](docs/history/AnchorFlow_V11_Fixed6_2026_10_14.md).

Weights, KITTI and teacher payloads stay local/Drive, not Git. The Orin package
contains code, **not checkpoint/exported ONNX**; exporter records the actual
checkpoint SHA and retains strict training contracts. Current model does not
depend on legacy `src/` or older-version folders.

[Docs](docs/README.md) · [Drive bundles](drive_upload/README.md) · [Cleanup](REPO_CLEANUP.md).
