# Drive bundles

## Current baseline

[AnchorFlow_S4_UPIR_NoRelative_30ep_A100](AnchorFlow_S4_UPIR_NoRelative_30ep_A100/README.md):
upload WHOLE folder to MyDrive and open `Train_S4_UPIR_NoRelative_Fresh30_A100.ipynb`.
Fresh30/resume, seed42, metric teacher ON, relative OFF; full400 TRT FP32 graph
benchmark and1000test predictions. Selected G best_policy epoch27.

Frozen source/config/manifest remain unchanged after training. Never upload helpers
inside frozen CODE: source hash includes every `.py`. Checkpoints/data remain outside Git.

Moving to another Colab/Drive account? Follow the
[metric-only dataset checklist](../docs/GeoLift_Data_MetricOnly_Transfer.md).
Only the teacher type is metric-only; RGB/sparse/GT/K and split JSON are still required.

## Deployment

[Orin NX / AGX package](../production/AnchorFlow_S4_UPIR_Orin/README.md): upload as
SEPARATE helper folder, export from completed Colab, copy ONNX/manifest/real samples
to Orin. Native build, inference only; no PyTorch needed on target. Orin not measured yet.

## New experiment — not promoted

[S4-U-PIR DecoderCapacity](AnchorFlow_S4_UPIR_DecoderCapacity_30ep_A100/README.md):
fresh30 ×3 on A100, matched G0 control + wider D16/D8 decoder H + H/D4 extension I.
Metric teacher only, no relative path; full400 TensorRT FP32 + CUDA Graph metric/
runtime comparison. Current baseline stays G until measured results justify a change.

## Controls

A/B/C FineAblation, D NoFine, E JPIR and F dual-teacher UPIR are controls, not
simultaneous current baselines. Retain for reproduction; G runtime does not depend
on them. Prior V11.3 adaptive/Fixed6 and S6/S3 audits are historical workflows.
[Current equations/results](../docs/AnchorFlow_V11_Baseline.md).
