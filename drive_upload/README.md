# Drive bundles

## Current baseline

[AnchorFlow_S4_UPIR_NoRelative_30ep_A100](AnchorFlow_S4_UPIR_NoRelative_30ep_A100/README.md):
upload WHOLE folder to MyDrive and open `Train_S4_UPIR_NoRelative_Fresh30_A100.ipynb`.
Fresh30/resume, seed42, metric teacher ON, relative OFF; full400 TRT FP32 graph
benchmark and1000test predictions. Selected G best_policy epoch27.

Frozen source/config/manifest remain unchanged after training. Never upload helpers
inside frozen CODE: source hash includes every `.py`. Checkpoints/data remain outside Git.

## Deployment

[Orin NX / AGX package](../production/AnchorFlow_S4_UPIR_Orin/README.md): upload as
SEPARATE helper folder, export from completed Colab, copy ONNX/manifest/real samples
to Orin. Native build, inference only; no PyTorch needed on target. Orin not measured yet.

## Controls

A/B/C FineAblation, D NoFine, E JPIR and F dual-teacher UPIR are controls, not
simultaneous current baselines. Retain for reproduction; G runtime does not depend
on them. Prior V11.3 adaptive/Fixed6 and S6/S3 audits are historical workflows.
[Current equations/results](../docs/AnchorFlow_V11_Baseline.md).
