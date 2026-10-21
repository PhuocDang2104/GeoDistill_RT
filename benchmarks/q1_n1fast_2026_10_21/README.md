# Q1 baseline evidence — 21/10/2026

Current baseline by explicit user decision: **Q1_s4_n1_fast**. The historical
pipeline emitted `subset_development_candidate_only/full_KITTI_baseline_promoted=false`;
that output is preserved verbatim. Adoption is a new external record, not a rewrite
of completed run contracts or a claim that full KITTI was evaluated.

Sources: [raw user console](user_console.txt), [structured evidence](evidence.json),
[same-session CSV](comparison.csv), [adoption record](../../production/AnchorFlow_S4_Q1_Fast/baseline.json).
Raw-byte SHA is in evidence.json; reproduce with `python scripts/summarize_q1_baseline.py`.

| Checkpoint / engine | RMSE m | iRMSE km⁻¹ | MAE m | Boundary3 m | Mean ms | FPS=1000/mean |
|---|---:|---:|---:|---:|---:|---:|
| N1 / original |0.979622|3.180601|0.257036|1.678923|4.211659|237.44|
| N1 / fast |0.979622|3.180601|0.257036|1.678923|3.850728|259.69|
| Q1 / original |0.978691|3.170880|0.255826|1.667166|4.229391|236.44|
| **Q1 / fast** |**0.978691**|**3.170880**|**0.255826**|**1.667166**|**3.846668**|**259.97**|

Session `ec751356850d497baefa6ef881c0eb57`. Q1 fast reverse-order repeat3.845360ms.
Unrounded RMSE<0.98/iRMSE<3.2/mean≤3.9 and reverse repeat≤3.9 all pass.

## What the comparison supports

- Same Q1 checkpoint original→fast reduces mean wall by about9.05%, with RMSE
  drift9.41e−9m/iRMSE drift8.10e−8km⁻¹. This supports a combined deployment
  improvement (operators + builder), not increased neural accuracy.
- Q1 fast versus N1 fast improves RMSE about0.0951%, but uses different trained
  weights. No new neural architecture/loss, no isolated causal accuracy claim.
- Both already use Graph; neither a Graph-vs-eager speedup nor each rewrite's
  individual contribution follows from this table.

Protocol: reported A100 pipeline, full400/25,424,992 valid GT pixels,B1×352×1216,
TRT10.13.3 strongly-typedFP32/TF32OFF; originallevel3/fastlevel5,auxstreams0.
100warmup/300iterations/100distinctGPUimages. Wall includesD2D, excludes decode/
preprocess/H2D/build/capture/metrics. Median/P95 exported separately. Not end-to-end
throughput or Orin latency. Boundary3 is the fixed GT-discontinuity protocol.

## Evidence limits and private artifacts

This is a **user-supplied console snapshot**, not a local copy of the full report,
weights,engine or all400 per-image records. Exact GPU model/software patch versions,
checkpoint epoch/hashes must be read from the actual Drive artifacts, not guessed.
Qualification/30epoch/source-data proof is reported by the frozen pipeline; the
local agent did not independently rerun the A100 training/validation.

Retain on Drive: runtime_config/data_contract/frozen_bundle, Q1 best_policy.pth,
training_status/train_log, full trt_benchmark.json, original/fastONNX/plans/layer
reports, replay_parity.json, comparison.csv/json, baseline_candidate.json,
kitti_test_predictions.zip and test_preview. See [artifact checklist](../../production/AnchorFlow_S4_Q1_Fast/README.md).

Source-level local verification remains the frozen19tests/full352×1216 CPU ONNX
report. New reverification lives outside that bundle. Current400validation was
repeatedly used for development; no independent-sequence/multi-seed/full-KITTI/
official-test/Orin claim is made.
