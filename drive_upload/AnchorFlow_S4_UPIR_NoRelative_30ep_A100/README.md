# G: S4-UPIR NoRelative · identical U-PIR architecture · A100 · Fresh30

Upload WHOLE folder to **MyDrive/AnchorFlow_S4_UPIR_NoRelative_30ep_A100**.
Open **Train_S4_UPIR_NoRelative_Fresh30_A100.ipynb**, choose A100, Run all.
Uses existing GeoLift_Data and old Drive run reports, no old source dependency.

Read [ANALYSIS.md](ANALYSIS.md) for evidence, rejected conclusions and rationale.

```text
RGB/sparse → unchanged encoder/decoder → S4 coarse Jet NODE (4NFE)
              → five-jet consensus → D2 (no fine NODE/CNN)
                         └ relative variance U2 ──────────────┐
              → phase lift/detail → existing PIR input + c(U2)
                                      → soft sensor fusion → Dfull
```

**Only first PIR projection input40→41**, hidden16 and output8 unchanged. Additional
16weights, initialized0; all old parameters copied after original constructor.
No beta gate injection; no new CNN block, field call, wider decoder or auxiliary loss.
Total **583,880params**, coarse4/fine0NFE. Bounded cue is U/(U+.002), NOT calibrated
probability. It conditions residual amplitude/sign AND gates through the existing
network. Same PIR correction bound (.5+.05D), .1–120m depth domain and sensor fusion.

## Training

Fresh30, seed42, ImageNet encoder only;1600train/400val/test1000, **metric teacher only**.
`relative_enabled=false`, `relative_weight=0`. Relative loss/pair building, cache
loading and archive preparation are skipped; DA3MONO archive is NOT required.
Original relative helpers remain lazy for historical tests, not default execution.
Metric KD remains .012→.006, no compensation/increase of another loss.
Same AdamW LR3e-4 encoder×.5/WD1e-5, B4, warmup1+cosine, BNfreeze,
flip/holdout. Eager nativeBF16 CNN + FP32geometry/solver/loss; TF32 OFF. All30epochs,
no early stop. Same remaining objectives/coefficients as U-PIR. Relative CSV terms
are exactly0 because disabled, not the old logging bug. Not fine-tuning old checkpoints.

Resume uses same RUN_TAG/source/config, last.pth, optimizer/RNG/counters restored at
epoch boundaries. A partial interrupted epoch repeats. Never upload helper .py
inside CODE. Source/config changes require new run; strict hashes are not bypassed.
Per-epoch logs include input U-weight magnitude, cue/raw-U statistics and average
absolute FIRST-PROJECTION contribution (not a gate-error/accuracy guarantee).

## Benchmark and comparisons

Identical successful FP32 TensorRT10 strong typing/TF32 OFF/non-default streams,
ONNX20→22 export. Untrained TRT graph smoke BEFORE training. After train: native
BF16/FP32 vs actual TRTgraph full400; numerical gates mean1mm/max6cm, global native
drift RMSE2mm/iRMSE.02km⁻¹, exact25,424,992GT support. No silent fallback/relaxation.
6cm diagnostic protocol is not the original5cm gate.

Timing B1 352×1216, warmup100/iterations300, same first100val scenes. PyTorch
eager/graph and TRT eager/graph, synchronized-wall median/P95 plus GPUstream/host.
Includes D2D; excludes I/O/decode/preprocessing/H2D/metrics/build/capture. Memory is
not isolated per-model VRAM and omits directTRT allocations. Not edge-device latency.

Outputs: full400 global/inverse/range/RGB-edge/tail/boundary metrics;1000test uint16
PNG predictions (/256meters) +10 fixed0..120m RGB/TURBO pairs; comparison.csv/json
A/B/C/D/E/G+MSG,10 stacked same-scene previews if old images exist.

Actual historical Drive reports/checkpoints verify split/metric-teacher data,
checkpoint SHA and initial shared weights (first40 projection columns). Only
relative cache metadata is excluded from cross-run data checks by explicit design.
Old A-E use relative KD, this run does not: `matched_training_verified=false` for
those comparisons; `shared_data_init_verified` is checked separately.
A-D30 match epoch budget; E26/MSG do not. Old timings
are historical NOT same-session; environment differences flagged. Missing old
reports use explicitly unverified numeric snapshots, not invented evidence.

`acceptance.json`: RMSE/iRMSE not worse than D, median≤E **3.815386ms**. This is a goal,
not guaranteed improvement. All valid benchmark numbers still exported if the
new model misses these goals. Small wins require same-session D and multiple seeds.

## Files / outputs

`GeoLift_RT_Runs/S4_UPIR_NoRelative_Fresh30_<RUN_TAG>/G_s4_upir_norel/`: last/best_policy/best_rmse/
best_inverse checkpoints, train_log.csv, val reports, trt_benchmark.json, engine,
test predictions/previews. Parent run has comparison and acceptance outputs.
The unchanged dual-metric best_policy rule selects ONE checkpoint; never combine
best-RMSE with best-iRMSE numbers from different epochs.

Local verification is CPU only: model/backward,4calls/no fine, zero-new-weight
functional identity, same-env init, U gradients and routing, FP32 ONNX checking,
logging/objective identity and comparison fixtures. No A100 training/parity,
real-data accuracy or latency improvement claimed before you run the notebook.
