# S4-U-PIR — decoder-capacity ablation · A100 · fresh30 ×3

Upload **WHOLE folder** to:
`MyDrive/AnchorFlow_S4_UPIR_DecoderCapacity_30ep_A100`.
Open **`Train_S4_UPIR_DecoderCapacity_Fresh30_A100.ipynb`**, select **A100**, Run all.
No old bundle/source dependency. Uses existing **GeoLift_Data**. No new results
are claimed before training and target benchmark run.

| Variant | Changes versus G0 | Params | Extra conv MACs B1×352×1216 |
|---|---|---:|---:|
| G0_s4_upir | Current G architecture, fresh matched control |583,880|0|
| H_s4_dec16_8 | Hidden expansion2→4 in D16 fusion refine and D8 top-down refine |641,288|181.81M|
| I_s4_dec16_8_4 | H + hidden expansion2→4 in D4 top-down refine |651,752|451.47M|

**No extra layer, module stage, NFE, fine NODE, wider output feature or decoder
depth bound.** Changes are inside existing inverted bottlenecks. U-PIR/readout,
encoder, sparse branch and coarse S4 unchanged. Details/rationale: [ANALYSIS.md](ANALYSIS.md).

## Training contract

- Same1,600train/400val, seed42, fresh30, B4, ImageNet encoder only; NOT fine-tuning G.
- Eager native BF16 CNN + FP32 geometry/loss. **TensorRT is inference, not training.**
- Metric teacher ON, same KD schedule/loss weights. **Relative teacher OFF**:
  no relative module, pair calculation, archive/cache loading or required DA3 files.
- Same LR3e−4 / encoder×0.5 / AdamW / warmup1+cosine / holdout/flip; TF32 OFF.
- First construct the entire original G, then widen. Retain all historical parameter
  slices and BN stats; extra hidden lanes have zero outgoing projection. Old
  function is preserved in exact arithmetic, subject to normal kernel roundoff.
- Shared-slice SHA checks do **not** claim every wider-model parameter is identical.
  New lanes are genuinely trainable and tested to learn; train RNG/order reset after construction.

All30epochs, **no early stop**: old G selected epoch27, so a26epoch cutoff can miss
its best. Same notebook/RUN_TAG resumes each `last.pth` with optimizer/RNG/loader
state at epoch boundaries. A completed arm does no further training on rerun.
Checkpoint source/config/data gates are NOT bypassed. Do not change CODE to resume.

## Data and quick start

Required `GeoLift_Data/teacher_subset_2000/`:

```text
selected_2000_ids.json
kitti_trainval_2000.tar
metric_coarse_train_2000.tar
```

Test: `GeoLift_Data/test_1000/kitti_test_1000.tar`, or original canonical download.
No relative teacher archive needed. Notebook can reuse the completed G metric-only
SSD cache; full data gates still verify the split/teacher content. First extraction
requires >30GiB free Colab SSD; cached run >4GiB. Sources are snapshotted into
`RUN/frozen_bundle` to recover after a notebook/runtime reset.

Run order: setup/contracts → prepare + **real backward/TRT build-capture smoke**
for all3 → train30 each → full benchmark each → compare. Stable eager training
matches G; there is no silent train-graph fallback or unmeasured promised speedup.
Historical G took about50min total epochs;3arms are roughly2.5h+ as a planning
reference, **not measured timing for the wider models**. Smoke/build/benchmark add time.

## Accuracy/runtime benchmark

Select each arm's **best_policy** after30epochs, same policy:
iRMSE≤3.2 then minimize RMSE; otherwise normalized dual-target ordering.
Do not mix best_inverse/last/GT test-oracle selection.

TensorRT10.13.3 **explicit FP32, TF32 OFF + CUDA Graph**:

- Metrics from the ACTUAL TRT graph predictions on full400 /25,424,992GTpixels.
- RMSE/iRMSE/MAE, GT-boundary bands, depth ranges, error-tail SSE and per-image SSE.
- Same A/B/A checks; pixel mean≤1mm / max≤6cm; native-BF16 metric drift limits
  RMSE≤2mm / iRMSE≤0.02km⁻¹. Fixed gates; no auto-relaxation/fallback.
- B1×352×1216,100distinctreal GPU-resident images; warmup100/iterations300;
  wall median/P95 and FPS, with D2D staging included. Disk/decode/preprocess/H2D,
  metric computation, engine build/capture excluded. **Not end-to-end or Orin time.**
- Non-default TRT work stream, stable device addresses; source/ckpt/ONNX/engine hashes.
- 1000test depth PNG predictions (no public GT),10 RGB/depth pairs per arm,
  full-resolution RGB/G0/H/I comparison previews.

Primary comparison is **new G0 versus H/I on the same A100 session**. Old G's
0.990151m /3.172074km⁻¹ /3.826771ms is stored as historical context, not relabeled
as a same-session measurement. Comparison verifies all3 training contracts and
retained initialization, software/device name, precision and timing recipe.

Predeclared decision report: both RMSE/iRMSE not worse than G0, latency≤1.05×G0.
All measured rows remain visible even if targets fail; no automatic baseline replacement.
Paired image-bootstrap intervals are diagnostic only (sequence dependence and
repeated validation selection mean they are NOT independent significance proof).

## Output / recovery

`MyDrive/GeoLift_RT_Runs/S4_UPIR_DecoderCapacity_Fresh30_pair01/`:

```text
runtime_config.json, data_contract.json, frozen_bundle/
G0_s4_upir/   H_s4_dec16_8/   I_s4_dec16_8_4/
  last.pth, best_policy.pth, best_rmse.pth, best_inverse.pth
  train_log.csv, best_policy_val_metrics.json, training_status.json
  trt_benchmark.json, replay_parity.json, deployment.onnx, deployment.plan
  kitti_test_predictions.zip, test_preview/
comparison.csv/json, paired_rmse_bootstrap.json, comparison_previews/
```

Train CSV now logs D16/D8/D0/D4/D2/D1/final native stage RMSE, so we can see
whether added decoder capacity improves the coarse seed or only downstream fits.
Do not compare values across differently pooled resolutions as if the GT targets
were identical. All inference comparisons use full-resolution common support.

The ONNX helper explicitly preserves **eval mode before/after export**, with a
regression test checking repeated predictions/BN buffers; the previous U-PIR
audit mode bug is not carried forward. Current frozen G/baseline files unchanged.

Only CPU structural/backward/export tests run locally. A100 train/TRT runtime
must be executed by this notebook. Send `comparison.csv`, `trt_benchmark.json`
and the train CSVs after completion.
