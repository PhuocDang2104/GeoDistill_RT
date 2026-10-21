# Adopted baseline: Q1_s4_n1_fast

**582,888 parameters, S4 coarse4NFE/fine0, Fusion512, Jet-context128, U-PIR.**
Same neural architecture as N1; new trained Q1 weights plus inference fastpath.
Current subset/development baseline, explicitly adopted21/10/2026. Not yet a
full-KITTI-trained or measured Orin release.

[Specification](../../docs/AnchorFlow_V11_Baseline.md) ·
[Theory](../../docs/AnchorFlow_Q1_Theory.md) · [Baseline JSON](baseline.json) ·
[Evidence](../../benchmarks/q1_n1fast_2026_10_21/README.md).

## What to retain to run the actual model

The code in Git contains NO trained checkpoint. Keep the original Drive run:

```text
GeoLift_RT_Runs/S4_UPIR_N1Fast_Fresh30_pair01/
  frozen_bundle/                 # exact completed source + manifest
  runtime_config.json
  data_contract.json
  history_provenance.json
  benchmark_session.json
  comparison.csv, comparison.json, baseline_candidate.json
  Q1_s4_n1_fast/
    best_policy.pth              # selected weights; NOT inferred from logs
    last.pth                     # optimizer/RNG/loader state for epoch resume
    training_status.json, train_log.csv
    trt_benchmark.json, replay_parity.json
    original.onnx, fast.onnx
    original.plan, fast.plan     # only valid on matching target/software
    original.layers.json, fast.layers.json
    kitti_test_predictions.zip, test_preview/
```

For reference-checkpoint remeasurement/preflight keep old N1's run/source on
Drive too. Local cold-storage of experimental folders does NOT delete Drive runs.
Checkpoint/source/ONNX/engine SHA should match the full actual report. The console
does not supply them, so baseline.json records unknown hashes/selected epoch as
null instead of inventing values. Do not upload private weights/KITTI/teachers to Git.

## Reproduce train/benchmark

Upload [the complete frozen package](../../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/README.md)
and open `Train_N1Fast_Fresh30_A100.ipynb`. Its historical README says candidate-only:
that text is deliberately frozen; current adoption is this separate document.
Same RUN_TAG resumes exact source and epoch state; never patch a completed bundle.

Seed42,B4,30epochs,no early stop,ImageNet encoder,metric teacher ON/relative OFF.
Train uses PyTorch autograd BF16CNN/FP32geometry+loss, AdamW fused. TensorRT is
inference-only. Packaging `python scripts/package_current_baseline.py` verifies
and ZIPs existing bytes without parent folders/source regeneration.

For final measurement rebench oldN1 AND Q1 in a fresh same session, each with
original/fast engines, full400 actual predictions and fixed parity gates.
100warmup/300timediterations/100residentimages; FPS=1000/wallmean. Do not replace
wall time with host-enqueue time or compare stale sessions as controlled speedups.

## Orin and full KITTI are next qualifications, not existing results

The [legacy G Orin folder](../AnchorFlow_S4_UPIR_Orin/README.md) is NOT a Q1 exporter:
it pins G583880 architecture/contracts. Renaming Q1 to G or bypassing its loader
would invalidate the result. A Q1 export/JetPack-compatible ONNX/native builder
must retain Q1 fastpath and pass parity against its actual checkpoint before
claiming target runtime. Never copy the A100 `.plan` to Orin or assume3.85ms there.

FullKITTI requires a new registered train/val recipe and locked independent split;
current1600/400 data is subset development evidence. Keep the Q1 checkpoint as
the reference; do not relabel its existing metric as a full-KITTI result.
