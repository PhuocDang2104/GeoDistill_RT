# Current baseline — Q1_s4_n1_fast

Adopted21/10/2026 at the user's request: **582,888 parameters, coarse4NFE/fine0**.
S4 JetODE + U-PIR, N1 low-resolution trims, D4Fusion512, Jet-context128.
The neural architecture is unchanged versus N1; fastpath is deployment engineering.

| Full400 development validation · reported A100/TRT FP32+Graph | Q1 |
|---|---:|
| RMSE m / iRMSE km⁻¹ |0.9786909355 /3.1708796716|
| MAE m / boundary3 RMSE m |0.2558258921 /1.6671664042|
| Wall mean / median / P95 ms |3.846668 /3.843643 /3.861209|
| FPS=1000/mean |259.9652|
| Reverse-order repeat mean ms |3.845360|

All three requested point-estimate thresholds pass without rounding. SameQ1
weights, original→fast mean4.229391→3.846668ms (about9.05% lower wall); quality
is effectively unchanged. Different fresh-training weights explain the comparison
axis versus oldN1; the deployment rewrite is not shown to improve accuracy.

- [Complete mathematical specification and limitations](AnchorFlow_Q1_Theory.md).
- [Raw log, attribution and evidence level](../benchmarks/q1_n1fast_2026_10_21/README.md).
- [Production artifact checklist](../production/AnchorFlow_S4_Q1_Fast/README.md).
- [Frozen train/inference notebook](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/README.md).
- [Previous G baseline, historical](history/AnchorFlow_S4_G_NoRelative_2026_10_19.md).

This promotion is a **subset/development baseline**. Full-KITTI training,
independent-sequence/seed validation and Orin latency are not yet supplied.
The current evidence is a user console snapshot, not local access to the actual
checkpoint/full GPU report. Private weights/data/exports do not belong in Git.
Neither the completed snapshot nor its historical candidate-only report is edited
to retroactively claim full-KITTI qualification. Selection is recorded separately.
