# G S4-U-PIR NoRelative — completed A100 evidence

Current deployment baseline by user decision,19/10/2026. **Not an isolated proof
of U-PIR improvement** or a new fullKITTI-trained model. Training snapshot is frozen.

## Evidence and provenance

- `train_log.csv`: original30-epoch CSV supplied from Downloads, copied unchanged.
- `comparison_user_log.txt`, `benchmark_user_log.txt`: original pasted outputs,
  copied unchanged (including warnings and failed target).
- `comparison.json`, `benchmark_snapshot.json`, `summary.json`: mechanically parsed
  snapshots, exact numeric values and original evidence SHA256s. Regenerate with
  `python scripts/summarize_s4_upir_evidence.py` from repository root.

No actual checkpoint/engine/full parity JSON was attached in this request; **do not
invent their hashes or claim independent reproduction**. Orin exporter must read
the actual strict-matching checkpoint and records its hash. Historical comparison
reports verified shared init/data but G differs in relative treatment; rows A–E
must not be called fully matched-training controls of G.

## Decision

G epoch27: RMSE0.990150559m, iRMSE3.172073529km⁻¹, MAE0.258984267m,
boundary3 RMSE1.685667680m;583880params. A100-SXM4-40GB, full400validation /
25424992GTpixels; FP32 TRT10.13.3+CUDA Graph, wall median/P953.826771/3.844566ms,
261.317FPS. Warmup100/iterations300,100distinctimages, D2D included; file IO,
decode/preprocess/H2D/metrics/build/capture excluded. Torch process memory908MiB
is NOT isolated model memory and excludes direct TRT allocations.

Versus D no-fine: RMSE−2.721mm/0.274%, iRMSE−0.520%, boundary−0.762mm;
MAE **+0.071mm**; wall−0.003ms (no meaningful speedup established).
Versus B: RMSE−0.752mm and latency−5.03%, but B has better MAE/boundary.
Versus A: latency−20.99%; A retains best boundary among these runs.
MSG-CHN fullKITTI pretrained uses different training/unaudited overlap, not an
architecture-only accuracy comparison.

**Unchanged old acceptance:** D RMSE/iRMSE targets pass; latency≤3.815386ms fails
(G+0.298%); `all_targets_met=false`. RMSE<0.98 still not achieved. Choosing G as
baseline does not change numerical tolerances or retroactively pass that target.

## What the learning curve says

30epochs, first iRMSE≤3.2 at epoch24; best native RMSE at epoch27:
0.990139440m /3.171911570km⁻¹. Native-versus-TRT drift≈0.011mm /0.000162km⁻¹.
Epoch29 RMSE0.990431966, so deploy best_policy rather than last.
Train-only epoch sums2142.6s; total epoch sums3004.0s; no training was run locally.
Relative raw/weighted terms are0 because deliberately disabled; metric KD remains.

At epoch27 mean|new U weights|≈0.505, normalized cue mean≈0.203, first-projection
absolute contribution≈0.10243. **This statistic is not a depth improvement,
gate probability or fraction of explained error.** It measures input-projection
activity, not whether that activity is useful. Rare |error|>10m pixels are0.178%
of support but57.12% of SSE; reducing outliers remains important for RMSE.

## Is U-PIR useful?

The **combined package** improves RMSE/iRMSE slightly at no-fine-like runtime.
Separate U-PIR contribution is **unidentified**: D→G also removes relative KD;
E differs in architecture, loss treatment and epoch budget. No completed F
dual-teacher UPIR result was supplied. One seed and repeated selection on400val
do not establish significance/generalization. Do not infer “relative KD hurts.”

Next minimum evidence: train **D-NoRelative** fresh30 with the same shared seed/
schedule/losses, compare with G. Before retraining, run `export_assets.py
--audit-upir`: full400 original vs same-checkpoint zeroed U-column, preserving
weights/source and reporting output change plus global/range/boundary metrics.
That intervention tests reliance, not the retrained architectural effect.

[Current architecture/math](../../docs/AnchorFlow_V11_Baseline.md) ·
[Orin export/benchmark](../../production/AnchorFlow_S4_UPIR_Orin/README.md).
