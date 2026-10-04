# V10.1 / V11 — completed paired benchmark

Canonical explanation: [AnchorFlow V11](../../../docs/AnchorFlow_V11.md). Small evidence only; no checkpoint, data or 1.000-image prediction ZIP is published here.

## Run identity

- Original archive: `Compare_V10_1_V11_Fresh40_bf16_pair01-20261004T151154Z-1-001.zip` (kept local/Drive).
- Archive SHA256: `a12a283f15a45f293c999642d2e9a46dbe038ec9eff0780cb07b7da14c1da470`.
- Source/split/cache/checkpoint provenance verified by [audit script](../../../scripts/audit_anchorflow_pair_completed.py). Frozen source bytes match the local V10.1 and V11 releases.
- Train1600 / val400 / anonymous1000; valid validation GT pixels **25,424,992**; ImageNet RGB initialization, fresh student, B4, seed42.
- Both models: **34 epochs / 13,600 optimizer updates**, early-stopped. Training shared one GPU; epoch duration is not isolated model speed.
- Isolated final evaluation/profile: **A100-SXM4-80GB**, Torch2.11+cu130, BF16 CNN + FP32 geometry, B1 352×1216. real100 wall profile excludes disk I/O/H2D.

| Model / solver | Epoch | RMSE m | iRMSE km⁻¹ | Wall median / P95 ms |
|---|---:|---:|---:|---:|
| V10.1 primary | 25 | 0.993694 | 3.180862 | 35.115 / 38.799 |
| V11 primary | 29 | 0.986021 | 3.215515 | 74.146 / 76.469 |
| V11 static midpoint8, same checkpoint | 29 | 0.985972 | 3.215281 | 50.327 / 51.805 |

## Evidence layout

`artifacts/` contains exact exported CSV/JSON bytes from the original archive, including both training logs, selected-checkpoint evaluation, stage/range/edge/tail/sensor metrics, profile and solver audit. Original [comparison dashboard](artifacts/comparison_dashboard.png) is retained. [artifact_manifest.json](artifact_manifest.json) lists their SHA256 values. [audit_summary.json](audit_summary.json) contains independently calculated deltas, loss shares and audit checks.

See [accuracy](artifacts/comparison_accuracy.csv), [checkpoint selections](artifacts/comparison_checkpoint_selections.csv), [efficiency](artifacts/comparison_efficiency.csv), [solver audit](artifacts/comparison_v11_solvers.csv) and [training budget](artifacts/comparison_training_budget.csv).

**Interpretation:** single seed, same validation used for checkpoint selection; not an independent test result or solver-only ablation. Native-stage supports differ across resolutions. V11 static and adaptive global scores being very close does not establish pixel-wise parity. Anonymous1000 predictions have no public test GT.

To reproduce the audit with the original local ZIP:

```bash
python scripts/audit_anchorflow_pair_completed.py
```

## Publication checks — 04/10/2026

Archive audit passed; 274 frozen bundle files and 36 evidence artifacts also match their SHA256 **inside the Git index**, so clone/push will not silently normalize their bytes. No weights, dataset, prediction ZIP or credentials are staged.

Local Windows CPU checks: V11 **32**, V10.1 **30**, V9.1 **31** contract tests passed; root `pytest tests --ignore=tests/test_server_process_recovery.py -q`: **63 passed**. POSIX signal/process-recovery tests were not run on Windows. These checks do not replace the archived A100 measurements or constitute a new GPU training run.
