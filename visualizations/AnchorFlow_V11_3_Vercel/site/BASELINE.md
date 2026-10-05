# V11_3 — selected internal research baseline

Checkpoint **best_policy**, fine-tune stage epoch **9**, cumulative epoch index **32**. Best-RMSE, best-iRMSE, best-joint and best-policy all contain the **same trained state**. Last is stage14, not the selected accuracy baseline.

| Full400 validation | Parent V11_2 epoch22 | V11_3 selected |
|---|---:|---:|
| RMSE m | 0.999491 | **0.986517** |
| iRMSE km⁻¹ | 3.207087 | **3.169735** |
| MAE m | 0.259726 | **0.255663** |

25,424,992 valid GT pixels; global accumulation, not average image RMSE. Target iRMSE<3.2 reached; **RMSE<0.98 not reached**. Anonymous KITTI1000 has no public GT. This is not a leaderboard claim.

```text
RGB pretrained encoder + compact sparse pyramid
→ gated fusion → D16 → D8 → D0
→ coarse bounded-chart Jet NODE / Bosh3 T=1 → D4
→ phase lift → five-jet inverse consensus → metric innovation
→ D2 fine chart NODE / one fixed midpoint / exactly2NFE
→ phase lift → detail → phase-aware sparse innovation (PIR)
→ D1 → learned sensor reliability fusion → Dfull
```

584,845 parameters. CNN BF16; RHS/chart/solver/loss FP32. Student forward uses RGB/S/M/K, never GT/teacher. V11_3 completed **15 extra epochs** from V11_2 epoch22 with fresh optimizer, not fresh student. Metric and relative teachers train-only; no cached val teachers.

| Same-grid transition, full400 native GT | RMSE m before → after |
|---|---|
| Coarse D0→D4 | 1.85103 → 1.67155 |
| D2_base→query→metric readout | 1.33821 → 1.27047 → 1.22663 |
| Fine D2 NODE | 1.22663 → 1.20692 |
| D1_base→detail→PIR | 1.03655 → 1.00202 → 0.99932 |
| D1→soft sensor fusion | 0.99932 → 0.98652 |

PIR-disabled counterfactual with the **same weights** gives RMSE0.987965 / iRMSE3.173899. Its gain is small and some tails worsen; not a matched retraining ablation. Hard-anchor diagnostic RMSE1.127354 is much worse than soft fusion.

Only **0.1757%** valid pixels with |error|>10m contribute **57.29%** of metric SSE. V11_3 improves near/mid ranges but 60–120m is slightly worse than parent. Relative weighted loss ~0.099% of total is not evidence of causality or uselessness. Focus next on rare tail, boundary/mixed-surface query and donor/sensor conflict rather than blindly increasing residual amplitude.

A100-SXM4-40GB, native BF16/FP32 geometry, B1 real100: adaptive **75.18 /79.17ms median/P95**; static coarse midpoint8 + fine2 **61.51 /62.62ms**. Full400 static accuracy is absent from the supplied ZIP. Full400 adaptive coarse NFE distribution:7×113 scenes /10×287; mean9.1525, plus exactly2 fine calls. No rejection; T=1.

Source archive: `dual_teacher-20261005T071424Z-1-001.zip`.

```text
Checkpoint SHA256:
c0914220e2c900eebf8fe918fdba21fde1faf2dbb3c73608d2a3efecaf7a5aad
Frozen trained source SHA256:
34d025bd12ca62603d52ac9e86d8f38572c62b038491b58466f389b0302f7b74
```

[pipeline_evidence.json](pipeline_evidence.json) embeds completed BF16 baseline numbers plus exact-model **10-scene CPU FP32** replay proofs. Viewer scene metrics are not re-evaluation of all400 and not anonymous test GT. Depth timelines retain all pixels; supplemental six-coefficient 3D traces retain all pixels with synchronized timelines and exact native-cell footprints. Fine playback uses an order-2 RK2 continuous extension, not an exact ODE solution or extra RHS calls. Full technical documentation is `docs/AnchorFlow_V11_3.md` in the repo. No model/loss/solver/config changed during this promotion or visualization work.
