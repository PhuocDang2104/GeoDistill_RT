# Documentation

## Current: Q1_s4_n1_fast

- [Canonical baseline](AnchorFlow_V11_Baseline.md): reported RMSE0.978691,
  iRMSE3.170880,wallmean3.846668ms,582888params,coarse4NFE/fine0.
- [Code-matched theory](AnchorFlow_Q1_Theory.md): bounded local jet/chart,
  midpoint solver,U-PIR,cardinal/stencil/moment equivalence and claim limits.
- [Evidence and attribution](../benchmarks/q1_n1fast_2026_10_21/README.md): old/fresh
  ×original/fast same-session table, raw console/hash, missing private artifacts.
- [Private deployment artifacts](../production/AnchorFlow_S4_Q1_Fast/README.md).
- [Frozen train/benchmark folder](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/README.md).
- [Metric-only GeoLift_Data checklist](GeoLift_Data_MetricOnly_Transfer.md).
- [Cleanup and recovery](../benchmarks/q1_n1fast_2026_10_21/CLEANUP.md).

## Historical — not current Q1 solver/architecture contracts

- [G NoRelative baseline](history/AnchorFlow_S4_G_NoRelative_2026_10_19.md).
- [G completed evidence](../benchmarks/s4_upir_norel_2026_10_19/README.md).
- [Legacy G Orin helper](../production/AnchorFlow_S4_UPIR_Orin/README.md): not Q1-compatible.
- [Fixed6 RK3+fine2](history/AnchorFlow_V11_Fixed6_2026_10_14.md).
- [V11.3 adaptive](AnchorFlow_V11_3.md),[V11 original](AnchorFlow_V11.md).
- [S6 A100 audit](AnchorFlow_S6_A100_Inference.md).
- [Old fullKITTI production](AnchorFlow_V11_3_Production.md),
  [Fixed6 server](AnchorFlow_V11_3_Fixed6_Production.md).

Historical D junctions can be disconnected; that is not proof their data is deleted
or unnecessary. Q1's self-contained frozen bundle does not depend on those paths.
