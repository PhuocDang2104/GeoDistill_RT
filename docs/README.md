# Documentation

## Current

- [V11-derived S4-U-PIR NoRelative](AnchorFlow_V11_Baseline.md): equations, best_policy
  epoch27, metrics, attribution limits, TensorRT/CUDA Graph.
- [Completed G run](../benchmarks/s4_upir_norel_2026_10_19/README.md): original logs,
  CSV, comparison and unchanged failed latency gate.
- [Orin NX / AGX inference](../production/AnchorFlow_S4_UPIR_Orin/README.md): required
  files, export, native GPU TensorRT build and runtime/parity audit.
- [Train folder](../drive_upload/AnchorFlow_S4_UPIR_NoRelative_30ep_A100/README.md):
  self-contained fresh30/resume, GeoLift_Data, metric teacher only.

## Historical, not current solver/metric contracts

- [Fixed6 RK3 coarse6 + fineRK2](history/AnchorFlow_V11_Fixed6_2026_10_14.md).
- [V11.3 adaptive fine-tune](AnchorFlow_V11_3.md), [V11 original](AnchorFlow_V11.md).
- [S6 A100 inference audit](AnchorFlow_S6_A100_Inference.md).
- [FullKITTI production](AnchorFlow_V11_3_Production.md),
  [Fixed6 server](AnchorFlow_V11_3_Fixed6_Production.md).

Historical dependencies/evidence may be on external D via junctions. Keep D attached;
do not interpret a disconnected archive as deleted source. [Cleanup/restore](../REPO_CLEANUP.md).
