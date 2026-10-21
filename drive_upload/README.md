# Active Drive package: Q1_s4_n1_fast

Upload the WHOLE [N1-Fast/Q1 frozen folder](AnchorFlow_S4_UPIR_N1Fast_30ep_A100/README.md)
to MyDrive and open `Train_N1Fast_Fresh30_A100.ipynb`. The README/notebook inside
are original completed experimental bytes; [current adoption is recorded separately](../production/AnchorFlow_S4_Q1_Fast/README.md).
Do not edit their sources/config/manifest to change the historical status text.

582888params,coarse4NFE/fine0,seed42,B4,fresh30/resume,ImageNet,metric teacher only,
relative OFF. Same400 TensorRT FP32+CUDA Graph accuracy/runtime protocol. Keep the
oldN1 Drive run for preflight/reference measurement; local archived control folders
are not required by this completed, self-contained snapshot.

```bash
python scripts/package_current_baseline.py
```

This command verifies/repackages Q1 without any retired parent generator and
without changing frozen bytes. Dataset/checkpoint/ONNX/engines are not in Git/ZIP.
[GeoLift_Data checklist](../docs/GeoLift_Data_MetricOnly_Transfer.md).

## Retained historical compatibility

- [G NoRelative folder](AnchorFlow_S4_UPIR_NoRelative_30ep_A100/README.md) remains
  immutable for historical reproduction and the old G Orin exporter.
- V11/V11.3 and prior device-audit folders are legacy controls, not current baseline.
- DecoderCapacity/1M/D4Split/Edge600/N1-parent/PhaseFusion source folders, ZIPs and
  generators were cold-stored, not irreversibly erased. [Archive receipt/recovery](../benchmarks/q1_n1fast_2026_10_21/CLEANUP.md).

Only [Q1](../docs/AnchorFlow_V11_Baseline.md) is the current adopted development
baseline. Full-KITTI and Q1 Orin qualification are pending.
