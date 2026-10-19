# AnchorFlow S4-U-PIR NoRelative — Orin GPU inference

Current deployment baseline: `G_s4_upir_norel`, **583,880 parameters, coarse4 NFE,
fine0**. Inference only. Supports the TensorRT10 GPU API used by JetPack6.x on
Orin NX / AGX Orin. **Not yet tested on physical Orin.** No claimed Orin latency.

## 1. Required files

On Orin, only this folder plus an exported `assets/` directory:

```text
benchmark.py, cuda_runtime.py
assets/
  model_fp32_op17.onnx       # SAME trained best_policy, FP32; new portable export
  io_manifest.json          # checkpoint/source hashes, IO contract, sample hashes
  samples/0000.npz ...      # RGB/S/M/K + PyTorch FP32 reference + GT/mask
```

**No checkpoint, teachers, full KITTI, trainer, timm or PyTorch needed on Orin.**
Use NumPy and the system TensorRT Python binding supplied by JetPack. Do not
install the A100 `requirements-trt.txt` or replace Jetson CUDA via pip.
100 real frames are default; export400 to also repeat the full validation audit.
Two frames are supported only for quick A/B/A smoke, not the 100-image protocol.

Weights/data are private local/Drive assets, **not included in Git or this ZIP**.
Actual checkpoint SHA is recorded during export; pasted metrics are not a substitute
for checkpoint verification. Never copy the A100 `.plan`: rebuild on each device/
TensorRT version. The original A100 opset22 graph is left untouched.

## 2. Export once, from the completed Colab runtime

Upload this helper folder OUTSIDE `/content/anchorflow_s4_upir_norel_code`.
Keep the original runtime config, frozen CODE, prepared local cache/data contract
and selected checkpoint. Example Colab cell (adjust helper path if needed):

```python
from pathlib import Path
import subprocess, sys
PACKAGE = Path('/content/drive/MyDrive/AnchorFlow_S4_UPIR_Orin')
RUN = Path('/content/drive/MyDrive/GeoLift_RT_Runs/S4_UPIR_NoRelative_Fresh30_pair01')
OUT = RUN/'orin_assets_fp32_400_v1'  # NEW directory; export never overwrites
subprocess.run([sys.executable, str(PACKAGE/'export_assets.py'),
    '--config', str(RUN/'G_s4_upir_norel/resolved_config.json'),
    '--output', str(OUT), '--frames', '400', '--audit-upir'], check=True)
```

Selected file is `G_s4_upir_norel/best_policy.pth` (**epoch27**, zero-based), NOT
`last.pth`. The helper retains the original strict source/config/data check.
If Colab SSD cache was lost, rerun the ORIGINAL notebook setup/prepare cells only;
do not rerun training or inject helpers into frozen CODE. A hash mismatch must be
investigated, never bypassed. ONNX17 is freshly exported and full-schema-checked;
numerical qualification happens on the target, not assumed from A100 opset22.

`--audit-upir` also evaluates full400 twice: original versus a COPY with the added
U-input weights set to0. `upir_intervention.json` measures learned reliance, **not**
the effect of retraining no-U-PIR. The real original checkpoint is not modified.

## 3. Copy to Orin and run

Install JetPack6.x / TensorRT10 Python bindings through NVIDIA's system packages.
For example on an existing correctly provisioned JetPack device:

```bash
sudo apt-get install python3-numpy python3-libnvinfer
python3 -c 'import tensorrt,numpy; print(tensorrt.__version__,numpy.__version__)'
# Copy the exported directory as ./assets (scp/SSD/rsync; no dataset regeneration).
python3 -m unittest discover -s . -p test_portable.py -v
python3 benchmark.py --assets ./assets --output ./runs/orin_fp32_01 \
  --workspace-gib 1 --timed-images 100 --warmup 100 --iterations 300 \
  --power-note 'Orin NX or AGX; RAM=?; nvpmodel=?; clocks=?; active cooling=?'
```

Use a fresh output name per run. Query `sudo nvpmodel -q`, record board/RAM,
power limits, clocks and temperatures (`tegrastats`). Select supported power mode
with the device owner; **no automatic overclock or power-mode change**. Log steady
thermal conditions. More CPU/GPU activity or different power profiles invalidate
matched-device speed comparisons. GPU only, no DLA or INT8/FP16 substitution.

100 resident FP32 inputs occupy about **816 MiB**, separate from model workspace,
engine/context and OS. If memory is insufficient, explicitly reduce
`--timed-images` and label the result as a different protocol. Workspace is a
builder limit, **not** a total-memory cap. No silent fallback on OOM/parity error.

## 4. What is measured

Native FP32 engine built ON Orin, strongly typed, TF32 OFF, fixed B1×352×1216.
Non-default CUDA stream; warm enqueue then CUDA graph capture with stable device
addresses. A/B/A and every exported frame must pass mean≤1mm / max≤6cm versus
same-checkpoint PyTorch FP32 reference; graph also checked against TRT eager.
Global RMSE/iRMSE drift limits are 0.002m / 0.02km⁻¹ versus FP32 references.
**These are qualification tolerances, not a claim of bitwise identity.**

`report.json` includes eager and graph stream/host/synchronized-wall timing,
median/P95, FPS, artifact hashes, target environment, qualification and errors.
D2D input staging is included; disk/decode/preprocessing/H2D/D2H/metrics/build/
capture excluded. CUDA event timing spans launch gaps; host enqueue is NOT FPS.
This is the same timing *scope* as A100, with a different native runner/device;
do not call it end-to-end or compare two devices as a CUDA-graph speedup.
Only400 exported frames reproduce the full400 support25,424,992;100-frame metrics
are labeled separately. Boundary/range diagnostics remain in the original A100
benchmark; this portable runner reports global RMSE/iRMSE/MAE only.

Eval-mode fix19/10/2026: exporter explicitly sets the wrapper and all BN children
to eval BEFORE/AFTER legacy ONNX export. Discard references/audits produced by
the older unfixed helper (original RMSE≈1.2246). Corrected user-reported full400
audit: original0.990151/3.172073/0.258984 versus U-input-zero
0.990889/3.191544/0.261689 (RMSE/iRMSE/MAE). This shows same-checkpoint reliance,
not retrained architectural superiority. Actual Orin parity remains required.

CPU tests/schema export are local checks; actual TensorRT parser/capture/performance
must pass on the device. Unsupported ops/errors stop and save an unqualified report.

References: [JetPack6.2 (CUDA12.6/TRT10.3)](https://developer.nvidia.com/embedded/jetpack-sdk-62),
[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.3.0/inference-library/python-api-docs.html),
[CUDA runtime graphs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html).
