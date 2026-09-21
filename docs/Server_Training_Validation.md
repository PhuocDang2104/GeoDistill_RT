# Server runtime validation — 2026-09-19

Published private image: `phatle0106/geodistill-train:server-v1`.
Exact digest and source identity are recorded in `docker/release.json`.
Docker Hub privacy and authenticated pull by the final digest were verified.

## Passed

- Linux repository suite: **44 tests passed**. Packaged recovery tests also passed.
- Mid-epoch CPU resume produced exactly equal model weights, optimizer state and
  scheduler state to uninterrupted training, using two workers and A3 augmentation.
- Separate Linux processes recovered from SIGTERM and SIGKILL and completed the
  remaining epochs without duplicate committed epoch-log rows.
- Corrupt checkpoint detection, archive traversal/link rejection, rclone local
  transport, verified bundle restore, and re-backup of restored best checkpoints.
- Full-resolution synthetic late-KD forward/backward/optimizer execution on the
  RTX 4060 Laptop 8 GB: batch 2, FP16 AMP, approximately 10.08 images/s and
  991,952,896 bytes peak reserved CUDA memory.
- All four supplied Drive assets downloaded and SHA256 verified. All extracted
  metric and geometry teacher files were decoded and checked.
- **All 1,600 train and 400 validation samples passed** the full local preflight:
  frozen IDs, no train/validation raw-drive overlap, required files, calibrated
  intrinsics, finite tensors, valid ground truth and teacher pixel coverage.
- Compose validation, source compilation and whitespace checks passed.

## Real-data GPU calibration

The packaged runtime prepared an 8-train/4-validation subset of actual TAR2000
samples into a native Linux Docker volume. It validated that subset and calibrated
the full late teacher objective, retaining the 352×1216 resolution and batch 2.

| Workers | Images/s | Peak reserved VRAM | Full-objective loss |
|---:|---:|---:|---:|
| 0 | 6.22 | 991,952,896 bytes | 19.8942 |
| 2 | **9.49** | 991,952,896 bytes | 19.8948 |
| 4 | 9.32 | 991,952,896 bytes | 19.8944 |

The runtime selected two workers. Each trial stabilized after one initial AMP
overflow skip. These are short, cached-subset measurements on this laptop, not a
full-dataset throughput or accuracy benchmark. Target servers must run calibration
themselves.

Full archive preparation and the complete sample audit ran natively on Windows
after Docker Desktop's Windows/OneDrive bind mounts proved slow. The GPU smoke
subset used native Linux volume storage. The complete sample audit report is in
`server-runs/full-dataset-preflight.json`; calibration records are in
`server-runs/real-kd-smoke/` (ignored by Git).

## Final real-data GPU checkpoint/resume check — passed

The published image was run by its exact digest on the RTX 4060 Laptop, using
the 8-train/4-validation TAR2000 subset, full 352×1216 resolution, batch 2, two
workers, FP16 AMP, and the existing 20-epoch KD schedule. Data used a native Linux
Docker volume; results and caches used persistent Windows host mounts.

The first container stopped cleanly at successful optimizer step 4, after batch 1
of zero-based epoch 1. A fresh container automatically loaded the checkpoint and
continued at batch 2, step 5, preserving the AMP scale. It completed the remaining
schedule, including active geometry and ordinal teacher losses, and exited with
code 0 and `status.json` set to `COMPLETE`.

- **80 consumed batches, 78 successful optimizer updates, two AMP overflow skips.**
  Scheduler position and all 256 populated optimizer states ended at step 78.
- All 20 committed epoch rows and all 80 batch positions were present exactly
  once. Training losses and validation RMSE remained finite.
- Final checkpoint model and optimizer tensors were finite; scaler state and
  Python, NumPy, PyTorch CPU and CUDA RNG states were present.
- Best checkpoint was epoch 17; its final subset validation RMSE was
  11.258173 m. This tiny smoke run is **not an accuracy benchmark**.
- Maximum reserved CUDA memory recorded in step telemetry was 2,237,661,184 bytes
  (about 2.08 GiB), including allocator reservations retained after validation.
  This differs from the isolated training-only calibration measurement above.
- TensorBoard files persisted across both process sessions. The latest three
  complete backup bundles were retained. Restoring the final bundle into an empty
  directory verified every file hash; restored last/best checkpoints matched the
  originals, and final evaluation results were included.

The audit report is `server-runs/final-validation/gpu-resume-audit.json`; the run,
logs, checkpoints and restored bundle are retained under `server-runs/` (ignored
by Git). These checks establish GPU restart continuity, not bitwise equivalence
to an uninterrupted GPU run. Exact uninterrupted/resumed equivalence and forced
SIGTERM/SIGKILL recovery were tested separately on CPU as described above.

No runtime code change or image rebuild was needed for the final test.

## Google Drive qualification — passed 2026-09-21

The user configured the `gdrive` remote with a custom OAuth client. Local Compose
now mounts the existing configuration from outside the OneDrive-backed repository;
no credentials were added to the image or source tree.

The published image authenticated in a fresh container, created and uploaded a
new complete backup through `publish_bundle` and `backup_once`, downloaded that
bundle from Google Drive, and restored it into an empty directory. All **17 file
SHA256 checks passed** (6,133,714 bytes). Restored last/best checkpoints matched
the originals. Model and optimizer tensors were finite, optimizer/scheduler step
was 78, AMP scale was 16,384, all four RNG state types were present, and the
20 epoch rows, 80 step rows, TensorBoard events and final evaluation survived.

Automatic OAuth refresh also passed inside the published image. An isolated
temporary configuration with an expired access token obtained a new token and
successfully accessed Drive. The user's original configuration was unchanged by
this test; the temporary credential copy was removed.

Verified remote bundle:
`gdrive:GeoLift_RT_runs/server/real-kd-smoke/000000000024-3ed7bb8d`.
Local evidence is `server-runs/drive-verification-20260921/audit.json` and
`server-runs/final-validation/drive-refresh-audit.json` (ignored by Git).
No further training or image rebuild was needed.

## External qualification still pending

Vast.ai testing is deferred at the user's request. RTX 5060 execution remains
unqualified. A new server still needs the authenticated rclone configuration
transferred securely to its writable secrets mount and a connectivity check there.
