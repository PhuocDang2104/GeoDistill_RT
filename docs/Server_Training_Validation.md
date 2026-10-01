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

## Vast.ai RTX 5060 Ti qualification — passed 2026-10-01

The short qualification used an RTX 5060 Ti 16 GB, driver 580.173.02, three
allocated CPU cores, about 42.5 GiB RAM, 15 GiB shared memory and a 100 GB container
disk. PyTorch 2.10.0+cu128 exercised compute capability 12.0 (`sm_120`) successfully.
The packaged source hash matched release `server-v1`:
`f469deac24b3c19fa8a3e288f3262b26294860c855052fa442dbc48ffdfc449e`.

Only 8 training and 4 validation samples from TAR2000 were downloaded and audited,
using the existing 352×1216 configuration and FP16 AMP. This was not a full training
run, a full-dataset server audit, or an accuracy benchmark.

| Loader workers | Calibration images/s | Peak reserved VRAM |
|---:|---:|---:|
| 0 | 3.506 | 1,086,324,736 bytes |
| 2 | **5.919** | 1,086,324,736 bytes |

Baseline calibration retained batch size 2 and selected two workers. It exercised
the full scheduled loss at epoch 23, including geometry objectives. Four workers
were correctly skipped because the rental exposes only three CPU cores. These
short cached-sample timings exclude warmup and do not predict full KITTI throughput.
Fresh training processes took roughly 89–93 seconds for the first training batch
and 74–75 seconds for the first validation batch; subsequent batches were much
faster. This startup overhead deserves separate profiling before repeated short
experiments; the qualification did not change the training protocol to avoid it.

Verified sequence:

- Clean stop at successful optimizer step 4; a new process resumed at step 5.
- Stop at step 8, upload immutable bundles to Google Drive, download the latest
  completed bundle, and restore into a separate run directory.
- All 15 downloaded file hashes passed; the restored checkpoint matched the
  original step-8 checkpoint byte for byte.
- A new process resumed the cloud-restored run at step 9 and stopped at step 12.
- All 256 populated optimizer states and the scheduler reached step 12. Model
  and optimizer tensors, recorded losses, and validation metrics were finite.
- The final log contained 13 consumed batches, 12 successful updates and one AMP
  skip, with no duplicate epoch/batch positions. Three TensorBoard session files
  survived the local and cloud resume sequence. Peak reserved memory recorded in
  telemetry was 2,262,827,008 bytes (about 2.11 GiB).
- Both run backup states were OK. Training processes exited and the GPU was idle.

Remote completed bundles:

- `gdrive:GeoLift_RT_runs/server/vast-5060ti-qual-20261001-01/000000000006-5becc1a0`
- `gdrive:GeoLift_RT_runs/server/vast-5060ti-qual-20261001-01-restored/000000000009-9d8ed87b`

Local evidence, checkpoints, logs, downloaded bundle, and test scripts are retained
under `server-runs/vast-qualification-20261001/` (ignored by Git). The transferred
archive SHA256 is `4cee28c3cf46a54ea12ffb51d391ef281cfc102fe0c7e652427caa545f470ec7`.
Credential files were excluded from the archive.
The eight evidence files were also uploaded to
`gdrive:GeoLift_RT_runs/qualification-evidence/vast-5060ti-qual-20261001-01`;
`rclone check --one-way` reported eight matches and zero differences.

The original run provenance correctly recorded source/data hashes but stored
`image: unrecorded`: the template's image variable did not reach the SSH shell.
The detailed audit separately records the expected image and observed packaged
source hash without rewriting the original provenance. Future launch instructions
explicitly export `TRAIN_IMAGE`.

This qualifies process restart and cloud restore on the 5060 Ti 16 GB. It does not
establish bitwise GPU equivalence, inject a physical host failure, qualify the
RTX 5060 8 GB, or replace a full TAR2000 preflight before a long server run. The
earlier CPU SIGTERM/SIGKILL tests remain separate evidence. Use a new run ID and
the full manifest when moving beyond this smoke subset.
