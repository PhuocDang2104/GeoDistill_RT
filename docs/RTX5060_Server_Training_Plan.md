# RTX 5060 server training deployment plan

Prepared 2026-09-18. Status: proposed implementation and operator handoff specification, not a built or tested deployment. Reviewed local branch `augmentation`, HEAD `5da7463`, including existing working-tree changes. Only this plan is added by this task.

## 1. Recommended decisions

Build a prepackaged Linux/amd64 Docker training image, distribute it by immutable digest, and operate it through Docker Compose plus a small `geolift` launcher. The operator should not need Python installation, notebook execution, source editing, package troubleshooting, or training-code knowledge.

Confirmed user choices: server specifications are unknown and must be probed automatically; start with TAR2000 while preparing for larger KITTI datasets; persist results locally and back them up to Google Drive. The deployment target is Linux (Ubuntu 24.04 LTS preferred) with the advertised RTX 5060; verify OS and exact GPU during onboarding. Actual CPU/RAM/VRAM/free space are discovered, not assumed. NVIDIA lists the desktop RTX 5060 as Blackwell with 8 GB; the 5060 Ti has 8/16 GB variants. Probe the exact device. [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/)

The first release reproduces the existing experiment. Architecture, losses, augmentation stages, split, normalization, and LR trajectory remain explicit experiment settings. Only reliability and deployment prerequisites are included. Previous model-improvement proposals stay separate.

Selected storage: local mounted directories are authoritative, with asynchronous Google Drive backups. TensorBoard provides a local dashboard; an optional W&B integration can be added when online comparisons are useful. Local-only operation remains a recovery capability when Drive is unavailable. No database service is needed initially.

Deliver a tested release image plus Compose/config/runbook files. A source build remains available for developers. The operator normally pulls an image rather than compiling it on the server.

## 2. Existing code to reuse and gaps to close

Reuse `src/train_student.py`, `src/dataset.py`, `src/augmentations.py`, `src/experiment_record.py`, and the S3 configs. Move notebook setup, manifest validation, teacher extraction, and augmentation presets into importable server modules.

Current gaps verified in source:

- `gpu-train-pod.yaml` targets a specific Kubernetes/MIG environment and PyTorch 2.6/CUDA 12.4, installs packages at startup, then sleeps indefinitely. It is not the single-5060 deployment base.
- Colab paths are embedded in configs/notebooks; no Compose deployment exists.
- S3 training batch defaults to 2, validation batch is hard-coded to 1, and worker count defaults to 2.
- Saving uses direct `torch.save` to final paths; no transactional checkpoint manifest, RNG capture, or graceful-stop handler exists.
- Resume does not enforce checkpoint/preprocessing identity, and LR advances even when GradScaler skips an update.
- Backup uses synchronous filesystem copying; it is not a resilient remote upload queue.
- The standard runner requires train/val/test. KD notebooks instead call the trainer directly and assign test_split to val_split as a placeholder. The server must support a genuine train/val-only protocol, not invent a test set or bypass leakage checks.
- A1/A2/A3 presets live in the augmentation notebook. Augmented workers deliberately restart each epoch for seed behavior; tuning persistent_workers blindly would change that behavior.
- The existing profiler measures inference, not peak training VRAM including teacher losses and optimizer state.

## 3. Host and image compatibility

The administrator installs a GPU driver supporting the exact card, Docker Engine, Compose v2, and NVIDIA Container Toolkit, configures Docker's NVIDIA runtime, and gives the operator access to the Docker deployment and writable data directories. The host does not need a Python environment or a separate CUDA development toolkit. Driver/runtime setup follows the [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Proposed qualification stack: Python 3.11, a pinned stable PyTorch 2.12.x CUDA 13.0 build, and its matching torchvision. PyTorch's 2.12 release documentation recommends CUDA 13.0+ wheels for Blackwell and states Linux driver 580.65.06 as the CUDA 13.0 wheel minimum. Use an appropriately maintained compatible driver at or above the documented requirement. This is a qualification candidate, not an already tested image tag. Resolve exact packages/base image and record their hashes/digest after testing. [PyTorch release notes](https://pytorch.org/blog/pytorch-2-12-release-blog/)

Do not reuse CUDA 12.4 just because nvidia-smi is visible. Doctor must run CUDA allocation, cuDNN convolutions, grid_sample forward/backward, and the actual S3 training step. Record compute capability and compiled architecture support. Stop with actionable compatibility diagnostics if these fail.

Image requirements:

- Install dependencies during build from a student-only lock, including headless OpenCV, timm, YAML, NumPy, metrics/plotting, system probing, TensorBoard, and the chosen transfer tools. Teacher-generation/Open3D/MMCV/Jupyter dependencies belong in a separate optional image.
- Pin Torch/torchvision as a pair; never let a general requirements install replace the CUDA stack.
- Bake source and source identity into the image. Use build labels and a source manifest because `.git` need not exist inside the image.
- Use a nonroot runtime UID/GID mapped to operator-owned bind mounts. Include a real entrypoint with signal forwarding and `init: true`.
- Exclude datasets, checkpoints, caches, credentials, and `.env` from the build context. No credentials in image layers or public release artifacts.
- Training has no package installation or runtime git pull. Prefetch and verify pretrained encoder weights in prepare; offline training must work afterward.
- Runtime contains the tools needed for eager training. A compile-enabled variant is qualified separately if compilation requires additional tooling; compile is off for the migration baseline.

## 4. Compose services and persistent layout

| Service | Role | GPU | Persistent access |
|---|---|---|---|
| prepare | Authenticated fetch, SHA256 verification, safe extraction, full dataset audit, encoder-weight prefetch | No | Data/cache write; credential state only here when needed |
| doctor/autotune | Actual-device health, isolated trial subprocesses, preflight report | One selected GPU | Data read; cache/report write |
| trainer | Training, validation, local checkpoints and final evaluation | Same selected GPU | Data read-only; runs/cache write |
| backup | Upload finalized artifacts, retry backlog, report remote completion | No | Runs read-only; backup state and OAuth config write |
| tensorboard | Optional local dashboard | No | Runs read-only; localhost port only |

Use Compose GPU device reservations with `driver: nvidia`, a selected device ID, and `capabilities: [gpu]`; do not combine device_ids and count. [Docker GPU documentation](https://docs.docker.com/compose/how-tos/gpu-support/)

Preparation must finish successfully before preflight/training. The launcher enforces those stages and their identity checks; it does not rely only on container process startup order. A separate utility profile runs tools without accidentally starting the trainer. A host bind-mounted GPU/run lock prevents two launchers from scheduling the same device or run concurrently. External GPU processes are checked separately.

Suggested host layout:

```text
/srv/geolift/
  deployment/                 # small versioned release bundle
  data/
    downloads/                # resumable transfers and verified archives
    datasets/<dataset-hash>/  # immutable prepared KITTI/splits/teachers
  cache/                      # pinned HF/Torch weights; bounded derived caches
  runs/<run-id>/              # results survive container removal
  state/                      # run locks, probe reports, backup queue/state
  credentials/                # restricted access; never included in backup
```

Offer `./artifacts/{data,cache,runs,state}` as a convenient repository-local bind-mount layout, all ignored by Git and Docker build context. Logs may live beside the repo but should not be committed automatically. Publish only intentional small result summaries to Git.

Compose should expose memory/CPU limits and a private shm_size, initially 2 GiB if budget permits and adjusted from measured prefetch use. Do not copy the old 16-GiB shared-memory allocation or use host IPC by default. Account for shm and pinned buffers in RAM limits. Rotate Docker stdout logs. Use a configurable graceful-stop timeout; no privileged container or Docker socket mount.

Use `restart: "no"` for finite training jobs in v1. Add an optional host systemd launcher for reboot recovery: it reads persisted desired state, resumes only RUNNING jobs, respects STOPPED/FAILED/COMPLETE, and has bounded retry policy. Do not use unconditional container restart policies that rerun completed training. A stale heartbeat reports an issue; automatic replay requires a validated checkpoint and an explicitly classified retryable failure.

## 5. Dataset acquisition and verification

Create a versioned `configs/data/tar2000.manifest.yaml` with exact source identifiers, filenames, bytes, SHA256, schema version, selected-ID hash, split identities, required roles, and expected extracted size/file counts. Real identifiers and checksums must be populated from the owner's actual archives before release; names and the known KITTI byte size alone are insufficient.

Initial required files:

```text
selected_2000_ids.json
kitti_trainval_2000.tar
metric_coarse_train_2000.tar       # required only for KD presets
geometry_fused_train_2000.tar      # required only for KD presets
```

Support provider adapters for local files, HTTPS, and rclone remotes (Google Drive initially; S3-compatible storage later). Source adapters only fetch files into the same verified local representation. Train from extracted SSD data, not a Drive/FUSE mount. A local import option handles servers without internet.

Preparation procedure:

1. Resolve immutable manifest and estimate downloads + extracted data + optional cache + run/checkpoint retention + temporary work + Docker image storage. Check free bytes and inodes on each actual mount, including Docker's root filesystem.
2. Acquire a dataset preparation lock. Download to `.partial` with retries/backoff; continue transfers when the backend supports it, otherwise restart only the affected file. Verify SHA256 before atomic promotion. Never retry forever.
3. Safely extract to a staging directory: reject absolute/traversal paths, links escaping the root, device entries, unexpected duplicates, and excess declared size/count. Keep the previous complete dataset until the new dataset is valid.
4. Require exactly 1,600 train and 400 val IDs; enforce sample and raw-drive disjointness; preserve the supplied split rather than regenerate a random one.
5. Fully audit all TAR2000 samples once: decodable RGB/sparse/GT; dimensions, depth units, finite K/positive focal lengths, valid depth/support; matching teacher IDs, NPZ keys, dtype/shape/finiteness and confidence bounds. Check per-sample teacher coverage for the selected objective, not merely archive presence.
6. Use `allow_pickle=False` for NPZ. Record a validation report, identities, and READY marker last, then atomically activate the dataset version. Later runs reuse it with fast identity checks; offer an explicit deep re-audit.
7. Prefetch the exact encoder weights, record revision/checksum and preprocessing metadata, and verify model creation with network access disabled.

Do not generate teachers on the 8-GB server by default. Use the already prepared maps. Larger KITTI ingestion is a separate profile with explicit storage budgeting and drive-level splits; the existing download/extraction helpers can be adapted for that profile.

Anonymous test download/export is optional and occurs for selected final runs. Train/validation-only runs must not require the extra 1,000 test images.

### Larger KITTI profile: build the extension points in the initial release

Use the same sample schema and runtime CLI for both `tar2000-v1` and a new `kitti-large-v1` manifest. Paths and sample counts are manifest-driven throughout preparation, preflight, training and reporting. Do not hard-code 1,600/400 assertions into the generic runner; those are TAR2000-profile checks.

- Acquire official depth/Velodyne archives and matching raw RGB/calibration through the existing KITTI download logic refactored into resumable steps. Fetch/extract only relevant RGB frames where practical. Track completion by archive/drive, so an interruption does not repeat the entire preparation job. Honor source access requirements; never assume a new endpoint needs no credentials.
- Support an intermediate larger subset before a complete dataset. Record its selection policy and IDs. Preserve the existing 400-image holdout only if all its raw drives are excluded from enlarged training, or declare a new official validation protocol and give it a new identity. Never mix incomparable results under the same dataset name.
- Estimate capacity from source metadata and per-sample schemas before transfers. Include peak archive-plus-extracted coexistence, teacher maps, derived caches, output retention and checkpoint-upload backlog. Archive cleanup is explicit and follows successful extraction verification; it does not remove the last recoverable input prematurely.
- Offline KD requires teachers for the expanded training IDs. TAR2000 teachers do not cover a larger training set. Define an external teacher-generation/import job and teacher-manifest version with full coverage checks; refuse a KD run with missing maps. A teacher-free larger-data preset is a separate explicit option. Heavy teacher generation stays outside the small student-training image and is not assumed to fit this GPU.
- Store per-sample schema/coverage/audit results in a compact local index. Deep-audit each immutable dataset once; subsequent runs validate the prepared version and required files cheaply. Reuse bounded per-worker/global-budget caches; support sharded or mmap caches behind the dataset adapter if measured small-file I/O becomes limiting.
- Recalibrate on the new dataset/objective identity because loading/decompression/cache hit rates may differ. Keep disk-budget monitoring, optional expensive evaluation/export cadence, and checkpoint retention configurable.
- Before promoting long-epoch training, implement and qualify mid-epoch recovery as specified below. Set checkpoint cadence in both time and successful optimizer steps, with the next safe optimizer boundary as the actual save point. Include a kill-and-resume equivalence test with the selected augmentation, loader workers, and sampler.

Deliver the larger-data manifest schema, provider/adapters and dry-run capacity report in the initial implementation. Downloading the entire dataset and generating its teachers are explicit jobs, not automatic side effects of a TAR2000 launch.

## 6. Augmentation and experiment presets

Represent the existing notebook presets in YAML:

| Preset | Augmentation | Schedule |
|---|---|---|
| s3-plain | None, teacher-free | Existing 30 epochs |
| s3-kd-plain | None, offline teacher KD | 20 epochs on existing 30-epoch LR trajectory |
| s3-kd-a1 | Horizontal flip p=0.5 | Same KD schedule |
| s3-kd-a2 | A1 + sparse dropout 0–20%, minimum support 64 | Same KD schedule |
| s3-kd-a3 | A2 + 1.0–1.1 scale/crop | Same KD schedule |

Apply augmentation on the fly to training only, using the existing aligned implementation. Do not multiply dataset size by materializing transformed copies. Each stage initializes independently from the specified pretrained encoder; it does not resume the previous stage. Preserve GT/teacher alignment and K updates. Save a fixed-seed preflight contact sheet showing RGB, sparse, GT, teachers, and augmentation parameters.

Autotuning must never select augmentation strength, LR, loss weights, resolution, or data split based on hardware. Keep preset choice separate from runtime choice. Preserve augmented-worker restart behavior until a stateless augmentation RNG derived from seed/epoch/sample ID is implemented and tested.

## 7. Hardware probing and bounded autotuning

Probe inside the same container image and resource limits used for training. A lightweight initial host/utility probe generates Compose resource overrides; perform the final measurement under those overrides. Respect cgroup CPU quota, CPU affinity, container memory limit, host MemAvailable, /dev/shm capacity, mount performance, and available rather than total VRAM.

Collect GPU identity/UUID/compute capability/driver, total/free VRAM and competing processes; CUDA/cuDNN/Torch versions; CPU and usable cores; RAM/swap; storage mount type, free bytes/inodes, bounded local read/write/decode performance; and transfer access. Observe temperature, utilization and power when available, but do not change GPU clocks/power settings automatically.

Two explicit tuning policies:

- **baseline (default):** preserve physical/effective training batch 2, precision, image size, LR and schedule. Tune worker/thread count, prefetch and caching; validation batching may be tuned after metric parity tests. Produce a throughput recommendation for larger batches, but do not apply it to the baseline run.
- **throughput:** explicitly selected new experiment family. Search physical batch size, optionally validated accumulation, precision/layout/compile choices within an approved set. Record effective batch and optimizer-update count. Accumulation does not reproduce larger-batch BatchNorm behavior; never claim exact baseline equivalence.

Calibration algorithm:

1. Use disposable subprocesses and a representative real subset, with early and fully activated KD objectives, validation, and the selected augmentation workload. Smoke-test maximum A3 geometry too when A3 is selected.
2. Include forward, loss, backward, scaler, clipping, and AdamW step. Warm up enough to initialize optimizer state and backend workspaces. Check finite losses/gradients, parameter updates, sparse anchoring, and normalization.
3. In baseline mode test batch 2 only for acceptance. In throughput mode increase 1/2/4/8 until limits/failure, then refine if worthwhile; never exhaustively search a large Cartesian product.
4. Search workers among 0/1/2/4, optionally 8 only with enough usable CPU/RAM; prefetch 1/2 when workers >0; cap OpenCV/OMP threads to avoid oversubscription. Compare loader wait and total examples/sec, not isolated decode time.
5. Compare SSD NPZ reads with a bounded pre-augmentation cache or mmap representation only when data preparation is a measured bottleneck. Four FP32 teacher channels for 1,600 images alone are about 10.2 GiB, so full-RAM caching is not an 8/16-GB-host default. Enforce a global cache budget across all worker processes.
6. Keep a configurable VRAM reserve, initially max(1 GiB, 15% of total), and RAM headroom for OS, loader workers, pinned buffers, backup and dashboard. Measure both allocated and reserved GPU memory plus device free memory; do not infer fit from parameter count.
7. Use short candidate windows followed by a 100-step or longer finalist stability check covering full KD; include cold-cache and warm-cache behavior. Initial tuning budget: approximately 10–15 minutes, configurable. Timings are to be measured, not promised.
8. Select the smallest-memory candidate within 5% of the best measured sustained throughput. Save every candidate/result, rejection reason, headroom, and the winning resolved config.
9. Cache tuning by GPU/CPU/resource-limit identity, image digest, dataset identity, model/objective/augmentation, shape and precision. Reuse unless those change or `--retune` is set. Recheck currently free resources on resume.

Keep FP16 AMP/channels-last/eager as the migration starting point. The prior review found reduced-precision coordinate risks; do not auto-enable BF16 for geometry. FP32-coordinate changes, BF16, and torch.compile belong to separately validated variants, not a hidden migration adjustment.

If baseline batch 2 does not fit, exit with evidence and a suggested new experiment. Do not silently shrink batch, drop teachers, lower resolution, or remove losses. A production OOM records the failure and preserves the last valid checkpoint; candidate-search OOMs are handled in isolated subprocesses without polluting subsequent trials.

## 8. Preflight and execution state machine

```text
NEW -> PREPARING -> DATA_READY -> PROBING -> PREFLIGHT_PASSED
    -> TRAINING -> VALIDATING -> CHECKPOINT_COMMITTED
    -> ... -> FINAL_EVALUATION -> COMPLETE

Explicit alternatives: STOPPED, FAILED, WAITING_FOR_RESOURCES
Backup state tracked separately: DISABLED / PENDING / SYNCED / ERROR
```

Before full training require: host/CUDA health, dataset readiness, pretrained weights available, writable persistent mounts, sufficient disk/RAM/shm/VRAM, full contract suite, actual-batch full-objective optimizer smoke, selected augmentation checks, serialization/reload check, and resolved run identity. Preflight produces JSON plus a human-readable PASS/FAIL summary and concrete remediation.

Migration qualification adds a brief overfit experiment on a fixed tiny subset to catch disconnected gradients or broken optimization; it is not repeated on every ordinary run or used as an accuracy benchmark. Trial models, optimizer state, RNG, and BN statistics are discarded before actual training initialization.

Separate `training_complete`, `evaluation_complete`, and `backup_complete`. A failed Drive upload must not label successful training as failed; failed evaluation must not label the whole requested experiment complete. A rerun resumes the failed stage without retraining a finished model.

## 9. Checkpoints, interruption and recovery

Make epoch checkpoints transactional: serialize to a temporary file on the same local filesystem, flush/fsync, atomically rename, compute identity, and publish the checkpoint manifest last. Keep the previous good checkpoint until the new one is committed.

Save model, optimizer, scheduler, AMP scaler, completed epoch and global optimizer step, best metric/checkpoint identity, RNG states, sampler/data-seed policy, resolved config/protocol, source/image identity, and committed metric-history boundary. Advance LR only after a successful optimizer update.

For v1, resume from the last **completed epoch**. On SIGTERM or launcher stop, request a graceful epoch-boundary checkpoint with a configured timeout; if termination is forced, retain the previous completed epoch. Explicitly tell the operator that abrupt power loss can lose up to one epoch. Logs from incomplete attempts are kept as attempt logs but excluded from committed epoch summaries to prevent duplicates. A fresh machine restores best/last/history together and verifies checksums before loading.

Mid-epoch recovery is a required qualification gate before enabling longer-epoch KITTI production runs: it needs a deterministic sample permutation, consumed-batch cursor (not prefetched cursor), stateless augmentation or equivalent worker-state design, and atomic step/log commits. Save only at completed optimizer boundaries, including scaler/scheduler/global-step and any accumulation-policy state. Intra-epoch metric accumulators and committed-log boundaries must resume consistently too. Periodically writing model weights alone is not exact mid-epoch resume. For nondeterministic CUDA kernels, test sequence/state continuity and numerical agreement within declared tolerances rather than claiming bitwise equality.

Retention default: best, two latest recoverable epoch checkpoints, and configurable milestone epochs. Do not prune the sole restorable checkpoint or artifacts pending required backup. Set a disk high-water threshold; pause cleanly before the filesystem fills. Unexpected NaN/Inf, repeated AMP skips, config mismatch, dataset corruption, or CUDA incompatibility fail clearly rather than restart indefinitely.

## 10. Logs, metrics, dashboards and backups

Every run writes:

```text
runs/<run-id>/
  run_manifest.json             # experiment identity, source/image/data hashes
  resolved_config.yaml
  hardware.json
  autotune.json
  preflight.json
  status.json                   # stage, epoch/step, heartbeat, progress
  experiment_record.json
  logs/
    console.log
    train_steps.jsonl
    train_epochs.csv
    system_metrics.csv
    events.jsonl                # checkpoints, retries, failures, resume history
  tensorboard/
  checkpoints/
  previews/
  evaluation/
  summary.json
```

Capture total/per-term losses and scheduled weights; LR; gradient norm and AMP skips/scale; global RMSE/MAE/iRMSE/iMAE/AbsRel/deltas; range/edge metrics; pre-/post-anchor diagnostics and nonfinite/clamp counts; epoch/step throughput and data wait; validation, checkpoint and sync durations; GPU memory/utilization/temperature/power where available; process-tree/cgroup RAM, CPU, disk free space and I/O; and checkpoint/best-epoch references. Include metric units and validity limits.

Log scalar summaries every 25–50 steps, epoch metrics each epoch, system metrics every 10–30 seconds, and a small fixed preview set every few epochs. Aggregate device statistics before transfers; do not add one CUDA synchronization per metric. Rotate console logs and bound preview/checkpoint retention. Default final validation to metrics + selected previews; full predictions/submission export is an explicit option.

Storage options:

| Option | Recommendation |
|---|---|
| Mounted local files + TensorBoard | Always available and authoritative; works without internet |
| Local + Google Drive via rclone | Recommended first release given existing inputs |
| Local + S3-compatible remote | Same artifact protocol, useful alternative if a bucket already exists |
| Local + optional W&B | Useful for interactive run comparisons; not the sole checkpoint backup |

Google Drive onboarding uses a prepared rclone OAuth configuration with a headless authorization procedure. Document ownership and scope of the input/output folders, token renewal, and expired-credential remediation. Keep credentials outside Git/image/runs and persist rclone's refreshed credential state securely; avoid treating a read-only secret file as the only writable OAuth config. Initial data access must succeed before training, but later backup outages do not stop local training. [rclone headless setup](https://rclone.org/remote_setup/) and [Drive backend](https://rclone.org/drive/)

Backup worker consumes immutable checkpoint/log segments or stable snapshots identified by completion manifests. It uploads files, verifies available checksums/size, and publishes the remote completion manifest last. Restore selects only complete manifests and verifies SHA256 locally. Upload metadata/log snapshots about every minute and checkpoint bundles after each committed epoch; cadence is configurable. Keep an acknowledged queue, exponential retry/backoff, and a visible backlog/last-success status.

Use non-deleting copy semantics and unique run/checkpoint paths, not a destructive mirror of a changing directory. Local pruning must not delete remote history. Do not race uploads against an actively overwritten last.pth or growing JSONL file. Limit transfer concurrency/bandwidth when it interferes with input I/O. [rclone copy semantics](https://rclone.org/commands/rclone_copy/)

TensorBoard binds to localhost and is accessed through SSH forwarding. Optional W&B logging must preserve local output, use explicit run/attempt identity, and support offline capture and later synchronization; test resumed-run behavior for the pinned SDK. [W&B offline guidance](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues)

## 11. Operator experience

Proposed commands below are the interface to implement; they do not exist yet:

```bash
cp .env.example .env
./geolift doctor
./geolift prepare --dataset tar2000
./geolift preflight --preset s3-kd-plain --tune baseline
./geolift start --preset s3-kd-plain --run-id migration-kd-01 --detach
./geolift status migration-kd-01
./geolift logs migration-kd-01 --follow
./geolift stop migration-kd-01
./geolift resume migration-kd-01 --detach
./geolift backup-status migration-kd-01
./geolift export migration-kd-01
```

`start` should run missing prepare/preflight steps automatically and reuse valid results, so an experienced operator can start from one command after onboarding. `resume` restores the exact run definition; it does not retune optimization settings. `start` refuses an existing run ID rather than overwriting it. Launch can optionally use a declared sequential experiment queue; no parallel GPU jobs by default. Disconnecting SSH must not stop a detached job.

The small .env asks only for host storage root, selected GPU, UID/GID, release image, dataset source profile, and backup mode/config path. Presets contain scientific settings. All credentials are files outside the deployment bundle. Error messages should name the failed stage and a copy-paste recovery command. `status` shows checkpoint age, epoch/step, ETA estimate, resource usage, backup lag and next action.

Support bundle export contains redacted environment/package identity, recent logs, hardware/preflight/tuning reports, resolved nonsecret config and checkpoint manifests. Exclude credentials, signed URLs, datasets and full weights unless explicitly requested. No external notification messages are sent by this planning task; future optional alert destinations must be configured by the owner.

## 12. Implementation work packages

| Order | Deliverables | Acceptance |
|---|---|---|
| 1. Freeze migration contract | Approved source snapshot, dataset manifest, server preset YAML, source/data identity schema | No untracked change can silently alter a release; exact inputs and experiment defined |
| 2. Package runtime | Dockerfile, student lock, .dockerignore, Compose/services/profiles, .env.example, build metadata, release image | Clean host pulls and passes real RTX 5060 CUDA/model smoke; no runtime pip |
| 3. Prepare inputs | Provider adapters, verified/staged extraction, schema/split audit, pretrained-weight cache | Interrupted transfer resumes/retries safely; corrupted/partial data rejected; second prepare reuses data |
| 4. Probe/preflight | Cgroup-aware hardware report, bounded tune search, worst-stage memory tests, resolved runtime config | Fits with headroom; correct objective tested; disposable tuning leaves actual initialization unchanged |
| 5. Harden trainer | Atomic checkpoints/RNG/config checks, committed logs, signal handling, LR-on-success, validation-only protocol | Kill/resume and complete-run recovery work; best checkpoint/history survive |
| 6. Observe and back up | Structured logs, TensorBoard, status heartbeat, immutable artifact queue, rclone worker, retention | Training survives offline periods; restore from a completed remote bundle works |
| 7. Larger-data readiness | Generic KITTI manifest/import, capacity dry-run, teacher-coverage contracts, bounded cache, validated mid-epoch recovery before long-run enablement | A larger subset prepares with the same interface; interrupted shuffled/augmented training resumes correctly |
| 8. Handoff | Launcher, quickstart, troubleshooting/restore guide, example presets, optional systemd recovery | A second person completes cold setup/start/stop/resume/export using only the runbook |

Suggested files/modules:

```text
docker/Dockerfile.train
docker/entrypoint.sh
requirements/train.lock
compose.yaml
.env.example
.dockerignore
geolift
configs/data/tar2000.manifest.yaml
configs/server/{s3_plain,s3_kd_plain,s3_kd_a1,s3_kd_a2,s3_kd_a3}.yaml
src/runtime/{cli,prepare,probe,autotune,preflight,checkpoint,telemetry,backup}.py
docs/SERVER_OPERATOR_GUIDE.md
tests/test_runtime_*.py
```

Implement reusable checkpoint/logging changes in the existing trainer rather than maintaining a separate server-only training loop. Extend experiment records to make absent test splits explicit. Leave the Colab notebooks available as historical references during migration, but make the CLI/configs authoritative for new runs.

## 13. Release and failure-injection tests

CPU/CI: current contracts; manifest/split identity; corrupt archive and safe extraction; cache invalidation; config precedence; checkpoint atomicity; incompatible resume; metric-history reconciliation; mocked transfer retries; secret redaction. Build/validate Compose and the pinned image in Linux CI. No RTX 5060 performance claim is accepted from CPU CI.

Target-server qualification:

1. Clean checkout/release bundle with empty cache: authenticated prepare, full audit, pretrained fetch, actual-device preflight.
2. Repeated prepare/start: no unnecessary extraction/download and no accidental second GPU job.
3. One short teacher-free and one KD training/evaluation run, with geometry loss explicitly active in preflight; exercise A3 preflight even if the first production run is plain.
4. Stop during a batch/epoch, kill during checkpoint writing, remove/recreate containers, and reboot under the optional recovery service. Restore the last committed epoch without corrupting best/history.
5. Inject missing teacher data, bad checksum, stale image/config, full disk, constrained RAM/shm, competing GPU allocation, and CUDA OOM. Each yields a specific state and recovery action, not silent fallback.
6. Disable internet after preparation: training/validation finish locally, backup shows pending, and queued artifacts upload after reconnection.
7. Restore to a clean run directory from remote backup and perform validation with the restored best checkpoint.
8. Run a sustained target-GPU burn-in with full KD/A3 workload before handoff; record memory trend and steady-state throughput. Keep GPU/precision accuracy differences and legacy pre-normalization results distinct.
9. Operator acceptance: someone other than the developer follows the runbook without editing Python or notebook cells.

Confirmed: automatic hardware discovery, TAR2000 initially with larger KITTI support, and local plus Google Drive persistence. Deployment-time inputs still needed: server access/OS verification, intended first experiment preset, real archive IDs/checksums, writable storage root, and backup account/folder authorization. Capacity and hardware-dependent batch/worker/cache values are intentionally resolved by probing; the artifact and recovery contracts are defined now. If the host is not Linux, doctor must report the unsupported deployment target with an explicit Linux/WSL2 migration path instead of trying Linux installation commands blindly.
