# GeoDistill server training

For the current Vast.ai setup, follow the
[step-by-step operator guide](Vast_Training_Step_by_Step.md), covering cloning,
credentials, full TAR2000 preparation, detached training and backup, monitoring,
resume/restore, final verification, and downloading results. Run training directly
from the image's `/app` inside Vast; the Docker Compose commands below apply to a
conventional Linux Docker host. The Vast guide also records which local startup
repairs have not yet been pushed to the repository.

The Docker image contains the training source, Python dependencies, CUDA runtime,
dataset preparation, hardware calibration, checkpoint recovery, TensorBoard, and
rclone backup. The host needs an NVIDIA driver, Docker Engine with Compose GPU
support, and NVIDIA Container Toolkit. No host Python or CUDA toolkit is required.

The first release targets one GPU. It uses PyTorch 2.10.0/CUDA 12.8, tested locally
on an RTX 4060 Laptop 8 GB and qualified on a Vast.ai RTX 5060 Ti 16 GB on
2026-10-01 (Asia/Saigon). The exact RTX 5060 8 GB target remains unqualified.
The installed wheel includes `sm_120`; actual GPU kernel execution is checked by
`doctor`. Use the published digest in `docker/release.json` to reproduce a release.

## First run on a Linux server

Copy this repository's `compose.yaml`, `docker/server.env.example`, and this guide
to the server. The source checkout itself is optional; Compose pulls the image.
Keep the following commands in the same directory as `compose.yaml`:

```bash
cp docker/server.env.example .env
mkdir -p server-data server-runs server-cache server-config server-secrets
docker login --username phatle0106
docker compose pull trainer
docker compose run --rm doctor
docker compose run --rm prepare
docker compose up -d trainer
docker compose logs -f trainer
```

Use a Docker Hub token with read permission when logging in on the training host.
Set `TRAIN_IMAGE` in `.env` to the digest recorded in the release file. Credentials
are entered through Docker's prompt and never belong in `.env`, the image, or Git.

`prepare` downloads the fixed TAR2000 assets, verifies SHA256 and size, extracts
them into a content-addressed directory, checks the frozen 1600/400 split and
drive separation, checks calibrated intrinsics and supervised files, decodes all
samples, validates teacher data, and benchmarks the actual training objective.
It also downloads/caches pretrained encoder weights. A failure stops preparation;
`server-runs/<RUN_ID>/failure.txt` explains it.

The dataset is about 14.2 GB of archives plus extracted data. Provision **at least
60 GB free**, preferably 80–100 GB for the image, download cache, checkpoints, and
backups. The runtime checks archive/extraction space before preparation. Store the
mounts on local SSD. Do not train from a live Drive mount. Point the directory
variables in `.env` to larger disks if needed.

The bundled manifest identifies the user-provided Drive files. If Drive download
quotas block anonymous access, configure rclone below, copy the bundled manifest to
`server-config/dataset.json`, and replace each `source` with its authenticated
`gdrive:path/to/file` location, keeping the verified sizes and hashes. This creates
a new dataset identity; finish this before starting a run. To seal a new TAR2000
source once, use:

```bash
docker compose run --rm catalog catalog --source gdrive:path/to/teacher_subset_2000
```

This downloads and calculates real hashes; it does not invent checksums or choose
new train/validation IDs. The provided `docker/tar2000-sources.json` is another
accepted `--source` argument inside the image.

## Calibration and experiment choices

Default `TRAIN_MODE=baseline` keeps the configured batch size (2), resolution,
learning rate, precision, objective, and schedule. It compares 0/2/4 workers where
CPU, RAM and `/dev/shm` allow, and retains 15% of available VRAM as headroom. Every
trial runs in a fresh process, warms AMP, activates the late teacher losses, and
executes forward/backward/optimizer steps. Actual data loading is included in its
throughput measurement. No fitting configuration means a clear failure, not a
silent experiment change.

`TRAIN_MODE=throughput` additionally searches batches 1/2/4/8/16. Use a **new RUN_ID**
because changing batch changes the optimization experiment. The learning rate is
not automatically scaled. The search finds the best measured candidate within
this bounded search, not a guarantee of the globally fastest configuration.
Compile and multi-GPU execution are disabled in this release. Precision is fixed
to the selected experiment; numerical correctness is prioritized over an
unvalidated precision switch.

Augmentation remains controlled by the training YAML. The initial run preserves
the existing no-augmentation baseline. For an A1/A2/A3 experiment, copy the chosen
config to `server-config/train.yaml`, add its `data.augmentation` settings, set
`TRAIN_CONFIG=/config/train.yaml`, and use a new run ID. Geometry-aligned transforms
run online; no expanded image dataset is materialized. A3 settings, for example:

```yaml
augmentation:
  enabled: true
  stage: A3
  horizontal_flip_prob: 0.5
  sparse_dropout: {enabled: true, min_rate: 0.05, max_rate: 0.20, min_points: 64}
  scale_jitter: {enabled: true, min_scale: 1.0, max_scale: 1.15}
```

Put this mapping **under `data`**, retaining the rest of the original config.
These are example settings; use the notebook's exact parameters for a matched
ablation. Augmentation RNG is tied to seed, epoch, and sample ID position so worker
prefetching cannot change resumed training inputs.

## Recovery

`last.pth` is committed every 100 consumed batches or five minutes, at epoch end,
and after a graceful stop request. It contains model, optimizer, scheduler, AMP
scaler, RNG states, sample order/cursor, running epoch loss totals, global optimizer
step, best validation score, and the experiment contract. Payloads are immutable,
hashed, fsynced and atomically published; training reads their verified pointer.
The scheduler advances only on successful optimizer updates.

Graceful shutdown:

```bash
docker compose stop trainer
docker compose up -d trainer
```

The five-minute stop grace period gives the trainer time to finish its current
batch/validation and save. A hard crash loses only work since the last committed
checkpoint; replay begins from that checkpoint. Restarting the same run ID resumes
automatically. Compose retries failed training at most three times to avoid an
unlimited failure loop. Logs from uncommitted epochs are reconciled on recovery.
Step logs include session IDs; replayed attempts remain visible there.

To request a persistent operator stop, create `server-runs/<RUN_ID>/STOP`. Remove
that file before resuming. Do not change epoch horizon, batch, objective,
preprocessing, or source files to resume; start a new experiment for those changes.
On a replacement host the runtime records new hardware and reduces workers to zero
if host/shared memory is smaller. It preserves batch and precision; insufficient
VRAM fails instead of silently changing training semantics. Numerical identity
across different GPU architectures is not guaranteed.

## Local metrics and Google Drive backup

All outputs survive container replacement in `server-runs/<RUN_ID>/`:

| Path | Contents |
|---|---|
| `status.json`, `failure.txt` | Current state or actionable failure |
| `resolved_config.json`, `resolved_paths.json`, `provenance.json` | Experiment and source/dataset identity |
| `hardware*.json`, `calibration.json`, `calibration_trials/` | Host inventory and measured candidates |
| `logs/train_log.csv`, `logs/train_log.jsonl` | Train losses, validation metrics, epoch timings |
| `steps.jsonl` | Losses, LR, scaler, skipped steps, grad norm, RAM/VRAM/disk telemetry |
| `tensorboard/` | Event files, separated by process session |
| `checkpoints/` | Last, best, epoch aliases, immutable payloads and checksums |
| `evaluation/best_val.json` | Final validation of the best checkpoint |
| `backup_queue/`, `backup_status.json` | Completed snapshots and latest backup result |

The final validation is on the existing validation split, not an independent test
score. TensorBoard is optional and bound to localhost:

```bash
docker compose --profile monitor up -d tensorboard
# From your laptop, use SSH port forwarding to server port 6006.
```

For Drive, authenticate once using rclone's normal browser/headless setup. The
config contains a refresh token and must stay outside Git:

Use your own Desktop OAuth client with the Drive API enabled. See
[rclone's client setup](https://rclone.org/drive/#making-your-own-client-id).
For unattended use, move an external OAuth app out of Testing mode to avoid the
weekly grant expiry. `drive.file` is suitable for a new backup-only remote;
accessing existing dataset files may require a broader scope or a separate remote.

```bash
docker compose run --rm --entrypoint rclone catalog config
docker compose run --rm --entrypoint rclone catalog lsd gdrive:
docker compose --profile backup up -d backup
```

The config is persisted at `server-secrets/rclone.conf`. Name the remote `gdrive`
or adjust `BACKUP_REMOTE`. On a headless server, complete OAuth on your laptop
using [rclone remote setup](https://rclone.org/remote_setup/) and transfer the
config securely. The writable secrets mount lets rclone refresh its token.

On Windows, you can authenticate using native rclone and set `SECRETS_DIR` in
the local `.env` to the directory containing its `rclone.conf` (use an absolute
path with forward slashes). Keep that directory outside OneDrive and Git. The
local verification used the existing configuration under `%APPDATA%/rclone`.
On the Linux training server, securely transfer the configuration into its own
secrets directory, restrict file access, and run the connectivity command above.

The published image passed a real Drive upload/download/restore and an isolated
expired-token refresh test on 2026-09-21; see `Server_Training_Validation.md`.

Backup uses `rclone copy`, never destructive synchronization. It uploads immutable
bundles, publishes `COMPLETE.json` last, and keeps the latest three local bundles.
Network failures leave local training running and appear in `backup_status.json`.
Cloud retention is deliberately manual in this release: complete remote bundles
accumulate, so monitor quota and prune older bundles after verifying newer ones.
Backup is asynchronous; host loss can lose the work since the most recent
**successfully uploaded** bundle. Before releasing a rented host, run:

```bash
docker compose run --rm backup backup --once
```

Restore a completed bundle from Drive to an empty run directory:

```bash
docker compose run --rm --entrypoint rclone catalog copy \
  gdrive:GeoLift_RT_runs/server/RUN_ID/BUNDLE_ID /data/restore-bundle
docker compose run --rm catalog restore \
  --bundle /data/restore-bundle --destination /runs/RUN_ID
docker compose up -d trainer
```

Replace `RUN_ID` and `BUNDLE_ID` with the actual completed bundle and use the same
RUN_ID in `.env`. Restore validates every file checksum. Keep the original image
digest, manifest, and train config. Missing datasets will be fetched again. Existing
run directories are never overwritten by restore.

## Larger KITTI datasets

Use the `prepared-kitti` manifest format for normalized, optionally sharded TARs.
It uses the same streaming dataset loader and calibration; no full dataset is
loaded into RAM. Each manifest asset has `name`, `source`, `sha256`, `bytes`, and
`expanded_bytes`. Each archive extracts relative to a shared root, with no duplicate
files or links. Supply a layout such as:

```json
{"version":1,"format":"prepared-kitti","assets":[],
 "layout":{"data":"kitti","splits":"splits","teachers":"teachers",
           "train_split":"train.jsonl","val_split":"val.jsonl"}}
```

Populate `assets` with real archives and hashes. Split JSONL records must contain
`id`, `rgb`, `sparse`, `gt` and `K` or `calib`; paths are relative to `data`.
Use explicit train/validation splits with no frame or raw-drive overlap. KD needs
canonical `teachers/metric_coarse/train/<id>.npz` and
`teachers/geometry_fused/train/<id>.npz` coverage. The image consumes offline
teachers; it does not automatically run large teacher models on an 8 GB GPU.
Official KITTI acquisition/normalization and new split selection must happen
before packing these archives; existing extraction scripts remain available.
Do not label a larger dataset run as the TAR2000 baseline.

## Vast.ai qualification

Use the private published image directly, with registry pull credentials, SSH
launch mode, at least 60 GB disk (80–100 preferred), and one GPU. Prefer an RTX 5060
8 GB to qualify the target; another 8 GB GPU only verifies the general workflow.
Paste the complete contents of `docker/vast-onstart.sh` into On-start Script.
It is also embedded as `onstart` in `docker/vast-template.json` and works with the
published `server-v1` image without requiring the new helper to exist in it.

Vast overrides ordinary image CMD/ENTRYPOINT in its managed launch workflow.
The image deliberately contains no SSH host keys; Vast can attempt to start SSH
before this hook and fail with `sshd: no hostkeys available -- exiting`. The hook
generates missing per-instance keys, validates SSH configuration and starts the
service. It also repairs root SSH directory/key-file ownership and permissions,
prints only public-key fingerprints and selected effective SSH settings, and sends
verbose authentication events to the container log. It does not add login keys,
disable StrictModes, change root login policy, or enable password authentication.
If a registered key still fails, compare the logged authorized-key fingerprint
with `ssh-keygen -lf ~/.ssh/id_ed25519.pub`, retry once, and inspect the new auth
log entry. The client error alone cannot distinguish missing keys, bad permissions,
account restrictions, or a provider configuration override.
Repeated execution preserves existing keys. It is required for the
existing published image even though newer source includes this recovery in
`vast-start.sh`. Editing a template does not repair an already-created instance;
run `docker/vast-ssh.sh` through an available browser terminal or ask
the provider to run them inside the container. Apply this hook to the existing
instance only if its controls support editing it without recreating storage.
Vast's `update instance` operation is documented as recreating the instance;
do not assume it preserves container data. Its constrained `execute` interface
is not a general replacement for a shell. Do not bake shared host keys into
an image. See [Vast's custom-image example](https://docs.vast.ai/examples/ner/gliner2).

The isolated integration check `tests/test_vast_ssh.sh` exercises real SSH login
and rejection in an Ubuntu 24.04 container with openssh-server/client installed.
It requires `GEODISTILL_DISPOSABLE_SSH_TEST=1`; never run it on a rented server or
the host because it deliberately creates test keys and insecure fixture modes.

Provision persistent paths and Drive credentials over SSH. Inside Vast, use the
runtime directly; **do not run Docker Compose inside its container**:

```bash
cd /app
export PATH=/opt/train-venv/bin:$PATH
export TRAIN_IMAGE=phatle0106/geodistill-train@sha256:04b67813e90d69d7f71866dbb8becb339022b64994539b683caccc7f1070dea9
export BACKUP_REMOTE=gdrive:GeoLift_RT_runs/server
export RCLONE_CONFIG=/secrets/rclone.conf
python -m src.runtime doctor
python -m src.runtime prepare --run-id vast-qualification
python -m src.runtime run --run-id vast-qualification --stop-after-steps 10
python -m src.runtime run --run-id vast-qualification --stop-after-steps 20
python -m src.runtime backup --once
```

The second `run` command must report resume and advance optimizer step from 10 to 20.
For full training, omit `--stop-after-steps`. Keep RUN_ID consistent when starting
the backup service. `AUTO_TRAIN=1` enables startup training after credentials and
storage are ready; `BACKUP_REMOTE` also enables its background backup process.
The default startup only probes and leaves training for the operator to start.
Use `/opt/train-venv/bin/python` explicitly if the SSH shell changes PATH.
Export the image reference and backup settings explicitly in SSH-launched jobs;
do not assume template variables are inherited by an SSH login shell. The first
Vast qualification recorded the correct source hash but `image: unrecorded` for
this reason. Existing evidence is preserved; the commands above correct future
launches.

Acceptance before a long run: full-objective calibration passes, VRAM margin is
recorded, SIGTERM resume works, a disposable SIGKILL test resumes from a committed
checkpoint, validation produces finite primary metrics, Drive contains a completed
bundle, and restoring that bundle into an empty directory resumes successfully.
Then run one full epoch and inspect loss, throughput and validation metrics. Cloud
backup credentials and this target-host qualification are separate from local
image tests. No paid Vast instance is provisioned by these scripts.

## Maintaining the image

```bash
docker build -t phatle0106/geodistill-train:NEW_TAG .
docker run --rm --entrypoint python phatle0106/geodistill-train:NEW_TAG -m pytest tests -q
docker run --rm --gpus all phatle0106/geodistill-train:NEW_TAG doctor
docker push phatle0106/geodistill-train:NEW_TAG
```

Confirm repository privacy before publishing. Use a new immutable version tag for
subsequent releases and retain old image digests for resumption. The image includes
the actual working-tree code, including the existing encoder normalization fix;
`/opt/source.sha256` records the exact source hash and `/opt/python-packages.txt`
records installed dependency versions. Dataset archives, training outputs, cached
weights, credentials, notebooks, and `.git` are excluded from the build context.

For local Docker Desktop checks, avoid placing large datasets on Windows/OneDrive
bind mounts: metadata access and extraction can be much slower than native Linux
SSD storage. Calibration measures the actual current storage path, so do not reuse
a laptop's worker-count recommendation as a server performance result. The same
runtime can validate/download data with host Python after installing the training
dependencies, while GPU training runs in the container.
