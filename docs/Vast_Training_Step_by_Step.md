# TAR2000 training on Vast.ai: operator guide

This guide uses the published GeoDistill image, one GPU, the full frozen TAR2000
dataset (1,600 train / 400 validation), local checkpoints, and Google Drive backup.
The baseline config runs 20 epochs. It retains its existing 30-epoch scheduler
horizon; do not change that setting midway through a run.

Commands marked **Laptop** run in Windows PowerShell. All other command blocks
run in **Bash inside the Vast instance**. Do not run Docker Compose inside the
Vast container. For a conventional Linux Docker host, use
[Server_Training_Runbook.md](Server_Training_Runbook.md) instead.

The short GPU qualification passed on the RTX 5060 Ti 16 GB instance on October
1, 2026. That was a small real-data smoke test, not a completed TAR2000 experiment.
This guide does not start training automatically.

## 1. Clone the repository on your laptop

**Laptop**, in a directory where you want a new checkout:

```powershell
git clone --branch augmentation https://github.com/PhuocDang2104/GeoDistill_RT.git
Set-Location GeoDistill_RT
```

If you already have the working checkout, use it instead. Do not overwrite local
changes. As of this guide's creation, the SSH startup repair and this guide are
local changes that have not been pushed; a new clone will not yet include them.
Use the existing working template/checkout for those files.

The clone is for configuration and documentation. The server's training source
and dependencies are already baked into `/app` in the image. Do not clone over
`/app`, run `pip install -U`, or replace model files while resuming an experiment.

## 2. Start or connect to the Vast instance

For the existing working instance, skip template creation. For a replacement:

| Setting | Value |
|---|---|
| Image repository/tag | `phatle0106/geodistill-train:server-v1` |
| Exact release | `phatle0106/geodistill-train@sha256:04b67813e90d69d7f71866dbb8becb339022b64994539b683caccc7f1070dea9` |
| Registry authentication | Docker Hub username and a read-capable access token, entered privately in Vast |
| Launch mode | Interactive shell server, SSH |
| Startup | Complete contents of the working checkout's `docker/vast-onstart.sh` |
| Automatic training | `AUTO_TRAIN=0` |
| Hardware | One compatible NVIDIA GPU; the qualified host was RTX 5060 Ti 16 GB |
| Storage | Prefer 80–100 GB; at least 60 GB free before dataset preparation |
| Template visibility | Private |

Use the pinned digest where the provider accepts it. `TRAIN_IMAGE` below is a
provenance label; setting it does not pull or change the running image.
Attach your laptop's public SSH key in Vast. Keep Docker tokens, private SSH keys,
and rclone credentials out of Git and template descriptions.

**Laptop** — current endpoint; replace port/IP if Vast recreates the instance:

```powershell
ssh -o IdentitiesOnly=yes -i "$env:USERPROFILE/.ssh/id_ed25519" -p 10495 root@61.76.96.126 -L 6006:localhost:6006
```

## 3. Define one full-training experiment

Use a new run ID, not a qualification/smoke run ID. For a deliberate new
experiment, change `RUN_ID` before its first preparation. For a resume, keep it.
Run the following once to create the operator environment:

```bash
mkdir -p /workspace/geodistill/{data,runs,cache,config,secrets}
chmod 700 /workspace/geodistill/secrets
test ! -e /workspace/geodistill/config/train.env && cat > /workspace/geodistill/config/train.env <<'EOF'
export ROOT=/workspace/geodistill
export RUN_ID=tar2000-kd-001
export TRAIN_MODE=baseline
export TRAIN_CONFIG=/app/configs/geolift_s3_lite_teacher_kd_tar2000.yaml
export DATASET_MANIFEST=/app/docker/tar2000-manifest.json
export TRAIN_IMAGE=phatle0106/geodistill-train@sha256:04b67813e90d69d7f71866dbb8becb339022b64994539b683caccc7f1070dea9
export RCLONE_CONFIG=/workspace/geodistill/secrets/rclone.conf
export BACKUP_REMOTE=gdrive:GeoLift_RT_runs/server
export HF_HOME=/workspace/geodistill/cache/huggingface
export TORCH_HOME=/workspace/geodistill/cache/torch
export MPLCONFIGDIR=/workspace/geodistill/cache/matplotlib
export PATH=/opt/train-venv/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
EOF
source /workspace/geodistill/config/train.env
mkdir -p "$ROOT/runs/$RUN_ID"
cd /app
test "$(cat /opt/source.sha256)" = f469deac24b3c19fa8a3e288f3262b26294860c855052fa442dbc48ffdfc449e
python -m src.runtime doctor --root "$ROOT/runs/$RUN_ID"
df -h "$ROOT" /dev/shm
```

The creation command preserves an existing `train.env`; inspect it before reuse.
The thread settings above match qualification. Hardware calibration chooses
feasible data-loader workers; it does not exhaustively tune CPU thread counts.
Check that CUDA is available and the doctor reports the expected GPU. Stop here
if the source check fails or there is insufficient storage.

Baseline preserves batch size 2, FP16 AMP, resolution, losses, and learning rate.
Preparation benchmarks feasible worker counts. `throughput` mode also searches
batch sizes, but should be a separate experiment with a new run ID. The baseline
uses the existing no-augmentation configuration; augmentation is not silently
added by the runtime.

## 4. Verify Google Drive access

The current qualified instance already has its `gdrive` credentials in the path
above. Verify them instead of configuring them again:

```bash
test -s "$RCLONE_CONFIG"
chmod 600 "$RCLONE_CONFIG"
rclone lsd gdrive:GeoLift_RT_runs
```

On a replacement instance only, transfer the authenticated rclone configuration
from the laptop before running that check. If the local config includes unrelated
remotes, export a separate config containing only the needed `gdrive` remote.
For a config containing only the intended remote, **Laptop**:

```powershell
scp -i "$env:USERPROFILE/.ssh/id_ed25519" -P 10495 "$env:APPDATA/rclone/rclone.conf" root@61.76.96.126:/workspace/geodistill/secrets/rclone.conf
```

Then repeat the remote permission/connectivity check. Do not paste tokens in chat.
Drive is backup storage; dataset extraction and training use the server's disk.

## 5. Create detached job launchers

These small scripts keep the same image/config/manifest/run identity across SSH
sessions. Create them on the server:

```bash
cat > /workspace/geodistill/config/job.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /workspace/geodistill/config/train.env
cd /app
action=${1:?Use prepare or run}
case "$action" in prepare|run) ;; *) exit 2 ;; esac
mkdir -p "$ROOT/runs/$RUN_ID"
exec python -m src.runtime "$action" \
  --manifest "$DATASET_MANIFEST" --config "$TRAIN_CONFIG" \
  --storage "$ROOT/data" --results "$ROOT/runs" \
  --run-id "$RUN_ID" --mode "$TRAIN_MODE" \
  >> "$ROOT/runs/$RUN_ID/$action.log" 2>&1
EOF

cat > /workspace/geodistill/config/backup.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /workspace/geodistill/config/train.env
cd /app
exec python -m src.runtime backup --results "$ROOT/runs" \
  --remote "$BACKUP_REMOTE" >> "$ROOT/backup.log" 2>&1
EOF
chmod 700 /workspace/geodistill/config/{job,backup}.sh
```

## 6. Download, validate, and calibrate before training

```bash
tmux new-session -d -s geodistill-prepare 'bash /workspace/geodistill/config/job.sh prepare'
tail -f "$ROOT/runs/$RUN_ID/prepare.log"
```

Press Ctrl+C to stop watching the log; the detached job continues. Preparation
downloads about 14.2 GB of archives, verifies checksums, extracts, validates all
2,000 samples and teacher coverage, caches encoder weights, and benchmarks loading.
Do not judge a slow first batch as a hang: qualification observed substantial
first-batch startup time. Use logs, GPU/process activity, and disk usage together.

Check progress at any time:

```bash
cat "$ROOT/runs/$RUN_ID/status.json"
tmux ls
```

Proceed only after `status.json` says `READY`. Inspect the selected configuration:

```bash
cat "$ROOT/runs/$RUN_ID/preflight.json"
cat "$ROOT/runs/$RUN_ID/calibration.json"
cat "$ROOT/runs/$RUN_ID/resolved_config.json"
```

Expect 1,600 training and 400 validation samples. `FAILED` means read `failure.txt`
and `prepare.log`, resolve the cause, then retry preparation. Do not run `prepare`
concurrently with training. Public Drive quota/download failures may require an
authenticated manifest; see the main runbook. Finish manifest changes before the
run is prepared, because its identity is part of the resume contract.

## 7. Start backup, then full training

```bash
tmux new-session -d -s geodistill-backup 'bash /workspace/geodistill/config/backup.sh'
tmux new-session -d -s geodistill-train 'bash /workspace/geodistill/config/job.sh run'
tail -f "$ROOT/runs/$RUN_ID/run.log"
```

This runs all 20 configured epochs, without the smoke test's step limit. Do not
start another trainer for the same run. You may disconnect SSH; tmux keeps the
jobs alive. A machine reboot does not restart these tmux sessions automatically.

## 8. Monitor training and backups

After reconnecting, load the environment first:

```bash
source /workspace/geodistill/config/train.env
cd /app
cat "$ROOT/runs/$RUN_ID/status.json"
tail -n 5 "$ROOT/runs/$RUN_ID/logs/train_log.csv"
tail -n 2 "$ROOT/runs/$RUN_ID/steps.jsonl"
cat "$ROOT/runs/$RUN_ID/backup_status.json"
nvidia-smi
df -h "$ROOT"
```

Some files appear only after the first step, validation, or checkpoint. Backup
polls once per minute; `backup_status.json` should eventually say `OK` with a recent
destination/time. For failures inspect `$ROOT/backup.log`; training can continue
locally despite a cloud backup error. A stale `OK` does not prove the latest
checkpoint was uploaded. Watch storage and Drive quota as snapshots accumulate.

Optional TensorBoard, using the SSH tunnel from step 2:

```bash
tmux new-session -d -s geodistill-tensorboard '/opt/train-venv/bin/tensorboard --logdir /workspace/geodistill/runs --host 127.0.0.1 --port 6006'
```

Open <http://localhost:6006> on the laptop. Check train/validation losses, validation
metrics, learning rate, skipped AMP steps, and memory. An isolated AMP skip is not
necessarily failure; persistent skips/nonfinite values need investigation.

## 9. Stop and resume safely when needed

Request a graceful stop:

```bash
touch "$ROOT/runs/$RUN_ID/STOP"
```

Wait for `status.json` to say `STOPPED` and the training tmux session to exit. It
finishes the current work boundary and saves state. Do not kill the instance while
waiting for a batch/validation to finish. Before releasing a host, also complete
the final backup verification in step 10.

Resume on the same server with the same run ID:

```bash
source /workspace/geodistill/config/train.env
cd /app
rm -f "$ROOT/runs/$RUN_ID/STOP"
tmux has-session -t geodistill-backup 2>/dev/null || tmux new-session -d -s geodistill-backup 'bash /workspace/geodistill/config/backup.sh'
tmux new-session -d -s geodistill-train 'bash /workspace/geodistill/config/job.sh run'
```

Use this only after confirming the previous trainer has exited. The same launch
also resumes after a crash. If it failed, inspect `failure.txt` before retrying.
Checkpoints include optimizer, scheduler, AMP scaler, RNG and data cursor. Normal
checkpoints occur every 100 consumed batches or five minutes, and at epoch ends
and graceful stops. A hard failure loses work since the last committed checkpoint.
Do not change source, dataset, mode, batch size, losses or schedule for a resume.

For a replacement/lost host, recreate the same image/environment and credentials,
then download a completed cloud bundle to a new directory:

```bash
# Replace BUNDLE_ID with an actual directory from this listing.
rclone lsf "$BACKUP_REMOTE/$RUN_ID/" --dirs-only
BUNDLE_ID=REPLACE_WITH_COMPLETED_BUNDLE_ID
rclone copy "$BACKUP_REMOTE/$RUN_ID/$BUNDLE_ID" "$ROOT/restore/$BUNDLE_ID"
test -f "$ROOT/restore/$BUNDLE_ID/COMPLETE.json"
python -m src.runtime restore --bundle "$ROOT/restore/$BUNDLE_ID" \
  --destination "$ROOT/runs/$RUN_ID"
```

The destination must be empty (do not run doctor/prepare into it before restore).
Restore verifies every bundle file's checksum. Reuse the original manifest/config
and start the backup/trainer from step 7. The runtime fetches missing data. Cloud
recovery can reach only the most recent successfully uploaded complete bundle.

## 10. Verify completion and final Drive backup

Training is finished only when `status.json` says `COMPLETE`, after best-checkpoint
validation. A tmux session disappearing alone is not proof of success.

```bash
cat "$ROOT/runs/$RUN_ID/status.json"
cat "$ROOT/runs/$RUN_ID/evaluation/best_val.json"
ls -lh "$ROOT/runs/$RUN_ID/checkpoints/"{best,last}.pth
tail -n 3 "$ROOT/runs/$RUN_ID/logs/train_log.csv"
```

Stop only the background backup session before a final synchronous upload, avoiding
concurrent backup locks. Training must already be `COMPLETE` (or `STOPPED` for a
planned pause):

```bash
tmux kill-session -t geodistill-backup 2>/dev/null || true
python -m src.runtime backup --results "$ROOT/runs" --remote "$BACKUP_REMOTE" --once
cat "$ROOT/runs/$RUN_ID/backup_status.json"

# Verify the latest reported cloud bundle against its immutable local files.
DEST=$(python -c 'import json,os; from pathlib import Path; p=Path(os.environ["ROOT"])/"runs"/os.environ["RUN_ID"]/"backup_status.json"; s=json.loads(p.read_text()); assert s["state"]=="OK",s; print(s["destination"])')
BUNDLE_ID=${DEST##*/}
rclone check "$ROOT/runs/$RUN_ID/backup_queue/$BUNDLE_ID" "$DEST" --one-way --exclude .uploaded
rclone lsf "$DEST" --include COMPLETE.json
```

If the synchronous command reports another process holds the backup lock, wait
for the previous uploader to exit and retry it. The synchronous upload must exit
successfully; the check must report no
differences, and the listing must contain `COMPLETE.json`. Do not release the
server on a failed check. Final validation is a validation-split score, not an
independent KITTI test benchmark.

## 11. Download results and release the rental

Archive the whole run, including checkpoint payloads and metadata. Do not copy
only `best.pth`/`last.pth`: keep their integrity metadata and experiment records too.
Keep enough free disk for this additional archive.

```bash
tar -czf "$ROOT/$RUN_ID-results.tgz" -C "$ROOT/runs" "$RUN_ID"
sha256sum "$ROOT/$RUN_ID-results.tgz"
```

**Laptop**, in the repository checkout (adjust ID/endpoint if changed):

```powershell
New-Item -ItemType Directory -Force server-runs | Out-Null
scp -i "$env:USERPROFILE/.ssh/id_ed25519" -P 10495 root@61.76.96.126:/workspace/geodistill/tar2000-kd-001-results.tgz ./server-runs/
Get-FileHash ./server-runs/tar2000-kd-001-results.tgz -Algorithm SHA256
```

Confirm that the SHA256 equals the server's value. The archive contains the run,
not the rclone credentials or dataset. Keep the original config/manifest/image
identity with the results. Then stop/release the instance through Vast. Container
disk can be lost when an instance is destroyed/recycled; it is not a persistent
volume, and an idle GPU rental can still cost money.

For an augmentation experiment or a larger prepared KITTI dataset, choose a new
run ID and appropriate config/manifest, then repeat preparation. See the main
runbook for the larger-dataset contract and offline-teacher requirements.
