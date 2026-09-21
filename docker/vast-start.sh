#!/usr/bin/env bash
set -euo pipefail
# Vast may replace the image ENTRYPOINT. Select this script as on-start command.
# Set AUTO_TRAIN=1 only after provisioning /config/dataset.json and Drive credentials.
cd /app
mkdir -p /data /runs /cache /config /secrets
python -m src.runtime doctor --root /runs > /runs/doctor.json
if [[ "${AUTO_TRAIN:-0}" == "1" ]]; then
  if [[ -n "${BACKUP_REMOTE:-}" ]]; then
    python -m src.runtime backup > /runs/backup-console.log 2>&1 &
  fi
  exec python -m src.runtime run
fi
echo 'GeoDistill ready. Provision dataset manifest and rclone config, then run python -m src.runtime prepare.'
