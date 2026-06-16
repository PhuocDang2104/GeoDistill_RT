#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/data/kitti_depth_completion}"
KEEP_ZIP="${KEEP_ZIP:-0}"

ANNOTATED_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
VELODYNE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip"
SELECTION_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"
DEVKIT_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_depth.zip"
RAW_DATA_BASE_URL="${RAW_DATA_BASE_URL:-https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data}"

DOWNLOAD_VELODYNE="${DOWNLOAD_VELODYNE:-0}"
DOWNLOAD_ANNOTATED="${DOWNLOAD_ANNOTATED:-0}"
DOWNLOAD_SELECTION="${DOWNLOAD_SELECTION:-0}"
DOWNLOAD_DEVKIT="${DOWNLOAD_DEVKIT:-1}"
DOWNLOAD_RAW_RGB="${DOWNLOAD_RAW_RGB:-0}"
DOWNLOAD_RAW_CALIB="${DOWNLOAD_RAW_CALIB:-1}"
RAW_DRIVES_FILE="${RAW_DRIVES_FILE:-}"
RAW_DRIVES="${RAW_DRIVES:-}"
RAW_MAX_DRIVES="${RAW_MAX_DRIVES:-0}"

mkdir -p "${ROOT}"
ROOT="$(cd "${ROOT}" && pwd)"
cd "${ROOT}"

echo "Install tools if missing..."
if ! command -v unzip >/dev/null 2>&1 || ! command -v wget >/dev/null 2>&1; then
  apt update
  apt install -y wget unzip ca-certificates
fi

download_and_unzip () {
  local url="$1"
  local zip_name="$2"

  echo
  echo "=================================================="
  echo "Downloading: ${zip_name}"
  echo "URL: ${url}"
  echo "Target: ${ROOT}"
  echo "=================================================="

  if [ ! -f "${zip_name}" ]; then
    wget -c --progress=bar:force:noscroll "${url}" -O "${zip_name}"
  else
    echo "Found existing ${zip_name}, resuming/checking with wget -c..."
    wget -c --progress=bar:force:noscroll "${url}" -O "${zip_name}"
  fi

  echo "Unzipping ${zip_name}..."
  unzip -q -o "${zip_name}"

  if [ "${KEEP_ZIP}" != "1" ]; then
    echo "Removing ${zip_name} to save disk..."
    rm -f "${zip_name}"
  fi
}

download_zip () {
  local url="$1"
  local zip_name="$2"

  if [ ! -f "${zip_name}" ]; then
    wget -c --progress=bar:force:noscroll "${url}" -O "${zip_name}"
  else
    echo "Found existing ${zip_name}, resuming/checking with wget -c..."
    wget -c --progress=bar:force:noscroll "${url}" -O "${zip_name}"
  fi
}

raw_drive_missing_count () {
  local drive="$1"
  python - "${ROOT}" "${drive}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
drive = sys.argv[2]
date = drive[:10]
missing = 0

for split in ("train", "val"):
    drive_root = root / split / drive / "proj_depth"
    if not drive_root.is_dir():
        continue
    for source in ("groundtruth", "velodyne_raw"):
        for cam in ("image_02", "image_03"):
            for depth_path in (drive_root / source / cam).glob("*.png"):
                rgb_path = root / date / drive / cam / "data" / depth_path.name
                if not rgb_path.exists():
                    missing += 1

print(missing)
PY
}

extract_raw_rgb_for_drive () {
  local zip_name="$1"
  local drive="$2"

  python - "${ROOT}" "${zip_name}" "${drive}" <<'PY'
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1])
zip_path = Path(sys.argv[2])
drive = sys.argv[3]
date = drive[:10]

wanted: set[str] = set()
for split in ("train", "val"):
    drive_root = root / split / drive / "proj_depth"
    if not drive_root.is_dir():
        continue
    for source in ("groundtruth", "velodyne_raw"):
        for cam in ("image_02", "image_03"):
            for depth_path in (drive_root / source / cam).glob("*.png"):
                wanted.add(f"{date}/{drive}/{cam}/data/{depth_path.name}")

if not wanted:
    raise SystemExit(f"No depth frames found for {drive}")

with zipfile.ZipFile(zip_path) as zf:
    names = set(zf.namelist())
    missing = sorted(wanted - names)
    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(f"{zip_path.name} is missing {len(missing)} expected RGB entries, for example:\n{preview}")

    extracted = 0
    skipped = 0
    for name in sorted(wanted):
        target = root / name
        if target.exists():
            skipped += 1
            continue
        zf.extract(name, root)
        extracted += 1

print(f"Extracted {extracted} RGB frames for {drive}; skipped {skipped} existing frames.")
PY
}

download_raw_calib () {
  local date="$1"
  local zip_name="${date}_calib.zip"
  local url="${RAW_DATA_BASE_URL}/${zip_name}"

  if [ -f "${ROOT}/${date}/calib_cam_to_cam.txt" ]; then
    echo "Calibration for ${date} already exists."
    return
  fi

  echo
  echo "Downloading raw calibration: ${date}"
  echo "URL: ${url}"
  download_zip "${url}" "${zip_name}"
  unzip -q -o "${zip_name}"

  if [ "${KEEP_ZIP}" != "1" ]; then
    rm -f "${zip_name}"
  fi
}

download_raw_rgb () {
  local drive_list
  drive_list="$(mktemp)"

  if [ -n "${RAW_DRIVES_FILE}" ]; then
    sed '/^[[:space:]]*$/d' "${RAW_DRIVES_FILE}" | sort -u > "${drive_list}"
  elif [ -n "${RAW_DRIVES}" ]; then
    printf '%s\n' ${RAW_DRIVES} | sed '/^[[:space:]]*$/d' | sort -u > "${drive_list}"
  else
    find "${ROOT}/train" "${ROOT}/val" -mindepth 1 -maxdepth 1 -type d -name '*_sync' -printf '%f\n' 2>/dev/null | sort -u > "${drive_list}"
  fi

  if [ "${RAW_MAX_DRIVES}" != "0" ]; then
    local limited_list
    limited_list="$(mktemp)"
    head -n "${RAW_MAX_DRIVES}" "${drive_list}" > "${limited_list}"
    mv "${limited_list}" "${drive_list}"
  fi

  local count
  count="$(wc -l < "${drive_list}" | tr -d ' ')"
  if [ "${count}" = "0" ]; then
    echo "No raw KITTI drives found to download under ${ROOT}/{train,val}."
    rm -f "${drive_list}"
    return
  fi

  echo
  echo "Raw RGB drive count: ${count}"
  echo "Raw RGB source: ${RAW_DATA_BASE_URL}"

  if [ "${DOWNLOAD_RAW_CALIB}" = "1" ]; then
    cut -c1-10 "${drive_list}" | sort -u | while read -r date; do
      download_raw_calib "${date}"
    done
  fi

  local drive
  while read -r drive; do
    local stem
    local zip_name
    local url
    local missing

    stem="${drive%_sync}"
    zip_name="${drive}.zip"
    url="${RAW_DATA_BASE_URL}/${stem}/${zip_name}"
    missing="$(raw_drive_missing_count "${drive}")"

    echo
    echo "=================================================="
    echo "Raw RGB drive: ${drive}"
    echo "Missing RGB frame count: ${missing}"
    echo "URL: ${url}"
    echo "=================================================="

    if [ "${missing}" = "0" ]; then
      echo "Skipping ${drive}; matching RGB frames already exist."
      continue
    fi

    download_zip "${url}" "${zip_name}"
    extract_raw_rgb_for_drive "${zip_name}" "${drive}"

    if [ "${KEEP_ZIP}" != "1" ]; then
      echo "Removing ${zip_name} to save disk..."
      rm -f "${zip_name}"
    fi
  done < "${drive_list}"

  rm -f "${drive_list}"
}

echo "Dataset root: ${ROOT}"
echo "Disk before:"
df -h "${ROOT}" || true

# Annotated dense depth maps. Unzip into the same ROOT as velodyne_raw so
# train/<drive>/proj_depth/{groundtruth,velodyne_raw}/... stay aligned.
if [ "${DOWNLOAD_ANNOTATED}" = "1" ]; then
  download_and_unzip "${ANNOTATED_URL}" "data_depth_annotated.zip"
fi

# Sparse LiDAR depth maps. Unzip into the same ROOT as annotated maps.
if [ "${DOWNLOAD_VELODYNE}" = "1" ]; then
  download_and_unzip "${VELODYNE_URL}" "data_depth_velodyne.zip"
fi

# Optional selected validation/test split
if [ "${DOWNLOAD_SELECTION}" = "1" ]; then
  download_and_unzip "${SELECTION_URL}" "data_depth_selection.zip"
fi

# Small devkit
if [ "${DOWNLOAD_DEVKIT}" = "1" ]; then
  download_and_unzip "${DEVKIT_URL}" "devkit_depth.zip"
fi

# Raw KITTI RGB frames. Each raw sync zip contains more than RGB, so this
# extracts only image_02/image_03 frames that correspond to existing depth maps.
if [ "${DOWNLOAD_RAW_RGB}" = "1" ]; then
  download_raw_rgb
fi

echo
echo "Disk after:"
df -h "${ROOT}" || true

echo
echo "Top-level structure:"
find "${ROOT}" -maxdepth 3 -type d | sort | head -80 || true

echo
echo "Annotated ground-truth PNG count:"
find "${ROOT}" -type f -path "*/proj_depth/groundtruth/*/*.png" | wc -l

echo
echo "Sparse velodyne PNG count:"
find "${ROOT}" -type f -path "*/proj_depth/velodyne_raw/*/*.png" | wc -l

echo
echo "Raw RGB PNG count:"
find "${ROOT}" -type f \( -path "*/image_02/data/*.png" -o -path "*/image_03/data/*.png" \) | wc -l

echo
echo "Example files:"
find "${ROOT}" -type f -path "*/proj_depth/groundtruth/*/*.png" | head -10 || true

echo
echo "Done."
