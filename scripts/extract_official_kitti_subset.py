from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path


SAMPLE_RE = re.compile(
    r"^(?P<date>\d{4}_\d{2}_\d{2})_drive_(?P<drive_num>\d{4})_sync_image_(?P<frame>\d{10})_(?P<camera>image_0[23])$"
)
RAW_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


def _wget(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-q", "-c", url, "-O", str(output)], check=True)


def _parse(sample_id: str) -> dict[str, str]:
    match = SAMPLE_RE.match(sample_id)
    if match is None:
        raise ValueError(f"Unsupported KITTI teacher sample ID: {sample_id}")
    result = match.groupdict()
    result["drive"] = f"{result['date']}_drive_{result['drive_num']}_sync"
    return result


def _zip_index(path: Path, source: str) -> dict[tuple[str, str, str], str]:
    index: dict[tuple[str, str, str], str] = {}
    marker = f"/proj_depth/{source}/"
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            normalized = "/" + name.lstrip("/")
            if marker not in normalized or not normalized.endswith(".png"):
                continue
            parts = name.split("/")
            drive = next((part for part in parts if part.endswith("_sync")), None)
            if drive is None:
                continue
            camera = parts[-2]
            frame = Path(parts[-1]).stem
            index[(drive, camera, frame)] = name
    return index


def _extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, open(destination, "wb") as output:
        while chunk := source.read(8 * 1024 * 1024):
            output.write(chunk)


def _intrinsics(calib_path: Path, camera: str) -> tuple[float, float, float, float]:
    suffix = camera[-2:]
    accepted = {f"P_rect_{suffix}", f"P{int(suffix)}", f"P_rect_{int(suffix)}"}
    for raw in calib_path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw:
            continue
        key, values = raw.split(":", 1)
        if key.strip() in accepted:
            nums = [float(value) for value in values.split()]
            return nums[0], nums[5], nums[2], nums[6]
    raise RuntimeError(f"No projection for {camera} in {calib_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract official KITTI inputs only for the selected teacher IDs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotated_zip", required=True)
    parser.add_argument("--velodyne_zip", required=True)
    parser.add_argument("--selection_zip", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--download_dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    roles = {sample_id: "train" for sample_id in manifest["train_ids"]}
    roles.update({sample_id: "val" for sample_id in manifest["val_ids"]})
    parsed = {sample_id: _parse(sample_id) for sample_id in roles}
    root, split_root, downloads = Path(args.data_root), Path(args.split_root), Path(args.download_dir)
    root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    depth_sources = {
        "groundtruth": (Path(args.annotated_zip), _zip_index(Path(args.annotated_zip), "groundtruth")),
        "velodyne_raw": (Path(args.velodyne_zip), _zip_index(Path(args.velodyne_zip), "velodyne_raw")),
    }
    locations: dict[str, dict[str, Path]] = {sample_id: {} for sample_id in roles}
    for source_name, (zip_path, index) in depth_sources.items():
        with zipfile.ZipFile(zip_path) as archive:
            for sample_id, fields in parsed.items():
                key = (fields["drive"], fields["camera"], fields["frame"])
                if key not in index:
                    raise RuntimeError(f"{key} not found in official {source_name} archive")
                member = index[key]
                split_name = member.split("/", 1)[0]
                target = root / split_name / fields["drive"] / "proj_depth" / source_name / fields["camera"] / f"{fields['frame']}.png"
                _extract_member(archive, member, target)
                locations[sample_id][source_name] = target
                locations[sample_id]["official_split"] = Path(split_name)

    for date in sorted({fields["date"] for fields in parsed.values()}):
        calib_zip = downloads / f"{date}_calib.zip"
        _wget(f"{RAW_BASE}/{date}_calib.zip", calib_zip)
        with zipfile.ZipFile(calib_zip) as archive:
            member = f"{date}/calib_cam_to_cam.txt"
            _extract_member(archive, member, root / member)
        calib_zip.unlink(missing_ok=True)

    by_drive: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for sample_id, fields in parsed.items():
        by_drive.setdefault(fields["drive"], []).append((sample_id, fields))
    for drive, samples in sorted(by_drive.items()):
        date = samples[0][1]["date"]
        raw_zip = downloads / f"{drive}.zip"
        stem = drive.removesuffix("_sync")
        _wget(f"{RAW_BASE}/{stem}/{drive}.zip", raw_zip)
        with zipfile.ZipFile(raw_zip) as archive:
            names = set(archive.namelist())
            for sample_id, fields in samples:
                member = f"{date}/{drive}/{fields['camera']}/data/{fields['frame']}.png"
                if member not in names:
                    raise RuntimeError(f"Missing raw RGB {member}")
                target = root / member
                _extract_member(archive, member, target)
                locations[sample_id]["rgb"] = target
        raw_zip.unlink(missing_ok=True)

    with zipfile.ZipFile(args.selection_zip) as archive:
        test_members = [name for name in archive.namelist() if "/test_depth_completion_anonymous/" in "/" + name and not name.endswith("/")]
        for member in test_members:
            # Official archive contains a depth_selection/ top folder; keep it under root.
            _extract_member(archive, member, root / member)

    def relative(path: Path) -> str:
        return path.relative_to(root).as_posix()

    split_lines: dict[str, list[str]] = {"train_800.txt": [], "val_200.txt": []}
    for sample_id, role in roles.items():
        fields = parsed[sample_id]
        calib = root / fields["date"] / "calib_cam_to_cam.txt"
        fx, fy, cx, cy = _intrinsics(calib, fields["camera"])
        record = locations[sample_id]
        line = (
            f"{sample_id} {relative(record['rgb'])} {relative(record['velodyne_raw'])} {relative(record['groundtruth'])} "
            f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
        )
        split_lines["train_800.txt" if role == "train" else "val_200.txt"].append(line)

    selection_root_candidates = [root / "depth_selection", root]
    selection_root = next((candidate for candidate in selection_root_candidates if (candidate / "test_depth_completion_anonymous").is_dir()), None)
    if selection_root is None:
        raise RuntimeError("Official test_depth_completion_anonymous was not extracted")
    test_root = selection_root / "test_depth_completion_anonymous"
    test_lines = []
    for rgb in sorted((test_root / "image").glob("*.png")):
        sparse = test_root / "velodyne_raw" / rgb.name
        intrinsics = test_root / "intrinsics" / f"{rgb.stem}.txt"
        if not sparse.exists() or not intrinsics.exists():
            raise RuntimeError(f"Incomplete official test sample {rgb.stem}")
        test_lines.append(f"{rgb.stem} {relative(rgb)} {relative(sparse)} none {relative(intrinsics)}")
    if len(test_lines) != 1000:
        raise RuntimeError(f"Expected official 1,000-image test split, found {len(test_lines)}")
    split_lines["test_1000.txt"] = test_lines
    for name, lines in split_lines.items():
        (split_root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: len(lines) for name, lines in split_lines.items()}, indent=2))


if __name__ == "__main__":
    main()

