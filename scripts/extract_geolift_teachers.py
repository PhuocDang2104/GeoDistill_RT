from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


SAMPLE_RE = re.compile(
    r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(?P<frame>\d{10})_(?P<camera>image_0[23])$"
)
GEOMETRY_SAMPLE_RE = re.compile(
    r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_(?P<camera>image_0[23])_(?P<frame>\d{10})$"
)


def canonical_sample_id(raw_id: str, kind: str) -> str | None:
    del kind  # All teacher archives may use either historical filename order.
    if SAMPLE_RE.match(raw_id):
        return raw_id
    match = GEOMETRY_SAMPLE_RE.match(raw_id)
    if match is not None:
        return f"{match.group('drive')}_image_{match.group('frame')}_{match.group('camera')}"
    return None


def _index(archive_path: Path, kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(".npz"):
                sample_id = canonical_sample_id(Path(member.name).stem, kind)
                if sample_id is None:
                    continue
                if sample_id in result:
                    raise RuntimeError(f"Duplicate canonical {kind} ID {sample_id} in {archive_path}")
                result[sample_id] = member.name
    return result


def _extract(archive_path: Path, index: dict[str, str], roles: dict[str, str], root: Path, kind: str) -> None:
    accepted = {
        "metric": ("D_cm", "D_full", "D_teacher"),
        "da": ("D_da_raw", "R_da", "R_i", "depth", "D_da_aligned", "D_full"),
        "geometry": ("R_G", "C_G"),
    }[kind]
    with tarfile.open(archive_path, "r:*") as archive:
        for sample_id, split in roles.items():
            source = archive.extractfile(archive.getmember(index[sample_id]))
            if source is None:
                raise RuntimeError(f"Could not extract {sample_id} from {archive_path}")
            if kind == "metric":
                target = root / "metric_coarse" / split / f"{sample_id}.npz"
            elif kind == "da":
                target = root / "depth_anything" / f"{split}_raw" / f"{sample_id}.npz"
            else:
                target = root / "geometry_fused" / split / f"{sample_id}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            with np.load(target) as data:
                if not any(key in data for key in accepted):
                    raise RuntimeError(f"Unexpected keys in {target}: {list(data.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 1,000 common teacher IDs and extract only those NPZ files.")
    parser.add_argument("--metric_tar", required=True)
    parser.add_argument("--da_tar", required=True)
    parser.add_argument("--geometry_tar", default="", help="Optional for legacy metric+DA extraction; required by final fused-geometry training.")
    parser.add_argument("--teacher_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--val_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("min_drives", "uniform"), default="min_drives")
    args = parser.parse_args()

    metric_tar, da_tar = Path(args.metric_tar), Path(args.da_tar)
    geometry_tar = Path(args.geometry_tar) if args.geometry_tar else None
    metric_index = _index(metric_tar, "metric")
    da_index = _index(da_tar, "da")
    geometry_index = _index(geometry_tar, "geometry") if geometry_tar is not None else {}
    common_ids = set(metric_index) & set(da_index)
    if geometry_tar is not None:
        common_ids &= set(geometry_index)
    common = sorted(common_ids)
    if len(common) < args.count:
        raise RuntimeError(f"Only {len(common)} valid common teacher IDs, need {args.count}")
    rng = random.Random(args.seed)
    by_drive: dict[str, list[str]] = defaultdict(list)
    for sample_id in common:
        match = SAMPLE_RE.match(sample_id)
        assert match is not None
        by_drive[match.group("drive")].append(sample_id)
    groups = list(by_drive.items())
    rng.shuffle(groups)
    if args.strategy == "min_drives":
        groups.sort(key=lambda pair: len(pair[1]), reverse=True)
    for _, ids in groups:
        rng.shuffle(ids)

    # Allocate whole drives to only one role. The final drive in each role may
    # be subsampled, but it is never reused by the other role.
    train_target = args.count - args.val_count
    train_ids: list[str] = []
    val_ids: list[str] = []
    for _, ids in groups:
        if len(train_ids) < train_target:
            train_ids.extend(ids[: train_target - len(train_ids)])
        elif len(val_ids) < args.val_count:
            val_ids.extend(ids[: args.val_count - len(val_ids)])
        if len(train_ids) == train_target and len(val_ids) == args.val_count:
            break
    if len(train_ids) != train_target or len(val_ids) != args.val_count:
        raise RuntimeError("Could not form drive-disjoint train/validation subsets with the requested counts")
    chosen = train_ids + val_ids
    roles = {sample_id: "train" for sample_id in train_ids}
    roles.update({sample_id: "val" for sample_id in val_ids})
    teacher_root = Path(args.teacher_root)
    _extract(metric_tar, metric_index, roles, teacher_root, "metric")
    _extract(da_tar, da_index, roles, teacher_root, "da")
    if geometry_tar is not None:
        _extract(geometry_tar, geometry_index, roles, teacher_root, "geometry")

    drives = sorted({SAMPLE_RE.match(sample_id).group("drive") for sample_id in chosen})  # type: ignore[union-attr]
    manifest = {
        "seed": args.seed,
        "strategy": args.strategy,
        "metric_archive_count": len(metric_index),
        "da_archive_count": len(da_index),
        "geometry_archive_count": len(geometry_index),
        "valid_common_count": len(common),
        "selected_count": len(chosen),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "unique_raw_drives": len(drives),
        "drive_overlap_train_val": len(
            {SAMPLE_RE.match(x).group("drive") for x in train_ids}  # type: ignore[union-attr]
            & {SAMPLE_RE.match(x).group("drive") for x in val_ids}  # type: ignore[union-attr]
        ),
        "drives": drives,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"train_ids", "val_ids", "drives"}}, indent=2))


if __name__ == "__main__":
    main()
