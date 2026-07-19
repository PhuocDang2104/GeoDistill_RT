from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_RE = re.compile(
    r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(?P<frame>\d{10})_(?P<camera>image_0[23])$"
)


def _split_ids(path: Path) -> list[str]:
    return [line.split(maxsplit=1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _coverage(ids: list[str], directory: Path) -> tuple[int, list[str]]:
    missing = [sample_id for sample_id in ids if not (directory / f"{sample_id}.npz").exists()]
    return len(ids) - len(missing), missing


def _summary(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {}
    flat = np.concatenate([value[np.isfinite(value)].reshape(-1) for value in values])
    return {
        "finite_values": int(flat.size),
        "min": float(flat.min()),
        "q01": float(np.quantile(flat, 0.01)),
        "median": float(np.median(flat)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extracted metric/DA/fused-geometry train and validation teachers.")
    parser.add_argument("--teacher_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--train_split", default="train_800.txt")
    parser.add_argument("--val_split", default="val_200.txt")
    parser.add_argument("--samples_per_split", type=int, default=16)
    parser.add_argument("--min_coverage", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    teacher_root, split_root = Path(args.teacher_root), Path(args.split_root)
    split_ids = {
        "train": _split_ids(split_root / args.train_split),
        "val": _split_ids(split_root / args.val_split),
    }
    overlap = set(split_ids["train"]) & set(split_ids["val"])
    train_drives = {CANONICAL_RE.match(value).group("drive") for value in split_ids["train"] if CANONICAL_RE.match(value)}
    val_drives = {CANONICAL_RE.match(value).group("drive") for value in split_ids["val"] if CANONICAL_RE.match(value)}
    drive_overlap = train_drives & val_drives
    rng = random.Random(args.seed)
    result: dict[str, Any] = {
        "id_overlap": sorted(overlap),
        "drive_overlap": sorted(drive_overlap),
        "splits": {},
    }
    contract_ok = not overlap and not drive_overlap

    for split, ids in split_ids.items():
        directories = {
            "metric": teacher_root / "metric_coarse" / split,
            "da_raw": teacher_root / "depth_anything" / f"{split}_raw",
            "geometry_fused": teacher_root / "geometry_fused" / split,
        }
        coverage: dict[str, Any] = {}
        for role, directory in directories.items():
            present, missing = _coverage(ids, directory)
            ratio = present / max(1, len(ids))
            coverage[role] = {
                "present": present,
                "total": len(ids),
                "ratio": ratio,
                "missing_preview": missing[:10],
            }
            contract_ok = contract_ok and ratio >= args.min_coverage

        inspected_ids = ids.copy()
        rng.shuffle(inspected_ids)
        inspected_ids = inspected_ids[: min(args.samples_per_split, len(inspected_ids))]
        r_values: list[np.ndarray] = []
        c_values: list[np.ndarray] = []
        sample_records = []
        for sample_id in inspected_ids:
            path = directories["geometry_fused"] / f"{sample_id}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                keys = list(data.files)
                if "R_G" not in data or "C_G" not in data:
                    contract_ok = False
                    sample_records.append({"id": sample_id, "keys": keys, "contract_ok": False})
                    continue
                r_g = np.asarray(data["R_G"], dtype=np.float32)
                c_g = np.asarray(data["C_G"], dtype=np.float32)
                sample_ok = bool(
                    r_g.ndim == 2
                    and c_g.shape == r_g.shape
                    and np.isfinite(r_g).all()
                    and np.isfinite(c_g).all()
                    and float(c_g.min()) >= 0.0
                    and float(c_g.max()) <= 1.0
                )
                contract_ok = contract_ok and sample_ok
                r_values.append(r_g)
                c_values.append(c_g)
                sample_records.append(
                    {
                        "id": sample_id,
                        "keys": keys,
                        "shape": list(r_g.shape),
                        "contract_ok": sample_ok,
                    }
                )
        result["splits"][split] = {
            "count": len(ids),
            "coverage": coverage,
            "inspected_samples": sample_records,
            "R_G": _summary(r_values),
            "C_G": _summary(c_values),
        }

    result["contract_ok"] = bool(contract_ok)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not contract_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

