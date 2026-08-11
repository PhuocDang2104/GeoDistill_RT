from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_drive(sample_id: str) -> str:
    return sample_id.split("_sync_image_", 1)[0] if "_sync_image_" in sample_id else sample_id


def split_fingerprint(path: str | Path) -> dict[str, Any]:
    split_path = Path(path).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    lines = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_ids = [line.split()[0] for line in lines]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in split: {split_path}")
    raw_drives = sorted({_raw_drive(sample_id) for sample_id in sample_ids})
    return {
        "path": str(split_path),
        "count": len(lines),
        "file_sha256": sha256_file(split_path),
        "ordered_sample_ids_sha256": hashlib.sha256(("\n".join(sample_ids) + "\n").encode("utf-8")).hexdigest(),
        "sample_set_sha256": hashlib.sha256(("\n".join(sorted(sample_ids)) + "\n").encode("utf-8")).hexdigest(),
        "raw_drive_count": len(raw_drives),
        "raw_drives_sha256": hashlib.sha256(("\n".join(raw_drives) + "\n").encode("utf-8")).hexdigest(),
        "sample_ids": sample_ids,
        "raw_drives": raw_drives,
    }


def build_protocol(cfg: Mapping[str, Any], paths: Mapping[str, str]) -> dict[str, Any]:
    split_root = Path(paths["split_root"])
    splits = {
        name: split_fingerprint(split_root / paths[f"{name}_split"])
        for name in ("train", "val", "test")
    }
    train_ids, val_ids, test_ids = (set(splits[name]["sample_ids"]) for name in ("train", "val", "test"))
    train_drives = set(splits["train"]["raw_drives"])
    val_drives = set(splits["val"]["raw_drives"])
    overlap = {
        "train_val_samples": sorted(train_ids & val_ids),
        "train_test_samples": sorted(train_ids & test_ids),
        "val_test_samples": sorted(val_ids & test_ids),
        "train_val_raw_drives": sorted(train_drives & val_drives),
    }
    if any(overlap.values()):
        raise ValueError(f"Split leakage detected: {overlap}")
    for split in splits.values():
        split.pop("sample_ids")
        split.pop("raw_drives")
    stable = {
        "image_size": list(cfg.get("data", {}).get("image_size", [])),
        "depth_scale": float(cfg.get("data", {}).get("depth_scale", 256.0)),
        "splits": {
            name: {
                key: value
                for key, value in split.items()
                if key not in {"path"}
            }
            for name, split in splits.items()
        },
    }
    return {
        **stable,
        "protocol_sha256": sha256_json(stable),
        "split_files": splits,
        "leakage_check": {"passed": True, **{key: 0 for key in overlap}},
        "test_has_ground_truth": False,
        "metric_aggregation": "global valid pixels; macro per-image metrics are diagnostic only",
    }


def _git_identity(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(status),
        "dirty_file_count": len(status.splitlines()) if status else 0,
    }


def build_initial_record(
    config_path: str | Path,
    cfg: Mapping[str, Any],
    paths: Mapping[str, str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    project_root = Path(str(paths.get("project_root", Path.cwd()))).resolve()
    config_path = Path(config_path).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "identity": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "seed": int(cfg.get("seed", 42)),
            "model_name": cfg.get("model", {}).get("name"),
            "architecture": cfg.get("model", {}).get("architecture"),
            "git": _git_identity(project_root),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "protocol": dict(protocol),
        "training": {
            "planned_epochs": int(cfg.get("train", {}).get("epochs", 0)),
            "batch_size": int(cfg.get("train", {}).get("batch_size", 0)),
            "history": [],
        },
        "evaluation": {},
        "performance": {},
        "timings": {"invocations": []},
        "artifacts": {},
        "errors": [],
    }


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def update_record(path: str | Path, patch: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    record = read_json(target) if target.is_file() else {}
    _deep_merge(record, patch)
    record["updated_at"] = utc_now()
    write_json_atomic(target, record)
    return record


def _coerce(value: str) -> Any:
    if value == "":
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def collect_training_log(path: str | Path) -> dict[str, Any]:
    log_path = Path(path)
    if not log_path.is_file():
        return {"history": [], "observed_epochs": 0}
    with log_path.open("r", encoding="utf-8", newline="") as stream:
        history = [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    valid = [row for row in history if isinstance(row.get("val_rmse"), (int, float))]
    best = min(valid, key=lambda row: float(row["val_rmse"])) if valid else None
    return {
        "history": history,
        "observed_epochs": len(history),
        "best_epoch": best.get("epoch") if best else None,
        "best_metrics": best,
        "epoch_time_seconds_total": sum(float(row.get("epoch_total_seconds") or 0.0) for row in history),
    }


def artifact_identity(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    return {"path": str(target.resolve()), "size_bytes": target.stat().st_size, "sha256": sha256_file(target)}

