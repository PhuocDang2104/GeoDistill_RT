from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_record import (
    artifact_identity,
    build_initial_record,
    build_protocol,
    collect_training_log,
    read_json,
    sha256_file,
    update_record,
    utc_now,
    write_json_atomic,
)
from src.utils import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical GeoLift train -> val -> anonymous test -> FP16 profile flow."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument(
        "--create-protocol-lock",
        action="store_true",
        help="Create the immutable split lock. Use only after manually auditing the intended split.",
    )
    parser.add_argument("--profile-warmup", type=int, default=100)
    parser.add_argument("--profile-runs", type=int, default=500)
    parser.add_argument("--skip-train", action="store_true", help="Debug/recovery only; marks the invocation accordingly.")
    parser.add_argument("--skip-infer", action="store_true", help="Debug/recovery only; record cannot become complete.")
    parser.add_argument("--skip-profile", action="store_true", help="Debug/recovery only; record cannot become complete.")
    return parser.parse_args()


def run_step(name: str, command: list[str], cwd: Path, timings: dict[str, float]) -> None:
    print(f"\n===== {name} =====", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True)
    timings[name] = time.perf_counter() - started


def maybe_read(path: Path) -> Any:
    return read_json(path) if path.is_file() else None


def mirror_record(record_path: Path, backup_root: Path | None) -> None:
    if backup_root is None:
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record_path, backup_root / record_path.name)


def restore_resume_logs(student_root: Path, backup_root: Path | None) -> None:
    if backup_root is None:
        return
    local_logs = student_root / "logs"
    backup_logs = backup_root / "logs"
    local_logs.mkdir(parents=True, exist_ok=True)
    for name in ("train_log.csv", "train_log.jsonl"):
        source = backup_logs / name
        target = local_logs / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg, paths = load_project_config(config_path)
    student_root = Path(paths["student_root"]).resolve()
    student_root.mkdir(parents=True, exist_ok=True)
    logs_root = student_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    record_path = student_root / "experiment_record.json"
    backup_value = cfg.get("outputs", {}).get("backup_root")
    backup_root = Path(str(backup_value)).resolve() if backup_value else None

    protocol = build_protocol(cfg, paths)
    lock_path = Path(args.protocol_lock).resolve()
    lock_payload = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "image_size": protocol["image_size"],
        "depth_scale": protocol["depth_scale"],
        "splits": {
            name: {
                key: value
                for key, value in fingerprint.items()
                if key != "path"
            }
            for name, fingerprint in protocol["split_files"].items()
        },
        "test_has_ground_truth": False,
    }
    if args.create_protocol_lock:
        if lock_path.exists():
            raise FileExistsError(f"Protocol lock already exists; refusing to overwrite: {lock_path}")
        write_json_atomic(lock_path, lock_payload)
        print(f"Created protocol lock: {lock_path}")
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"Protocol lock is required: {lock_path}. Audit the split, then rerun once with --create-protocol-lock."
        )
    locked = read_json(lock_path)
    if locked.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(
            "Current train/val/test split or image protocol differs from the immutable protocol lock. "
            f"locked={locked.get('protocol_sha256')} current={protocol['protocol_sha256']}"
        )
    protocol["lock"] = {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path),
        "protocol_sha256": locked["protocol_sha256"],
    }

    if record_path.is_file():
        existing = read_json(record_path)
        existing_hash = existing.get("protocol", {}).get("protocol_sha256")
        if existing_hash and existing_hash != protocol["protocol_sha256"]:
            raise RuntimeError("Existing run root contains a record from a different evaluation protocol.")
        update_record(record_path, {"status": "running", "protocol": protocol})
    else:
        write_json_atomic(record_path, build_initial_record(config_path, cfg, paths, protocol))
    mirror_record(record_path, backup_root)

    restore_resume_logs(student_root, backup_root)
    invocation_started_at = utc_now()
    invocation_started = time.perf_counter()
    step_timings: dict[str, float] = {}
    invocation: dict[str, Any] = {
        "started_at": invocation_started_at,
        "skip_train": bool(args.skip_train),
        "skip_infer": bool(args.skip_infer),
        "skip_profile": bool(args.skip_profile),
    }

    try:
        if not args.skip_train:
            run_step(
                "train",
                [sys.executable, "-m", "src.train_student", "--config", str(config_path)],
                PROJECT_ROOT,
                step_timings,
            )
        best_checkpoint = student_root / "checkpoints" / "best.pth"
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")

        if not args.skip_infer:
            for split in ("val", "test"):
                run_step(
                    f"infer_{split}",
                    [
                        sys.executable,
                        "-m",
                        "src.infer_student",
                        "--config",
                        str(config_path),
                        "--checkpoint",
                        str(best_checkpoint),
                        "--split",
                        split,
                    ],
                    PROJECT_ROOT,
                    step_timings,
                )

        test_dir = student_root / "test_predictions" / "benchmark_png"
        test_png = sorted(test_dir.glob("*.png")) if test_dir.is_dir() else []
        expected_test = int(protocol["split_files"]["test"]["count"])
        if not args.skip_infer and len(test_png) != expected_test:
            raise RuntimeError(f"Expected {expected_test} anonymous test PNGs, found {len(test_png)}")
        test_archive = student_root / "kitti_test_predictions.zip"
        if test_png:
            archive_base = test_archive.with_suffix("")
            shutil.make_archive(str(archive_base), "zip", test_dir)

        profile_path = logs_root / "geolift_component_profile.json"
        if not args.skip_profile:
            architecture = str(cfg.get("model", {}).get("architecture", "")).lower().replace("-", "_")
            profile_script = (
                "scripts/profile_geolift_s3.py"
                if architecture in {
                    "geolift_s3",
                    "geolift_s3_lite",
                    "geolift_s3_v1",
                    "geolift_decoder_v2",
                    "geolift_s3_decoder_v2",
                    "decoder_v2",
                }
                else "scripts/profile_geolift_s2.py"
            )
            run_step(
                "profile_fp16",
                [
                    sys.executable,
                    profile_script,
                    "--config",
                    str(config_path),
                    "--warmup",
                    str(args.profile_warmup),
                    "--runs",
                    str(args.profile_runs),
                    "--output",
                    str(profile_path),
                ],
                PROJECT_ROOT,
                step_timings,
            )

        training = collect_training_log(logs_root / "train_log.csv")
        validation = {
            "global_metrics": maybe_read(logs_root / "infer_val_metrics_global.json"),
            "macro_per_image_metrics": maybe_read(logs_root / "infer_val_metrics.json"),
            "runtime": maybe_read(logs_root / "infer_val_runtime.json"),
        }
        anonymous_test = {
            "has_ground_truth": False,
            "metrics": None,
            "prediction_count": len(test_png),
            "runtime": maybe_read(logs_root / "infer_test_runtime.json"),
            "submission_archive": artifact_identity(test_archive),
        }
        profile = maybe_read(profile_path)
        invocation.update(
            {
                "completed_at": utc_now(),
                "status": "complete",
                "wall_seconds": time.perf_counter() - invocation_started,
                "steps_seconds": step_timings,
            }
        )
        record = read_json(record_path)
        invocations = list(record.get("timings", {}).get("invocations", [])) + [invocation]
        complete = not (args.skip_infer or args.skip_profile)
        update_record(
            record_path,
            {
                "status": "complete" if complete else "incomplete",
                "training": training,
                "evaluation": {"validation": validation, "anonymous_test": anonymous_test},
                "performance": {"fp16_batch1_model_profile": profile},
                "timings": {
                    "invocations": invocations,
                    "wall_seconds_all_invocations": sum(float(item.get("wall_seconds", 0.0)) for item in invocations),
                },
                "artifacts": {
                    "best_checkpoint": artifact_identity(best_checkpoint),
                    "train_log": artifact_identity(logs_root / "train_log.csv"),
                    "profile": artifact_identity(profile_path),
                    "experiment_record": {"path": str(record_path)},
                },
            },
        )
        mirror_record(record_path, backup_root)
        print(f"\nCanonical experiment record: {record_path}")
    except Exception as exc:
        invocation.update(
            {
                "completed_at": utc_now(),
                "status": "failed",
                "wall_seconds": time.perf_counter() - invocation_started,
                "steps_seconds": step_timings,
                "error": str(exc),
            }
        )
        record = read_json(record_path)
        invocations = list(record.get("timings", {}).get("invocations", [])) + [invocation]
        errors = list(record.get("errors", [])) + [
            {"at": utc_now(), "message": str(exc), "traceback": traceback.format_exc()}
        ]
        update_record(
            record_path,
            {
                "status": "failed",
                "training": collect_training_log(logs_root / "train_log.csv"),
                "evaluation": {
                    "validation": {
                        "global_metrics": maybe_read(logs_root / "infer_val_metrics_global.json"),
                        "macro_per_image_metrics": maybe_read(logs_root / "infer_val_metrics.json"),
                        "runtime": maybe_read(logs_root / "infer_val_runtime.json"),
                    },
                    "anonymous_test": {
                        "has_ground_truth": False,
                        "metrics": None,
                        "runtime": maybe_read(logs_root / "infer_test_runtime.json"),
                    },
                },
                "performance": {
                    "fp16_batch1_model_profile": maybe_read(logs_root / "geolift_component_profile.json")
                },
                "timings": {"invocations": invocations},
                "errors": errors,
            },
        )
        mirror_record(record_path, backup_root)
        raise


if __name__ == "__main__":
    main()
