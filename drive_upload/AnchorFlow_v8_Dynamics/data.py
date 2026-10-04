"""Drive TAR preparation and a teacher-free validation/test data interface."""
from __future__ import annotations

import io
import json
import hashlib
import re
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)
SIZE = (352, 1216)
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"
ID_NEW = re.compile(r"^(\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(\d{10})_(image_0[23])$")
ID_OLD = re.compile(r"^(\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_(image_0[23])_(\d{10})$")


def canonical_id(name):
    stem = Path(name).stem
    if ID_NEW.fullmatch(stem):
        return stem
    old = ID_OLD.fullmatch(stem)
    return f"{old[1]}_image_{old[3]}_{old[2]}" if old else None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    partial.replace(path)


def read_lines(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_extract(archive_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def validate_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    train, val = manifest["train_ids"], manifest["val_ids"]
    if (len(train), len(val), len(set(train + val))) != (1600, 400, 2000):
        raise RuntimeError("Expected exactly 1600 train + 400 distinct val IDs")
    for sid in train + val:
        if not ID_NEW.fullmatch(sid):
            raise RuntimeError(f"Unexpected sample ID: {sid}")
    drives = lambda ids: {ID_NEW.fullmatch(s)[1] for s in ids}
    if drives(train) & drives(val):
        raise RuntimeError("Raw drive overlap between train and val")
    return train, val


def verify_bundle(source, root):
    train, val = validate_manifest(source / "selected_2000_ids.json")
    if digest(source / "selected_2000_ids.json") != digest(root / "selected_2000_ids.json"):
        raise RuntimeError("Bundle manifest differs from selected_2000_ids.json on Drive")
    for split, expected in (("train_1600", train), ("val_400", val)):
        rows = [line.split() for line in read_lines(root / "splits" / f"{split}.txt")]
        if [row[0] for row in rows] != expected:
            raise RuntimeError(f"Split order/IDs changed: {split}")
        for row in rows:
            if len(row) != 8:
                raise RuntimeError(f"Unexpected split row: {row}")
            for relative in row[1:4]:
                path = (root / relative).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise RuntimeError(f"Missing or unsafe paired KITTI path: {relative}")
    return train, val


def teacher_arrays(payload, size=SIZE):
    if not {"D_cm", "C_cm"}.issubset(payload.files):
        raise RuntimeError(f"Metric archive must contain D_cm and C_cm, got {payload.files}")
    d, c = [np.asarray(payload[key], np.float32).squeeze() for key in ("D_cm", "C_cm")]
    if d.ndim != 2 or c.shape != d.shape:
        raise RuntimeError(f"Teacher shape mismatch: {d.shape}, {c.shape}")
    if not np.isfinite(d).all() or not np.isfinite(c).all() or c.min() < 0 or c.max() > 1:
        raise RuntimeError("Nonfinite teacher or C_cm outside [0,1]")
    valid = (d > 0.1) & (d < 120)
    if d.shape != size:
        weight = cv2.resize(valid.astype(np.float32), size[::-1], interpolation=cv2.INTER_LINEAR)
        numerator = cv2.resize(np.where(valid, d, 0), size[::-1], interpolation=cv2.INTER_LINEAR)
        d = numerator / np.maximum(weight, 1e-6)
        c = cv2.resize(c * valid, size[::-1], interpolation=cv2.INTER_LINEAR)
        valid = (weight > 0.99) & (d > 0.1) & (d < 120)
    return np.stack((np.where(valid, d, 0), np.where(valid, c, 0))).astype(np.float32)


def extract_metric(archive_path, train_ids, val_ids, cache, threshold):
    """Read all selected records once; cache only train as uncompressed float32."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    expected = set(train_ids + val_ids)
    train_set = set(train_ids)
    signature = {"archive_bytes": archive_path.stat().st_size,
                 "archive_mtime_ns": archive_path.stat().st_mtime_ns,
                 "selected_sha256": hashlib.sha256("\n".join(train_ids + val_ids).encode()).hexdigest(),
                 "schema": 3, "shape": list(SIZE), "confidence_threshold": threshold}
    report_path = cache / "teacher_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["signature"] == signature and all((cache / f"{sid}.npy").is_file() for sid in train_ids):
            print("Metric cache restored: 1600 train records; val/test never load teachers", flush=True)
            return report
    seen = set()
    summaries = []
    selected_content = hashlib.sha256()
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            sid = canonical_id(member.name)
            if sid not in expected:
                continue
            if sid in seen:
                raise RuntimeError(f"Duplicate canonical teacher ID: {sid}")
            seen.add(sid)
            with archive.extractfile(member) as stream:
                raw = stream.read()
            selected_content.update(sid.encode())
            selected_content.update(hashlib.sha256(raw).digest())
            with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
                arrays = teacher_arrays(payload)
            valid = (arrays[0] > 0.1) & (arrays[0] < 120)
            summaries.append((float(valid.mean()), float((valid & (arrays[1] >= threshold)).mean())))
            if sid in train_set:
                target = cache / f"{sid}.npy"
                partial = target.with_suffix(".npy.partial")
                with partial.open("wb") as handle:
                    np.save(handle, arrays, allow_pickle=False)
                partial.replace(target)
            if len(seen) % 200 == 0:
                print(f"Metric D_cm/C_cm inspected: {len(seen)}/2000", flush=True)
    if seen != expected:
        raise RuntimeError(f"Metric TAR missing {len(expected - seen)} IDs: {sorted(expected - seen)[:3]}")
    report = {"signature": signature, "inspected": len(expected), "cached_train": len(train_ids), "cached_val": 0,
              "selected_teacher_content_sha256": selected_content.hexdigest(),
              "mean_valid_ratio": float(np.mean(summaries, axis=0)[0]),
              "mean_confident_ratio": float(np.mean(summaries, axis=0)[1]),
              "teacher_is_privileged_training_target": True,
              "GT_overlap_is_not_independent_teacher_accuracy": True}
    write_json(report_path, report)
    return report


def test_rows(root):
    root = Path(root)
    images = sorted((root / "test_1000" / "image").glob("*.png"))
    if len(images) != 1000:
        raise RuntimeError(f"Expected 1000 anonymous test images, found {len(images)}")
    rows = []
    for rgb in images:
        sparse = root / "test_1000" / "velodyne_raw" / rgb.name
        intr = root / "test_1000" / "intrinsics" / f"{rgb.stem}.txt"
        if not sparse.is_file() or not intr.is_file():
            raise RuntimeError(f"Incomplete test sample: {rgb.stem}")
        rows.append((rgb.stem, rgb, sparse, intr))
    return rows


def prepare_test(drive_data, work):
    archive_path = drive_data / "test_1000" / "kitti_test_1000.tar"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.is_file():
        download = work / "data_depth_selection.zip"
        print("Downloading official KITTI anonymous test once...", flush=True)
        urllib.request.urlretrieve(KITTI_URL, download)
        partial = archive_path.with_suffix(".tar.partial")
        with zipfile.ZipFile(download) as source, tarfile.open(partial, "w") as output:
            names = set(source.namelist())
            images = sorted(n for n in names if "/test_depth_completion_anonymous/image/" in "/" + n and n.endswith(".png"))
            if len(images) != 1000:
                raise RuntimeError(f"Unexpected official ZIP: {len(images)} test images")
            for image in images:
                base = image.rsplit("/image/", 1)[0]
                stem = Path(image).stem
                for role, suffix in (("image", ".png"), ("velodyne_raw", ".png"), ("intrinsics", ".txt")):
                    name = f"{base}/{role}/{stem}{suffix}"
                    info = tarfile.TarInfo(f"test_1000/{role}/{stem}{suffix}")
                    info.size = source.getinfo(name).file_size
                    with source.open(name) as handle:
                        output.addfile(info, handle)
        partial.replace(archive_path)
        download.unlink()
    marker = work / "test_ready.json"
    if not marker.is_file():
        safe_extract(archive_path, work / "kitti")
        test_rows(work / "kitti")
        write_json(marker, {"samples": 1000, "archive_bytes": archive_path.stat().st_size})
    test_rows(work / "kitti")


def prepare(config):
    drive_data, work = Path(config["drive_data"]), Path(config["work"])
    source, root = drive_data / "teacher_subset_2000", work / "kitti"
    for name in ("selected_2000_ids.json", "kitti_trainval_2000.tar", config["metric_tar"]):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "bundle_ready.json").is_file():
        safe_extract(source / "kitti_trainval_2000.tar", root)
        verify_bundle(source, root)
        write_json(work / "bundle_ready.json", {"manifest_sha256": digest(source / "selected_2000_ids.json")})
    train, val = verify_bundle(source, root)
    report = extract_metric(source / config["metric_tar"], train, val, work / "metric_train", config["teacher_conf_min"])
    # Per-sample KD coverage after excluding GT and every original sensor pixel.
    coverage = []
    effective_threshold = config.get("kd_conf_min", config["teacher_conf_min"])
    for index, sample in enumerate(KITTIDataset(config, "train", teacher=True), 1):
        eligible = ((sample["gt_mask"] == 0) & (sample["mask"] == 0)
                    & (sample["teacher"] > 0.1) & (sample["confidence"] >= effective_threshold))
        count = int(eligible.sum())
        if count < config["min_teacher_pixels"]:
            raise RuntimeError(f"Insufficient non-GT teacher pixels: {sample['sid']} -> {count}")
        coverage.append(count)
        if index % 400 == 0:
            print(f"Paired RGB/sparse/GT/teacher gate: {index}/1600", flush=True)
    prepare_test(drive_data, work)
    contract = {"train": 1600, "val": 400, "test": 1000,
                "manifest_sha256": digest(source / "selected_2000_ids.json"),
                "teacher_report": report, "min_non_gt_teacher_pixels": min(coverage),
                "median_non_gt_teacher_pixels": float(np.median(coverage)),
                "effective_kd_confidence_threshold": effective_threshold,
                "size": list(SIZE), "depth_scale": 256, "validation_teacher_used": False}
    write_json(work / "data_contract.json", contract)
    print("DATA GATE PASS: 1600/400/1000; metric teacher ready for training.", flush=True)


def read_depth(path):
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None or data.dtype != np.uint16:
        raise RuntimeError(f"Not a KITTI uint16 depth image: {path}")
    return data.astype(np.float32) / 256


class KITTIDataset(Dataset):
    def __init__(self, config, split, teacher=False):
        self.config, self.split = config, split
        self.root = Path(config["work"]) / "kitti"
        self.teacher = bool(teacher)
        if self.teacher and split != "train":
            raise ValueError("Teacher access is forbidden for validation/test")
        if split == "test":
            self.rows = test_rows(self.root)
        else:
            name = "train_1600.txt" if split == "train" else "val_400.txt"
            self.rows = [r.split() for r in read_lines(self.root / "splits" / name)]
        if len(self.rows) != {"train": 1600, "val": 400, "test": 1000}[split]:
            raise RuntimeError(f"Wrong {split} count: {len(self.rows)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sid = row[0]
        if self.split == "test":
            _, rgb_path, sparse_path, intr = row
            gt_path = None
            values = np.fromstring(intr.read_text(), sep=" ")
        else:
            rgb_path, sparse_path, gt_path = [self.root / r for r in row[1:4]]
            values = np.array(row[4:], dtype=np.float32)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Unreadable RGB: {rgb_path}")
        original = rgb.shape[:2]
        if self.split == "test" and original != SIZE:
            raise RuntimeError(f"Test output contract requires {SIZE}, got {original}")
        sparse = read_depth(sparse_path)
        gt = read_depth(gt_path) if gt_path else np.zeros(original, np.float32)
        if sparse.shape != original or gt.shape != original:
            raise RuntimeError(f"RGB/depth dimensions differ: {sid}")
        rgb = cv2.cvtColor(cv2.resize(rgb, SIZE[::-1], interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        sparse = cv2.resize(sparse, SIZE[::-1], interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, SIZE[::-1], interpolation=cv2.INTER_NEAREST)
        mask, gt_mask = (sparse > 0.1) & (sparse < 120), (gt > 0.1) & (gt < 120)
        sparse, gt = np.where(mask, sparse, 0), np.where(gt_mask, gt, 0)
        if len(values) == 9:
            fx, fy, cx, cy = values[0], values[4], values[2], values[5]
        elif len(values) == 4:
            fx, fy, cx, cy = values
        else:
            raise RuntimeError(f"Invalid K for {sid}")
        sx, sy = SIZE[1] / original[1], SIZE[0] / original[0]
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float32)
        output = {"sid": sid, "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255,
                  "sparse": torch.from_numpy(sparse[None]), "mask": torch.from_numpy(mask[None].astype(np.float32)),
                  "gt": torch.from_numpy(gt[None]), "gt_mask": torch.from_numpy(gt_mask[None].astype(np.float32)),
                  "K": torch.from_numpy(K)}
        if self.teacher:
            file = Path(self.config["work"]) / "metric_train" / f"{sid}.npy"
            teacher = np.load(file, allow_pickle=False)
            if teacher.shape != (2, *SIZE) or not np.isfinite(teacher).all():
                raise RuntimeError(f"Invalid metric cache: {file}")
            output["teacher"] = torch.from_numpy(teacher[:1])
            output["confidence"] = torch.from_numpy(teacher[1:])
        return output
