"""Extract only small reports/student checkpoints; never duplicate the prediction ZIP."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        files = {}
        for member in archive.infolist():
            name = Path(member.filename).name
            if member.is_dir() or Path(name).suffix not in (".json", ".csv", ".jsonl", ".log", ".pth"):
                continue
            if member.file_size > 32 * 1024 ** 2 or name in files:
                raise RuntimeError(f"Unexpected/duplicate report: {member.filename}")
            contents = archive.read(member)
            (target / name).write_bytes(contents)
            files[name] = hashlib.sha256(contents).hexdigest()
    report = {"source": str(args.archive.resolve()), "files_sha256": files}
    (target / "archive_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = list(csv.DictReader((target / "train_log.csv").open(encoding="utf-8")))
    print("EPOCHS")
    for row in rows:
        fields = ("epoch", "val_rmse_m", "val_pre_anchor_rmse_m", "val_legacy_hard_rmse_m", "loss_total", "train_seconds")
        print({k: row[k] for k in fields})
    print("LAST TRAIN ROW", json.dumps(rows[-1], indent=2))
    for name in ("initial_val_metrics.json", "val_metrics.json"):
        values = json.loads((target / name).read_text())
        print(name, json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
