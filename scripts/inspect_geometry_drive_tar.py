from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from gdown.download import get_url_from_gdrive_confirmation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_geolift_teachers import canonical_sample_id


def _open_drive_stream(file_id: str) -> tuple[requests.Session, requests.Response]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    last_disposition = ""
    for _ in range(5):
        response = session.get(url, stream=True, timeout=90)
        response.raise_for_status()
        last_disposition = response.headers.get("Content-Disposition", "")
        if "attachment" in last_disposition:
            response.raw.decode_content = True
            return session, response
        url = get_url_from_gdrive_confirmation(response.text)
        response.close()
    session.close()
    raise RuntimeError(f"Google Drive did not return an attachment after confirmations: {last_disposition!r}")


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    values = arr[finite]
    result: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_ratio": float(finite.mean()),
    }
    if values.size:
        result.update(
            {
                "min": float(values.min()),
                "q01": float(np.quantile(values, 0.01)),
                "median": float(np.median(values)),
                "q99": float(np.quantile(values, 0.99)),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-inspect a few NPZ members from a large public Drive geometry tar.")
    parser.add_argument("--file_id", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    session, response = _open_drive_stream(args.file_id)
    records: list[dict[str, Any]] = []
    members_seen: list[str] = []
    try:
        with tarfile.open(fileobj=response.raw, mode="r|") as archive:
            for member in archive:
                members_seen.append(member.name)
                if not member.isfile() or not member.name.lower().endswith(".npz"):
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read {member.name}")
                payload = source.read()
                with np.load(io.BytesIO(payload)) as data:
                    keys = list(data.files)
                    arrays = {key: _array_stats(data[key]) for key in keys}
                    r_ok = "R_G" in data and data["R_G"].ndim == 2
                    c_ok = "C_G" in data and data["C_G"].ndim == 2
                    shape_ok = r_ok and c_ok and data["R_G"].shape == data["C_G"].shape
                    c_range_ok = c_ok and bool(
                        np.isfinite(data["C_G"]).all()
                        and float(data["C_G"].min()) >= 0.0
                        and float(data["C_G"].max()) <= 1.0
                    )
                raw_id = Path(member.name).stem
                records.append(
                    {
                        "member": member.name,
                        "member_bytes": member.size,
                        "raw_id": raw_id,
                        "canonical_id": canonical_sample_id(raw_id, "geometry"),
                        "keys": keys,
                        "arrays": arrays,
                        "contract_ok": bool(r_ok and c_ok and shape_ok and c_range_ok),
                    }
                )
                if len(records) >= args.samples:
                    break
    finally:
        response.close()
        session.close()

    result = {
        "file_id": args.file_id,
        "content_length": int(response.headers.get("Content-Length", 0)),
        "accept_ranges": response.headers.get("Accept-Ranges"),
        "content_disposition": response.headers.get("Content-Disposition"),
        "first_members": members_seen[:10],
        "samples": records,
        "all_inspected_samples_pass": len(records) == args.samples and all(record["contract_ok"] for record in records),
        "expected_contract": {
            "layout": "geometry_fused/train/*.npz",
            "keys": ["R_G", "C_G"],
            "canonical_output_name": "<drive>_image_<frame>_image_0X.npz",
        },
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_inspected_samples_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
