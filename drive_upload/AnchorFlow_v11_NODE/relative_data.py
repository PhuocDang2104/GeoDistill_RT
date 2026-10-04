"""Paired relative teacher cache: strict manifest/hash gate, train-only loading."""
import io
import json
import hashlib
import tarfile
from pathlib import Path
import numpy as np
from data import SIZE, canonical_id, write_json, digest


def relative_arrays(payload):
    if not {"R_T", "C_T"}.issubset(payload.files):
        raise RuntimeError("Relative teacher must contain R_T,C_T (not metric metres)")
    values = [np.asarray(payload[k]) for k in ("R_T","C_T")]
    if any(x.dtype != np.float32 for x in values):
        raise RuntimeError("Relative targets must be float32; refusing silent dtype conversion")
    result = np.stack([x.squeeze() for x in values])
    if result.shape != (2,*SIZE) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid relative tensor shape/finite values: {result.shape}")
    if result[1].min()<0 or result[1].max()>1 or (result[1]>=.35).sum()<1024:
        raise RuntimeError("Invalid confidence or insufficient relative support")
    if result[0][result[1]>0].std()<1e-4:
        raise RuntimeError("Relative map is constant; no structural signal")
    return result


def extract_relative(config, train, val):
    source=Path(config["drive_data"])/"teacher_subset_2000"
    archive_path=source/config["relative_tar"]
    cache=Path(config["work"])/"relative_train"
    cache.mkdir(parents=True,exist_ok=True)
    expected=set(train+val)
    train_set=set(train)
    if len(expected)!=len(train)+len(val):
        raise RuntimeError("Relative IDs duplicated or train/val overlap")
    seen=set()
    total_confident=0
    content=hashlib.sha256()
    with tarfile.open(archive_path,"r|*") as archive:
        iterator=iter(archive)
        first=next(iterator)
        if first.name!="relative_manifest.json" or not first.isfile():
            raise RuntimeError("Relative archive must start with relative_manifest.json")
        metadata=json.loads(archive.extractfile(first).read())
        if metadata["selected_manifest_sha256"]!=digest(source/"selected_2000_ids.json"):
            raise RuntimeError("Relative teacher generated from a different subset")
        if metadata["train_ids"]!=train or metadata["val_ids"]!=val or metadata["representation"]!="standardized_relative_inverse_depth_near_high":
            raise RuntimeError("Relative split order or depth convention changed")
        if metadata["model_id"]!=config["relative_model_id"] or metadata["model_revision"]!=config["relative_model_revision"]:
            raise RuntimeError("Wrong relative foundation model/revision")
        if set(metadata["records"])!=expected:
            raise RuntimeError("Relative manifest does not cover exactly selected 2000 IDs")
        for member in iterator:
            if not member.isfile() or not member.name.endswith(".npz"):
                raise RuntimeError(f"Unexpected relative archive member: {member.name}")
            sid=canonical_id(member.name)
            if sid not in expected or sid in seen:
                raise RuntimeError(f"Unexpected/duplicate relative ID: {sid}")
            seen.add(sid)
            raw=archive.extractfile(member).read()
            sha=hashlib.sha256(raw).hexdigest()
            if sha!=metadata["records"][sid]:
                raise RuntimeError(f"Relative content checksum failed: {sid}")
            content.update(sid.encode()); content.update(bytes.fromhex(sha))
            with np.load(io.BytesIO(raw),allow_pickle=False) as payload:
                arrays=relative_arrays(payload)
            total_confident+=float((arrays[1]>=.35).mean())
            if sid in train_set:
                target=cache/f"{sid}.npy"
                partial=target.with_suffix(".npy.partial")
                with partial.open("wb") as handle: np.save(handle,arrays,allow_pickle=False)
                partial.replace(target)
            if len(seen)%400==0: print(f"Relative audited {len(seen)}/2000; train-only cache",flush=True)
    if seen!=expected:
        raise RuntimeError(f"Relative archive missing {len(expected-seen)} IDs")
    report={"records":len(seen),"cached_train":len(train),"cached_val":0,"validation_teacher_used":False,
            "metadata":metadata,"selected_content_sha256":content.hexdigest(),
            "mean_confident_fraction":total_confident/max(1,len(seen))}
    write_json(cache/"relative_report.json",report)
    return report
