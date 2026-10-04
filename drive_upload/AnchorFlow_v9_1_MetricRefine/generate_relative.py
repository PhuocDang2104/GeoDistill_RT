"""Single-RGB DA3MONO-LARGE generation; no metric/sparse/GT input to teacher.

Pinned official network + safetensors, small adapter avoids optional API exporters
and xformers/open3d/gsplat dependencies. Attention uses native PyTorch SDPA.
"""
import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from data import SIZE, safe_extract, verify_bundle, validate_manifest, write_json, digest
from relative_data import relative_arrays

MODEL_ID="depth-anything/DA3MONO-LARGE"
MODEL_REVISION="f465978e618db8cc79c83b8bbf24964857db1875"
CODE_REVISION="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


def standardize(depth, sky=None):
    valid=np.isfinite(depth)&(depth>1e-6)
    if sky is not None: valid &= ~sky.astype(bool)
    if valid.sum()<1024: raise RuntimeError("Teacher has insufficient non-sky support")
    inv=np.zeros_like(depth,dtype=np.float32)
    inv[valid]=1/depth[valid]
    mean,std=float(inv[valid].mean()),float(inv[valid].std())
    if std<1e-8: raise RuntimeError("Teacher inverse depth is constant")
    return np.where(valid,(inv-mean)/std,0).astype(np.float32),valid


def fuse_tta(depth, flipped_depth, sky=None, flipped_sky=None):
    a,ma=standardize(depth,sky)
    b,mb=standardize(flipped_depth,flipped_sky)
    support=ma&mb
    confidence=np.exp(-np.abs(a-b)).astype(np.float32)*support
    relative=(a+b)*.5
    # Heuristic TTA agreement, NOT a learned/calibrated metric confidence.
    return np.where(support,relative,0).astype(np.float32),confidence.astype(np.float32)


def load_teacher(third_party, weight_root, device):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    sys.path.insert(0,str(third_party/"src"))
    from depth_anything_3.cfg import create_object,load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    location=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(weight_root),
                                   allow_patterns=["config.json","model.safetensors"]))
    cfg=json.loads((location/"config.json").read_text())
    if cfg["model_name"]!="da3mono-large": raise RuntimeError(f"Unexpected HF config: {cfg}")
    model=create_object(load_config(MODEL_REGISTRY["da3mono-large"]))
    payload=load_file(str(location/"model.safetensors"))
    if not payload or not all(k.startswith("model.") for k in payload):
        raise RuntimeError("Official checkpoint prefix changed; refusing non-strict adapter load")
    model.load_state_dict({k[len("model."):]:v for k,v in payload.items()},strict=True)
    del payload
    return model.eval().to(device)


@torch.inference_mode()
def predict(model,rgb,device,long_side):
    h,w=rgb.shape[:2]
    scale=long_side/max(h,w)
    hw=(max(14,round(h*scale/14)*14),max(14,round(w*scale/14)*14))
    image=cv2.resize(rgb,hw[::-1],interpolation=cv2.INTER_CUBIC)
    x=torch.from_numpy(image.transpose(2,0,1).copy()).to(device).float()/255
    mean=x.new_tensor([.485,.456,.406])[:,None,None]
    std=x.new_tensor([.229,.224,.225])[:,None,None]
    x=((x-mean)/std)[None,None]
    amp=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device.type,dtype=amp):
        out=model(x,infer_gs=False,use_ray_pose=False)
    d=out["depth"].float().reshape(1,1,*hw)
    depth=F.interpolate(d,size=SIZE,mode="bilinear",align_corners=False)[0,0].cpu().numpy()
    sky=out.get("sky")
    if sky is not None:
        sky=F.interpolate(sky.float().reshape(1,1,*hw),size=SIZE,mode="bilinear",align_corners=False)[0,0].cpu().numpy()>=.3
    return depth,sky


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-data",type=Path,required=True)
    parser.add_argument("--work",type=Path,default=Path("/content/anchorflow_v9_teacher_work"))
    parser.add_argument("--long-side",type=int,default=1232)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Teacher generation requires Colab GPU")
    if args.long_side<504: raise ValueError("Use long side >=504; default1232 preserves KITTI fine detail")
    source=args.drive_data/"teacher_subset_2000"
    train,val=validate_manifest(source/"selected_2000_ids.json")
    root=args.work/"kitti"
    if not (root/"selected_2000_ids.json").is_file(): safe_extract(source/"kitti_trainval_2000.tar",root)
    verify_bundle(source,root)
    rows=[]
    for split in ("train_1600","val_400"):
        rows.extend([line.split() for line in (root/"splits"/(split+".txt")).read_text().splitlines() if line.strip()])
    recipe={"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"code_revision":CODE_REVISION,
            "generator_sha256":digest(Path(__file__)),"selected_manifest_sha256":digest(source/"selected_2000_ids.json"),
            "long_side":args.long_side,"shape":list(SIZE),"tta":"RGB horizontal flip; per-pass standardized inverse depth",
            "resize":"student RGB full-frame352x1216 -> patch14 full-frame resize -> full-frame back; no crop",
            "representation":"standardized_relative_inverse_depth_near_high",
            "teacher_inputs":"RGB only; one view at a time; no GT, sparse, K or metric teacher",
            "confidence":"exp(-abs(normalized inverse depth TTA disagreement)), non-sky; heuristic"}
    recipe_hash=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    staging=source/"relative_DA3MONO_LARGE_generation"
    staging.mkdir(parents=True,exist_ok=True)
    frozen=staging/"recipe.json"
    if frozen.exists() and json.loads(frozen.read_text())!=recipe:
        raise RuntimeError("Existing teacher recipe differs; rename/archive generation folder before a new recipe")
    write_json(frozen,recipe)
    third_party=args.work/"Depth-Anything-3"
    if not third_party.exists():
        subprocess.run(["git","clone","https://github.com/ByteDance-Seed/Depth-Anything-3.git",str(third_party)],check=True)
    subprocess.run(["git","-C",str(third_party),"checkout","--detach",CODE_REVISION],check=True)
    model=load_teacher(third_party,args.work/"hf_cache",torch.device("cuda"))
    records={}
    for index,row in enumerate(rows,1):
        sid=row[0]; target=staging/(sid+".npz")
        if target.is_file():
            with np.load(target,allow_pickle=False) as payload:
                relative_arrays(payload)
                if str(payload["recipe_sha256"].item())!=recipe_hash: raise RuntimeError(f"Stale relative file: {sid}")
        else:
            # Only RGB path read; GT/sparse paths in the split are NOT read by teacher.
            rgb=cv2.imread(str(root/row[1]))
            if rgb is None: raise RuntimeError(f"Unreadable RGB: {sid}")
            rgb=cv2.cvtColor(cv2.resize(rgb,SIZE[::-1],interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
            depth,sky=predict(model,rgb,torch.device("cuda"),args.long_side)
            depth_flip,sky_flip=predict(model,rgb[:,::-1].copy(),torch.device("cuda"),args.long_side)
            r,c=fuse_tta(depth,depth_flip[:,::-1],sky,None if sky_flip is None else sky_flip[:,::-1])
            partial=target.with_suffix(".npz.partial")
            with partial.open("wb") as handle: np.savez_compressed(handle,R_T=r,C_T=c,recipe_sha256=recipe_hash)
            with np.load(partial,allow_pickle=False) as payload: relative_arrays(payload)
            partial.replace(target)
        records[sid]=digest(target)
        if index%20==0 or index==len(rows):
            write_json(staging/"progress.json",{"completed":index,"total":2000,"recipe_sha256":recipe_hash})
            print(f"Relative teacher {index}/2000; per-image Drive resume",flush=True)
    metadata={**recipe,"train_ids":train,"val_ids":val,"records":records,"validation_targets":"audit only; never loaded by student val/test"}
    destination=source/"relative_teacher_2000_DA3MONO_LARGE.tar"
    partial=destination.with_suffix(".tar.partial")
    with tarfile.open(partial,"w") as archive:
        raw=json.dumps(metadata,sort_keys=True).encode()
        info=tarfile.TarInfo("relative_manifest.json"); info.size=len(raw)
        archive.addfile(info,io.BytesIO(raw))
        for sid in train+val: archive.add(staging/(sid+".npz"),arcname=f"relative/{sid}.npz",recursive=False)
    partial.replace(destination)
    write_json(source/"relative_teacher_2000_report.json",{"tar":str(destination),"bytes":destination.stat().st_size,
               "records":2000,"recipe":recipe,"recipe_sha256":recipe_hash,"student_training_records":1600,"audit_only_val_records":400})
    print("DONE:",destination,flush=True)


if __name__=="__main__": main()
