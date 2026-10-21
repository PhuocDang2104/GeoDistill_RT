"""Verify and ZIP immutable Q1 bytes. No parent generator, mutation or training."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest_path=TRAIN/'bundle_manifest.json'
    baseline=json.loads((ROOT/'production/AnchorFlow_S4_Q1_Fast/baseline.json').read_text(encoding='utf-8'))
    if sha(manifest_path)!=baseline['frozen_bundle_manifest_sha256']:raise RuntimeError('Adopted manifest identity changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_python={n for n in manifest['files'] if n.endswith('.py')}
    actual_python={p.relative_to(TRAIN).as_posix() for p in TRAIN.rglob('*.py') if '__pycache__' not in p.parts}
    if actual_python!=expected_python:raise RuntimeError('Extra/missing frozen Python source')
    for name,expected in manifest['files'].items():
        path=(TRAIN/name).resolve()
        if not path.is_relative_to(TRAIN.resolve()) or not path.is_file() or sha(path)!=expected:
            raise RuntimeError('Frozen bundle differs: '+name)
    return manifest


def package(directory,names):
    dest=directory.with_suffix('.zip');partial=dest.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:archive.write(directory/name,directory.name+'/'+name)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:raise RuntimeError('ZIP CRC failed')
        for name in names:
            if hashlib.sha256(archive.read(directory.name+'/'+name)).hexdigest()!=sha(directory/name):raise RuntimeError('ZIP content differs')
    partial.replace(dest)
    print(f'{dest} ({dest.stat().st_size:,} bytes) SHA256={sha(dest)}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-orin',action='store_true')
    args=parser.parse_args();manifest=verify()
    package(TRAIN,[*manifest['files'],'bundle_manifest.json'])
    if args.legacy_orin:
        old=ROOT/'production/AnchorFlow_S4_UPIR_Orin'
        print('WARNING: historical G Orin helper, NOT Q1-compatible; no Q1 target measurement.')
        package(old,[p.name for p in sorted(old.iterdir()) if p.is_file() and p.suffix in ('.py','.md','.json')])
    print('Frozen Q1 verified/repacked unchanged. No weights/data/ONNX in ZIP; no old parent folder needed.')


if __name__=='__main__':main()
