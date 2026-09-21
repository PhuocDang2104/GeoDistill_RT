from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .common import atomic_json, identity, lock, read_json, sha256, under


def fetch(source: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + '.partial')
    if source.startswith('drive-id:'):
        import gdown
        result = gdown.download(id=source.split(':', 1)[1], output=str(part), quiet=False, resume=True)
        if result is None:
            raise RuntimeError('Drive download failed; use an authenticated rclone remote if quota/sharing blocks it')
    elif source.startswith(('https://', 'http://')):
        with urllib.request.urlopen(source, timeout=120) as response, part.open('wb') as output:
            shutil.copyfileobj(response, output, 8 * 1024 * 1024)
    elif Path(source).is_file():
        shutil.copyfile(source, part)
    elif ':' in source:
        subprocess.run(['rclone', 'copyto', source, str(part), '--retries', '3'], check=True)
    else:
        raise ValueError(f'Unknown source: {source}')
    os.replace(part, destination)


def safe_extract(source: Path, destination: Path, byte_limit: int):
    total = 0
    with tarfile.open(source, 'r:*') as archive:
        for member in archive:
            target = under(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.name.startswith('/') or '\\' in member.name:
                raise ValueError(f'Unsupported archive entry: {member.name}')
            if target.exists():
                raise ValueError(f'Duplicate archive path: {member.name}')
            total += member.size
            if total > byte_limit:
                raise ValueError('Archive exceeds declared extraction byte limit')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, target.open('wb') as output:
                shutil.copyfileobj(stream, output, 8 * 1024 * 1024)


def canonical_id(name: str) -> str:
    stem = Path(name).stem
    match = re.match(r'^(.*)_sync_image_(\d{2})_(\d{10})$', stem)
    if match:
        return f'{match[1]}_sync_image_{match[3]}_image_{match[2]}'
    match = re.match(r'^(.*)_sync_image_0([23])_(\d{10})$', stem)
    if match:
        return f'{match[1]}_sync_image_{match[3]}_image_0{match[2]}'
    if re.match(r'^.+_sync_image_\d{10}_image_0[23]$', stem):
        return stem
    match = re.match(r'^(.*)_sync_(image_0[23])_(\d{10})$', stem)
    if match:
        return f'{match[1]}_sync_image_{match[3]}_{match[2]}'
    raise ValueError(f'Unsupported teacher ID: {name}')


def extract_teacher(source: Path, destination: Path, roles: dict, byte_limit: int):
    seen, total = set(), 0
    with tarfile.open(source, 'r:*') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.npz'):
                continue
            sid = canonical_id(member.name)
            if sid not in roles:
                continue
            if sid in seen:
                raise ValueError(f'Duplicate teacher ID {sid}')
            seen.add(sid)
            total += member.size
            if total > byte_limit:
                raise ValueError('Teacher archive exceeds declared extraction byte limit')
            output = under(destination, f'{roles[sid]}/{sid}.npz')
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, output.open('wb') as out:
                shutil.copyfileobj(stream, out)
            with np.load(output, allow_pickle=False) as arrays:
                keys = ('R_G', 'C_G') if destination.name == 'geometry_fused' else ('D_cm', 'C_cm')
                if not all(key in arrays for key in keys):
                    raise ValueError(f'Missing required teacher keys {keys}: {output}')
                for key in keys:
                    if arrays[key].size == 0 or not np.isfinite(arrays[key]).all():
                        raise ValueError(f'Invalid teacher tensor {key}: {output}')
    if seen != set(roles):
        raise ValueError(f'Teacher coverage missing {len(set(roles) - seen)} samples')


def prepare(manifest_path: Path, storage: Path) -> Path:
    manifest = read_json(manifest_path)
    digest = identity(manifest)
    root = storage / 'datasets' / digest
    with lock(storage / '.prepare.lock'):
        if (root / 'READY.json').exists():
            print(f'Using prepared dataset {digest}', flush=True)
            return root
        stage = storage / 'datasets' / (digest + '.staging')
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        assets = {}
        total_required = sum(int(a['bytes']) + int(a.get('expanded_bytes', a['bytes'])) for a in manifest['assets'])
        if shutil.disk_usage(storage).free < total_required + 2 * 1024**3:
            raise RuntimeError('Insufficient space for archives, extraction, and 2 GiB reserve')
        for asset in manifest['assets']:
            print(f'Verifying dataset asset: {asset["name"]}', flush=True)
            checksum = asset['sha256']
            if not re.fullmatch('[a-f0-9]{64}', checksum):
                raise ValueError('Every asset needs a real SHA256; use catalog to create the manifest')
            archive = storage / 'downloads' / checksum
            actual = sha256(archive) if archive.exists() else None
            if actual != checksum:
                fetch(asset['source'], archive)
                actual = sha256(archive)
            if archive.stat().st_size != int(asset['bytes']) or actual != checksum:
                raise ValueError(f'Asset checksum/size mismatch: {asset["name"]}')
            assets[asset['name']] = (archive, int(asset.get('expanded_bytes', asset['bytes'])))
        if manifest['format'] == 'tar2000':
            selected = read_json(assets['selected_2000_ids.json'][0])
            train, val = selected['train_ids'], selected['val_ids']
            if (len(train), len(val), len(set(train + val))) != (1600, 400, 2000):
                raise ValueError('TAR2000 requires the frozen 1600/400 unique split')
            if {s.split('_sync_image_')[0] for s in train} & {s.split('_sync_image_')[0] for s in val}:
                raise ValueError('Train/validation raw-drive leakage')
            roles = {**dict.fromkeys(train, 'train'), **dict.fromkeys(val, 'val')}
            print('Extracting KITTI inputs', flush=True)
            safe_extract(assets['kitti_trainval_2000.tar'][0], stage / 'kitti', assets['kitti_trainval_2000.tar'][1])
            bundled = read_json(stage / 'kitti' / 'selected_2000_ids.json')
            if any(bundled[k] != selected[k] for k in ('train_ids', 'val_ids')):
                raise ValueError('KITTI bundle selection differs from teacher selection')
            report = read_json(stage / 'kitti' / 'kitti_bundle_report.json')
            if report.get('contract_ok') is not True or report.get('selected_count') != 2000:
                raise ValueError('KITTI bundle report failed its source contract')
            for name, role in [('metric_coarse_train_2000.tar', 'metric_coarse'), ('geometry_fused_train_2000.tar', 'geometry_fused')]:
                print(f'Extracting and validating {role}', flush=True)
                extract_teacher(assets[name][0], stage / 'teachers' / role, roles, assets[name][1])
            layout = {'data': 'kitti', 'splits': 'kitti/splits', 'teachers': 'teachers',
                      'train_split': 'train_1600.txt', 'val_split': 'val_400.txt'}
        elif manifest['format'] == 'prepared-kitti':
            # Shards share one normalized root and explicit JSONL splits with calibrated K.
            for asset in manifest['assets']:
                safe_extract(assets[asset['name']][0], stage, assets[asset['name']][1])
            layout = manifest['layout']
        else:
            raise ValueError('Supported formats: tar2000, prepared-kitti')
        for key in ('data', 'splits', 'teachers'):
            under(stage, layout[key])
        atomic_json(stage / 'READY.json', {'manifest': manifest, 'digest': digest, 'layout': layout})
        os.replace(stage, root)
    return root


def catalog(source: str, output: Path, storage: Path):
    """Pin a trusted TAR2000 source once, before deployment to another machine."""
    assets = []
    sources = read_json(Path(source)) if Path(source).is_file() else None
    for name in ('selected_2000_ids.json', 'kitti_trainval_2000.tar',
                 'metric_coarse_train_2000.tar', 'geometry_fused_train_2000.tar'):
        local = storage / 'catalog' / name
        remote = sources[name] if sources else source.rstrip('/') + '/' + name
        if not local.exists():
            fetch(remote, local)
        expanded = local.stat().st_size
        if name.endswith('.tar'):
            with tarfile.open(local, 'r:*') as archive:
                expanded = sum(m.size for m in archive if m.isfile())
        checksum = sha256(local)
        cached = storage / 'downloads' / checksum
        cached.parent.mkdir(parents=True, exist_ok=True)
        os.replace(local, cached)
        assets.append({'name': name, 'source': remote, 'sha256': checksum,
                       'bytes': cached.stat().st_size, 'expanded_bytes': expanded})
    atomic_json(output, {'version': 1, 'format': 'tar2000', 'assets': assets})
