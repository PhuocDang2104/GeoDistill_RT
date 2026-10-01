"""Local metrics and immutable, independently restorable backup bundles."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import psutil
import torch

from .common import atomic_json, lock, read_json, sha256, under
from .recovery import atomic_alias


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Telemetry:
    def __init__(self, root: Path, enabled=True):
        self.root, self.enabled = root, enabled
        self.last = time.monotonic()
        self.session = uuid.uuid4().hex[:12]
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(root / 'tensorboard' / self.session), flush_secs=30)

    def step(self, epoch, batch, step, items, lr, scale, skipped, grad_norm):
        now = time.monotonic()
        row = {'time': time.time(), 'session': self.session, 'epoch': epoch, 'batch': batch,
               'step': step, 'seconds': now - self.last, 'lr': lr, 'scale': scale,
               'skipped': skipped, 'grad_norm': float(grad_norm) if grad_norm is not None else None,
               'rss_bytes': psutil.Process().memory_info().rss,
               'ram_available_bytes': psutil.virtual_memory().available,
               'disk_free_bytes': shutil.disk_usage(self.root).free, **items}
        if torch.cuda.is_available():
            row.update(cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                       cuda_peak=torch.cuda.max_memory_reserved())
        with (self.root / 'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(clean(row), allow_nan=False) + '\n')
        if self.writer and batch % 10 == 0:
            for key in ('loss', 'lr', 'grad_norm', 'seconds', 'cuda_allocated'):
                value = row.get(key)
                if value is not None and math.isfinite(float(value)):
                    self.writer.add_scalar('step/' + key, value, step)
        self.last = now

    def epoch(self, record):
        if self.writer:
            for key, value in record.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.writer.add_scalar('epoch/' + key, value, record['epoch'])
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()


def publish_bundle(root: Path):
    """Publish only closed snapshots. Backup never reads live checkpoint aliases."""
    queue = root / 'backup_queue'
    queue.mkdir(exist_ok=True)
    counter_file = root / 'bundle_sequence.json'
    sequence = read_json(counter_file)['sequence'] + 1 if counter_file.exists() else 1
    atomic_json(counter_file, {'sequence': sequence})
    token = f'{sequence:012d}-{uuid.uuid4().hex[:8]}'
    stage = queue / ('.' + token)
    stage.mkdir()
    # Include both last and best so a recovered run keeps its selected model.
    for alias in ('last', 'best'):
        pointer = root / 'checkpoints' / (alias + '.json')
        if pointer.exists():
            meta = read_json(pointer)
            source = under(pointer.parent, meta['file'])
            atomic_alias(source, stage / 'checkpoints' / (alias + '.pth'))
        elif pointer.with_suffix('.pth').exists():
            # Restored bundles contain already-verified standalone aliases.
            # Preserve best even when the resumed run has not improved on it.
            atomic_alias(pointer.with_suffix('.pth'), stage / 'checkpoints' / (alias + '.pth'))
    for name in ('resolved_config.json', 'resolved_paths.json', 'hardware.json', 'calibration.json',
                 'provenance.json', 'bundle_sequence.json', 'train_log.csv', 'train_log.jsonl', 'steps.jsonl', 'status.json'):
        source = root / name
        if source.exists():
            shutil.copyfile(source, stage / name)
    for name in ('logs', 'tensorboard', 'evaluation'):
        if (root / name).exists():
            shutil.copytree(root / name, stage / name)
    files = {str(p.relative_to(stage)).replace('\\', '/'): sha256(p)
             for p in stage.rglob('*') if p.is_file()}
    atomic_json(stage / 'COMPLETE.json', {'files': files, 'created': time.time()})
    os.replace(stage, queue / token)
    # Bound disk use during an outage: keep latest 3 complete bundles. Use the
    # same lock as backup so we never remove a bundle being uploaded.
    try:
        with lock(root / '.backup.lock'):
            bundles = sorted(p for p in queue.iterdir() if not p.name.startswith('.') and p.is_dir())
            for old in bundles[:-3]:
                shutil.rmtree(old)
            prune_snapshots(root)
    except RuntimeError:
        pass


def prune_snapshots(root: Path):
    ckpts = root / 'checkpoints'
    keep = {read_json(p)['file'] for p in ckpts.glob('*.json')}
    for p in (ckpts / 'snapshots').glob('*.pth'):
        if 'snapshots/' + p.name not in keep:
            p.unlink()
            p.with_suffix('.json').unlink(missing_ok=True)


def backup_once(root: Path, remote: str):
    if ':' not in remote or remote.startswith(':'):
        raise ValueError('Use a configured rclone remote:path (Google Drive recommended)')
    with lock(root / '.backup.lock'):
        for bundle in sorted((root / 'backup_queue').glob('*')):
            if bundle.name.startswith('.') or not (bundle / 'COMPLETE.json').exists():
                continue
            if (bundle / '.uploaded').exists():
                continue
            destination = remote.rstrip('/') + '/' + root.name + '/' + bundle.name
            # Publish the completion marker last; incomplete cloud bundles are ignored.
            subprocess.run(['rclone', 'copy', str(bundle), destination, '--exclude', 'COMPLETE.json',
                            '--exclude', '.uploaded', '--transfers', '2', '--retries', '3'], check=True)
            subprocess.run(['rclone', 'copyto', str(bundle / 'COMPLETE.json'), destination + '/COMPLETE.json'], check=True)
            atomic_json(bundle / '.uploaded', {'destination': destination, 'time': time.time()})
            atomic_json(root / 'backup_status.json', {'state': 'OK', 'destination': destination, 'time': time.time()})


def restore_bundle(bundle: Path, destination: Path):
    meta = read_json(bundle / 'COMPLETE.json')
    for name, checksum in meta['files'].items():
        if sha256(under(bundle, name)) != checksum:
            raise ValueError(f'Backup checksum mismatch: {name}')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Restore destination must be empty; existing runs are never overwritten')
    destination.mkdir(parents=True, exist_ok=True)
    for name in meta['files']:
        output = under(destination, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(under(bundle, name), output)
    atomic_json(destination / 'restored.json', {'bundle': str(bundle), 'time': time.time()})
