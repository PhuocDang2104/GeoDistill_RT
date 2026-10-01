from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import shutil
import signal
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from .common import atomic_json, identity, read_json, sha256, under, sync_directory


class EpochBatchSampler(Sampler):
    """Deterministic permutation and sample RNG independent of worker prefetch."""
    def __init__(self, length: int, batch_size: int, seed: int):
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.epoch, self.start_batch = 0, 0

    def __len__(self):
        return self.length // self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.length, generator=g).tolist()
        for batch in range(self.start_batch, len(self)):
            result = []
            for index in indices[batch * self.batch_size:(batch + 1) * self.batch_size]:
                seed = int(identity([self.seed, self.epoch, index])[:8], 16)
                result.append((index, seed))
            yield result


def capture_rng() -> dict:
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def training_contract(cfg: dict) -> dict:
    train = copy.deepcopy(cfg.get('train', {}))
    for key in ('resume', 'checkpoint_steps', 'checkpoint_seconds', 'stop_after_steps'):
        train.pop(key, None)
    data = copy.deepcopy(cfg.get('data', {}))
    for key in ('num_workers', 'prefetch_factor', 'worker_threads'):
        data.pop(key, None)
    return {**{k: cfg.get(k) for k in ('seed', 'model', 'student', 'loss', 'schedule', 'mono_ssi', 'sparse_propagation')},
            'train': train, 'data': data, 'protocol': cfg.get('runtime', {}).get('protocol'),
            'source': cfg.get('runtime', {}).get('source')}


def verify_resume(saved: dict, cfg: dict) -> None:
    if training_contract(saved) != training_contract(cfg):
        raise ValueError('Resume contract mismatch: model/preprocessing, objective, batch, schedule, data, or source changed. Start a new run.')


def atomic_alias(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name('.' + target.name + '.' + uuid.uuid4().hex)
    try:
        try:
            os.link(source, temp)
        except OSError:
            shutil.copyfile(source, temp)
        os.replace(temp, target)
        sync_directory(target.parent)
    finally:
        temp.unlink(missing_ok=True)


def commit_checkpoint(path: Path, payload: dict) -> Path:
    """Immutable payload + manifest, then replace the conventional last/best alias."""
    snapshots = path.parent / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    name = f"step-{payload.get('progress', {}).get('global_step', 0):09d}-{uuid.uuid4().hex[:12]}.pth"
    target = snapshots / name
    fd, temp = tempfile.mkstemp(prefix='.checkpoint-', dir=snapshots)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        manifest = {'file': name, 'sha256': sha256(target), 'bytes': target.stat().st_size,
                    'epoch': payload['epoch'], 'progress': payload.get('progress', {})}
        atomic_json(target.with_suffix('.json'), manifest)
        atomic_alias(target, path)
        atomic_json(path.with_suffix('.json'), {**manifest, 'file': f'snapshots/{name}'})
        return target
    finally:
        Path(temp).unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict:
    manifest_path = path.with_suffix('.json')
    if manifest_path.is_file():
        meta = read_json(manifest_path)
        path = under(path.parent, meta['file'])
        if sha256(path) != meta['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {path}')
    return torch.load(path, map_location='cpu', weights_only=True)


def alias_checkpoint(source: Path, target: Path) -> None:
    meta = read_json(source.with_suffix('.json'))
    atomic_alias(under(source.parent, meta['file']), target)
    atomic_json(target.with_suffix('.json'), meta)


def reconcile_logs(root: Path, completed_epoch: int) -> None:
    """Drop uncommitted epoch rows after power loss between log and checkpoint writes."""
    path = root / 'train_log.csv'
    if path.exists():
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [r for r in reader if int(r['epoch']) <= completed_epoch]
        temp = path.with_suffix('.tmp')
        with temp.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    path = root / 'train_log.jsonl'
    if path.exists():
        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row['epoch']) <= completed_epoch:
                rows.append(row)
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        os.replace(temp, path)


class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = {}
        self.cleanup = None

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous[sig] = signal.signal(sig, self.handle)
        return self

    def handle(self, *_):
        self.requested = True

    def __exit__(self, *_):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
        if self.cleanup is not None:
            self.cleanup()
