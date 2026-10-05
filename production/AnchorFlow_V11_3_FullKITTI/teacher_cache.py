"""Crash-safe train-only float32 NPY cache + transactional SQLite provenance."""
import os
from contextlib import closing
import shutil
import sqlite3
import tempfile
from pathlib import Path
import numpy as np
from full_data import SIZE, load_index, read_sample
from utils import atomic_json, freeze, identity, read_json, sha256


def cache_path(cfg, role, sid):
    token = identity(sid)[:2]
    return Path(cfg['teacher_root']) / role / token / (sid + '.npy')


def connection(cfg, role):
    root = Path(cfg['teacher_root']) / role
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'records.sqlite3', timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS records(sid TEXT PRIMARY KEY, input_sha TEXT, recipe TEXT, sha TEXT, bytes INTEGER)')
    return db


def validate_arrays(a, role):
    if a.shape != (2, *SIZE) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise RuntimeError(f'Invalid {role} teacher shape/dtype/finite values: {a.shape}/{a.dtype}')
    if a[1].min() < 0 or a[1].max() > 1:
        raise RuntimeError('Teacher confidence outside [0,1]')
    if role == 'metric' and not ((a[0] > .1) & (a[0] < 120)).any():
        raise RuntimeError('Metric teacher has no valid metric depth')
    if role == 'relative':
        support = a[1] > 0
        if int(support.sum()) < 1024 or a[0][support].std() < 1e-4:
            raise RuntimeError('Relative teacher is constant/has insufficient non-sky support')


def predict_row(cfg, role, row, teacher):
    # RGB-only relative inference never reads GT/sparse arrays.
    if role == 'relative':
        import cv2
        meta = row['rgb']; path = Path(cfg['dataset_root']) / meta['path']
        if sha256(path) != meta['sha256']:
            raise RuntimeError('RGB changed since dataset index')
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError('Relative input RGB decode failed')
        rgb = cv2.cvtColor(cv2.resize(image, SIZE[::-1]), cv2.COLOR_BGR2RGB)
        return teacher.predict(rgb)
    sample = read_sample(row, cfg['dataset_root'], include_gt=True, verify_hash=True)
    return teacher.predict(sample)


def teacher_smoke(cfg, role):
    """Qualify BOTH providers on real input before either full generation starts."""
    import time
    import torch
    from teachers import build_teacher, teacher_recipe
    row = load_index(cfg, 'train')[0]
    started = time.monotonic(); teacher = build_teacher(cfg, role)
    torch.cuda.synchronize(); load_seconds = time.monotonic() - started
    started = time.monotonic(); arrays = predict_row(cfg, role, row, teacher)
    torch.cuda.synchronize(); inference_seconds = time.monotonic() - started
    validate_arrays(arrays, role)
    sample = read_sample(row, cfg['dataset_root'], verify_hash=True)
    eligible = (arrays[1] >= (.5 if role == 'metric' else .35))
    eligible &= (sample['gt_mask'][0].numpy() == 0) & (sample['mask'][0].numpy() == 0)
    count = int(eligible.sum())
    if count < cfg.get('min_teacher_pixels', 1024):
        raise RuntimeError(f'{role} real GPU smoke has insufficient KD support: {count}')
    report = {'passed': True, 'role': role, 'sample_id': row['sid'],
              'recipe': teacher_recipe(cfg, role), 'eligible_pixels': count,
              'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
              'load_seconds': load_seconds, 'inference_seconds': inference_seconds,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
    atomic_json(Path(cfg['work']) / (role + '_gpu_smoke.json'), report)
    print(f'{role} real GPU forward/target smoke PASS; KD support={count}', flush=True)


def commit_record(cfg, role, row, recipe, arrays, db):
    validate_arrays(arrays, role)
    target = cache_path(cfg, role, row['sid'])
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.teacher-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.save(f, arrays, allow_pickle=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, target)
        db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?,?)',
                   (row['sid'], row['fingerprint'], recipe, sha256(target), target.stat().st_size))
        db.commit()  # Publish AFTER a complete, checksum-validated array exists.
    finally:
        Path(temporary).unlink(missing_ok=True)


def already_done(cfg, role, row, recipe, db):
    record = db.execute('SELECT input_sha,recipe,sha,bytes FROM records WHERE sid=?', (row['sid'],)).fetchone()
    path = cache_path(cfg, role, row['sid'])
    if record is None:
        return False
    if record[:2] != (row['fingerprint'], recipe):
        raise RuntimeError(f'Cache recipe/input conflict for {row["sid"]}; new cache root required')
    if not path.is_file() or path.stat().st_size != record[3] or sha256(path) != record[2]:
        raise RuntimeError(f'Committed cache corruption: {path}; repair explicitly, never silently skip')
    validate_arrays(np.load(path, allow_pickle=False), role)
    return True


def storage_gate(cfg, rows):
    root = Path(cfg['teacher_root']); root.mkdir(parents=True, exist_ok=True)
    required = 0
    for role in ('metric', 'relative'):
        missing = sum(not cache_path(cfg, role, r['sid']).is_file() for r in rows)
        required += missing * (2 * SIZE[0] * SIZE[1] * 4 + 128)
    required += int(cfg.get('teacher_reserve_gib', 20) * 2**30)
    free = shutil.disk_usage(root).free
    if required > free:
        raise RuntimeError(f'Teacher storage needs {required/2**30:.1f} GiB free, found {free/2**30:.1f}. '
                           'Float32 NPY intentionally avoids training decompression. Use a larger persistent SSD.')
    return {'free_gib': free/2**30, 'required_gib': required/2**30}


def generate(cfg, role, stop):
    from teachers import build_teacher, teacher_recipe
    rows = load_index(cfg, 'train')
    recipe = teacher_recipe(cfg, role)
    recipe_sha = identity(recipe)
    freeze(Path(cfg['teacher_root']) / role / 'recipe.json', recipe)
    db = connection(cfg, role)
    teacher = None
    try:
        for i, row in enumerate(rows, 1):
            if stop.requested or (Path(cfg['work']) / 'STOP').exists():
                return False
            if not already_done(cfg, role, row, recipe_sha, db):
                if teacher is None:
                    teacher = build_teacher(cfg, role)
                arrays = predict_row(cfg, role, row, teacher)
                commit_record(cfg, role, row, recipe_sha, arrays, db)
            if i % 100 == 0 or i == len(rows):
                atomic_json(Path(cfg['teacher_root']) / role / 'progress.json',
                            {'verified': i, 'total_train': len(rows), 'recipe_sha256': recipe_sha})
                print(f'{role} committed/verified {i}/{len(rows)}', flush=True)
    finally:
        db.close()
    return True


def audit(cfg):
    from teachers import teacher_recipe
    rows = load_index(cfg, 'train'); result = {}
    # A failed re-audit must not leave an earlier successful gate authoritative.
    atomic_json(Path(cfg['work']) / 'data_gate.json', {'passed': False, 'status': 'auditing'})
    for role in ('metric', 'relative'):
        recipe = identity(read_json(Path(cfg['teacher_root']) / role / 'recipe.json'))
        if recipe != identity(teacher_recipe(cfg, role)):
            raise RuntimeError(f'{role} cache recipe differs from current pinned provider')
        coverage = []
        content = []
        with closing(connection(cfg, role)) as db:
            expected = {r['sid'] for r in rows}
            if {r[0] for r in db.execute('SELECT sid FROM records')} != expected:
                raise RuntimeError(f'{role} cache must cover EXACTLY all train IDs, no val/test or subset')
            for i, row in enumerate(rows, 1):
                if not already_done(cfg, role, row, recipe, db):
                    raise RuntimeError('Missing teacher; train blocked')
                sample = read_sample(row, cfg['dataset_root'], verify_hash=True)
                a = np.load(cache_path(cfg, role, row['sid']), allow_pickle=False)
                eligible = (a[1] >= (.5 if role == 'metric' else .35))
                eligible &= (sample['gt_mask'][0].numpy() == 0) & (sample['mask'][0].numpy() == 0)
                count = int(eligible.sum())
                if count < cfg.get('min_teacher_pixels', 1024):
                    raise RuntimeError(f'Insufficient non-GT/non-sensor {role} pixels: {row["sid"]}: {count}')
                coverage.append(count)
                content.append(db.execute('SELECT sha FROM records WHERE sid=?', (row['sid'],)).fetchone()[0])
                if i % 1000 == 0:
                    print(f'{role} full content/support audit {i}/{len(rows)}', flush=True)
        result[role] = {'cached_train': len(rows), 'cached_val': 0, 'cached_test': 0,
                        'min_eligible_pixels': min(coverage), 'median_eligible_pixels': float(np.median(coverage)),
                        'recipe_sha256': recipe, 'content_sha256': identity(content)}
    # Also recheck all held-out input contents before declaring a locked protocol.
    for split in ('val', 'test'):
        for row in load_index(cfg, split):
            read_sample(row, cfg['dataset_root'], split != 'test', verify_hash=True)
    report = {'passed': True, 'dataset': read_json(Path(cfg['work'])/'dataset_contract.json'), 'teachers': result}
    atomic_json(Path(cfg['work']) / 'data_gate.json', report)
    return report
