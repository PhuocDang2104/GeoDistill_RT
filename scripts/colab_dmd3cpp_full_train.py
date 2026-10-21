"""Train-only DMD3C++ export, using the pinned V11.3 production teacher.

Colab notebook embeds this file and its production dependencies. No credentials,
relative teacher, student model, validation or anonymous-test generation.
"""
from contextlib import ExitStack, closing
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import time
import zipfile

EXPECTED_TRAIN = 85898
BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti'
FIXED_MTIME = 1600000000


def zip_bytes(archive, name, data):
    info = zipfile.ZipInfo(name, date_time=(2020, 9, 13, 12, 26, 40))
    info.compress_type = archive.compression
    archive.writestr(info, data)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)


def freeze(path, value):
    if Path(path).exists() and read_json(path) != value:
        raise RuntimeError(f'Recipe/source conflict: {path}. Use a new export folder.')
    save_json(path, value)


def disk_gate(root, gib):
    free = shutil.disk_usage(root).free / 2**30
    if free < gib:
        raise RuntimeError(f'Local disk: need {gib:.1f} GiB free, found {free:.1f}')


def publish(local, destination):
    """Commit only after reopening and hashing the Drive-side copy."""
    local, destination = Path(local), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {'file': destination.name, 'bytes': local.stat().st_size, 'sha256': digest(local)}
    if destination.exists():
        if destination.stat().st_size == record['bytes'] and digest(destination) == record['sha256']:
            return record
        raise RuntimeError(f'Existing committed-name artifact differs: {destination}')
    temporary = destination.with_name(destination.name + '.partial')
    with local.open('rb') as src, temporary.open('wb') as dst:
        shutil.copyfileobj(src, dst, 8 * 2**20)
        dst.flush()
        os.fsync(dst.fileno())
    if temporary.stat().st_size != record['bytes'] or digest(temporary) != record['sha256']:
        raise RuntimeError(f'Drive copy verification failed: {temporary}')
    os.replace(temporary, destination)
    return record


class RangeZip:
    """Read official ZIP central directories and required PNG members over HTTP ranges."""
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        import fsspec
        import requests
        response = requests.get(self.url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'},
                                stream=True, timeout=120)
        try:
            response.raise_for_status()
            if response.status_code != 206 or 'Content-Range' not in response.headers:
                raise RuntimeError('Official server does not support ranged ZIP reads; no full-archive fallback')
            self.source = {'url': self.url, 'bytes': int(response.headers['Content-Range'].split('/')[-1]),
                           'etag': response.headers.get('ETag')}
        finally:
            response.close()
        self.handle = fsspec.open(self.url, mode='rb', block_size=2**20, cache_type='blockcache',
                                 cache_options={'maxblocks': 8}).open()
        self.z = zipfile.ZipFile(self.handle)
        return self

    def __exit__(self, *args):
        self.z.close()
        self.handle.close()

    def extract(self, member, root, relative=None):
        target = (Path(root) / (relative or member)).resolve()
        if not target.is_relative_to(Path(root).resolve()):
            raise ValueError(f'Unsafe ZIP member: {member}')
        info = self.z.getinfo(member)
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('ZIP symlink forbidden')
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.z.open(info) as source, target.open('wb') as dest:
            shutil.copyfileobj(source, dest, 2**20)  # ZipExtFile validates CRC.
        os.utime(target, (FIXED_MTIME, FIXED_MTIME))
        return target


def enumerate_train(gt_zip, sparse_zip):
    names = sorted(n for n in gt_zip.z.namelist()
                   if n.startswith('train/') and '/proj_depth/groundtruth/image_0' in n and n.endswith('.png'))
    if len(names) != EXPECTED_TRAIN or len(set(names)) != EXPECTED_TRAIN:
        raise RuntimeError(f'Full official train requires {EXPECTED_TRAIN} unique GT images; found {len(names)}')
    sparse_names = set(sparse_zip.z.namelist())
    grouped = defaultdict(list)
    sids = set()
    for name in names:
        p = PurePosixPath(name)
        drive, cam = p.parts[-5], p.parts[-2]
        if cam not in ('image_02', 'image_03'):
            raise ValueError(cam)
        sparse = name.replace('/groundtruth/', '/velodyne_raw/')
        if sparse not in sparse_names:
            raise RuntimeError(f'Missing paired sparse input: {sparse}')
        sid = f'{drive}_image_{p.stem}_{cam}'
        if sid in sids:
            raise RuntimeError(f'Duplicate sample: {sid}')
        sids.add(sid)
        grouped[drive].append({'sid': sid, 'drive': drive, 'camera': cam, 'gt': name, 'sparse': sparse,
                              'rgb_member': f'{drive[:10]}/{drive}/{cam}/data/{p.name}'})
    val_drives = {PurePosixPath(n).parts[-5] for n in gt_zip.z.namelist()
                  if n.startswith('val/') and '/proj_depth/groundtruth/' in n and n.endswith('.png')}
    if set(grouped) & val_drives:
        raise RuntimeError('Official train/validation drive overlap')
    return dict(grouped)


def plan_shards(groups, chunk):
    if chunk < 1 or chunk > 1024:
        raise ValueError('shard_samples must be between 1 and 1024')
    plan = []
    for drive, rows in sorted(groups.items()):
        for start in range(0, len(rows), chunk):
            plan.append({'key': f'part-{len(plan):05d}', 'drive': drive, 'samples': rows[start:start+chunk]})
    return plan


def check_committed(root, part, recipe_sha):
    marker = Path(root) / 'commits' / (part['key'] + '.json')
    if not marker.exists():
        return None
    commit = read_json(marker)
    if commit['recipe_sha256'] != recipe_sha or commit['sample_ids'] != [s['sid'] for s in part['samples']]:
        raise RuntimeError(f'Shard resume conflict: {part["key"]}')
    for kind in ('inputs', 'teachers'):
        record = commit[kind]
        path = Path(root) / kind / record['file']
        if not path.exists() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise RuntimeError(f'Committed Drive shard corrupt/missing: {path}')
    return commit


def make_rows(part, stage, gt, sparse, raw, calib):
    from full_data import intrinsics, file_record, read_sample
    rows = []
    date = part['drive'][:10]
    cpath = f'{date}/calib_cam_to_cam.txt'
    calib.extract(cpath, stage, 'raw/' + cpath)
    for item in part['samples']:
        gt.extract(item['gt'], stage)
        sparse.extract(item['sparse'], stage)
        raw.extract(item['rgb_member'], stage, 'raw/' + item['rgb_member'])
        row = {'sid': item['sid'], 'drive': item['drive'],
               'K': intrinsics(stage / 'raw' / cpath, item['camera']),
               **{role: file_record(stage / name, stage) for role, name in (
                   ('rgb', 'raw/' + item['rgb_member']), ('gt', item['gt']), ('sparse', item['sparse']))}}
        row['fingerprint'] = identity(row)
        read_sample(row, stage, include_gt=True, verify_hash=True)
        rows.append(row)
    return rows


def make_teacher_shard(rows, stage, output, teacher, recipe_sha, minimum_pixels):
    import numpy as np
    import torch
    from full_data import read_sample
    from teacher_cache import validate_arrays
    contents = []
    started = time.monotonic()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for i, row in enumerate(rows, 1):
            sample = read_sample(row, stage, include_gt=True, verify_hash=True)
            arrays = teacher.predict(sample)
            validate_arrays(arrays, 'metric')
            eligible = (arrays[1] >= .5) & (sample['gt_mask'][0].numpy() == 0) & (sample['mask'][0].numpy() == 0)
            count = int(eligible.sum())
            if count < minimum_pixels:
                raise RuntimeError(f'Insufficient teacher KD support: {row["sid"]}: {count}')
            member = f'metric/{identity(row["sid"])[:2]}/{row["sid"]}.npy'
            buffer = io.BytesIO()
            np.save(buffer, arrays, allow_pickle=False)
            payload = buffer.getvalue()
            zip_bytes(z, member, payload)
            contents.append({'sid': row['sid'], 'path': member, 'input_sha256': row['fingerprint'],
                             'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
                             'eligible_pixels': count, 'recipe_sha256': recipe_sha})
            if i % 25 == 0 or i == len(rows):
                print(f'  DMD3C++ {i}/{len(rows)}; elapsed {time.monotonic()-started:.1f}s', flush=True)
        zip_bytes(z, 'records.json', json.dumps(contents))
    # Independent read-back, CRC, NPY decode and exact ID checks before transfer.
    with zipfile.ZipFile(output) as z:
        if {c['sid'] for c in contents} != {r['sid'] for r in rows}:
            raise RuntimeError('Teacher shard ID mismatch')
        for record in contents:
            payload = z.read(record['path'])
            if hashlib.sha256(payload).hexdigest() != record['sha256']:
                raise RuntimeError('Teacher shard content checksum mismatch')
            validate_arrays(np.load(io.BytesIO(payload), allow_pickle=False), 'metric')
    torch.cuda.synchronize()
    return contents


def make_input_shard(rows, stage, output):
    files = {p for row in rows for p in (row['rgb']['path'], row['gt']['path'], row['sparse']['path'])}
    files.update(p.relative_to(stage).as_posix() for p in (stage/'raw').glob('*/calib_cam_to_cam.txt'))
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name in sorted(files):
            z.write(stage/name, name)
        zip_bytes(z, 'train_index.json', json.dumps(rows))
    with zipfile.ZipFile(output) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f'Input ZIP failed CRC: {bad}')


def generate(cfg):
    from teachers import MetricTeacher, teacher_recipe
    import torch
    out, local = Path(cfg['drive_output']), Path(cfg['local_root'])
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    disk_gate(local, 15)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Select A100/H100/L4 native BF16 CUDA. T4 is unsupported by the pinned teacher.')
    teacher_cfg = {'third_party_root': str(local/'third_party'), 'hf_cache': str(local/'hf_cache')}
    recipe = {'schema': 1, 'teacher': teacher_recipe(teacher_cfg, 'metric'),
              'export_script_sha256': digest(__file__), 'expected_train': EXPECTED_TRAIN,
              'shard_samples': cfg['shard_samples'], 'min_teacher_pixels': cfg['min_teacher_pixels'],
              'teacher_format': 'ZIP shards containing production-layout float32 (2,352,1216) NPY',
              'input_mtime_seconds': FIXED_MTIME, 'other_teachers': [], 'val_test_teachers': False}
    freeze(out/'recipe.json', recipe)
    recipe_sha = identity(recipe)
    teacher = None
    with ExitStack() as stack:
        gt = stack.enter_context(RangeZip(BASE+'/data_depth_annotated.zip'))
        sparse = stack.enter_context(RangeZip(BASE+'/data_depth_velodyne.zip'))
        groups = enumerate_train(gt, sparse)
        plan = plan_shards(groups, cfg['shard_samples'])
        freeze(out/'plan.json', {'expected_train': EXPECTED_TRAIN, 'parts': plan,
                                'sources': [gt.source, sparse.source], 'recipe_sha256': recipe_sha})
        total_done = 0
        for drive in sorted(groups):
            parts = [p for p in plan if p['drive'] == drive]
            pending = []
            for part in parts:
                commit = check_committed(out, part, recipe_sha)
                if commit is None:
                    pending.append(part)
                else:
                    total_done += len(part['samples'])
            if not pending:
                continue
            date = drive[:10]
            url = BASE+'/raw_data/'+drive.removesuffix('_sync')+'/'+drive+'.zip'
            with RangeZip(url) as raw, RangeZip(BASE+'/raw_data/'+date+'_calib.zip') as calib:
                freeze(out/'sources'/(drive+'.json'), {'raw': raw.source, 'calib': calib.source})
                for part in pending:
                    if (out/'STOP').exists():
                        save_json(out/'status.json', {'state': 'stopped', 'committed_train': total_done,
                                                     'expected_train': EXPECTED_TRAIN})
                        return
                    disk_gate(local, 15)
                    stage = local/'stage'/part['key']
                    if stage.exists():
                        assert stage.resolve().is_relative_to((local/'stage').resolve())
                        shutil.rmtree(stage)
                    stage.mkdir(parents=True)
                    started = time.monotonic()
                    save_json(out/'status.json', {'state': 'processing', 'part': part['key'],
                                                 'committed_train': total_done, 'expected_train': EXPECTED_TRAIN})
                    rows = make_rows(part, stage, gt, sparse, raw, calib)
                    if teacher is None:
                        torch.cuda.reset_peak_memory_stats()
                        teacher = MetricTeacher(teacher_cfg)
                        # Mandatory real-data GPU smoke before the first full shard.
                        from full_data import read_sample
                        from teacher_cache import validate_arrays
                        smoke_start = time.monotonic()
                        arrays = teacher.predict(read_sample(rows[0], stage, verify_hash=True))
                        validate_arrays(arrays, 'metric')
                        torch.cuda.synchronize()
                        save_json(out/'gpu_smoke.json', {'passed': True, 'sample_id': rows[0]['sid'],
                                  'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
                                  'inference_seconds': time.monotonic()-smoke_start,
                                  'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20})
                    input_zip, teacher_zip = stage/'inputs.zip', stage/'teachers.zip'
                    make_input_shard(rows, stage, input_zip)
                    records = make_teacher_shard(rows, stage, teacher_zip, teacher, recipe_sha, cfg['min_teacher_pixels'])
                    # Content names let an interrupted uncommitted publication coexist
                    # with a recomputed GPU result without overwriting or trusting it.
                    inputs = publish(input_zip, out/'inputs'/(part['key']+'-'+digest(input_zip)[:16]+'.zip'))
                    teachers = publish(teacher_zip, out/'teachers'/(part['key']+'-'+digest(teacher_zip)[:16]+'.zip'))
                    commit = {'key': part['key'], 'recipe_sha256': recipe_sha,
                              'sample_ids': [r['sid'] for r in rows], 'inputs': inputs, 'teachers': teachers,
                              'train_rows': rows, 'teacher_records': records,
                              'elapsed_seconds': time.monotonic()-started}
                    save_json(out/'commits'/(part['key']+'.json'), commit)
                    total_done += len(rows)
                    save_json(out/'status.json', {'state': 'generating', 'committed_train': total_done,
                                                 'expected_train': EXPECTED_TRAIN, 'last_part': part['key']})
                    print(f'Drive VERIFIED {total_done}/{EXPECTED_TRAIN}: {part["key"]}', flush=True)
                    shutil.rmtree(stage)
    save_json(out/'status.json', {'state': 'generated_pending_final_audit',
                                 'committed_train': total_done, 'expected_train': EXPECTED_TRAIN})


def audit(cfg):
    out = Path(cfg['drive_output'])
    if (out/'COMPLETE.json').exists():
        save_json(out/'COMPLETE.json', {'passed': False, 'state': 'reauditing'})
    save_json(out/'audit.json', {'passed': False, 'state': 'auditing'})
    plan = read_json(out/'plan.json')
    recipe_sha = identity(read_json(out/'recipe.json'))
    ids, records, rows, commits = [], [], [], []
    for part in plan['parts']:
        commit = check_committed(out, part, recipe_sha)
        if commit is None:
            raise RuntimeError(f'Incomplete export: {part["key"]}')
        ids.extend(commit['sample_ids'])
        rows.extend(commit['train_rows'])
        records.extend(commit['teacher_records'])
        commits.append(commit)
    expected = [s['sid'] for p in plan['parts'] for s in p['samples']]
    if len(ids) != EXPECTED_TRAIN or len(set(ids)) != EXPECTED_TRAIN or ids != expected:
        raise RuntimeError('Full train coverage/duplicate/order check failed')
    if {r['sid'] for r in records} != set(ids) or len(records) != EXPECTED_TRAIN:
        raise RuntimeError('Teacher exact coverage failed')
    save_json(out/'train_index.json', rows)
    save_json(out/'teacher_records.json', records)
    report = {'passed': True, 'train_samples': len(ids), 'val_teachers': 0, 'test_teachers': 0,
              'recipe_sha256': recipe_sha, 'train_ids_sha256': identity(ids),
              'shards': len(commits), 'minimum_eligible_pixels': min(r['eligible_pixels'] for r in records),
              'teacher_content_sha256': identity([r['sha256'] for r in records])}
    save_json(out/'audit.json', report)
    print('FULL TRAIN TEACHER AUDIT PASS', report, flush=True)
    return commits, rows, records, report


def finalize(cfg):
    """Assemble one complete train ZIP on Colab SSD, then copy and hash it on Drive."""
    out, local = Path(cfg['drive_output']), Path(cfg['local_root'])
    commits, rows, records, report = audit(cfg)
    required = sum(c['inputs']['bytes'] for c in commits) + max(c['inputs']['bytes'] for c in commits)
    disk_gate(local, required/2**30 + 10)
    target = local/'kitti_train_full_85898.zip'
    seen = {}
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as dest:
        for commit in commits:
            staged = local/'assemble-shard.zip'
            shutil.copyfile(out/'inputs'/commit['inputs']['file'], staged)
            if digest(staged) != commit['inputs']['sha256']:
                raise RuntimeError('Assembly source checksum failed')
            with zipfile.ZipFile(staged) as source:
                for info in source.infolist():
                    if info.filename == 'train_index.json':
                        continue
                    signature = (info.CRC, info.file_size)
                    if info.filename in seen:
                        if seen[info.filename] != signature:
                            raise RuntimeError('Conflicting repeated calibration file')
                        continue
                    with source.open(info) as src, dest.open(info.filename, 'w', force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 8*2**20)
                    seen[info.filename] = signature
            staged.unlink()
        zip_bytes(dest, 'train_index.json', json.dumps(rows))
        zip_bytes(dest, 'dataset_contract.json', json.dumps(report))
        zip_bytes(dest, 'README.txt', 'Official KITTI train only: 85898 RGB/sparse/GT/K pairs. '
                      'Depth PNG uint16/256 metres. Index paths relative to extraction root. '
                      'After extraction set input file mtimes to 1600000000 seconds before using '
                      'the production read_sample adapter, or rebuild file metadata/fingerprints. '
                      'KITTI terms: https://www.cvlibs.net/datasets/kitti/\n')
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise RuntimeError('Full train ZIP CRC failed')
        if len([n for n in z.namelist() if '/groundtruth/' in n and n.endswith('.png')]) != EXPECTED_TRAIN:
            raise RuntimeError('Full train ZIP sample count failed')
    archive = publish(target, out/target.name)
    # Production-compatible metadata, ready to unpack metric/ shards alongside this database.
    dbpath = local/'records.sqlite3'
    dbpath.unlink(missing_ok=True)
    with closing(sqlite3.connect(dbpath)) as db:
        db.execute('CREATE TABLE records(sid TEXT PRIMARY KEY,input_sha TEXT,recipe TEXT,sha TEXT,bytes INTEGER)')
        teacher_sha = identity(read_json(out/'recipe.json')['teacher'])
        db.executemany('INSERT INTO records VALUES(?,?,?,?,?)',
                       [(r['sid'], r['input_sha256'], teacher_sha, r['sha256'], r['bytes']) for r in records])
        db.commit()
    db_record = publish(dbpath, out/'metric_metadata'/'records.sqlite3')
    freeze(out/'metric_metadata'/'recipe.json', read_json(out/'recipe.json')['teacher'])
    save_json(out/'COMPLETE.json', {'passed': True, 'full_train_zip': archive,
                                  'metric_database': db_record, 'audit': report,
                                  'teacher_shards': [c['teachers'] for c in commits]})
    save_json(out/'status.json', {'state': 'complete', 'committed_train': EXPECTED_TRAIN,
                                 'full_train_zip': archive, 'teacher_shards': len(commits)})
    print('COMPLETE: full train ZIP and all DMD3C++ teacher shards verified on Drive.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate', 'audit', 'finalize'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    globals()[args.action](read_json(args.config))
