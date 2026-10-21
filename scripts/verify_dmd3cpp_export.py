"""Read-only manifest and stratified payload checks for the Colab DMD3C++ export.

Does not call exporter.audit(), invalidate COMPLETE.json, or load a GPU model.
One paired input/teacher sample per shard is decoded; all shard directories and
all record fingerprints are checked. This is not a fresh full-payload hash pass.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import zipfile


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(root, payloads=True):
    import numpy as np
    import cv2
    cv2.setNumThreads(0)
    root = Path(root)
    recipe = read(root/'recipe.json')
    recipe_sha = identity(recipe)
    plan = read(root/'plan.json')
    require(recipe['other_teachers'] == [] and recipe['val_test_teachers'] is False, 'Teacher scope mismatch')
    require(recipe['expected_train'] == plan['expected_train'] == 85898, 'Expected count mismatch')
    require(plan['recipe_sha256'] == recipe_sha, 'Plan recipe mismatch')
    rows, records, ids, inventory, samples = [], [], [], [], []
    expected = [s['sid'] for p in plan['parts'] for s in p['samples']]
    def check_part(item):
        i, part = item
        inventory, samples = [], []
        c = read(root/'commits'/(part['key']+'.json'))
        require(c['key'] == part['key'] and c['recipe_sha256'] == recipe_sha, 'Commit recipe/key mismatch')
        part_ids = [s['sid'] for s in part['samples']]
        require(c['sample_ids'] == part_ids == [r['sid'] for r in c['train_rows']], 'Commit input IDs mismatch')
        require(part_ids == [r['sid'] for r in c['teacher_records']], 'Commit teacher IDs mismatch')
        for row, record in zip(c['train_rows'], c['teacher_records']):
            require(row['fingerprint'] == identity({k:v for k,v in row.items() if k != 'fingerprint'}), 'Input fingerprint mismatch')
            require(record['input_sha256'] == row['fingerprint'] and record['recipe_sha256'] == recipe_sha, 'Teacher lineage mismatch')
            require(record['path'] == f'metric/{identity(row["sid"])[:2]}/{row["sid"]}.npy', 'Teacher path mismatch')
            require(record['bytes'] == 2*352*1216*4+128 and record['eligible_pixels'] >= recipe['min_teacher_pixels'], 'Teacher byte/support mismatch')
            for role in ('rgb', 'gt', 'sparse'):
                rel = PurePosixPath(row[role]['path'])
                require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe input path')
                require(row[role]['mtime_ns'] == 1600000000*10**9, 'Unexpected indexed mtime')
        paths = {}
        for kind in ('inputs', 'teachers'):
            record = c[kind]
            require(Path(record['file']).name == record['file'], 'Unsafe shard name')
            path = root/kind/record['file']
            require(path.stat().st_size == record['bytes'], 'Shard size mismatch: '+str(path))
            inventory.append({'kind':kind, **record})
            paths[kind] = path
        with zipfile.ZipFile(paths['teachers']) as tz, zipfile.ZipFile(paths['inputs']) as iz:
            require(len(tz.namelist()) == len(set(tz.namelist())), 'Duplicate teacher ZIP member')
            require(set(tz.namelist()) == {r['path'] for r in c['teacher_records']} | {'records.json'}, 'Teacher directory mismatch')
            require(read_zip_json(tz, 'records.json') == c['teacher_records'], 'Embedded teacher manifest mismatch')
            require(read_zip_json(iz, 'train_index.json') == c['train_rows'], 'Embedded input index mismatch')
            names = iz.namelist()
            require(len(names) == len(set(names)), 'Duplicate input ZIP member')
            for row in c['train_rows']:
                for role in ('rgb', 'gt', 'sparse'):
                    require(iz.getinfo(row[role]['path']).file_size == row[role]['bytes'], 'Input directory/size mismatch')
            if payloads:
                j = len(c['train_rows'])//2
                row, record = c['train_rows'][j], c['teacher_records'][j]
                data = tz.read(record['path'])  # ZipExtFile verifies CRC on full read.
                require(hashlib.sha256(data).hexdigest() == record['sha256'], 'Sample NPY SHA mismatch')
                a = np.load(io.BytesIO(data), allow_pickle=False)
                require(a.shape == (2,352,1216) and a.dtype == np.float32 and np.isfinite(a).all(), 'Sample array contract failed')
                require(a[1].min() >= 0 and a[1].max() <= 1, 'Sample confidence bounds failed')
                require(((a[0] > .1) & (a[0] < 120)).any(), 'No valid sample depth')
                images = {}
                for role in ('rgb', 'gt', 'sparse'):
                    data = iz.read(row[role]['path'])
                    require(hashlib.sha256(data).hexdigest() == row[role]['sha256'], 'Sample input SHA mismatch')
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR if role == 'rgb' else cv2.IMREAD_UNCHANGED)
                    require(image is not None, 'Sample PNG decode failed')
                    if role != 'rgb':
                        require(image.dtype == np.uint16, 'Sample depth PNG dtype failed')
                    images[role] = image
                require(len({x.shape[:2] for x in images.values()}) == 1, 'Paired image dimensions differ')
                k = np.asarray(row['K'])
                require(k.shape == (3,3) and np.isfinite(k).all() and k[0,0] > 0 and k[1,1] > 0 and np.allclose(k[2], [0,0,1]), 'Invalid intrinsics')
                mask = lambda role: cv2.resize(images[role].astype(np.float32)/256, (1216,352), interpolation=cv2.INTER_NEAREST)
                g, s = mask('gt'), mask('sparse')
                eligible = int(((a[1] >= .5) & ~((g > .1)&(g < 120)) & ~((s > .1)&(s < 120))).sum())
                require(eligible == record['eligible_pixels'], 'Eligible support differs from committed record')
                samples.append({'part':part['key'], 'sid':row['sid'], 'eligible_pixels':eligible,
                                'depth_min_m':float(a[0].min()), 'depth_max_m':float(a[0].max()),
                                'confidence_min':float(a[1].min()), 'confidence_max':float(a[1].max())})
        return c, inventory, samples
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (c, part_inventory, part_samples) in enumerate(pool.map(check_part, enumerate(plan['parts']))):
            rows.extend(c['train_rows']); records.extend(c['teacher_records']); ids.extend(c['sample_ids'])
            inventory.extend(part_inventory); samples.extend(part_samples)
            if (i+1) % 20 == 0 or i+1 == len(plan['parts']):
                print(f'Checked directories/lineage {i+1}/{len(plan["parts"])} shards; sampled payloads={len(samples)}', flush=True)
    require(ids == expected and len(ids) == len(set(ids)) == 85898, 'Full coverage/uniqueness failed')
    sources = read(root/'source_manifest.json')
    require(all(digest(root/'source'/n) == sha for n,sha in sources.items()), 'Saved source SHA mismatch')
    repair = root/'repairs'/'confidence_roundoff_v1'
    policy = read(repair/'policy.json') if (repair/'policy.json').exists() else None
    events = [json.loads(x) for x in (repair/'events.jsonl').read_text().splitlines() if x.strip()] if (repair/'events.jsonl').exists() else []
    if policy:
        require(digest(repair/'dmd3cpp_confidence_repair.py') == policy['repair_source_sha256'], 'Repair source SHA mismatch')
        id_set = set(ids)
        for e in events:
            require(e['sid'] in id_set and e['policy_sha256'] == identity(policy) and e['base_recipe_sha256'] == recipe_sha, 'Repair event lineage mismatch')
            require(e['confidence_min_before'] >= -policy['tolerance'] and e['confidence_max_before'] <= 1+policy['tolerance'], 'Material confidence correction')
    result = {'schema':1, 'verified_utc':datetime.now(timezone.utc).isoformat(),
              'manifest_directory_checks_passed':True, 'full_shard_sha256_recheck':False,
              'samples_checked':len(samples), 'train_samples':len(ids), 'shards':len(plan['parts']),
              'drives':len({r['drive'] for r in rows}), 'cameras':dict(Counter(r['sid'][-8:] for r in rows)),
              'recipe_sha256':recipe_sha, 'train_ids_sha256':identity(ids),
              'teacher_content_sha256':identity([r['sha256'] for r in records]),
              'minimum_eligible_pixels':min(r['eligible_pixels'] for r in records),
              'input_zip_bytes':sum(c['bytes'] for c in inventory if c['kind']=='inputs'),
              'teacher_zip_bytes':sum(c['bytes'] for c in inventory if c['kind']=='teachers'),
              'teacher_npy_bytes':sum(r['bytes'] for r in records),
              'source_manifest':sources, 'recipe':recipe, 'repair_policy':policy,
              'repair_event_count':len(events), 'repair_unique_samples':len({e['sid'] for e in events}),
              'repair_corrected_pixels':sum(e['corrected_pixels'] for e in events),
              'repair_min_confidence_before':min((e['confidence_min_before'] for e in events), default=None),
              'repair_max_confidence_before':max((e['confidence_max_before'] for e in events), default=None),
              'observed_status':read(root/'status.json'),
              'observed_audit':read(root/'audit.json') if (root/'audit.json').exists() else None,
              'complete_present':(root/'COMPLETE.json').exists(),
              'full_train_zip_present':(root/'kitti_train_full_85898.zip').exists(),
              'inventory':inventory, 'sample_checks':samples}
    for n in ('train_index.json', 'teacher_records.json'):
        if (root/n).exists():
            require(read(root/n) == (rows if n == 'train_index.json' else records), 'Top-level manifest differs')
    if (root/'COMPLETE.json').exists():
        complete = read(root/'COMPLETE.json')
        require(complete['passed'] and complete['audit']['train_samples'] == 85898, 'Completion marker failed')
        require(complete['teacher_shards'] == [{k:v for k,v in x.items() if k != 'kind'} for x in inventory if x['kind']=='teachers'], 'Completion inventory differs')
        for key, folder in (('full_train_zip', root), ('metric_database', root/'metric_metadata')):
            record = complete[key]
            require((folder/record['file']).stat().st_size == record['bytes'], 'Completion file size differs')
        dbpath = root/'metric_metadata'/'records.sqlite3'
        require(digest(dbpath) == complete['metric_database']['sha256'], 'Database SHA differs')
        with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro', uri=True) as db:
            require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'SQLite integrity failed')
            actual = {r[0]:r[1:] for r in db.execute('SELECT sid,input_sha,recipe,sha,bytes FROM records')}
        require(actual == {r['sid']:(r['input_sha256'],identity(recipe['teacher']),r['sha256'],r['bytes']) for r in records}, 'SQLite lineage differs')
        result['completion'] = complete
        result['metric_database_integrity'] = 'ok'
    return result


def read_zip_json(z, name):
    return json.loads(z.read(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.root.resolve()), 'Write report outside the export root')
    report = verify(args.root, payloads=not args.metadata_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inventory','sample_checks','completion')}, indent=2))
