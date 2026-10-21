"""Read Google Drive export ZIP inventories; copy ONLY canonical metric-only inputs.

Never deletes ZIPs or original data. Missing metric teacher remains explicitly missing;
never substitute old DA2/relative caches or generate a teacher from another model.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

NEEDED = ('teacher_subset_2000/selected_2000_ids.json',
          'teacher_subset_2000/kitti_trainval_2000.tar',
          'teacher_subset_2000/metric_coarse_train_2000.tar',
          'test_1000/kitti_test_1000.tar')
EXPECTED_SPLIT = 'f0f482799db63b1e89f4ef21be0aff71cc409bac8bde20259192f3bc1cd40ddb'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def inventory(root):
    found = {name: [] for name in NEEDED}
    summaries = []
    for path in sorted(root.glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            summaries.append({'zip': path.name, 'bytes': path.stat().st_size, 'entries': len(members)})
            for entry in members:
                parts = PurePosixPath(entry.filename.replace('\\', '/')).parts
                if '..' in parts or entry.is_dir(): continue
                for name in NEEDED:
                    if entry.filename.replace('\\', '/').endswith('/GeoLift_Data/'+name) or entry.filename.replace('\\', '/') == 'GeoLift_Data/'+name:
                        found[name].append({'zip': str(path.resolve()), 'member': entry.filename,
                                            'bytes': entry.file_size, 'crc32': entry.CRC})
    return found, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('D:/GeoLift_Data'))
    p.add_argument('--output', type=Path, default=Path('D:/GeoLift_MetricOnly_Transfer'))
    p.add_argument('--extract', action='store_true')
    a = p.parse_args()
    found, zips = inventory(a.source)
    if a.output.resolve() == a.source.resolve() or a.source.resolve() in a.output.resolve().parents:
        raise ValueError('Output must be a separate transfer folder')
    if a.output.exists(): raise FileExistsError('Use a new output folder; no overwrite/merge')
    a.output.mkdir(parents=True)
    result = {'source': str(a.source.resolve()), 'transfer_root': str(a.output.resolve()),
        'zip_inventory': zips, 'required': found, 'copied': [],
        'missing': [name for name, locations in found.items() if not locations],
        'note': 'Only metric teacher is needed, but RGB/sparse/GT/K + split JSON are still essential. Test1000 can be auto-downloaded by prepare.'}
    if a.extract:
        required_space = sum(entries[0]['bytes'] for entries in found.values() if entries)
        if shutil.disk_usage(a.output).free < required_space+512*1024**2:
            raise RuntimeError('Not enough space for verified copy + 512MiB reserve')
        for name, locations in found.items():
            if not locations: continue
            if len(locations) != 1: raise RuntimeError(f'Ambiguous duplicate candidates; inspect manually: {name}')
            location = locations[0]
            target = a.output/'GeoLift_Data'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f'Extracting ONLY {name} ({location["bytes"]:,} bytes)', flush=True)
            h = hashlib.sha256()
            with zipfile.ZipFile(location['zip']) as archive, archive.open(location['member']) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(8*1024*1024), b''):
                    output.write(block);h.update(block)
            actual = sha(target)
            if target.stat().st_size != location['bytes'] or actual != h.hexdigest():
                raise RuntimeError(f'Output size/checksum mismatch: {name}')
            if name.endswith('selected_2000_ids.json') and actual != EXPECTED_SPLIT:
                raise RuntimeError('Wrong selected2000 split; do not upload as matched baseline data')
            result['copied'].append({'path': 'GeoLift_Data/'+name, 'bytes': target.stat().st_size, 'sha256': actual})
            print(f'Verified {name}: {actual}', flush=True)
    result['ready_for_current_training'] = all(found[name] for name in NEEDED[:3]) and a.extract
    (a.output/'transfer_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    message = ('Upload the GeoLift_Data subfolder to the NEW account MyDrive.\n'
        'Do NOT upload relative generation/cache, anchorflow_cache or old checkpoints for fresh capacity training.\n'
        'Original ZIPs retained. Files missing below must be copied from the OLD Drive before training:\n'
        + '\n'.join(result['missing'])+'\n')
    (a.output/'READ_BEFORE_UPLOAD.txt').write_text(message, encoding='utf-8')
    print(message, flush=True)


if __name__ == '__main__': main()
