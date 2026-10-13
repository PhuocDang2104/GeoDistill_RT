"""Clone once / seal a separate fixed-six-NFE full-KITTI production package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'production/AnchorFlow_V11_3_FullKITTI'
OUT = ROOT/'production/AnchorFlow_V11_3_Fixed6_FullKITTI'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', action='store_true', help='Copy original sources once; refuses existing target')
    args = parser.parse_args()
    if args.init:
        if OUT.exists(): raise RuntimeError('Refuse to overwrite existing fixed6 bundle')
        original = json.loads((PARENT/'bundle_manifest.json').read_text(encoding='utf8'))
        for name, expected in original['files'].items():
            if sha(PARENT/name) != expected: raise RuntimeError(f'Parent source changed: {name}')
        shutil.copytree(PARENT, OUT, ignore=shutil.ignore_patterns('__pycache__','*.pyc',
                        'bundle_manifest.json','verification.json','server-*'))
        print(OUT); return
    original = json.loads((PARENT/'bundle_manifest.json').read_text(encoding='utf8'))
    # Only numerical integration differs within the vendored model tree.
    preserved = {}
    for name, expected in original['files'].items():
        if name.startswith(('v11_model/','server_runtime/')) and name != 'v11_model/ode_solver.py':
            assert sha(OUT/name) == expected, f'Unexpected model/loss/runtime change: {name}'
            preserved[name] = expected
    for name in ('teachers.py','teacher_cache.py','bootstrap_teachers.py','extension_compat.py',
                 'full_data.py','acquire.py','requirements.txt'):
        assert sha(OUT/name) == original['files'][name], f'Teacher/data recipe changed: {name}'
        preserved[name] = original['files'][name]
    files = {p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
             and p.name not in ('bundle_manifest.json','verification.json')}
    manifest = {'schema':1, 'variant':'V11.3 Fixed6 FullKITTI',
        'frozen_model':'V11.3 learned field/readout; coarse solver changed to fixed RK3',
        'model_sources_unchanged':False, 'solver':'classic_rk3_two_half_steps',
        'coarse_nfe':6, 'fine_nfe':2, 'total_field_nfe':8,
        'original_bundle_manifest_sha256':sha(PARENT/'bundle_manifest.json'),
        'preserved_original_file_sha256':preserved,
        'gpu_training_verified':False, 'files':files}
    (OUT/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive = OUT.with_suffix('.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for name in files: z.write(OUT/name,OUT.name+'/'+name)
        z.write(OUT/'bundle_manifest.json',OUT.name+'/bundle_manifest.json')
        if (OUT/'verification.json').exists(): z.write(OUT/'verification.json',OUT.name+'/verification.json')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in files.items():
            assert hashlib.sha256(z.read(OUT.name+'/'+name)).hexdigest() == expected
    print(json.dumps({'folder':str(OUT),'zip':str(archive),'bytes':archive.stat().st_size,
                      'preserved_files':len(preserved),'files':len(files)},indent=2))

if __name__ == '__main__': main()
