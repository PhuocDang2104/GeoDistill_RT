"""Assemble a standalone server bundle without touching frozen research sources."""
import hashlib
import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'production' / 'AnchorFlow_V11_3_FullKITTI'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standalone',action='store_true',help='Repackage committed vendored model without the local research bundle')
    args=parser.parse_args()
    parent = ROOT / 'drive_upload' / 'AnchorFlow_v11_3_PhaseInnovation'
    dest = OUT / 'v11_model'
    dest.mkdir(parents=True, exist_ok=True)
    if (parent/'bundle_manifest.json').exists() and not args.standalone:
        manifest = json.loads((parent / 'bundle_manifest.json').read_text(encoding='utf-8'))
        for name, checksum in manifest['files'].items():
            if name.endswith('.py') and hashlib.sha256((parent / name).read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f'Frozen V11.3 source changed: {name}')
        for source in parent.glob('*.py'):
            if source.name.startswith(('test_', 'retained_')):
                continue
            shutil.copy2(source, dest / source.name)
        shutil.copy2(parent / 'config.json', dest / 'reference_config.json')
    else:
        # Clean Git clones already contain the complete model. No research assets required.
        manifest=json.loads((OUT/'bundle_manifest.json').read_text(encoding='utf-8'))
        model_files={n:h for n,h in manifest['files'].items() if n.startswith('v11_model/')}
        if not model_files:raise RuntimeError('No committed frozen model in standalone manifest')
        for name,checksum in model_files.items():
            if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=checksum:
                raise RuntimeError(f'Committed frozen model modified: {name}')
    # Reuse the exact battle-tested S3 atomic checkpoint/RNG/stop primitives.
    runtime = OUT / 'server_runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ('__init__.py', 'common.py', 'recovery.py'):
        shutil.copy2(ROOT / 'src' / 'runtime' / name, runtime / name)
    files = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.rglob('*')) if p.is_file()
             and '__pycache__' not in p.parts and p.name not in ('bundle_manifest.json', 'verification.json')}
    (OUT / 'bundle_manifest.json').write_text(json.dumps({
        'schema': 1, 'frozen_model': 'V11.3 PhaseInnovation',
        'model_sources_unchanged': True, 'files': files}, indent=2), encoding='utf-8')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(OUT / name, arcname=OUT.name + '/' + name)
        z.write(OUT / 'bundle_manifest.json', arcname=OUT.name + '/bundle_manifest.json')
        if (OUT / 'verification.json').exists():
            z.write(OUT / 'verification.json', arcname=OUT.name + '/verification.json')
    print(json.dumps({'folder': str(OUT), 'zip': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
