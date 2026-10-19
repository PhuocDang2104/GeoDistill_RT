"""Verify frozen train bytes, package train + Orin code. No parent-bundle dependency."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT/'drive_upload/AnchorFlow_S4_UPIR_NoRelative_30ep_A100'
ORIN = ROOT/'production/AnchorFlow_S4_UPIR_Orin'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((TRAIN/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = TRAIN/name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f'Frozen training bundle changed: {name}')
    # No regeneration of original notebook/source/manifest, no cross-OS rehash
    # inside the checkpoint contract. Orin helpers always stay separate.
    for directory, names in ((TRAIN, [*manifest['files'], 'bundle_manifest.json']),
            (ORIN, [p.name for p in sorted(ORIN.iterdir()) if p.is_file() and p.suffix in ('.py', '.md', '.json')])):
        dest = directory.with_suffix('.zip')
        with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names: archive.write(directory/name, directory.name+'/'+name)
        with zipfile.ZipFile(dest) as archive:
            assert archive.testzip() is None
        print(f'{dest} ({dest.stat().st_size:,} bytes) SHA256={sha(dest)}')
    print('Frozen training manifest verified. ZIPs contain NO trained weights or ONNX assets.')


if __name__ == '__main__': main()
