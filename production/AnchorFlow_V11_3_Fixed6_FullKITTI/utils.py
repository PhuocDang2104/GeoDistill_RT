"""Small, portable transaction helpers; no dependency on Drive or repo cwd."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Unsafe path: {name}')
    return path


def freeze(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f'Frozen recipe changed: {path}. Use a NEW work/cache/run directory.')
    atomic_json(path, value)


def source_identity():
    root = Path(__file__).parent
    return identity({p.relative_to(root).as_posix(): sha256(p)
                     for p in sorted(root.rglob('*.py'))
                     if '__pycache__' not in p.parts and not p.name.startswith('test_')})


def numerical_environment():
    from importlib.metadata import version
    import cv2
    result={name: version(name) for name in ('torch','torchvision','timm','torchdiffeq','numpy','safetensors')}
    result['opencv']={'version':cv2.__version__,
                       'build_sha256':hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()}
    return result
