"""Official KITTI acquisition, resumable HTTP + selective, CRC-checked extraction."""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from utils import atomic_json, read_json, safe_path, sha256

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti'


def extract_selected(archive, root, wanted=None):
    root = Path(root)
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        if wanted is not None and not set(wanted).issubset(names):
            raise RuntimeError(f'Archive missing required entries: {list(set(wanted)-names)[:3]}')
        for item in z.infolist():
            if item.is_dir() or (wanted is not None and item.filename not in wanted):
                continue
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError('ZIP symlink forbidden')
            target = safe_path(root, item.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + '.download-partial')
            with z.open(item) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                output.flush(); os.fsync(output.fileno())
            partial.replace(target)  # ZipExtFile verified the CRC before publish.


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    partial = path.with_suffix('.zip.partial')
    # Public official assets only. Authentication/quota errors remain visible.
    subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--continue-at', '-',
                    '--output', str(partial), url], check=True)
    with zipfile.ZipFile(partial) as z:
        if not z.namelist():
            raise RuntimeError('Empty download')
    partial.replace(path)


def fetch_extract(cfg, name, url, destination, wanted=None):
    downloads = Path(cfg['dataset_root']) / '.downloads'
    marker = downloads / (name + '.complete.json')
    # Extraction may be interrupted; only the completion transaction is authoritative.
    if marker.exists():
        meta = read_json(marker)
        if meta['url'] != url:
            raise RuntimeError('Download URL changed under an existing completion marker')
        return
    if shutil.disk_usage(Path(cfg['dataset_root'])).free < cfg.get('dataset_download_min_free_gib', 120) * 2**30:
        raise RuntimeError('Insufficient free SSD space for remaining KITTI download/extraction')
    archive = downloads / name
    download(url, archive)
    checksum = sha256(archive)
    extract_selected(archive, destination, wanted)
    atomic_json(marker, {'url': url, 'download_sha256': checksum,
                         'bytes': archive.stat().st_size, 'destination': str(destination)})
    if not cfg.get('keep_download_archives', False):
        archive.unlink()  # Only our exact completed ZIP, never source data.


def acquire(cfg):
    if not cfg.get('download_dataset', True):
        return
    if not cfg.get('accept_kitti_license', False):
        raise RuntimeError('Read KITTI terms, then set accept_kitti_license=true before downloading.')
    root = Path(cfg['dataset_root'])
    root.mkdir(parents=True, exist_ok=True)
    for name in ('data_depth_annotated.zip', 'data_depth_velodyne.zip', 'data_depth_selection.zip'):
        fetch_extract(cfg, name, BASE + '/' + name, root)
    gt_files = sorted(root.glob('train/*/proj_depth/groundtruth/image_0[23]/*.png'))
    gt_files += sorted(root.glob('val/*/proj_depth/groundtruth/image_0[23]/*.png'))
    if not gt_files:
        raise RuntimeError('Official depth layout missing after download')
    drives = {p.parts[-5] for p in gt_files}
    raw = root / 'raw'
    for date in sorted({d[:10] for d in drives}):
        fetch_extract(cfg, date + '_calib.zip', BASE + '/raw_data/' + date + '_calib.zip', raw)
    for i, drive in enumerate(sorted(drives), 1):
        wanted = {f'{drive[:10]}/{drive}/{p.parts[-2]}/data/{p.name}'
                  for p in gt_files if p.parts[-5] == drive}
        name = drive + '.zip'
        fetch_extract(cfg, name, BASE + '/raw_data/' + drive.removesuffix('_sync') + '/' + name, raw, wanted)
        print(f'RGB drive {i}/{len(drives)}: {drive}', flush=True)
    print('KITTI downloaded; archives removed only after verified extraction. Dataset PNGs retained.', flush=True)
