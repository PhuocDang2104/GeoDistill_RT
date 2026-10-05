"""Full official split adapter; no TAR2000 assumptions, no validation teacher access."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from utils import atomic_json, freeze, identity, read_json, sha256

SIZE = (352, 1216)
cv2.setNumThreads(0)


def intrinsics(path, camera):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(':')
        if key == 'P_rect_' + camera[-2:]:
            values[key] = np.fromstring(value, sep=' ')
    p = values['P_rect_' + camera[-2:]].reshape(3, 4)
    return p[:, :3].tolist()


def file_record(path, root):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {'path': path.relative_to(root).as_posix(), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'sha256': sha256(path)}


def make_index(cfg):
    root, work = Path(cfg['dataset_root']).resolve(), Path(cfg['work'])
    work.mkdir(parents=True, exist_ok=True)
    rows, drive_sets = {}, {}
    for split in ('train', 'val'):
        # Supports both direct official extraction and DMD3Cpp's reorganized layout.
        folder = root / split
        if not folder.is_dir():
            folder = root / 'data_depth_annotated' / split
        depths = sorted(folder.glob('*/proj_depth/groundtruth/image_0[23]/*.png'))
        rows[split] = []
        drive_sets[split] = set()
        for i, gt in enumerate(depths, 1):
            drive, cam = gt.parts[-5], gt.parts[-2]
            drive_sets[split].add(drive)
            sparse = gt.parent.parent.parent / 'velodyne_raw' / cam / gt.name
            if not sparse.is_file():
                sparse = root / 'data_depth_velodyne' / split / drive / 'proj_depth' / 'velodyne_raw' / cam / gt.name
            raw = root / 'raw'
            if not raw.is_dir():
                raw = root
            rgb = raw / drive[:10] / drive / cam / 'data' / gt.name
            calib = raw / drive[:10] / 'calib_cam_to_cam.txt'
            sid = f'{drive}_image_{gt.stem}_{cam}'
            row = {'sid': sid, 'drive': drive, 'K': intrinsics(calib, cam),
                   **{k: file_record(p, root) for k, p in (('rgb', rgb), ('sparse', sparse), ('gt', gt))}}
            row['fingerprint'] = identity(row)
            # Decode paired pixels and K before expensive teacher inference.
            read_sample(row, root, include_gt=True)
            rows[split].append(row)
            if i % 1000 == 0:
                print(f'Paired KITTI index + content SHA {split} {i}/{len(depths)}', flush=True)
    if drive_sets['train'] & drive_sets['val']:
        raise RuntimeError('Raw drive leakage between official train and validation')
    rows['test'] = []
    test = root / 'depth_selection' / 'test_depth_completion_anonymous'
    if not test.is_dir():
        test = root / 'test_depth_completion_anonymous'
    for rgb in sorted((test / 'image').glob('*.png')):
        kpath = test / 'intrinsics' / (rgb.stem + '.txt')
        k = np.fromstring(kpath.read_text(), sep=' ').reshape(3, 3)
        row = {'sid': rgb.stem, 'drive': None, 'K': k.tolist(),
               'rgb': file_record(rgb, root),
               'sparse': file_record(test / 'velodyne_raw' / rgb.name, root)}
        row['fingerprint'] = identity(row)
        read_sample(row, root, include_gt=False)
        rows['test'].append(row)
    for split in ('train', 'val', 'test'):
        n = len(rows[split])
        expected = cfg['expected_counts'][split]
        if n != expected or len({r['sid'] for r in rows[split]}) != n:
            raise RuntimeError(f'{split}: expected {expected} unique paired records, found {n}. No silent subset.')
    if {r['sid'] for r in rows['train']} & {r['sid'] for r in rows['val']}:
        raise RuntimeError('Duplicate train/val IDs')
    data = {'schema': 1, 'counts': {k: len(v) for k, v in rows.items()},
            'shape': list(SIZE), 'resize': 'full-frame linear RGB/nearest depth; K scaled; matches V11.3',
            'depth_scale': 256, 'bounds_m': [.1, 120],
            'split_ids_sha256': {k: identity([r['sid'] for r in v]) for k, v in rows.items()},
            'input_content_sha256': {k: identity([r['fingerprint'] for r in v]) for k, v in rows.items()},
            'train_val_drive_disjoint': True, 'teacher_on_val_test': False}
    freeze(work / 'dataset_contract.json', data)
    for split, records in rows.items():
        atomic_json(work / (split + '_index.json'), records)
    return data


def load_index(cfg, split):
    return read_json(Path(cfg['work']) / (split + '_index.json'))


def read_sample(row, root, include_gt=True, verify_hash=False):
    arrays = {}
    for role in ('rgb', 'sparse', *(['gt'] if include_gt else [])):
        meta = row[role]; path = Path(root) / meta['path']
        stat = path.stat()
        if stat.st_size != meta['bytes'] or stat.st_mtime_ns != meta['mtime_ns']:
            raise RuntimeError(f'Dataset modified after index: {path}; rebuild as a new protocol')
        if verify_hash and sha256(path) != meta['sha256']:
            raise RuntimeError(f'Dataset content checksum mismatch: {path}')
        image = cv2.imread(str(path), cv2.IMREAD_COLOR if role == 'rgb' else cv2.IMREAD_UNCHANGED)
        if image is None or (role != 'rgb' and image.dtype != np.uint16):
            raise RuntimeError(f'Invalid KITTI {role}: {path}')
        arrays[role] = image
    original = arrays['rgb'].shape[:2]
    if any(x.shape[:2] != original for x in arrays.values()):
        raise RuntimeError(f'Paired spatial mismatch: {row["sid"]}')
    rgb = cv2.cvtColor(cv2.resize(arrays['rgb'], SIZE[::-1], interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    result = {'sid': row['sid'], 'rgb': torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()/255}
    for role in ('sparse', 'gt'):
        depth = arrays.get(role, np.zeros(original, np.uint16)).astype(np.float32)/256
        depth = cv2.resize(depth, SIZE[::-1], interpolation=cv2.INTER_NEAREST)
        valid = (depth > .1) & (depth < 120)
        result[role] = torch.from_numpy(np.where(valid, depth, 0)[None])
        result['mask' if role == 'sparse' else 'gt_mask'] = torch.from_numpy(valid[None].astype(np.float32))
    k = np.asarray(row['K'], np.float32).copy()
    k[0] *= SIZE[1]/original[1]; k[1] *= SIZE[0]/original[0]
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise RuntimeError(f'Invalid camera intrinsics: {row["sid"]}')
    result['K'] = torch.from_numpy(k)
    return result


class FullKITTI(Dataset):
    def __init__(self, cfg, split, teachers=False):
        if teachers and split != 'train':
            raise ValueError('Teacher loading forbidden for val/test')
        self.cfg, self.split, self.teachers = cfg, split, teachers
        self.rows = load_index(cfg, split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]  # Legacy deterministic sampler is supported too.
        row = self.rows[index]
        result = read_sample(row, self.cfg['dataset_root'], self.split != 'test')
        if self.teachers:
            from teacher_cache import cache_path
            for role, keys in (('metric', ('teacher', 'confidence')), ('relative', ('relative', 'relative_confidence'))):
                a = np.load(cache_path(self.cfg, role, row['sid']), mmap_mode='r', allow_pickle=False)
                if a.shape != (2, *SIZE) or a.dtype != np.float32:
                    raise RuntimeError(f'Invalid {role} cache: {row["sid"]}')
                # Copy read-only mmap once; no decompression, bounded prefetch memory.
                for key, channel in zip(keys, a):
                    result[key] = torch.from_numpy(np.array(channel[None], copy=True))
        return result


class ResumableBatches(Sampler):
    """Every sample once per epoch, including final short batch; consumed cursor, not prefetched."""
    def __init__(self, length, batch_size, seed, epoch=0, start=0):
        self.length, self.batch_size, self.seed, self.epoch, self.start = length, batch_size, seed, epoch, start

    def __len__(self):
        return (self.length + self.batch_size - 1)//self.batch_size

    def __iter__(self):
        order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        for batch in range(self.start, len(self)):
            yield order[batch*self.batch_size:(batch+1)*self.batch_size]
