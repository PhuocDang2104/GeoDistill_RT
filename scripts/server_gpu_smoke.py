"""Full-resolution synthetic KD backward check for a newly provisioned GPU.

Not an accuracy benchmark. Real-data calibration remains required.
"""
import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.runtime.common import atomic_json
from src.runtime.probe import trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/runs/gpu-smoke.json')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rng = np.random.RandomState(42)
        rows = []
        h, w = 352, 1216
        yy, xx = np.mgrid[:h, :w]
        for index in range(4):
            depth = (10 + xx / 50 + yy / 40).astype('float32')
            gt = np.zeros((h, w), np.uint16)
            gt[::4, ::4] = (depth[::4, ::4] * 256).astype('uint16')
            sparse = np.zeros_like(gt)
            sparse[::8, ::8] = gt[::8, ::8]
            cv2.imwrite(str(root / f'{index}-rgb.png'), rng.randint(0, 255, (h, w, 3), dtype='uint8'))
            cv2.imwrite(str(root / f'{index}-gt.png'), gt)
            cv2.imwrite(str(root / f'{index}-sparse.png'), sparse)
            sid = f'sample-{index}'
            rows.append(json.dumps({'id': sid, 'rgb': f'{index}-rgb.png', 'gt': f'{index}-gt.png',
                                   'sparse': f'{index}-sparse.png', 'K': [721.5, 721.5, 609.5, 172.8]}))
            metric, geometry = root / 'metric_coarse/train', root / 'geometry_fused/train'
            metric.mkdir(parents=True, exist_ok=True)
            geometry.mkdir(parents=True, exist_ok=True)
            np.savez(metric / (sid + '.npz'), D_cm=depth, C_cm=np.ones_like(depth))
            np.savez(geometry / (sid + '.npz'), R_G=1.0 / depth, C_G=np.ones_like(depth))
        (root / 'train.txt').write_text('\n'.join(rows))
        cfg = yaml.safe_load(Path('configs/geolift_s3_lite_teacher_kd_tar2000.yaml').read_text())
        cfg['model']['encoder_pretrained'] = False
        cfg['data']['num_workers'] = 0
        cfg['runtime'] = {'resumable': True}
        paths = {'data_root': str(root), 'split_root': str(root), 'teacher_root': str(root), 'train_split': 'train.txt'}
        atomic_json(root / 'cfg.json', cfg)
        atomic_json(root / 'paths.json', paths)
        trial(root / 'cfg.json', root / 'paths.json', Path(args.output), 4)
    print(Path(args.output).read_text())


if __name__ == '__main__':
    main()
