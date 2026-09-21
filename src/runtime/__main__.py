"""Server entry point; also runs directly inside a Vast.ai container."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

from .common import atomic_json, identity, lock, read_json, sha256, under
from .data import catalog, fetch, prepare
from .probe import calibrate, hardware, trial
from .telemetry import backup_once, publish_bundle, restore_bundle, clean


def source_hash():
    root = Path(__file__).resolve().parents[2]
    files = sorted(p for directory in ('src', 'configs') for p in (root / directory).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.yaml', '.json'))
    return identity({str(p.relative_to(root)): sha256(p) for p in files})


def preflight(cfg, paths, expected=None):
    from ..train_student import make_loader, preflight_teacher_coverage
    seen, rgb_seen, drives, result = set(), set(), {}, {}
    for split in ('train', 'val'):
        dataset = make_loader(cfg, paths, split, split == 'train').dataset
        ids, images = [s.sample_id for s in dataset.samples], [str(s.rgb_path.resolve()) for s in dataset.samples]
        if len(set(ids)) != len(ids) or len(set(images)) != len(images) or set(ids) & seen or set(images) & rgb_seen:
            raise ValueError(f'Duplicate IDs or train/validation leakage in {split}')
        seen.update(ids)
        rgb_seen.update(images)
        drives[split] = {s.split('_sync_image_')[0] for s in ids if '_sync_image_' in s}
        if expected and set(ids) != set(expected[split + '_ids']):
            raise ValueError(f'{split} split differs from frozen selection')
        for index, info in enumerate(dataset.samples):
            if index % 200 == 0:
                print(f'Validating {split}: {index}/{len(dataset)}', flush=True)
            for path in (info.rgb_path, info.sparse_path, info.gt_path):
                if path is None or not path.is_file():
                    raise ValueError(f'Missing supervised {split} input: {path}')
            if info.K is None and info.calib_path is None:
                raise ValueError(f'Missing calibrated intrinsics for {info.sample_id}; default K is prohibited')
            sample = dataset[index]
            for key, value in sample.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite {key} for {info.sample_id}')
            if sample['gt_mask'].sum() == 0:
                raise ValueError(f'Empty ground truth for {info.sample_id}')
            if split == 'train' and cfg['loss'].get('require_dense_metric_teacher', False):
                valid = ((sample['D_cm'] > cfg['loss'].get('min_depth', .001)) &
                         (sample['D_cm'] < cfg['loss'].get('max_depth', 120)) &
                         (sample['C_cm'] > cfg['loss'].get('metric_conf_min', .05)) &
                         (sample['gt_mask'] < .5))
                if int(valid.sum()) < cfg['loss'].get('min_dense_metric_pixels', 1024):
                    raise ValueError(f'Insufficient non-GT teacher pixels: {info.sample_id}')
            if split == 'train' and cfg['loss'].get('require_geometry_teacher', False):
                count = int((sample['C_G'] > cfg['loss'].get('geometry_conf_threshold', .4)).sum())
                if count < cfg['loss'].get('geometry_min_valid_pixels', 512):
                    raise ValueError(f'Insufficient geometry confidence: {info.sample_id}')
        if split == 'train':
            preflight_teacher_coverage(cfg, dataset, logging.getLogger('preflight'))
        result[split] = {'count': len(ids), 'ids_sha256': identity(ids)}
    if drives['train'] & drives['val']:
        raise ValueError('Raw-drive overlap between train and validation')
    if result['train']['count'] < cfg['train']['batch_size'] or result['val']['count'] == 0:
        raise ValueError('Insufficient samples')
    return result


def run(args):
    root = under(Path(args.results), args.run_id)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / '.train.lock'):
        try:
            cfg_file, paths_file = root / 'resolved_config.json', root / 'resolved_paths.json'
            atomic_json(root / 'status.json', {'state': 'PREPARING'})
            manifest = Path(args.manifest)
            if not manifest.exists() and args.manifest == '/config/dataset.json':
                manifest = Path('/app/docker/tar2000-manifest.json')
            dataset_root = prepare(manifest, Path(args.storage))
            ready = read_json(dataset_root / 'READY.json')
            layout = ready['layout']
            paths = {'project_root': str(Path.cwd()), 'data_root': str(under(dataset_root, layout['data'])),
                     'split_root': str(under(dataset_root, layout['splits'])),
                     'teacher_root': str(under(dataset_root, layout['teachers'])), 'student_root': str(root),
                     'weights_root': os.environ.get('HF_HOME', '/cache/huggingface'),
                     'train_split': layout['train_split'], 'val_split': layout['val_split'],
                     'test_split': layout.get('test_split', layout['val_split'])}
            source = {'code': source_hash(), 'dataset': ready['digest']}
            if cfg_file.exists():
                cfg = read_json(cfg_file)
                if cfg['runtime']['source'] != source:
                    raise ValueError('Existing run source or data changed. Use its original image/manifest, or a new run ID.')
                if args.mode != cfg['runtime']['mode']:
                    raise ValueError('Run mode changed; use a new run ID')
            else:
                cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
                cfg['runtime'] = {'resumable': True, 'source': source, 'mode': args.mode, 'protocol': 'pending'}
                cfg['train'].update(resume=None, checkpoint_steps=100, checkpoint_seconds=300,
                                    data_parallel=False, distributed=False, compile=False)
                cfg['outputs'].pop('backup_root', None)
                cfg['data']['num_workers'] = 0
                atomic_json(root / 'status.json', {'state': 'PREFLIGHT'})
                expected_path = dataset_root / 'kitti' / 'selected_2000_ids.json'
                expected = read_json(expected_path) if ready['manifest']['format'] == 'tar2000' else None
                report = preflight(cfg, paths, expected)
                atomic_json(root / 'preflight.json', report)
                atomic_json(root / 'status.json', {'state': 'CALIBRATING'})
                cfg = calibrate(cfg, paths, root, args.mode)
                cfg['paths_file'] = str(paths_file)
                atomic_json(cfg_file, cfg)
                atomic_json(root / 'provenance.json', {'source': source, 'created': time.time(),
                            'image': os.environ.get('TRAIN_IMAGE', 'unrecorded'), 'preflight': report})
            atomic_json(paths_file, paths)
            if args.prepare_only:
                atomic_json(root / 'status.json', {'state': 'READY'})
                return
            if (root / 'STOP').exists():
                raise RuntimeError('STOP file exists. Remove it when intentionally resuming.')
            current = hardware(root)
            atomic_json(root / 'hardware_current.json', current)
            if cfg['device'] == 'cuda' and not current['cuda_available']:
                raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
            last = root / 'checkpoints' / 'last.pth'
            cfg['paths_file'] = str(paths_file)
            cfg['train']['resume'] = str(last) if last.exists() else None
            cfg['train']['stop_after_steps'] = args.stop_after_steps
            # Resumes retain batch/precision/schedule; zero workers is a safe fallback
            # when the replacement server has less host RAM or shared memory.
            previous = read_json(root / 'hardware.json') if (root / 'hardware.json').exists() else {}
            if current.get('ram_bytes', 0) < previous.get('ram_bytes', 0) or current.get('shm_free_bytes', 0) < previous.get('shm_free_bytes', 0):
                cfg['data']['num_workers'] = 0
            atomic_json(root / 'status.json', {'state': 'TRAINING', 'resuming': last.exists()})
            from ..train_student import train, make_loader, validate
            train(cfg, paths)
            status = read_json(root / 'status.json')
            if status['state'] == 'TRAINING_COMPLETE':
                from .recovery import load_checkpoint
                from ..model_factory import build_student
                evaluation_cfg = copy.deepcopy(cfg)
                evaluation_cfg['model']['encoder_pretrained'] = False
                model = build_student(evaluation_cfg).to(cfg['device'])
                checkpoint = load_checkpoint(root / 'checkpoints' / 'best.pth')
                model.load_state_dict(checkpoint['model'])
                metrics = validate(model, make_loader(cfg, paths, 'val', False), torch.device(cfg['device']), cfg)
                atomic_json(root / 'evaluation' / 'best_val.json', {'checkpoint_epoch': checkpoint['epoch'], 'metrics': clean(metrics)})
                atomic_json(root / 'status.json', {**status, 'state': 'COMPLETE'})
            publish_bundle(root)
        except BaseException as exc:
            atomic_json(root / 'status.json', {'state': 'FAILED', 'error': str(exc), 'time': time.time()})
            (root / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise


def main():
    parser = argparse.ArgumentParser(description='GeoDistill reproducible server training')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'prepare'):
        p = sub.add_parser(command)
        p.add_argument('--manifest', default=os.environ.get('DATASET_MANIFEST', '/config/dataset.json'))
        p.add_argument('--config', default=os.environ.get('TRAIN_CONFIG', '/app/configs/geolift_s3_lite_teacher_kd_tar2000.yaml'))
        p.add_argument('--storage', default='/data')
        p.add_argument('--results', default='/runs')
        p.add_argument('--run-id', default=os.environ.get('RUN_ID', 'tar2000-kd-001'))
        p.add_argument('--mode', choices=('baseline', 'throughput'), default=os.environ.get('TRAIN_MODE', 'baseline'))
        p.add_argument('--stop-after-steps', type=int, default=0)
        p.set_defaults(prepare_only=command == 'prepare')
    p = sub.add_parser('doctor')
    p.add_argument('--root', default='/runs')
    p = sub.add_parser('catalog')
    p.add_argument('--source', required=True)
    p.add_argument('--output', default='/config/dataset.json')
    p.add_argument('--storage', default='/data')
    p = sub.add_parser('_trial')
    for name in ('config', 'paths', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--steps', type=int, default=6)
    p = sub.add_parser('backup')
    p.add_argument('--results', default='/runs')
    p.add_argument('--remote', default=os.environ.get('BACKUP_REMOTE', ''))
    p.add_argument('--once', action='store_true')
    p = sub.add_parser('restore')
    p.add_argument('--bundle', required=True, help='Downloaded local COMPLETE bundle directory')
    p.add_argument('--destination', required=True)
    args = parser.parse_args()
    if args.command in ('run', 'prepare'):
        run(args)
    elif args.command == 'doctor':
        print(json.dumps(hardware(Path(args.root)), indent=2))
    elif args.command == 'catalog':
        catalog(args.source, Path(args.output), Path(args.storage))
    elif args.command == '_trial':
        trial(Path(args.config), Path(args.paths), Path(args.output), args.steps)
    elif args.command == 'restore':
        restore_bundle(Path(args.bundle), Path(args.destination))
    elif args.command == 'backup':
        if not args.remote:
            parser.error('Set BACKUP_REMOTE or --remote before starting backup')
        while True:
            failed = False
            for root in Path(args.results).iterdir():
                if not root.is_dir() or not (root / 'backup_queue').exists():
                    continue
                try:
                    backup_once(root, args.remote)
                except Exception as exc:
                    failed = True
                    atomic_json(root / 'backup_status.json', {'state': 'FAILED', 'error': str(exc), 'time': time.time()})
                    print(f'Backup failed for {root.name}: {exc}', file=sys.stderr, flush=True)
            if args.once:
                if failed:
                    sys.exit(1)
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
