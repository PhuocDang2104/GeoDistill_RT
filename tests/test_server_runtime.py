import copy
import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.runtime.common import atomic_json, sha256
from src.runtime.data import safe_extract
from src.runtime.recovery import (EpochBatchSampler, commit_checkpoint, load_checkpoint,
                                  verify_resume, capture_rng, restore_rng)
from src.runtime.telemetry import publish_bundle, restore_bundle
from src.train_student import train


def fixture_config(root, device='cpu', workers=2):
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    rows = []
    for i in range(10):
        rgb = rng.randint(0, 255, (64, 128, 3), dtype=np.uint8)
        gt = np.full((64, 128), (12 + i) * 256, np.uint16)
        sparse = np.zeros_like(gt)
        sparse[::4, ::4] = gt[::4, ::4]
        for kind, array in [('rgb', rgb), ('gt', gt), ('sparse', sparse)]:
            cv2.imwrite(str(data / f'{i}-{kind}.png'), array)
        rows.append(json.dumps({'id': f'sample-{i}', 'rgb': f'{i}-rgb.png', 'gt': f'{i}-gt.png',
                               'sparse': f'{i}-sparse.png', 'K': [100, 100, 64, 32]}))
    (data / 'train.txt').write_text('\n'.join(rows[:8]))
    (data / 'val.txt').write_text('\n'.join(rows[8:]))
    cfg = yaml.safe_load(Path('configs/geolift_s3_lite_tar2000.yaml').read_text())
    cfg['device'] = device
    cfg['model']['encoder_pretrained'] = False
    cfg['train'].update(epochs=2, scheduler_total_epochs=2, amp=device == 'cuda', fused_adamw=False,
                        deterministic=True, checkpoint_steps=1, checkpoint_seconds=300, resume=None)
    cfg['data'].update(image_size=[64, 128], num_workers=workers,
                       augmentation={'enabled': True, 'stage': 'A3', 'horizontal_flip_prob': .5,
                       'sparse_dropout': {'enabled': True, 'min_rate': .05, 'max_rate': .2},
                       'scale_jitter': {'enabled': True, 'min_scale': 1., 'max_scale': 1.15}})
    cfg['outputs'] = {'save_every': 0}
    cfg['runtime'] = {'resumable': True, 'protocol': 'test', 'source': 'synthetic'}
    paths = {'data_root': str(data), 'split_root': str(data), 'train_split': 'train.txt',
             'val_split': 'val.txt', 'teacher_root': str(root / 'teachers'), 'student_root': str(root / 'run')}
    return cfg, paths


def test_archive_rejects_traversal_and_symlink(tmp_path):
    for name, kind in [('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
        archive = tmp_path / 'bad.tar'
        with tarfile.open(archive, 'w') as t:
            entry = tarfile.TarInfo(name)
            entry.type, entry.size, entry.linkname = kind, 1 if kind == tarfile.REGTYPE else 0, '/tmp/escape'
            t.addfile(entry, io.BytesIO(b'x') if entry.size else None)
        with pytest.raises(ValueError):
            safe_extract(archive, tmp_path / 'output', 100)


def test_checkpoint_integrity_rng_and_resume_contract(tmp_path):
    state = capture_rng()
    expected = torch.rand(8)
    restore_rng(state)
    assert torch.equal(expected, torch.rand(8))
    path = tmp_path / 'last.pth'
    target = commit_checkpoint(path, {'epoch': 0, 'model': {'x': torch.ones(1)}, 'rng': state})
    assert torch.equal(load_checkpoint(path)['model']['x'], torch.ones(1))
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(path)
    with pytest.raises(ValueError, match='contract'):
        verify_resume({'train': {'batch_size': 2}}, {'train': {'batch_size': 4}})


def test_prefetch_cursor_is_consumption_based():
    sampler = EpochBatchSampler(20, 2, 42)
    first = list(sampler)
    sampler.start_batch = 3
    assert list(sampler) == first[3:]
    sampler.epoch = 1
    assert list(sampler) != first[3:]


def test_real_trainer_mid_epoch_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2)
    cfg, paths = fixture_config(tmp_path)
    train(cfg, paths)
    expected = load_checkpoint(Path(paths['student_root']) / 'checkpoints/last.pth')
    resumed_paths = {**paths, 'student_root': str(tmp_path / 'resumed')}
    interrupted = copy.deepcopy(cfg)
    interrupted['train']['stop_after_steps'] = 2
    train(interrupted, resumed_paths)
    checkpoint = Path(resumed_paths['student_root']) / 'checkpoints/last.pth'
    assert load_checkpoint(checkpoint)['progress']['next_batch'] == 2
    resumed = copy.deepcopy(cfg)
    resumed['train']['resume'] = str(checkpoint)
    train(resumed, resumed_paths)
    actual = load_checkpoint(checkpoint)
    assert actual['progress']['global_step'] == expected['progress']['global_step'] == 8
    assert actual['scheduler'] == expected['scheduler']
    for name in expected['model']:
        assert torch.equal(actual['model'][name], expected['model'][name]), name
    for index, state in expected['optimizer']['state'].items():
        for key, value in state.items():
            assert torch.equal(actual['optimizer']['state'][index][key], value)
    # Backup restore uses self-contained last/best and verified copied logs.
    publish_bundle(Path(resumed_paths['student_root']))
    bundle = sorted((Path(resumed_paths['student_root']) / 'backup_queue').glob('[!.]*'))[-1]
    restore = tmp_path / 'restored'
    restore_bundle(bundle, restore)
    assert sha256(restore / 'checkpoints/last.pth') == sha256(checkpoint)
    assert (restore / 'logs/train_log.csv').read_text() == (Path(resumed_paths['student_root']) / 'logs/train_log.csv').read_text()
    publish_bundle(restore)
    second_bundle = sorted((restore / 'backup_queue').glob('[!.]*'))[-1]
    assert sha256(second_bundle / 'checkpoints/best.pth') == sha256(restore / 'checkpoints/best.pth')
