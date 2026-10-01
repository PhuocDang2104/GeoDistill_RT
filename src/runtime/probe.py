from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

from .common import atomic_json, identity, read_json


def hardware(root: Path):
    vm = psutil.virtual_memory()
    ram = vm.total
    limit = Path('/sys/fs/cgroup/memory.max')
    if limit.exists() and limit.read_text().strip().isdigit():
        ram = min(ram, int(limit.read_text()))
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        value, period = quota.read_text().split()
        if value != 'max':
            cpus = min(cpus, max(1, int(value) // int(period)))
    result = {'platform': platform.platform(), 'python': platform.python_version(),
              'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'cpus': cpus, 'ram_bytes': ram, 'ram_available_bytes': min(ram, vm.available),
              'disk_free_bytes': shutil.disk_usage(root).free, 'cuda_available': torch.cuda.is_available()}
    shm = Path('/dev/shm')
    result['shm_free_bytes'] = shutil.disk_usage(shm).free if shm.exists() else 0
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        result.update(gpu=p.name, vram_bytes=total, vram_free_bytes=free,
                      compute_capability=list(torch.cuda.get_device_capability()),
                      supported_arches=torch.cuda.get_arch_list())
        # Exercise CUDA kernels, not just device enumeration.
        a = torch.randn(128, 128, device='cuda')
        (a @ a).sum().item()
        result['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,uuid',
                                              '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    return result


def trial(config: Path, paths_file: Path, output: Path, steps: int):
    from ..train_student import make_loader, to_device, _make_optimizer, _make_grad_scaler
    from ..model_factory import build_student
    from ..losses import geort_loss
    from ..utils import seed_everything
    import logging
    cfg, paths = read_json(config), read_json(paths_file)
    seed_everything(int(cfg['seed']), bool(cfg['train'].get('deterministic', False)))
    torch.set_float32_matmul_precision(cfg['train'].get('float32_matmul_precision', 'high'))
    if not cfg['train'].get('deterministic', False):
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(cfg['device'])
    loader = make_loader(cfg, paths, 'train', True)
    if len(loader) == 0:
        raise ValueError('Batch exceeds dataset length')
    model = build_student(cfg).to(device).train()
    channels_last = bool(cfg['train'].get('channels_last', False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = _make_optimizer(model, cfg, device, logging.getLogger('probe'))
    amp = device.type == 'cuda' and cfg['train'].get('amp', True)
    dtype = torch.bfloat16 if cfg['train'].get('amp_dtype') == 'bfloat16' else torch.float16
    scaler = _make_grad_scaler(device, amp and dtype == torch.float16)
    active_epoch = max([int(v) for k, v in cfg['schedule'].items() if k.endswith('_epoch') or k.endswith('_epochs')] + [0]) + 20
    iterator, durations = iter(loader), []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    successful, skipped = 0, 0
    for index in range(steps + 18):
        started = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = to_device(batch, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=dtype):
            pred = model(batch['rgb'], batch['sparse'], batch['mask'], batch['ray'], batch['uv'], batch.get('K'))
            loss, items = geort_loss(pred, batch, cfg['loss'], cfg['schedule'], active_epoch, cfg.get('mono_ssi', {}))
        if not torch.isfinite(loss):
            raise ValueError('Calibration found a nonfinite full-objective loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train'].get('grad_clip_norm', 1.0))
        if not torch.isfinite(norm) and not scaler.is_enabled():
            raise ValueError('Calibration found nonfinite gradients')
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if scaler.get_scale() < old_scale:
            skipped += 1
            continue
        successful += 1
        if successful > 2:
            durations.append(time.monotonic() - started)
        if len(durations) >= steps:
            break
    if len(durations) < steps:
        raise ValueError('AMP did not stabilize during calibration')
    atomic_json(output, {'images_per_second': cfg['train']['batch_size'] * steps / sum(durations),
                        'peak_vram_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0,
                        'rss_bytes': psutil.Process().memory_info().rss,
                        'workers': cfg['data']['num_workers'], 'batch_size': cfg['train']['batch_size'],
                        'loss': float(loss), 'full_loss_epoch': active_epoch, 'amp_warmup_skips': skipped})


def calibrate(cfg: dict, paths: dict, root: Path, mode='baseline', steps=6):
    report = hardware(root)
    atomic_json(root / 'hardware.json', report)
    if cfg['device'] == 'cuda' and not report['cuda_available']:
        raise RuntimeError('GPU required but unavailable. Install NVIDIA driver/container toolkit; CPU fallback is disabled.')
    trials = root / 'calibration_trials'
    trials.mkdir(exist_ok=True)
    paths_file = trials / 'paths.json'
    atomic_json(paths_file, paths)
    # Limit worker/prefetch memory using actual collated tensor size.
    from ..train_student import make_loader
    sizing = copy.deepcopy(cfg)
    sizing['data']['num_workers'] = 0
    example = make_loader(sizing, paths, 'train', True).dataset[0]
    sample_bytes = sum(t.nelement() * t.element_size() for t in example.values() if torch.is_tensor(t))
    batches = [int(cfg['train']['batch_size'])] if mode == 'baseline' else [1, 2, 4, 8, 16]
    observations = []
    for batch in batches:
        candidates = [0, 2, 4]
        for workers in candidates:
            if workers > report['cpus']:
                continue
            queued = sample_bytes * batch * max(1, workers) * 2
            if queued > report['ram_available_bytes'] * .15:
                continue
            if workers and queued > report['shm_free_bytes'] * .65:
                continue
            candidate = copy.deepcopy(cfg)
            candidate['train']['batch_size'] = batch
            candidate['data'].update(num_workers=workers, prefetch_factor=2)
            name = f'b{batch}-w{workers}'
            config, result = trials / (name + '.config.json'), trials / (name + '.result.json')
            atomic_json(config, candidate)
            started = time.monotonic()
            try:
                with (trials / (name + '.log')).open('w', encoding='utf-8') as log:
                    run = subprocess.run([sys.executable, '-m', 'src.runtime', '_trial', '--config', str(config),
                                          '--paths', str(paths_file), '--output', str(result), '--steps', str(steps)],
                                         stdout=log, stderr=subprocess.STDOUT, timeout=600)
                observation = read_json(result) if run.returncode == 0 else {'error': 'trial failed; see ' + name + '.log'}
            except subprocess.TimeoutExpired:
                observation = {'error': 'trial timed out'}
            observation.update(batch_size=batch, workers=workers, elapsed_seconds=time.monotonic() - started)
            # Reserve 15% of currently available VRAM for validation and runtime variation.
            if observation.get('peak_vram_bytes', 0) > report.get('vram_free_bytes', 0) * .85:
                observation['error'] = 'less than 15% VRAM headroom'
            observations.append(observation)
            print(json.dumps(observation), flush=True)
    passed = [o for o in observations if 'error' not in o]
    if not passed:
        atomic_json(root / 'calibration.json', {'trials': observations, 'mode': mode})
        raise RuntimeError('No safe configuration found; inspect calibration_trials/*.log. Baseline batch is never silently changed.')
    best = max(passed, key=lambda o: o['images_per_second'])
    resolved = copy.deepcopy(cfg)
    resolved['train']['batch_size'] = best['batch_size']
    resolved['data'].update(num_workers=best['workers'], prefetch_factor=2)
    resolved['runtime']['protocol'] = 'baseline-b2' if mode == 'baseline' else 'throughput-b' + str(best['batch_size'])
    atomic_json(root / 'calibration.json', {'trials': observations, 'selected': best, 'mode': mode,
                                         'hardware_fingerprint': identity(report), 'sample_tensor_bytes': sample_bytes})
    return resolved
