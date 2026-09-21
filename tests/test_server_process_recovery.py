"""Linux process failure tests, using the real student and training loop."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from src.runtime.common import atomic_json, read_json
from src.runtime.recovery import load_checkpoint
from src.runtime.telemetry import backup_once, restore_bundle
from test_server_runtime import fixture_config


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process signals tested in Linux image')
@pytest.mark.parametrize('kill_signal', [signal.SIGTERM, signal.SIGKILL])
def test_process_restart_after_signal(tmp_path, kill_signal):
    cfg, paths = fixture_config(tmp_path, workers=0)
    cfg['train'].update(epochs=5, scheduler_total_epochs=5)
    config = tmp_path / 'config.json'
    paths_file = tmp_path / 'paths.json'
    atomic_json(config, cfg)
    atomic_json(paths_file, paths)
    command = [sys.executable, '-c',
               'import torch; torch.set_num_threads(2); from src.runtime.common import read_json; '
               'from pathlib import Path; from src.train_student import train; '
               'import sys; train(read_json(Path(sys.argv[1])),read_json(Path(sys.argv[2])))',
               str(config), str(paths_file)]
    root = Path(paths['student_root'])
    with (tmp_path / 'process.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            checkpoint_meta = root / 'checkpoints/last.json'
            while time.monotonic() < deadline:
                if checkpoint_meta.exists():
                    break
                if process.poll() is not None:
                    pytest.fail((tmp_path / 'process.log').read_text())
                time.sleep(.02)
            else:
                pytest.fail('Checkpoint did not appear')
            os.killpg(process.pid, kill_signal)
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    last = root / 'checkpoints/last.pth'
    before = load_checkpoint(last)['progress']['global_step']
    assert before < 20
    cfg['train']['resume'] = str(last)
    atomic_json(config, cfg)
    with (tmp_path / 'resume.log').open('w') as log:
        resumed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert resumed.returncode == 0, (tmp_path / 'resume.log').read_text()
    assert load_checkpoint(last)['progress']['global_step'] == 20
    rows = [json.loads(s) for s in (root / 'logs/train_log.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == list(range(5))
    # Exercise the actual rclone transport, offline with its local backend.
    if shutil.which('rclone'):
        remote_config = tmp_path / 'rclone.conf'
        remote_config.write_text('[test]\ntype = local\n')
        old = os.environ.get('RCLONE_CONFIG')
        os.environ['RCLONE_CONFIG'] = str(remote_config)
        try:
            backup_once(root, 'test:' + str(tmp_path / 'remote'))
        finally:
            if old is None:
                os.environ.pop('RCLONE_CONFIG')
            else:
                os.environ['RCLONE_CONFIG'] = old
        bundle = sorted((tmp_path / 'remote' / root.name).iterdir())[-1]
        restore_bundle(bundle, tmp_path / 'recovered')
        recovered = load_checkpoint(tmp_path / 'recovered/checkpoints/last.pth')
        assert recovered['progress']['global_step'] == 20
