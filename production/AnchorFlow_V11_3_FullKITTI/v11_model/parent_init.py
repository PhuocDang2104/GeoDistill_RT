"""Strict model-only migration from byte-pinned V11_2 trained checkpoints."""
import hashlib
import json
from pathlib import Path
import torch


def initialize_from_parent(model, config, check_data=True):
    path = Path(config['init_checkpoint'])
    if not path.is_absolute(): path = Path(__file__).parent/path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != config['parent_checkpoint_sha256']:
        raise RuntimeError('Parent checkpoint checksum mismatch; refusing arbitrary warm-start')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    protocol = payload['protocol']
    if protocol['source_sha256'] != config['parent_source_sha256']:
        raise RuntimeError('Parent source is not pinned trained V11_2')
    if protocol['config']['architecture'] != 'AnchorFlow-V11_2-CoarseNODE-FineRK2-MetricOnly':
        raise RuntimeError('Wrong parent architecture')
    expected = config['expected_subset_sha256']
    if protocol['data']['manifest_sha256'] != expected:
        raise RuntimeError('Parent subset mismatch')
    if check_data:
        data = json.loads((Path(config['work'])/'data_contract.json').read_text(encoding='utf-8'))
        if data['manifest_sha256'] != expected: raise RuntimeError('Fine-tune subset mismatch')
    state = payload['model']
    for name, value in state.items():
        if not torch.isfinite(value).all(): raise RuntimeError('Nonfinite parent parameter: '+name)
    result = model.load_state_dict(state, strict=False)
    allowed = {key for key in model.state_dict() if key.startswith('detail1.pir.')}
    if result.unexpected_keys or set(result.missing_keys) != allowed:
        raise RuntimeError(f'Unsafe architecture migration: {result}')
    info = {'loaded': True, 'parent_epoch': payload['epoch'],
        'parent_completed_epochs': payload['epoch']+1, 'parent_global_step': payload['global_step'],
        'parent_checkpoint_sha256': actual, 'parent_label': config['parent_label'],
        'missing_new_keys': result.missing_keys,
        'optimizer_restored': False, 'schedule_restored': False, 'early_stopping_restored': False}
    print('MODEL-only parent initialization:', json.dumps(info), flush=True)
    return info
