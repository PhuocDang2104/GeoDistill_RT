"""V11.3 model/loss adapter with a locked six-NFE coarse RK3 solver."""
import contextlib
import sys
from pathlib import Path
import torch

MODEL_ROOT = Path(__file__).parent / 'v11_model'
sys.path.insert(0, str(MODEL_ROOT))
from model import AnchorFlowEdge, Deploy
from losses import objective
from run import validate, augment, input_with_holdout, move_batch, learning_rate, worker_setup
from selection import update_policy_best, early_stop_update


def model_config(cfg):
    validate_recipe(cfg)
    from utils import read_json
    model = read_json(MODEL_ROOT/'reference_config.json')
    for key in ('drive_data','drive_runs','run_name','metric_tar','relative_tar',
                'expected_subset_sha256','parent_label','parent_checkpoint_sha256','parent_source_sha256'):
        model.pop(key,None)
    model.update(cfg['training'])
    model.update(encoder_pretrained=True, init_checkpoint=None, teacher_enabled=True,
                 relative_enabled=True, compile=False, initialization='fresh_imagenet_encoder',
                 architecture='AnchorFlow-V11_3-Fixed6-FullKITTI-Fresh',work=cfg['work'])
    return model


def validate_recipe(cfg):
    tc = cfg['training']
    for key, expected in {'ode_method':'rk3', 'ode_fixed_steps':2, 'ode_max_nfe':6,
                          'ode_terminal_time':1., 'ode_field_precision':'fp32'}.items():
        if tc.get(key) != expected:
            raise ValueError(f'Fixed6 contract requires {key}={expected!r}; no solver fallback')
    if tc.get('accumulation',1) != 1:
        raise ValueError('This trainer performs one optimizer update per batch; accumulation is not implemented')
    if tc['batch_size'] not in (1,2,4):
        raise ValueError('Choose explicit batch 1, 2 or 4 before starting a fresh run')
    if tc['amp'] != 'bf16' and tc['amp'] != 'fp32':
        raise ValueError('No silent FP16 fallback')


def assert_solver_contract(network):
    coarse, fine = network.dynamics.last_solver_report, network.fine_node.last_solver_report
    if coarse.get('method') != 'rk3' or coarse.get('nfe') != 6 or coarse.get('terminal_time') != 1.:
        raise RuntimeError(f'Coarse fixed6 solver contract failed: {coarse}')
    if fine.get('nfe') != 2 or fine.get('terminal_time') != 1.:
        raise RuntimeError(f'Fine RK2 contract failed: {fine}')


def build(cfg, device, pretrained=False):
    mc = model_config(cfg)
    net = AnchorFlowEdge(pretrained=pretrained, flow_steps=2, encoder=mc['encoder'],
                         model_name='v11_node', phase_context_enabled=True,
                         fine_node_enabled=True, pir_enabled=True).to(device)
    net.dynamics.configure(mc)
    if mc['channels_last']:
        net = net.to(memory_format=torch.channels_last)
    return net


def precision(cfg, device):
    amp = cfg['training']['amp']
    if amp == 'bf16':
        if device.type != 'cuda' or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError('Native BF16 GPU required. No FP16/CPU fallback.')
        return torch.autocast('cuda', dtype=torch.bfloat16)
    if amp != 'fp32':
        raise ValueError('Only explicit bf16 or fp32; no silent fallback')
    return contextlib.nullcontext()


class ModelAdapter:
    """Export/import contract suitable for GPU members, not tied to notebooks."""
    input_names = ('rgb', 'sparse', 'mask', 'K')
    output_name = 'D_full'

    def __init__(self, cfg, device, pretrained=False):
        self.cfg, self.device = cfg, device
        self.network = build(cfg, device, pretrained)

    def forward(self, batch, mask=None):
        with precision(self.cfg, self.device):
            output = self.network(batch['rgb'],batch['sparse'],batch['mask'] if mask is None else mask,batch['K'])
        assert_solver_contract(self.network)
        return output

    def loss(self, output, batch, holdout, progress):
        return objective(output,batch,holdout,progress,model_config(self.cfg))

    def deployment(self):
        return Deploy(self.network.eval())
