"""Production I/O adapter around the byte-identical V11.3 model and objective."""
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
    from utils import read_json
    model = read_json(MODEL_ROOT/'reference_config.json')
    for key in ('drive_data','drive_runs','run_name','metric_tar','relative_tar',
                'expected_subset_sha256','parent_label','parent_checkpoint_sha256','parent_source_sha256'):
        model.pop(key,None)
    model.update(cfg['training'])
    model.update(encoder_pretrained=True, init_checkpoint=None, teacher_enabled=True,
                 relative_enabled=True, compile=False, initialization='fresh_imagenet_encoder',
                 architecture='AnchorFlow-V11_3-FullKITTI-Fresh',work=cfg['work'])
    return model


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
            return self.network(batch['rgb'],batch['sparse'],batch['mask'] if mask is None else mask,batch['K'])

    def loss(self, output, batch, holdout, progress):
        return objective(output,batch,holdout,progress,model_config(self.cfg))

    def deployment(self):
        return Deploy(self.network.eval())
