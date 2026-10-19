"""F: S4 no-fine; one bounded uncertainty feature in existing PIR."""
from pathlib import Path
import hashlib
import random
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import torch
from torch import nn
from model import AnchorFlowEdge

VARIANTS=('G_s4_upir_norel',)
INPUTS=('rgb','sparse','mask','K')
TRAIN_OUTPUTS=('D16','D8','D4','D2','D1','D_full','sensor_logits','sensor_gate',
               'delta4','delta1','surface_barrier_logits','D4_step1','D4_step2')

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def assert_config(cfg):
    if cfg['epochs']!=30 or cfg['early_stopping']:
        raise ValueError('Full30, early stop disabled')
    if (cfg['ode_method'],cfg['ode_fixed_steps'],cfg['ode_max_nfe'])!=('midpoint',2,4):
        raise ValueError('S4 = 2 midpoint steps = 4 coarse NFE, NOT RK4')
    if cfg['amp']!='bf16' or cfg.get('init_checkpoint'):
        raise ValueError('Fresh student, ImageNet encoder only, BF16 train')
    if cfg.get('fine_mode')!='none':raise ValueError('D must have no fine module')
    if cfg.get('upir_uncertainty_scale')!=.002:raise ValueError('Frozen U/(U+.002) normalization')
    if cfg.get('relative_enabled') or cfg.get('relative_weight')!=0.:
        raise ValueError('Registered no-relative ablation requires relative_enabled=False, relative_weight=0')
    if cfg['tf32']:raise ValueError('Registered FP32 TensorRT comparison uses TF32 OFF')

def make_model(cfg,variant,device='cpu',pretrained=False):
    if variant not in VARIANTS:raise ValueError(variant)
    assert_config(cfg)
    seed_all(cfg['seed'])
    # Construct fine FIRST: PIR/detail layers created after it must consume the
    # identical RNG sequence as historical A/B/C. Then REMOVE its parameters/path.
    model=AnchorFlowEdge(pretrained=pretrained,flow_steps=2,encoder=cfg['encoder'],
        model_name='v11_node',phase_context_enabled=True,fine_node_enabled=True,pir_enabled=True)
    model.fine_node=None
    model.detail1.pir.enable_uncertainty_feature(cfg['upir_uncertainty_scale'])
    model.dynamics.configure(cfg)
    model.to(device)
    if cfg['channels_last']:model.to(memory_format=torch.channels_last)
    return model

def common_weight_hash(model):
    sha=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if not name.startswith('fine_node.'):
            if name=='detail1.pir.body.0.weight':value=value[:,:40]
            sha.update(name.encode());sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()

def source_hash():
    sha=hashlib.sha256()
    for path in sorted(ROOT.rglob('*.py')):
        if '__pycache__' not in path.parts:
            sha.update(str(path.relative_to(ROOT)).encode());sha.update(path.read_bytes())
    return sha.hexdigest()

def amp(device,enabled=True):
    return torch.autocast(device.type,dtype=torch.bfloat16,
                          enabled=enabled and device.type=='cuda',cache_enabled=False)

def freeze_encoder_bn(model):model.freeze_encoder_bn()

class TensorForward(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,rgb,sparse,mask,K):
        with amp(rgb.device):pred=self.model(rgb,sparse,mask,K)
        return tuple(pred[k] for k in TRAIN_OUTPUTS)

def unpack(values):return dict(zip(TRAIN_OUTPUTS,values))
