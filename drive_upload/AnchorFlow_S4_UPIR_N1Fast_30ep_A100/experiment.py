"""One fresh30 candidate: M3 with Fusion768->512, unchanged Jet-context128."""
import hashlib
from pathlib import Path
import torch
from torch import nn
import base_experiment as B
from base_experiment import INPUTS,TRAIN_OUTPUTS,TensorForward,unpack,amp,seed_all,freeze_encoder_bn
from core import ConvBN,LiteBlock

ROOT=Path(__file__).resolve().parent
REFERENCE_VARIANT='N1_s4_m3_f512'
VARIANTS=('Q1_s4_n1_fast',)
TRAIN_VARIANTS=VARIANTS
# full module path, external C, H,W at B1x352x1216
BLOCKS=(('decoder.fuse.2.refine',96,22,76),('decoder.fuse.1.refine',64,44,152),
        ('decoder.refine8',64,44,152),('context32.body.1',64,11,38),
        ('decoder.fuse.0.refine',48,88,304),('dynamics.context.1',32,88,304))
WIDTHS={VARIANTS[0]:(96,64,64,32,512,128)}
EXPECTED_PARAMETERS={VARIANTS[0]:582888}
# EXACT historical Edge600 intersection, not min(widths) of this single candidate.
# Keeps the shared-init proof comparable to M3's saved initial_shared_sha256.
MIN_HIDDEN=(96,64,64,32,96,64)


def assert_config(cfg):
    B.assert_config(cfg)
    if cfg.get('capacity_design')!='n1_exact_fastpath_v1' or cfg['training_backend']!='eager':raise ValueError('Frozen N1 fastpath/eager training required')
    if not cfg['teacher_enabled'] or not cfg['encoder_pretrained']:raise ValueError('AnchorFlow requires metric teacher and ImageNet encoder')
    if cfg['seed']!=42 or cfg['batch_size']!=4:raise ValueError('Seed42/B4 required')
    if cfg.get('experiment_targets')!={'parameters':582888,'rmse_m_strict_max':.98,
      'irmse_km_inv_strict_max':3.2,'mean_wall_ms_max':3.9}:
        raise ValueError('Predeclared metric/parameter/mean-latency targets changed')
    if cfg.get('deployment_optimization')!={'design':'exact_ops_v1','original_builder_level':3,'fast_builder_level':5,'max_aux_streams':0}:
        raise ValueError('Deployment optimization protocol changed')


def resize_block(old,hidden):
    c=old.body[0][0].in_channels
    if hidden%16 or hidden<16:raise ValueError('Hidden width must be a positive multiple of16')
    ref=old.body[0][0].weight
    new=LiteBlock(c)
    new.body=nn.Sequential(ConvBN(c,hidden),ConvBN(hidden,hidden,3,hidden),ConvBN(hidden,c,activation=False))
    new.to(device=ref.device,dtype=ref.dtype)
    with torch.no_grad():
        new.body[2][0].weight.zero_()
        for name,value in old.state_dict().items():
            dest=new.state_dict()[name]
            slices=tuple(slice(0,min(a,b)) for a,b in zip(value.shape,dest.shape))
            dest[slices].copy_(value[slices])
    new.train(old.training)
    return new


def shared_shapes(model):
    shapes={n:list(v.shape) for n,v in model.state_dict().items()}
    for (path,_,_,_),hidden in zip(BLOCKS,MIN_HIDDEN):
        for name,shape in shapes.items():
            if name.startswith((path+'.body.0.',path+'.body.1.')) and shape:shape[0]=hidden
            if name==path+'.body.2.0.weight':shape[1]=hidden
    return shapes


def make_model(cfg,variant,device='cpu',pretrained=False):
    assert_config(cfg)
    if variant not in VARIANTS:raise ValueError(variant)
    model=B.make_model(cfg,'G_s4_upir_norel',device,pretrained)
    # Legacy full starting hash BEFORE any resize, for old K1 provenance only.
    model._starting_sha256=B.common_weight_hash(model)
    model._shared_shapes=shared_shapes(model)
    for index,((path,c,_,_),hidden) in enumerate(zip(BLOCKS,WIDTHS[variant])):
        if hidden==2*c:continue
        with torch.random.fork_rng(devices=[]):
            # Preserve D4 Fusion added lanes from K1's registered D4Split init.
            offset=1009 if path=='decoder.fuse.0.refine' else 3001+index*97
            torch.default_generator.manual_seed(cfg['seed']+offset)
            block=resize_block(model.get_submodule(path),hidden)
        parent,_,leaf=path.rpartition('.')
        setattr(model.get_submodule(parent),leaf,block)
    if cfg['channels_last']:model.to(memory_format=torch.channels_last)
    model._capacity_variant=variant
    if sum(p.numel() for p in model.parameters())!=EXPECTED_PARAMETERS[variant]:raise RuntimeError('Actual parameter budget differs')
    return model


def common_weight_hash(model):
    """Actual retained tensor intersection, NOT old full583k starting digest."""
    sha=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('fine_node.'):continue
        value=value[tuple(slice(0,n) for n in model._shared_shapes[name])]
        if name=='detail1.pir.body.0.weight':value=value[:,:40]
        sha.update(name.encode());sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def architecture(model):
    detail=[]
    for path,c,h,w in BLOCKS:
        hidden=model.get_submodule(path).body[0][0].out_channels
        delta=hidden-2*c
        detail.append({'block':path,'hidden':hidden,'base_hidden':2*c,'resolution':[h,w],
                       'delta_parameters':delta*(2*c+13),'delta_conv_macs':h*w*delta*(2*c+9)})
    return {'variant':model._capacity_variant,'parameters':sum(p.numel() for p in model.parameters()),
      'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'coarse_nfe':4,'fine_nfe':0,'fine_module':None,'new_layers':0,'blocks':detail,
      'extra_decoder_conv_macs_at_352x1216':sum(r['delta_conv_macs'] for r in detail),
      'extra_parameters':sum(r['delta_parameters'] for r in detail),
      'mac_scope':'Additional/removed conv MACs across listed blocks vs G0; not total model FLOPs or latency',
      'initialization':'Copied retained slices, zero outgoing added lanes; shrinking is NOT function preserving',
      'common_hash_scope':'Historical Edge600 retained intersection, same scope as M3',
      'parent_variant':'M3_s4_trimlow_fuse768_jetctx','parent_parameters':610792,
      'only_change_vs_M3':'decoder.fuse.0.refine hidden768->512; Jet-context128 unchanged',
      'change_vs_N1':'No neural architecture/training change; deployment-only operator rewrites and TRT builder search',
      'removed_parameters_vs_M3':27904,'removed_conv_macs_vs_M3':719093760}


def source_hash():
    sha=hashlib.sha256()
    for path in sorted(ROOT.rglob('*.py')):
        if '__pycache__' not in path.parts:
            sha.update(str(path.relative_to(ROOT)).encode());sha.update(path.read_bytes())
    return sha.hexdigest()
