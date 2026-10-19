"""Three architecture-only arms: same G, widen existing decoder hidden lanes.

No extra layer, field call, teacher, loss, phase/readout bound or output channel.
Added hidden lanes start with zero projection, preserving the old function in
exact arithmetic; old parameter slices and BN statistics are retained.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
import base_experiment as B
from base_experiment import (INPUTS, TRAIN_OUTPUTS, TensorForward, unpack, amp,
                             seed_all, freeze_encoder_bn)
from core import LiteBlock

ROOT = Path(__file__).resolve().parent
VARIANTS = ('G0_s4_upir', 'H_s4_dec16_8', 'I_s4_dec16_8_4')
TARGETS = {
    VARIANTS[0]: (),
    VARIANTS[1]: ('fuse.2.refine', 'refine8'),
    VARIANTS[2]: ('fuse.2.refine', 'refine8', 'refine4'),
}


def assert_config(cfg):
    B.assert_config(cfg)
    if cfg.get('decoder_expansion') != 4 or cfg.get('training_backend') != 'eager':
        raise ValueError('Registered expansion4 / matched eager-BF16 training only')
    if not cfg.get('teacher_enabled'):
        raise ValueError('Metric teacher remains ON; only relative teacher removed')
    if cfg['seed'] != 42 or cfg['batch_size'] != 4:
        raise ValueError('Matched seed42 / B4 required')


def widen_block(old, expansion=4):
    channels = old.body[0][0].in_channels
    old_hidden = old.body[0][0].out_channels
    if channels*expansion <= old_hidden:
        raise ValueError('Expected genuine widening')
    reference = old.body[0][0].weight
    new = LiteBlock(channels, expansion).to(device=reference.device, dtype=reference.dtype)
    old_state, new_state = old.state_dict(), new.state_dict()
    with torch.no_grad():
        # Zero outgoing weights of every added lane, not the old lanes.
        new.body[2][0].weight.zero_()
        for name, value in old_state.items():
            dest = new_state[name]
            slices = tuple(slice(0, size) for size in value.shape)
            dest[slices].copy_(value)
    new.train(old.training)
    return new


def make_model(cfg, variant, device='cpu', pretrained=False):
    assert_config(cfg)
    if variant not in VARIANTS: raise ValueError(variant)
    # Construct complete G FIRST, so every original parameter/RNG draw matches.
    model = B.make_model(cfg, 'G_s4_upir_norel', device, pretrained)
    model._baseline_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    for path in TARGETS[variant]:
        old = model.decoder.get_submodule(path)
        replacement = widen_block(old, cfg['decoder_expansion'])
        if path == 'fuse.2.refine': model.decoder.fuse[2].refine = replacement
        else: setattr(model.decoder, path, replacement)
    if cfg['channels_last']: model.to(memory_format=torch.channels_last)
    model._capacity_variant = variant
    return model


def common_weight_hash(model):
    """Hash ONLY retained historical tensor slices, never new hidden lanes.

    Same as historical G's common hash when no widening is applied. Equality
    proves shared initial old parameters, not equality of all wider parameters.
    """
    sha = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('fine_node.'): continue
        shape = model._baseline_shapes[name]
        value = value[tuple(slice(0, n) for n in shape)]
        if name == 'detail1.pir.body.0.weight': value = value[:, :40]
        sha.update(name.encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def architecture(model):
    widths = {path: model.decoder.get_submodule(path).body[0][0].out_channels
              for path in ('fuse.2.refine', 'refine8', 'refine4')}
    # Additional convolution MACs, NOT total model FLOPs or a latency prediction.
    macs = 0
    for path, channels, area in (('fuse.2.refine', 96, 22*76), ('refine8', 64, 44*152),
                                 ('refine4', 48, 88*304)):
        delta = widths[path]-2*channels
        macs += area*delta*(2*channels+9)
    return {'coarse_nfe': 4, 'fine_nfe': 0, 'fine_module': None,
            'decoder_hidden': widths, 'new_layers': 0,
            'parameters': sum(p.numel() for p in model.parameters()),
            'decoder_parameters': sum(p.numel() for p in model.decoder.parameters()),
            'extra_decoder_conv_macs_at_352x1216': macs,
            'initialization': 'Retained old slices; added hidden lanes have zero outgoing projection'}


def source_hash():
    sha = hashlib.sha256()
    for path in sorted(ROOT.rglob('*.py')):
        if '__pycache__' not in path.parts:
            sha.update(str(path.relative_to(ROOT)).encode());sha.update(path.read_bytes())
    return sha.hexdigest()
