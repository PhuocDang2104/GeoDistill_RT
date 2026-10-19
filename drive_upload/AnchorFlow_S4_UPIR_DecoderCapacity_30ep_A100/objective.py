"""Metric-only training, identical losses in every arm. No relative import/path."""
import torch
from losses_parent import objective


def loss_fn(pred, batch, holdout, progress, cfg):
    if cfg.get('relative_enabled') or cfg.get('relative_weight', 0) != 0:
        raise ValueError('Relative teacher is forbidden in this experiment')
    with torch.autocast(pred['D_full'].device.type, enabled=False):
        return objective(pred, batch, holdout, progress, cfg)
