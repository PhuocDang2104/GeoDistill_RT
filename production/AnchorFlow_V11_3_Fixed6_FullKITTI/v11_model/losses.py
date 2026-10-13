"""Fine-tune schedules plus train-only relative structure; FP32 loss reductions."""
import torch
from losses_parent import objective as parent_objective, smooth_rmse
from loss_helpers import pooled
from relative_loss import relative_structure


def scheduled_config(config, progress):
    fraction = min(1., max(0., float(progress))/max(1, config['epochs']))
    ramp = min(1., max(0., float(progress))/max(1e-6, config['finetune_loss_ramp_epochs']))
    cfg = dict(config)
    for key, prefix in [('inverse_rmse_weight', 'inverse_rmse'), ('tail_weight', 'tail'),
                        ('dynamics_aux_weight', 'dynamics_aux')]:
        cfg[key] = config[prefix+'_start']+(config[prefix+'_end']-config[prefix+'_start'])*ramp
    desired_kd = config['kd_start']+(config['kd_end']-config['kd_start'])*fraction
    # Parent receives nonzero observation progress: old inverse/tail ramps stay ON.
    offset_progress = max(1., float(progress))
    old_fraction = min(1., offset_progress/max(1, config['epochs']))
    cfg['kd_weight'] = desired_kd/(1-.5*old_fraction)
    cfg.update(relative_enabled=False, relative_weight=0., inverse_warmup_epochs=1e-6,
               tail_warmup_epochs=1e-6)
    return cfg, offset_progress


def objective(pred, batch, holdout_mask, epoch_progress, config):
    with torch.autocast(pred['D_full'].device.type, enabled=False):
        cfg, progress = scheduled_config(config, epoch_progress)
        total, stats = parent_objective(pred, batch, holdout_mask, progress, cfg)
        relative = total.new_zeros(()); coverage = relative
        ramp = min(1., max(0., float(epoch_progress))/max(1e-6, config['relative_warmup_epochs']))
        if config.get('relative_enabled') and config.get('relative_weight', 0)>0:
            if not {'relative', 'relative_confidence'}.issubset(batch):
                raise RuntimeError('Audited relative cache required; no metric/GT fallback')
            forbidden = ((batch['gt_mask']>0)|(batch['mask']>0)).float()
            relative, _, coverage = relative_structure(pred['D2'].float(), batch['relative'].float(),
                batch['relative_confidence'].float(), batch['rgb'].float(), forbidden)
        weighted = config['relative_weight']*ramp*relative
        total = total+weighted
        stats.update(total=total.detach(), relative_gradient=relative.detach(),
                     relative_pair_coverage=coverage.detach(), relative_ramp=total.new_tensor(ramp),
                     weighted_relative=weighted.detach(),
                     lambda_inverse=total.new_tensor(cfg['inverse_rmse_weight']),
                     lambda_tail=total.new_tensor(cfg['tail_weight']),
                     lambda_aux=total.new_tensor(cfg['dynamics_aux_weight']))
        stats.update({k:v.detach() for k,v in pred.items() if k.startswith('pir_')})
        return total, stats
