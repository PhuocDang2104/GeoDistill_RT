"""Identical fresh-training objective in all three arms; no refinement-specific loss."""
import torch
from experiment import ROOT
from losses_parent import objective as baseline_objective


def loss_fn(pred, batch, holdout, progress, cfg):
    with torch.autocast(pred["D_full"].device.type, enabled=False):
        parent = dict(cfg, relative_enabled=False, relative_weight=0.)
        loss, stats = baseline_objective(pred, batch, holdout, progress, parent)
        relative = loss.new_zeros(())
        coverage = loss.new_zeros(())
        if cfg["relative_enabled"]:
            from relative_loss import relative_structure
            forbidden = ((batch["gt_mask"] > 0) | (batch["mask"] > 0)).float()
            relative, _, coverage = relative_structure(pred["D2"].float(), batch["relative"],
                batch["relative_confidence"], batch["rgb"], forbidden)
        ramp = min(1., max(0., progress) / cfg["relative_warmup_epochs"]) if cfg['relative_enabled'] else 0.
        loss = loss + cfg["relative_weight"] * ramp * relative
        stats.update(total=loss.detach(),relative_gradient=relative.detach(),
            relative_pair_coverage=coverage.detach(),relative_ramp=loss.new_full((),ramp),weighted_relative=(
            cfg["relative_weight"] * ramp * relative).detach())
    return loss, stats
