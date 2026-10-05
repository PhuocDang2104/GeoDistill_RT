"""Phase-preserving sensor-innovation residual, no learned full-res features."""
import torch
from torch import nn
from torch.nn import functional as F
from model_parent import ContextPhaseDetail


def phase_innovation(depth, sparse, mask):
    """Donor errors and moments separately for each PixelUnshuffle phase.

    3x3 half-grid support = nearby same-phase donors on the full grid.
    Empty neighborhoods give exactly zero moments. Never uses GT/teacher.
    """
    with torch.autocast(depth.device.type, enabled=False):
        d, s, m = depth.float(), sparse.float(), mask.float()
        valid = F.pixel_unshuffle(m, 2)
        err = F.pixel_unshuffle(m*(s-d), 2)
        mass = F.avg_pool2d(valid, 3, 1, 1)
        mean = F.avg_pool2d(err, 3, 1, 1)/mass.clamp_min(1e-6)
        second = F.avg_pool2d(err.square(), 3, 1, 1)/mass.clamp_min(1e-6)
        support = (mass > 0).float()
        std = ((second-mean.square()).clamp_min(0)+1e-6).sqrt()-.001
        packed_log = F.pixel_unshuffle(d.clamp_min(.1).log(), 2)
        contrast = packed_log-packed_log.mean(1, keepdim=True)
        features = torch.cat(((mean/20).clamp(-6, 6)*support,
                              (std/20).clamp(0, 6)*support, mass,
                              contrast.clamp(-4, 4)), 1)
        return features, support


class PhaseInnovationResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(40, 16, 1), nn.SiLU(),
                                  nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.SiLU())
        self.head = nn.Conv2d(16, 8, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad(): self.head.bias[4:].fill_(-3.)

    def forward(self, hidden, depth, sparse, mask):
        features, support = phase_innovation(depth, sparse, mask)
        output = self.head(self.body(torch.cat((hidden, features), 1))).float()
        delta, gates = output[:, :4], output[:, 4:].sigmoid()
        correction = F.pixel_shuffle(delta.tanh()*gates, 2)*(.5+.05*depth.float())
        return correction, gates, support


class InnovationPhaseDetail(ContextPhaseDetail):
    def __init__(self):
        super().__init__()
        self.pir = PhaseInnovationResidual()
        self.diagnostics = True
        self.last_diagnostics = {}

    def forward(self, rgb, sparse, mask, depth, p2, g2, rich=None):
        packed = F.pixel_unshuffle(torch.cat((sparse/120, mask, mask*(sparse-depth)/20, depth/120), 1), 2)
        hidden = self.body(torch.cat((p2, g2, F.pixel_unshuffle(rgb, 2), packed), 1))
        if rich is not None: hidden = hidden+rich
        raw = F.pixel_shuffle(self.delta(hidden).float(), 2)
        old_delta = (.5+.05*depth)*raw.tanh()
        before = (depth+old_delta).clamp(.1, 120)
        correction, gates, support = self.pir(hidden, before, sparse, mask)
        refined = (before+correction).clamp(.1, 120)
        logits = F.pixel_shuffle(self.trust_logits(hidden).float(), 2)

        def fusion(d):
            tolerance = (.5+.02*d)*self.log_tolerance.clamp(-1, 2).exp()
            confidence = logits-torch.log1p(((sparse-d)/tolerance).square())
            gate = mask*confidence.sigmoid()
            return (1-gate)*d+gate*sparse, gate, confidence

        final, gate, confidence = fusion(refined)
        # Counterfactual is telemetry only, never fed into train loss.
        self.last_diagnostics = {}
        if self.diagnostics:
            with torch.no_grad(): pre_final = fusion(before.detach())[0]
            self.last_diagnostics = {'D1_pre_pir': before, 'D_full_pre_pir': pre_final,
                'pir_abs_delta_mean_m': correction.abs().mean(),
                'pir_gate_mean': gates.mean(), 'pir_support_fraction': support.mean()}
        return refined, final, gate, confidence, old_delta+correction
