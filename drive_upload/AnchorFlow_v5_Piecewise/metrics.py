"""Global pixel metrics without per-batch CPU synchronization."""
import torch
from torch.nn import functional as F


class Metrics:
    def __init__(self, device):
        self.names = ["all", "0-20", "20-40", "40-60", "60-80", "80-120", "edge", "non_edge"]
        self.sums = torch.zeros(8, 8, dtype=torch.float64, device=device)
        self.nonfinite = torch.zeros((), dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred, gt, valid, rgb):
        p, g, valid = pred.float(), gt.float(), valid.bool()
        self.nonfinite += (~torch.isfinite(p)).sum()
        # Same grayscale/gradient alignment as src/metrics.py in the S3 baseline.
        grey = rgb.float().mean(1, keepdim=True)
        edge = torch.maximum(F.pad((grey[..., 1:] - grey[..., :-1]).abs(), (1, 0, 0, 0)),
                             F.pad((grey[..., 1:, :] - grey[..., :-1, :]).abs(), (0, 0, 1, 0))) > 0.05
        masks = [valid, valid & (g < 20), valid & (g >= 20) & (g < 40),
                 valid & (g >= 40) & (g < 60), valid & (g >= 60) & (g < 80),
                 valid & (g >= 80), valid & edge, valid & ~edge]
        # eps guards invalid zero GT; no evaluator-side prediction clamp to 0.1 m.
        error = p - g
        inverse_error = 1000 * (p.clamp_min(1e-6).reciprocal() - g.clamp_min(1e-6).reciprocal())
        rel = error.abs() / g.clamp_min(1e-6)
        ratio = torch.maximum(p / g.clamp_min(1e-6), g / p.clamp_min(1e-6))
        for i, mask in enumerate(masks):
            values = [mask.sum(), (error.square() * mask).sum(dtype=torch.float64),
                      (error.abs() * mask).sum(dtype=torch.float64),
                      (inverse_error.square() * mask).sum(dtype=torch.float64),
                      (inverse_error.abs() * mask).sum(dtype=torch.float64),
                      (rel * mask).sum(dtype=torch.float64), ((ratio < 1.25) & mask).sum(),
                      ((p < 0.1) & mask).sum()]
            self.sums[i] += torch.stack(values)

    def report(self):
        values = self.sums.cpu().tolist()
        if int(self.nonfinite.cpu()):
            raise RuntimeError("Nonfinite predictions encountered during evaluation")
        result = {}
        for name, v in zip(self.names, values):
            count = int(v[0])
            result[name] = {"pixels": count,
                            "squared_error_fraction_global": v[1] / values[0][1] if values[0][1] else 0,
                            "rmse_m": (v[1] / count) ** 0.5 if count else None,
                            "mae_m": v[2] / count if count else None,
                            "irmse_km_inv": (v[3] / count) ** 0.5 if count else None,
                            "imae_km_inv": v[4] / count if count else None,
                            "abs_rel": v[5] / count if count else None,
                            "delta1": v[6] / count if count else None,
                            "pred_below_0_1": int(v[7])}
        return result
