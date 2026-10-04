"""V9: five transported jets, phase-dependent consensus and disagreement gating.

No teachers/GT in forward, no additional dynamics steps or full-res feature CNN.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model_v8 import AnchorFlowEdge as V8, JetPhaseReadout, adjacent, Deploy


def candidate_queries(j):
    """Return inverse depths B,5,4,H,W and legal-neighbour mask B,5,1,H,W.

    Candidate order center,E,W,S,N. Each jet is evaluated at the target phase
    expressed relative to that candidate's origin. No interpolation/grid_sample.
    """
    jets = torch.cat((j[:, None], adjacent(j)), 1).float()
    ox = j.new_tensor([0, 1, -1, 0, 0]).view(1, 5, 1, 1, 1)
    oy = j.new_tensor([0, 0, 0, 1, -1]).view(1, 5, 1, 1, 1)
    # Query only the value component, all 5 origins x 4 phases in one broadcast.
    # Equivalent to translate(...)[..., :1], without materializing unused
    # transported gradient/Hessian components or four separate six-state cats.
    sx = jets.new_tensor([-.25,.25,-.25,.25]).view(1,1,4,1,1)-ox
    sy = jets.new_tensor([-.25,-.25,.25,.25]).view(1,1,4,1,1)-oy
    v,gx,gy,hxx,hxy,hyy = jets.split(1,dim=2)
    candidates = (v+gx*sx+gy*sy+.5*hxx*sx*sx+hxy*sx*sy+.5*hyy*sy*sy).clamp(1/120,10.)
    one = torch.ones_like(j[:, :1])
    valid = torch.cat((one[:, None],
                      torch.stack((F.pad(one[..., 1:], (0, 1, 0, 0)),
                                   F.pad(one[..., :-1], (1, 0, 0, 0)),
                                   F.pad(one[..., 1:, :], (0, 0, 0, 1)),
                                   F.pad(one[..., :-1, :], (0, 0, 1, 0))), 1)), 1)
    return candidates, valid


class MultiJetReadout(JetPhaseReadout):
    def __init__(self):
        super().__init__()
        self.candidates = nn.Conv2d(24, 20, 1)
        self.log_uncertainty_strength = nn.Parameter(torch.full((4,), math.log(math.expm1(.5))))
        nn.init.zeros_(self.candidates.weight)
        nn.init.zeros_(self.candidates.bias)
        with torch.no_grad():
            self.candidates.bias[:4].fill_(1.5)

    def forward(self, j, context, g4, base, sparse, mask, barrier):
        sensor = F.pixel_unshuffle(torch.cat((base/120, mask, mask*(sparse-base)/20, sparse/120), 1), 2)
        hidden = self.body(torch.cat((context, g4, sensor), 1))
        inv, valid = candidate_queries(j)
        bphase = F.pixel_unshuffle(base.float(), 2)[:, None]
        penalty = ((inv.reciprocal()-bphase).abs()/(1+.1*bphase)).clamp_max(6)
        # Two undirected edge barriers -> center/E/W/S/N costs.
        b = barrier.float().sigmoid()
        east, south = b[:, :1], b[:, 1:2]
        west = F.pad(east[..., :-1], (1, 0, 0, 0))
        north = F.pad(south[..., :-1, :], (0, 0, 1, 0))
        edge_cost = torch.stack((torch.zeros_like(east), east, west, south, north), 1)
        logits = 4*torch.tanh(self.candidates(hidden).float().reshape(j.shape[0], 5, 4, *j.shape[-2:])/4)
        logits = logits-penalty-2*edge_cost
        weights = (logits+(1-valid)*(-10000.)).softmax(1)
        consensus = (weights*inv).sum(1)
        # Dimensionless relative inverse-depth variance. Not calibrated aleatoric uncertainty.
        uncertainty = (weights*(inv-consensus[:, None]).square()).sum(1)/consensus.square().clamp_min(1e-6)
        query = F.pixel_shuffle(consensus.reciprocal(), 2)
        strength = F.softplus(self.log_uncertainty_strength).view(1, 4, 1, 1)
        gate = F.pixel_shuffle((self.blend(hidden).float()-strength*torch.log1p(uncertainty)).sigmoid(), 2)
        geometric = gate*(query-base).clamp(-(1+.1*base), 1+.1*base)
        delta = (1+.05*base)*F.pixel_shuffle(self.delta(hidden).float().tanh(), 2)
        final = (base+geometric+delta).clamp(.1, 120)
        diagnostics = {
            "query_uncertainty": F.pixel_shuffle(uncertainty, 2),
            "query_uncertainty_mean": uncertainty.mean(),
            "query_gate_mean": gate.mean(),
            "query_center_weight_mean": weights[:, 0].mean(),
            "query_entropy_mean": -(weights*weights.clamp_min(1e-8).log()).sum(1).mean(),
        }
        return final, query, geometric, delta, diagnostics


class AnchorFlowEdge(V8):
    def __init__(self, pretrained=False, flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", model_name="v9_consensus"):
        if model_name != "v9_consensus":
            raise ValueError("V9 model class supports only v9_consensus")
        super().__init__(pretrained, flow_steps, encoder, "v8_dynamics")
        self.phase2 = MultiJetReadout()

    def forward(self, rgb, sparse, mask, K):
        valid = (mask>.5)&(sparse>.1)&(sparse<120)&torch.isfinite(sparse)
        sparse = torch.where(valid, sparse.float(), torch.zeros_like(sparse, dtype=torch.float32))
        mask = valid.float()
        f4, f8, f16, f32 = self.encoder((rgb-self.rgb_mean)/self.rgb_std)
        f16 = self.context32(f32, f16)
        sparse_features, states = self.sparse(sparse, mask, K.float())
        p4, d16, d8, z0 = self.decoder((f4, f8, f16), sparse_features, states[0])
        d0 = z0.exp()
        g2 = self.guidance(F.pixel_unshuffle(rgb, 2))
        g4 = F.pixel_unshuffle(g2, 2)
        j, context, diagnostics = self.dynamics(p4, g4, d0, states[0])
        d4 = j[:, :1].reciprocal()
        d2_base = self.up4_2(d4, torch.cat((p4, g4, self.sensor_state(d4, sparse, mask, 4)), 1))
        sensor2 = self.sensor_state(d2_base, sparse, mask, 2)
        d2, query, geometric, phase_delta, query_stats = self.phase2(
            j, context, g4, d2_base, (sensor2[:, 2:3]*120+d2_base)*sensor2[:, 1:2],
            sensor2[:, 1:2], diagnostics["surface_barrier_logits"])
        p2 = F.interpolate(self.context2(p4), size=g2.shape[-2:], mode="nearest")
        d1_base = self.up2_1(d2, torch.cat((p2, g2, self.sensor_state(d2, sparse, mask, 2)), 1))
        d1, final, gate, logits, delta1 = self.detail1(rgb, sparse, mask, d1_base, p2, g2)
        return {"D16":d16, "D8":d8, "D0":d0, "D4":d4, "D2_base":d2_base,
                "D2_query":query, "D2":d2, "D1_base":d1_base, "D1":d1,
                "D_full":final, "D_hard":torch.where(valid, sparse, d1),
                "sensor_gate":gate, "sensor_logits":logits, "delta1":delta1,
                "phase2_delta":phase_delta, "jet_phase_delta":geometric, **diagnostics, **query_stats}
