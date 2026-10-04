"""Residual projective 2-jets: matrix-free translation and shared phase query.

Coordinates are D4 cells, NOT intrinsic surface coordinates. Six coefficients
are a local Taylor model, not a globally integrable field or a physics PDE.
The trained V6 context/phase CNNs are retained; its vector head is removed.
Zero heads reproduce ReducedV6, deliberately NOT the full V6 checkpoint.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model_v5 import AnchorFlowEdge as V5
from model_v6 import AnchorFlowEdge as V6, ConnectionMetric4, PhaseMetric2, adjacent, rays, normalize


def derivatives(x):
    east, west, south, north = adjacent(x).squeeze(2).split(1, 1)
    left, right, up, down = x-west, east-x, x-north, south-x
    gx = torch.where(left.abs()<right.abs(), left, right)
    gy = torch.where(up.abs()<down.abs(), up, down)
    gx = torch.cat((right[..., :1], gx[..., 1:-1], left[..., -1:]), -1)
    gy = torch.cat((down[..., :1, :], gy[..., 1:-1, :], up[..., -1:, :]), -2)
    return gx, gy


def descriptors(depth, K):
    """Exactly the V6 ray/normal input, without tangent frames or cross products."""
    d = depth.detach().float()
    xi = d.reciprocal()
    gx, gy = derivatives(xi)
    ray = rays(d, K.float())
    a, b = gx*K[:, 0, 0, None, None, None]/4, gy*K[:, 1, 1, None, None, None]/4
    normal = normalize(torch.cat((a, b, xi-a*ray[:, :1]-b*ray[:, 1:2]), 1), 1)
    return ray, normal, xi, gx, gy


def context_input(feature, depth, state, rgb, ray, normal):
    mean, valid, density, _, _, spread = state
    sensor = torch.cat((depth/120, (valid*(mean-depth)/20).clamp(-6, 6),
                        valid, density, spread.clamp_max(2)), 1)
    return torch.cat((feature, F.avg_pool2d(rgb, 4, 4), sensor, ray, normal), 1)


def translate(jet, dx, dy):
    """Recenter a quadratic at displacement p-q. Supports tensor broadcasting."""
    v, gx, gy, xx, xy, yy = jet.split(1, dim=-3)
    return torch.cat((v+gx*dx+gy*dy+.5*xx*dx*dx+xy*dx*dy+.5*yy*dy*dy,
                      gx+xx*dx+xy*dy, gy+xy*dx+yy*dy, xx, xy, yy), dim=-3)


def phase_query(jet, include_value=True):
    """Four child centers, PixelShuffle order 00,01,10,11; offsets +/-1/4."""
    v, gx, gy, xx, xy, yy = jet.split(1, 1)
    center = v if include_value else v*0
    common = center + (xx+yy)/32
    return torch.cat((common-gx/4-gy/4+xy/16, common+gx/4-gy/4-xy/16,
                      common-gx/4+gy/4-xy/16, common+gx/4+gy/4+xy/16), 1)


def inverse_correction(depth, delta_xi):
    # Stable exact no-op when delta_xi=0; FP32 denominator guard before reciprocal.
    return (depth.float()/(1+depth.float()*delta_xi.float()).clamp(.2, 5)).clamp(.1, 120)


class JetMetric4(nn.Module):
    def __init__(self, steps=2):
        super().__init__()
        # Reuse all trained V6 body weights/shapes, not a random replacement CNN.
        self.body = ConnectionMetric4().body
        self.jet = nn.Conv2d(64, 6, 1)
        nn.init.zeros_(self.jet.weight)
        nn.init.zeros_(self.jet.bias)
        self.steps = int(steps)
        self.register_buffer("residual_bounds", torch.tensor([.50, .10, .10, .04, .04, .04]).view(1, 6, 1, 1))
        # Device-resident constants: no Python-list -> GPU transfer each forward.
        # Keep buffers rank-1: Module.to(channels_last) must not receive rank-5.
        self.register_buffer("offset_dx", torch.tensor([-1., 1., 0., 0.]), persistent=False)
        self.register_buffer("offset_dy", torch.tensor([0., 0., -1., 1.]), persistent=False)

    def forward(self, feature, depth, state, rgb, K, old_weights):
        ray, normal, xi, gx, gy = descriptors(depth, K)
        hidden = self.body(context_input(feature, depth, state, rgb, ray, normal))
        # Detached scale prevents an uncontrolled feedback path through 1/D.
        delta = xi*self.residual_bounds*self.jet(hidden).float().tanh()
        # Base derivatives are only a proposal/compatibility descriptor. Mixed
        # cells near occlusion boundaries are not guaranteed physical surfaces.
        hxx, hxy1 = derivatives(gx)
        hxy2, hyy = derivatives(gy)
        limit = xi*.25
        base = torch.cat((xi, gx.clamp(-limit, limit), gy.clamp(-limit, limit),
                          hxx.clamp(-limit, limit), ((hxy1+hxy2)*.5).clamp(-limit, limit),
                          hyy.clamp(-limit, limit)), 1)
        dx, dy = self.offset_dx.view(1, 4, 1, 1, 1), self.offset_dy.view(1, 4, 1, 1, 1)
        base_neighbors = adjacent(base)
        neighbor_base = translate(base_neighbors, dx, dy)
        # Symmetric p<->q inverse-plane agreement, approximately converted to m.
        outgoing = translate(base[:, None].expand(-1, 4, -1, -1, -1), -dx, -dy)[:, :, :1]
        dq = base_neighbors[:, :, :1].reciprocal()
        disagreement = .5*((neighbor_base[:, :, :1]-xi[:, None]).abs()+(outgoing-base_neighbors[:, :, :1]).abs())
        discrepancy = disagreement*depth.detach().float()[:, None]*dq
        tolerance = .25+.01*(depth.detach().float()[:, None]+dq)
        compatibility = .05+.95*torch.exp(-(discrepancy/tolerance).clamp_max(20))
        # Inherited V5 absolute weights sum<=1. Retain at least 50% center mass.
        weights = .5*old_weights[:, :, None].float()*compatibility
        center = 1-weights.sum(1)
        for _ in range(self.steps):
            delta = center*delta+(weights*translate(adjacent(delta), dx, dy)).sum(1)
        refined = inverse_correction(depth, delta[:, :1])
        return refined, {"D4_surface": depth, "jet_delta4": refined-depth,
                         "jet_neighbour_mass": weights.sum(1), "jet_gradient_abs": delta[:, 1:3].abs().mean(1, keepdim=True),
                         "jet_hessian_abs": delta[:, 3:].abs().mean(1, keepdim=True),
                         "_connection_context": hidden, "_jet_delta": delta, "_jet_base": base}


class JetPhase2(PhaseMetric2):
    def __init__(self):
        super().__init__()
        self.proposal_adapter = nn.Conv2d(4, 32, 1, bias=False)
        nn.init.zeros_(self.proposal_adapter.weight)

    def forward(self, depth, g2, context, sparse, mask, residual_jet, base_jet):
        original = depth
        # The center correction already entered D4/up4_2; don't add it twice.
        shape = F.pixel_shuffle(phase_query(residual_jet, include_value=False), 2)
        depth = inverse_correction(depth, shape)
        absolute_xi = F.pixel_shuffle(phase_query(base_jet+residual_jet), 2)
        proposal = absolute_xi.clamp(1/120, 10).reciprocal()
        proposal_error = F.pixel_unshuffle((proposal-original).clamp(-120, 120)/20, 2)
        density = F.avg_pool2d(mask, 2, 2)
        mean = F.avg_pool2d(sparse*mask, 2, 2)/density.clamp_min(1e-6)
        valid = (density>0).float()
        state = torch.cat((depth/120, valid, density, (valid*(mean-depth)/20).clamp(-6, 6)), 1)
        inputs = torch.cat((context, F.pixel_unshuffle(g2, 2), F.pixel_unshuffle(state, 2)), 1)
        first = self.body[0]
        hidden = first[2](first[1](first[0](inputs)+self.proposal_adapter(proposal_error)))
        hidden = self.body[1](hidden)
        delta = (1+.05*depth)*F.pixel_shuffle(self.delta(hidden).float(), 2).tanh()
        refined = (depth+delta).clamp(.1, 120)
        return refined, {"D2_base": original, "D2_jet": depth,
                         "phase2_delta": refined-depth, "jet_phase_delta": depth-original}


class ReducedV6(V6):
    """Causal control: vector correction OFF; learned context/phase still ON."""
    def refine_surface(self, p4, d4, state, rgb, K):
        d4, diag = V5.refine_surface(self, p4, d4, state, rgb, K)
        ray, normal, *_ = descriptors(d4, K)
        context = self.connection4.body(context_input(p4, d4, state, rgb, ray, normal))
        return d4, {**diag, "D4_surface": d4, "_connection_context": context}


class AnchorFlowEdge(V5):
    def __init__(self, pretrained=False, flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", jet_steps=2):
        super().__init__(pretrained=pretrained, flow_steps=flow_steps, encoder=encoder, barrier_enabled=True)
        self.connection4 = JetMetric4(jet_steps)
        self.phase2 = JetPhase2()

    def refine_surface(self, p4, d4, state, rgb, K):
        d4, diag = super().refine_surface(p4, d4, state, rgb, K)
        d4, new = self.connection4(p4, d4, state, rgb, K, diag["surface_weights"])
        return d4, {**diag, **new}

    def refine_half(self, d2, p4, g2, sparse, mask, K, diagnostics):
        context = diagnostics.pop("_connection_context")
        residual, base = diagnostics.pop("_jet_delta"), diagnostics.pop("_jet_base")
        return self.phase2(d2, g2, context, sparse, mask, residual, base)
