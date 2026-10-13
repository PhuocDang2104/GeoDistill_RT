from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN,LiteBlock

def adjacent(x):
    """E/W/S/N neighbours. Boundary rates, not padding, enforce zero flux."""
    return torch.stack((F.pad(x[..., 1:], (0, 1, 0, 0), mode="replicate"),
                        F.pad(x[..., :-1], (1, 0, 0, 0), mode="replicate"),
                        F.pad(x[..., 1:, :], (0, 0, 0, 1), mode="replicate"),
                        F.pad(x[..., :-1, :], (0, 0, 1, 0), mode="replicate")), 1)



def translate(j, dx, dy):
    """Exact change of origin for a quadratic jet; offsets in quarter-grid cells."""
    v, gx, gy, hxx, hxy, hyy = j.split(1, dim=-3)
    return torch.cat((v+gx*dx+gy*dy+.5*hxx*dx*dx+hxy*dx*dy+.5*hyy*dy*dy,
                      gx+hxx*dx+hxy*dy, gy+hxy*dx+hyy*dy, hxx, hxy, hyy), dim=-3)



def edge_weights(rates):
    """Two undirected edge rates -> four neighbours, reciprocal/zero-flux."""
    east = F.pad(rates[:, :1, :, :-1], (0, 1, 0, 0))
    west = F.pad(rates[:, :1, :, :-1], (1, 0, 0, 0))
    south = F.pad(rates[:, 1:, :-1, :], (0, 0, 0, 1))
    north = F.pad(rates[:, 1:, :-1, :], (0, 0, 1, 0))
    return torch.stack((east, west, south, north), 1)

def minmod(a,b):
    # Continuous equivalent of 0.5*(sign(a)+sign(b))*min(abs(a),abs(b)).
    # No tie-sensitive selection of oppositely signed slopes, no Sign operator.
    return F.relu(torch.minimum(a,b))-F.relu(torch.minimum(-a,-b))



def limited_derivatives(x):
    east,west,south,north=adjacent(x).unbind(1)
    a,b,c,d=east-x,x-west,south-x,x-north
    gx,gy=minmod(a,b),minmod(c,d)
    gx=torch.cat((a[...,:1],gx[...,1:-1],b[...,-1:]),-1)
    gy=torch.cat((c[...,:1,:],gy[...,1:-1,:],d[...,-1:,:]),-2)
    return gx,gy

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


