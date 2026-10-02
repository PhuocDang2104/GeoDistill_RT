"""Connection-aware metric residual at D4 + phase-resolved correction at D2.

Normal alignment is a DISCRETE APPROXIMATION, not exact manifold parallel
transport. Tangent vectors and normal scalars are transported separately. All
geometry is FP32, cached, fixed four-neighbour stencil; no point-cloud warp.
"""
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN,LiteBlock
from model_v5 import AnchorFlowEdge as V5


def adjacent(t):
    # [B,4,C,H,W], E/W/S/N. No pixel loop or NxN matrix.
    return torch.stack((F.pad(t[...,1:],(0,1,0,0),mode="replicate"),
                        F.pad(t[...,:-1],(1,0,0,0),mode="replicate"),
                        F.pad(t[...,1:,:],(0,0,0,1),mode="replicate"),
                        F.pad(t[...,:-1,:],(0,0,1,0),mode="replicate")),1)


def normalize(v,dim):
    return v/(v.square().sum(dim,keepdim=True)+1e-12).sqrt()


def rays(depth,K):
    h,w = depth.shape[-2:]
    u = (torch.arange(w,device=depth.device,dtype=torch.float32)+.5)*4-.5
    v = (torch.arange(h,device=depth.device,dtype=torch.float32)+.5)*4-.5
    x = (u.view(1,1,1,w)-K[:,0,2,None,None,None])/K[:,0,0,None,None,None].clamp_min(1)
    y = (v.view(1,1,h,1)-K[:,1,2,None,None,None])/K[:,1,1,None,None,None].clamp_min(1)
    return torch.cat((x.expand_as(depth),y.expand_as(depth),torch.ones_like(depth)),1)


def surface_geometry(depth,K):
    """Robust one-sided inverse-depth derivatives; reject cross-edge gradients.

Normals are detached descriptors: no backprop through finite-difference normal
estimation. This avoids unstable second-order geometry gradients under AMP.
"""
    d = depth.detach().float()
    ray = rays(d,K.float())
    xi = d.reciprocal()
    ne = adjacent(xi).squeeze(2)
    east,west,south,north = ne.split(1,1)
    left,right = xi-west,east-xi
    up,down = xi-north,south-xi
    gx = torch.where(left.abs()<right.abs(),left,right)
    gy = torch.where(up.abs()<down.abs(),up,down)
    # Correct one-sided derivative at image edges, not replicated zero.
    gx = torch.cat((right[...,:1],gx[...,1:-1],left[...,-1:]),-1)
    gy = torch.cat((down[...,:1,:],gy[...,1:-1,:],up[...,-1:,:]),-2)
    a = gx*K[:,0,0,None,None,None].float()/4
    b = gy*K[:,1,1,None,None,None].float()/4
    normal = normalize(torch.cat((a,b,xi-a*ray[:,:1]-b*ray[:,1:2]),1),1)
    axis_x = torch.cat((torch.ones_like(d),d*0,d*0),1)
    axis_y = torch.cat((d*0,torch.ones_like(d),d*0),1)
    axis = torch.where(normal[:,:1].abs()>.9,axis_y,axis_x)
    tangent1 = normalize(axis-(axis*normal).sum(1,keepdim=True)*normal,1)
    tangent2 = torch.cross(normal,tangent1,dim=1)
    return ray,normal,tangent1,tangent2,d*ray


def rotate_tangent(vector,source_normal,target_normal):
    """Minimal normal-aligning rotation, Rodrigues WITHOUT 3x3 matrices.

Caller masks nearly antipodal pairs. Return identity there (finite fallback),
never divide by 1+cos(theta) near zero. Not a geodesic/path-defined connection.
"""
    cosine = (source_normal*target_normal).sum(2,keepdim=True)
    axis = torch.cross(source_normal,target_normal,dim=2)
    cross1 = torch.cross(axis,vector,dim=2)
    rotated = vector+cross1+torch.cross(axis,cross1,dim=2)/(1+cosine).clamp_min(.05)
    return torch.where(cosine>-.95,rotated,vector)


def project_to_ray(vector,ray):
    # Least-squares constrained update X'=D' r. Not just vector.z.
    return (vector*ray).sum(1,keepdim=True)/ray.square().sum(1,keepdim=True).clamp_min(1e-6)


class ConnectionMetric4(nn.Module):
    def __init__(self,transport_mode="connection"):
        super().__init__()
        self.transport_mode = transport_mode
        # P4(48)+RGB4(3)+depth/innovation/valid/density/spread(5)+ray(3)+normal(3).
        self.body = nn.Sequential(ConvBN(62,64),LiteBlock(64),
                                  nn.Conv2d(64,64,5,padding=4,dilation=2,groups=64),nn.SiLU())
        self.field = nn.Conv2d(64,3,1)
        nn.init.zeros_(self.field.weight)
        nn.init.zeros_(self.field.bias)

    def forward(self,feature,depth,state,rgb,K,old_weights):
        mean,valid,density,_,_,spread = state
        ray,n,t1,t2,points = surface_geometry(depth,K)
        sensor = torch.cat((depth/120,(valid*(mean-depth)/20).clamp(-6,6),valid,density,spread.clamp_max(2)),1)
        hidden = self.body(torch.cat((feature,F.avg_pool2d(rgb,4,4),sensor,ray,n),1))
        coefficients = (1+.08*depth)*self.field(hidden).float().tanh()
        ca,cb,cn = coefficients.split(1,1)
        tangent = ca*t1+cb*t2
        local_vector = tangent+cn*n
        nq = adjacent(n)
        npair = n[:,None].expand_as(nq)
        difference = adjacent(points)-points[:,None]
        plane_error = (difference*npair).sum(2,keepdim=True).abs()+(difference*nq).sum(2,keepdim=True).abs()
        tolerance = (.25+.02*depth)[:,None]
        compatibility = .05+.95*torch.exp(-(plane_error/tolerance).clamp_max(20))
        cosine = (nq*npair).sum(2,keepdim=True)
        weights = old_weights[:, :, None].float()*compatibility*(cosine>-.95)
        if self.transport_mode=="connection":
            neighbour = rotate_tangent(adjacent(tangent),nq,npair)+adjacent(cn)*npair
        else:
            neighbour = adjacent(local_vector)  # valid common-camera-frame ablation
        mixed = (1-weights.sum(1))*local_vector+(weights*neighbour).sum(1)
        delta = project_to_ray(mixed,ray)
        refined = (depth+delta).clamp(.1,120)
        diagnostics = {"D4_surface":depth,"connection_delta":refined-depth,
                       "connection_tangent_magnitude":torch.linalg.vector_norm(tangent,dim=1,keepdim=True),
                       "connection_normal_magnitude":cn.abs(),
                       "connection_neighbour_mass":weights.sum(1),
                       "connection_plane_error":plane_error.mean(1),"_connection_context":hidden}
        return refined,diagnostics


class PhaseMetric2(nn.Module):
    def __init__(self):
        super().__init__()
        # Quarter context(64), G2 packed(32), D2 + S2 validity/density/innovation packed(16).
        self.body = nn.Sequential(ConvBN(112,32),LiteBlock(32))
        self.delta = nn.Conv2d(32,4,1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self,depth,g2,context,sparse,mask):
        density = F.avg_pool2d(mask,2,2)
        mean = F.avg_pool2d(sparse*mask,2,2)/density.clamp_min(1e-6)
        valid = (density>0).float()
        state = torch.cat((depth/120,valid,density,(valid*(mean-depth)/20).clamp(-6,6)),1)
        hidden = self.body(torch.cat((context,F.pixel_unshuffle(g2,2),F.pixel_unshuffle(state,2)),1))
        delta = (1+.05*depth)*F.pixel_shuffle(self.delta(hidden).float(),2).tanh()
        refined = (depth+delta).clamp(.1,120)
        return refined,{"D2_base":depth,"phase2_delta":refined-depth}


class AnchorFlowEdge(V5):
    def __init__(self,pretrained=False,flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k",transport_mode="connection"):
        super().__init__(pretrained=pretrained,flow_steps=flow_steps,encoder=encoder,barrier_enabled=True)
        self.connection4 = ConnectionMetric4(transport_mode)
        self.phase2 = PhaseMetric2()

    def refine_surface(self,p4,d4,state,rgb,K):
        d4,diagnostics = super().refine_surface(p4,d4,state,rgb,K)
        d4,new = self.connection4(p4,d4,state,rgb,K,diagnostics["surface_weights"])
        return d4,{**diagnostics,**new}

    def refine_half(self,d2,p4,g2,sparse,mask,K,diagnostics):
        context = diagnostics.pop("_connection_context")
        return self.phase2(d2,g2,context,sparse,mask)
