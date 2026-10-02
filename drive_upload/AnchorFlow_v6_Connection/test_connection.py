"""Geometry, warm-start, opened-head gradients and compute contracts for V6."""
import json
import unittest
from pathlib import Path
import torch
from model import AnchorFlowEdge,load_parent_state
from model_v6 import surface_geometry,rotate_tangent,project_to_ray,normalize
from run import count_operations,sample_inputs
from losses import objective
from loss_helpers import huber,mean_masked


class ConnectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_planar_normal_and_frame(self):
        depth = torch.ones(2,1,16,32)*20
        K = torch.tensor([[[80.,0,64],[0,80,32],[0,0,1]]]).repeat(2,1,1)
        ray,n,t1,t2,_ = surface_geometry(depth,K)
        torch.testing.assert_close(n[:,:2],n[:,:2]*0,atol=1e-6,rtol=0)
        torch.testing.assert_close(n[:,2:],torch.ones_like(depth))
        xi = .05+.01*ray[:,:1]-.008*ray[:,1:2]
        _,n,t1,t2,_ = surface_geometry(xi.reciprocal(),K)
        expected = normalize(torch.tensor([.01,-.008,.05]).view(1,3,1,1),1).expand_as(n)
        torch.testing.assert_close(n,expected,atol=3e-6,rtol=1e-4)
        torch.testing.assert_close((n*t1).sum(1),torch.zeros_like(depth[:,0]),atol=1e-6,rtol=0)
        torch.testing.assert_close(torch.cross(t1,t2,dim=1),n,atol=1e-6,rtol=1e-5)

    def test_rotation_preserves_tangency_norm_and_reverse(self):
        q = normalize(torch.randn(2,4,3,4,8),2)
        p = normalize(q+.2*torch.randn_like(q),2)
        v = torch.randn_like(q)
        v = v-(v*q).sum(2,keepdim=True)*q
        transported = rotate_tangent(v,q,p)
        torch.testing.assert_close((transported*p).sum(2),torch.zeros_like(v[:,:,0]),atol=2e-6,rtol=0)
        torch.testing.assert_close(transported.square().sum(2),v.square().sum(2),atol=4e-6,rtol=2e-6)
        torch.testing.assert_close(rotate_tangent(transported,p,q),v,atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.isfinite(rotate_tangent(v,q,-q)).all())

    def test_ray_projection_is_not_z_component(self):
        ray = torch.tensor([2.,0.,1.]).view(1,3,1,1)
        vector = torch.tensor([0.,0.,1.]).view(1,3,1,1)
        delta = project_to_ray(vector,ray)
        self.assertAlmostEqual(float(delta),.2,places=6)
        self.assertAlmostEqual(float(((vector-delta*ray)*ray).sum()),0,places=6)

    def test_v5_noop_is_exact_with_nonzero_parent_heads(self):
        parent = AnchorFlowEdge(model_name="v5_piecewise").eval()
        with torch.no_grad():
            parent.surface.amplitude.fill_(.15)
            torch.nn.init.normal_(parent.surface.reaction.weight,std=.005)
        inputs = sample_inputs(torch.device("cpu"),height=64,width=128,channels_last=False)
        with torch.no_grad():
            reference = parent(*inputs)
            for mode in ("v6_connection","v6_ambient"):
                model = AnchorFlowEdge(model_name=mode).eval()
                migration = load_parent_state(model,parent.state_dict())
                self.assertEqual(migration["parent_parameters_loaded"],579993)
                actual = model(*inputs)
                for stage in ("D16","D8","D4","D2","D1","D_full"):
                    torch.testing.assert_close(actual[stage],reference[stage],atol=0,rtol=0)
                self.assertEqual(float(actual["connection_delta"].abs().max()),0)
                self.assertEqual(float(actual["phase2_delta"].abs().max()),0)
                self.assertNotIn("_connection_context",actual)

    def test_v6_amp_backward_and_new_head_opening(self):
        model = AnchorFlowEdge().train()
        model.freeze_encoder_bn()
        rgb,sparse,mask,K = sample_inputs(torch.device("cpu"),height=64,width=128,channels_last=False)
        rgb,sparse,mask,K = (t.repeat(2,*([1]*(t.ndim-1))) for t in (rgb,sparse,mask,K))
        cfg = json.loads(Path(__file__).with_name("config.json").read_text())
        cfg["teacher_enabled"] = False
        batch = dict(gt=torch.ones_like(mask)*20,gt_mask=torch.ones_like(mask),sparse=sparse,mask=mask)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu",dtype=torch.bfloat16):
                out = model(rgb,sparse,mask,K)
            self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
            loss,_ = objective(out,batch,mask*0,step,cfg)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.connection4.field.weight.grad.abs().sum()),0)
            self.assertGreater(float(model.phase2.delta.weight.grad.abs().sum()),0)
            if step==1:
                self.assertGreater(float(model.connection4.body[0][0].weight.grad.abs().sum()),0)
            optimizer.step()

    def test_robust_mse_matches_metric_square_and_masks_invalid(self):
        residual = torch.tensor([1.,10.,30.,1000.],requires_grad=True)
        valid = torch.tensor([1.,1.,1.,0.])
        loss = mean_masked(2*huber(residual,20),valid)
        self.assertAlmostEqual(float(loss.detach()), (1+100+800)/3,places=4)
        loss.backward()
        self.assertEqual(float(residual.grad[-1]),0)
        self.assertAlmostEqual(float(residual.grad[-2]),40/3,places=5)

    def test_v6_full_size_compute_and_empty_sparse(self):
        model = AnchorFlowEdge().eval()
        inputs = sample_inputs(torch.device("cpu"))
        report = count_operations(model,inputs)
        self.assertLess(report["total_parameters"],650000)
        self.assertLess(report["total_conv_linear_macs"],4.5e9)
        with torch.no_grad():
            output = model(inputs[0],inputs[1]*0,inputs[2]*0,inputs[3])
        self.assertTrue(torch.isfinite(output["D_full"]).all())
        self.assertEqual(output["D_full"].shape,(1,1,352,1216))

    def test_missing_old_keys_rejected(self):
        parent = AnchorFlowEdge(model_name="v5_piecewise").state_dict()
        parent.pop("metric4.delta.bias")
        with self.assertRaisesRegex(RuntimeError,"Incomplete parent"):
            load_parent_state(AnchorFlowEdge(),parent)


if __name__ == "__main__":
    unittest.main()
