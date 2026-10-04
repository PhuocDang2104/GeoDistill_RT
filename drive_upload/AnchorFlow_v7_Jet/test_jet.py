"""Translation algebra, phase ordering, tail gradient and strict V6 migration."""
import json
import unittest
from pathlib import Path
import torch
from model import AnchorFlowEdge, load_parent_state, DROPPED_KEYS
from model_v6 import surface_geometry
from model_v7 import translate, phase_query, descriptors, inverse_correction
from run import sample_inputs, count_operations
from losses import objective, tail_risk


class JetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_translation_composition_and_inverse(self):
        j = torch.randn(2,6,8,12)
        torch.testing.assert_close(translate(translate(j,.3,-.7),-.2,.4),translate(j,.1,-.3),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(translate(translate(j,1.,-1.),-1.,1.),j,atol=1e-6,rtol=1e-6)

    def test_quadratic_reproduction_and_phase_order(self):
        x = torch.arange(6).view(1,1,1,6).float()
        y = torch.arange(4).view(1,1,4,1).float()
        v = .1+.01*x-.02*y+.003*x*x+.004*x*y+.002*y*y
        j = torch.cat((v, (.01+.006*x+.004*y).expand_as(v),
                       (-.02+.004*x+.004*y).expand_as(v),
                       v*0+.006,v*0+.004,v*0+.004),1)
        shifted = translate(j,.7,-.3)
        expected = .1+.01*(x+.7)-.02*(y-.3)+.003*(x+.7)**2+.004*(x+.7)*(y-.3)+.002*(y-.3)**2
        torch.testing.assert_close(shifted[:,:1],expected,atol=2e-7,rtol=1e-6)
        packed = phase_query(j)
        for i,(dx,dy) in enumerate(((-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25))):
            torch.testing.assert_close(packed[:,i:i+1],translate(j,dx,dy)[:,:1],atol=2e-7,rtol=1e-6)
        full = torch.nn.functional.pixel_shuffle(packed,2)
        torch.testing.assert_close(full[...,0::2,1::2],packed[:,1:2])
        torch.testing.assert_close(full[...,1::2,0::2],packed[:,2:3])

    def test_descriptor_parity_without_tangent_frame(self):
        _,d,_,K = sample_inputs(torch.device("cpu"),64,128,False)
        d = torch.nn.functional.avg_pool2d(d.clamp_min(.1),4,4)
        old_ray,old_n,*_ = surface_geometry(d,K)
        ray,n,*_ = descriptors(d,K)
        torch.testing.assert_close(ray,old_ray,atol=0,rtol=0)
        torch.testing.assert_close(n,old_n,atol=0,rtol=0)

    def test_tail_does_not_saturate_or_normalize_by_tail_count(self):
        residual = torch.tensor([1.,10.,30.,1000.],requires_grad=True)
        valid = torch.tensor([1.,1.,1.,0.])
        loss = tail_risk(residual,valid,2.)
        self.assertAlmostEqual(float(loss.detach()),(64+784)/3,places=4)
        loss.backward()
        self.assertEqual(float(residual.grad[-1]),0)
        self.assertAlmostEqual(float(residual.grad[-2]),56/3,places=5)
        self.assertEqual(float(tail_risk(residual,valid*0).detach()),0)

    def test_stable_positive_inverse_correction_and_exact_zero(self):
        depth = torch.tensor([.1,1.,20.,120.])
        torch.testing.assert_close(inverse_correction(depth,depth*0),depth,atol=0,rtol=0)
        result = inverse_correction(depth,torch.tensor([-1e6,-10.,10.,1e6]))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result>=.1)&(result<=120)).all())

    def test_actual_v6_migration_rejects_other_missing_or_extra_keys(self):
        state = torch.load(Path(__file__).with_name("init_v6_best.pth"),map_location="cpu",weights_only=False)["model"]
        model = AnchorFlowEdge()
        report = load_parent_state(model,state)
        self.assertEqual(set(report["old_keys_dropped"]),DROPPED_KEYS)
        self.assertEqual(report["parent_parameters_loaded"],612701)
        self.assertEqual(report["new_parameters"],518)
        reduced = load_parent_state(AnchorFlowEdge(model_name="v6_reduced"),state)
        self.assertEqual(reduced["no_op_reference"],"v6_reduced")
        self.assertFalse(reduced["full_v6_no_op_claimed"])
        for field in ("metric4.delta.bias","connection4.field.weight"):
            broken = dict(state)
            broken.pop(field)
            with self.assertRaisesRegex(RuntimeError,"Incomplete V6 parent"):
                load_parent_state(model,broken)
        broken = dict(state,wrong_tensor=torch.zeros(1))
        with self.assertRaises(RuntimeError):
            load_parent_state(model,broken)

    def test_opened_jet_gradients_amp_and_mass_bound(self):
        model = AnchorFlowEdge().train()
        state = torch.load(Path(__file__).with_name("init_v6_best.pth"),map_location="cpu",weights_only=False)["model"]
        load_parent_state(model,state)
        model.freeze_encoder_bn()
        inputs = tuple(t.repeat(2,*([1]*(t.ndim-1))) for t in sample_inputs(torch.device("cpu"),64,128,False))
        cfg = json.loads(Path(__file__).with_name("config.json").read_text())
        cfg["teacher_enabled"] = False
        batch = dict(gt=inputs[1]*0+20,gt_mask=inputs[2]*0+1,sparse=inputs[1],mask=inputs[2])
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        for i in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu",dtype=torch.bfloat16):
                out = model(*inputs)
            loss,_ = objective(out,batch,inputs[2]*0,i,cfg)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.connection4.jet.weight.grad.abs().sum()),0)
            self.assertGreater(float(model.phase2.proposal_adapter.weight.grad.abs().sum()),0)
            self.assertTrue(((out["jet_neighbour_mass"]>=0)&(out["jet_neighbour_mass"]<=.500001)).all())
            for key in out:
                self.assertFalse(key.startswith("_"))
            optimizer.step()

    def test_compute_and_empty_sparse(self):
        model = AnchorFlowEdge().eval()
        inputs = sample_inputs(torch.device("cpu"))
        report = count_operations(model,inputs)
        self.assertEqual(report["total_parameters"],613219)
        self.assertLess(report["total_conv_linear_macs"],4.1e9)
        with torch.no_grad():
            output = model(inputs[0],inputs[1]*0,inputs[2]*0,inputs[3])
        self.assertTrue(torch.isfinite(output["D_full"]).all())
        torch.testing.assert_close(output["D_full"],output["D1"],atol=0,rtol=0)


if __name__=="__main__":
    unittest.main()
