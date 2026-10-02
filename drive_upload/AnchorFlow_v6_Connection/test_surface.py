import inspect
import json
import unittest
from pathlib import Path
import torch
from model import AnchorFlowEdge,load_parent_state
from model_v3 import AnchorFlowEdge as V3
from model_v5 import directional_weights,plane_transport
from boundaries import boundary_mask,quarter_targets,balanced_barrier_loss,BoundaryMetrics
from losses import objective
from run import sample_inputs,count_operations


class SurfaceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.state = torch.load(Path(__file__).with_name("init_v3_best.pth"),map_location="cpu",weights_only=False)["model"]

    def test_actual_parent_exact_noop_both_modes_all_old_stages(self):
        old = V3().eval()
        old.load_state_dict(self.state,strict=True)
        inputs = sample_inputs(torch.device("cpu"),64,128,False)
        with torch.no_grad():
            expected = old(*inputs)
            for mode in ("v5_transport","v5_piecewise"):
                new = AnchorFlowEdge(model_name=mode).eval()
                report = load_parent_state(new,self.state)
                self.assertEqual(report["parent_tensors_loaded"],697)
                self.assertEqual(report["parent_parameters_loaded"],572017)
                actual = new(*inputs)
                for key in expected:
                    torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
                self.assertEqual(float(actual["surface_delta"].abs().max()),0)
                broken = dict(self.state)
                broken.pop(next(iter(broken)))
                with self.assertRaises(RuntimeError):
                    load_parent_state(new,broken)

    def test_plane_transport_preserves_affine_inverse_depth(self):
        K = torch.tensor([[[80.,0,10.],[0,100.,8.],[0,0,1.]]])
        x = torch.arange(12).view(1,1,1,12)*4/80
        y = torch.arange(8).view(1,1,8,1)*4/100
        xi = .1+.01*x-.02*y
        slopes = torch.cat((torch.ones_like(xi)*.01,torch.ones_like(xi)*-.02),1)
        weights = directional_weights(torch.zeros(1,2,8,12),torch.zeros(1,2,8,12))
        out = plane_transport(xi,slopes,K,weights)
        torch.testing.assert_close(out,xi,atol=1e-7,rtol=0)
        zero = torch.zeros_like(xi)
        empty = plane_transport(torch.ones_like(xi)/20,torch.cat((zero,zero),1),K,weights)
        torch.testing.assert_close(empty,torch.ones_like(xi)/20,atol=1e-7,rtol=0)

    def test_barriers_symmetric_directional_and_not_normalized_away(self):
        affinity = torch.zeros(1,2,8,12)
        barrier = torch.full_like(affinity,-20)
        opened = directional_weights(affinity,barrier)
        barrier[:,0] = 20
        closed_x = directional_weights(affinity,barrier)
        self.assertLess(float(closed_x[:,:2].sum()),1e-6)
        torch.testing.assert_close(closed_x[:,2:],opened[:,2:])
        torch.testing.assert_close(opened[:,0,:,:-1],opened[:,1,:,1:])
        torch.testing.assert_close(opened[:,2,:-1,:],opened[:,3,1:,:])
        blocked = directional_weights(affinity,barrier*0+20)
        self.assertLess(float(blocked.sum()),1e-6)
        self.assertTrue((opened.sum(1)<=1).all())

    def test_gt_boundary_does_not_use_missing_values_as_edges(self):
        gt = torch.ones(1,1,16,32)*10
        gt[...,:,16:] = 30
        valid = torch.ones_like(gt)
        valid[...,0,:] = 0
        gt[...,0,:] = float("nan")
        mask = boundary_mask(gt,valid)
        self.assertEqual(int(mask[...,0,:].sum()),0)
        self.assertEqual(int(mask[...,1:,15:17].sum()),30)
        qlabels,support = quarter_targets(gt,valid)
        self.assertTrue(torch.isfinite(qlabels).all())
        self.assertGreater(float(qlabels.sum()),0)
        logits = torch.zeros_like(qlabels,requires_grad=True)
        loss,_,_ = balanced_barrier_loss(logits,gt,valid)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        empty,_,_ = balanced_barrier_loss(logits,gt,valid*0)
        self.assertEqual(float(empty.detach()),0)

    def test_boundary_metrics_counts_bands_and_ring_union(self):
        g = torch.ones(1,1,16,32)*10
        g[...,16:] = 30
        metrics = BoundaryMetrics("cpu")
        metrics.update(g+2,g,torch.ones_like(g))
        report = metrics.report()
        self.assertAlmostEqual(report["bands"]["3"]["rmse_m"],2)
        self.assertEqual(sum(r["pixels"] for r in report["rings"].values()),report["bands"]["10"]["pixels"])

    def test_amp_finite_new_gradients_and_opened_slopes(self):
        torch.manual_seed(3)
        model = AnchorFlowEdge().train()
        load_parent_state(model,self.state)
        model.freeze_encoder_bn()
        inputs = sample_inputs(torch.device("cpu"),64,128,False)
        inputs = tuple(t.repeat(2,*([1]*(t.ndim-1))) for t in inputs)
        gt = inputs[1]*0+25
        gt[...,64:] = 50
        valid = torch.ones_like(gt)
        batch = dict(rgb=inputs[0],sparse=inputs[1],mask=inputs[2],gt=gt,gt_mask=valid,
                     teacher=gt,confidence=valid*.8)
        cfg = json.loads(Path(__file__).with_name("config.json").read_text())
        with torch.autocast("cpu",dtype=torch.bfloat16):
            output = model(*inputs)
        loss,stats = objective(output,batch,valid*0,0,cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.surface.reaction.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.surface.barriers.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.surface.amplitude.grad.abs()),0)
        # A real optimizer update opens zero-init heads before testing trunk grads.
        with torch.no_grad():
            for parameter in model.surface.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad,alpha=-.001)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.surface.amplitude.fill_(.1)
        model(*inputs)["D_full"].mean().backward()
        self.assertGreater(float(model.surface.slopes.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.surface.body[0][0].weight.grad.abs().sum()),0)

    def test_inference_inputs_empty_sparse_and_compute_budget(self):
        model = AnchorFlowEdge().eval()
        self.assertEqual(list(inspect.signature(model.forward).parameters),["rgb","sparse","mask","K"])
        rgb,s,m,K = sample_inputs(torch.device("cpu"),64,128,False)
        with torch.no_grad():
            output = model(rgb,s*0,m*0,K)
        torch.testing.assert_close(output["D_full"],output["D1"],atol=0,rtol=0)
        self.assertTrue(torch.isfinite(output["D_full"]).all())
        extra = sum(p.numel() for p in model.surface.parameters())
        self.assertLess(extra,100000)
        complexity = count_operations(model,(rgb,s,m,K))
        quarter_pixels = (352//4)*(1216//4)
        macs = complexity["conv_linear_macs"]["surface"]*quarter_pixels/((64//4)*(128//4))
        self.assertLess(macs,500000000)


if __name__=="__main__":
    unittest.main()
