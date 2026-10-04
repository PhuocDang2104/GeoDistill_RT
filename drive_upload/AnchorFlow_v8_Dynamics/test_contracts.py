"""Synthetic contracts, including feedback sensitivity and real trainer resume."""
import ast
import inspect
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from model import AnchorFlowEdge, JetDynamics, translate, phase_values, edge_weights
from data import KITTIDataset, canonical_id, safe_extract, teacher_arrays, write_json
from losses import objective, tail_risk
from loss_helpers import teacher_weights
from metrics import Metrics
import run


def config():
    return json.loads(Path(__file__).with_name("config.json").read_text())


def fixture():
    inputs=run.sample_inputs(torch.device("cpu"),64,128)
    rgb,sparse,mask,K=inputs
    batch={"rgb":rgb,"sparse":sparse,"mask":mask,"K":K,
           "gt":torch.ones_like(mask)*25,"gt_mask":(torch.rand_like(mask)<.2).float(),
           "teacher":torch.ones_like(mask)*22,"confidence":torch.ones_like(mask)*.8}
    return inputs,batch


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_interface_and_no_old_correction_stack(self):
        self.assertEqual(list(inspect.signature(AnchorFlowEdge.forward).parameters),["self","rgb","sparse","mask","K"])
        m=AnchorFlowEdge()
        for old in ("flow","metric4","surface","connection4"):
            self.assertFalse(hasattr(m,old))
        self.assertEqual(m.dynamics.steps,3)
        for branch in (m.dynamics.state,m.dynamics.update,m.dynamics.reaction,m.dynamics.conductance):
            self.assertFalse(any(isinstance(x,nn.BatchNorm2d) for x in branch.modules()))

    def test_channels_last_bf16_gradients_and_six_components(self):
        inputs,batch=fixture()
        m=AnchorFlowEdge().train().to(memory_format=torch.channels_last)
        m.freeze_encoder_bn()
        with torch.autocast("cpu",dtype=torch.bfloat16):
            out=m(*inputs)
        loss,stats=objective(out,batch,torch.zeros_like(batch["mask"]),1.,config())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertTrue((m.dynamics.reaction.weight.grad.abs().flatten(1).sum(1)>0).all())
        self.assertGreater(float(m.dynamics.state.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.dynamics.conductance.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.decoder.head16.weight.grad.abs().sum()),0)
        for k in (1,2,3):
            self.assertGreater(float(stats[f"dynamics_state_change_{k}"]),0)
            self.assertLessEqual(float(stats[f"dynamics_mass_{k}"]),.8)

    def test_field_depends_on_current_state(self):
        block=JetDynamics().eval()
        context=torch.randn(1,32,8,16)
        barrier=torch.zeros(1,2,8,16)
        depth=torch.full((1,1,8,16),20.)
        j0=block.base_jet(depth)
        state=(depth*1.5,torch.ones_like(depth),torch.ones_like(depth)*.1,depth,torch.ones_like(depth),torch.zeros_like(depth))
        f0,w0=block.vector_field(context,barrier,j0,j0,state,0.)
        j1=j0.clone(); j1[:, :1]*=1.2; j1[:,1:3]=.002
        f1,w1=block.vector_field(context,barrier,j1,j0,state,0.)
        self.assertGreater(float((f0-f1).abs().max()),1e-6)
        self.assertGreater(float((w0-w1).abs().max()),1e-8)
        differentiable=j1.clone().requires_grad_()
        forcing,weights=block.vector_field(context,barrier,differentiable,j0,state,.3)
        grad=torch.autograd.grad(forcing.square().sum()+weights.square().sum(),differentiable)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()),0)

    def test_vector_field_is_called_three_times(self):
        inputs,_=fixture()
        m=AnchorFlowEdge().eval()
        with patch.object(m.dynamics,"vector_field",wraps=m.dynamics.vector_field) as field,torch.no_grad():
            out=m(*inputs)
        self.assertEqual(field.call_count,3)
        j0,j1,j2=[call.args[2] for call in field.call_args_list]
        self.assertFalse(torch.equal(j0,j1))
        self.assertFalse(torch.equal(j1,j2))
        self.assertEqual(out["D4_step3"].shape,(1,1,16,32))

    def test_frozen_feedback_control_same_parameterization(self):
        m=AnchorFlowEdge().eval()
        control=AnchorFlowEdge(model_name="v8_frozen_feedback").eval()
        control.load_state_dict(m.state_dict(),strict=True)
        inputs,_=fixture()
        with patch.object(control.dynamics,"vector_field",wraps=control.dynamics.vector_field) as field,torch.no_grad():
            a,b=m(*inputs),control(*inputs)
        self.assertEqual(sum(p.numel() for p in m.parameters()),sum(p.numel() for p in control.parameters()))
        self.assertTrue(all(torch.equal(c.args[2],field.call_args_list[0].args[2]) for c in field.call_args_list))
        self.assertGreater(float((a["D4"]-b["D4"]).abs().max()),1e-5)

    def test_translation_composition_and_phase_order(self):
        j=torch.randn(1,6,5,8)
        torch.testing.assert_close(translate(translate(j,.3,-.2),-.5,.7),translate(j,-.2,.5))
        polynomial=torch.tensor([1.,.2,.3,.1,.07,.05]).view(1,6,1,1)
        phase=phase_values(polynomial)
        for i,(y,x) in enumerate(((-.25,-.25),(-.25,.25),(.25,-.25),(.25,.25))):
            self.assertAlmostEqual(float(phase[:,i]),1+.2*x+.3*y+.05*x*x+.07*x*y+.025*y*y,places=6)

    def test_edge_reciprocity_and_boundary_flux(self):
        w=edge_weights(torch.rand(1,2,8,12)*.6)
        self.assertEqual(float(w[:,0,...,-1].abs().max()),0)
        self.assertEqual(float(w[:,1,...,0].abs().max()),0)
        self.assertEqual(float(w[:,2,...,-1,:].abs().max()),0)
        self.assertEqual(float(w[:,3,...,0,:].abs().max()),0)
        torch.testing.assert_close(w[:,0,...,:-1],w[:,1,...,1:])
        torch.testing.assert_close(w[:,2,...,:-1,:],w[:,3,...,1:,:])
        self.assertLessEqual(float(w.sum(1).max()/3),.8)

    def test_empty_sparse_and_invalid_sensor_bounds(self):
        inputs,_=fixture()
        rgb,s,m,K=inputs
        model=AnchorFlowEdge().eval()
        with torch.no_grad():
            empty=model(rgb,s*0,m*0,K)
            contaminated=model(rgb,torch.full_like(s,float("nan")),torch.ones_like(m),K)
        torch.testing.assert_close(empty["D_full"],contaminated["D_full"])
        self.assertTrue(all(torch.isfinite(x).all() for x in empty.values()))
        self.assertGreaterEqual(float(empty["D_full"].min()),.09999)
        self.assertLessEqual(float(empty["D_full"].max()),120.0001)
        self.assertEqual(float(empty["sensor_gate"].max()),0)

    def test_soft_sensor_fusion_and_hard_diagnostic(self):
        inputs,_=fixture()
        model=AnchorFlowEdge().eval()
        with torch.no_grad(): out=model(*inputs)
        _,s,mask,_=inputs
        self.assertTrue(torch.equal(out["D_hard"][mask.bool()],s[mask.bool()]))
        torch.testing.assert_close(out["D_full"],(1-out["sensor_gate"])*out["D1"]+out["sensor_gate"]*s)

    def test_teacher_forbidden_on_val_test(self):
        for split in ("val","test"):
            with self.assertRaises(ValueError): KITTIDataset({"work":"unused"},split,teacher=True)

    def test_kd_forbids_gt_and_original_holdout_sensor(self):
        inputs,batch=fixture()
        batch["gt_mask"].zero_(); batch["gt_mask"][...,0,0]=1
        batch["mask"].zero_(); batch["mask"][...,7,7]=1
        teacher,weights,forbidden,eligible=teacher_weights(batch,.5)
        self.assertEqual(float(weights[...,0,0]),0)
        self.assertEqual(float(weights[...,7,7]),0)
        self.assertEqual(float(forbidden[...,7,7]),1)
        self.assertGreater(float(weights[...,9,9]),0)

    def test_teacher_schema_and_canonical_id(self):
        old="2011_09_26_drive_0001_sync_image_03_0000000038.npz"
        self.assertEqual(canonical_id(old),"2011_09_26_drive_0001_sync_image_0000000038_image_03")
        with io.BytesIO() as b:
            np.savez(b,D_cm=np.ones((4,8),np.float32)*20,C_cm=np.ones((4,8),np.float32)*.7)
            b.seek(0)
            with np.load(b) as p: x=teacher_arrays(p,(8,16))
        self.assertTrue(np.allclose(x[0],20))
        self.assertTrue(np.allclose(x[1],.7))

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/"bad.tar"
            with tarfile.open(path,"w") as z:
                item=tarfile.TarInfo("../escape.txt"); item.size=1; z.addfile(item,io.BytesIO(b"x"))
            with self.assertRaises(RuntimeError): safe_extract(path,root/"out")

    def test_global_metrics_and_inverse_units(self):
        m=Metrics("cpu")
        gt=torch.tensor([[[[1.,2.]]]])
        m.update(gt*2,gt,torch.ones_like(gt),torch.zeros(1,3,1,2))
        x=m.report()["all"]
        self.assertAlmostEqual(x["rmse_m"],2.5**.5)
        self.assertAlmostEqual(x["irmse_km_inv"],(312500/2)**.5)
        self.assertEqual(x["pixels"],2)

    def test_loss_tail_ramp_and_aux_supervision(self):
        inputs,batch=fixture()
        out=AnchorFlowEdge().eval()(*inputs)
        a,stats0=objective(out,batch,batch["mask"]*0,0,config())
        b,stats2=objective(out,batch,batch["mask"]*0,2,config())
        self.assertEqual(float(stats0["weighted_tail"]),0)
        self.assertGreater(float(stats2["weighted_tail"]),0)
        self.assertGreater(float(stats0["weighted_dynamics_aux"]),0)
        self.assertAlmostEqual(float(stats2["tail_ramp"]),1.)
        self.assertTrue(torch.isfinite(a+b))
        residual=torch.tensor([0.,1.,3.,10.])
        self.assertAlmostEqual(float(tail_risk(residual,torch.ones_like(residual),2)),16.25)

    def test_lr_and_flip_intrinsics(self):
        self.assertEqual(run.learning_rate(9,100,10,.05),1)
        self.assertAlmostEqual(run.learning_rate(99,100,10,.05),.05)
        inputs,batch=fixture()
        with patch.object(torch,"rand",return_value=torch.zeros(1)):
            changed=run.augment(batch,config())
        self.assertAlmostEqual(float(changed["K"][0,0,2]),127-float(batch["K"][0,0,2]))
        self.assertTrue(torch.equal(changed["rgb"],batch["rgb"].flip(-1)))

    def test_trainer_fresh_initialization_and_resume(self):
        inputs,batch=fixture()
        class Loader(list): dataset=[0]
        loader=Loader([batch])
        cfg=config()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            cfg.update(work=str(root/"work"),drive_runs=str(root/"drive"),epochs=1,workers=0,
                       encoder_pretrained=False,fused_adamw=False,amp="fp32",log_every=1)
            write_json(root/"work/data_contract.json",{"fixture":True})
            with patch.object(run,"device_setup",return_value=torch.device("cpu")), \
                 patch.object(run,"data_loader",return_value=loader), \
                 patch.object(torch.cuda,"get_device_name",return_value="CPU fixture"), \
                 patch.object(run,"make_model",wraps=run.make_model) as builder:
                run.train(cfg,"metric_kd")
                run.train(cfg,"metric_kd")
            self.assertTrue(all(not c.kwargs["pretrained"] for c in builder.call_args_list))
            folder=root/"drive"/cfg["run_name"]/"metric_kd"
            saved=torch.load(folder/"last.pth",weights_only=False)
            self.assertEqual(saved["epoch"],0)
            self.assertEqual(saved["global_step"],1)
            self.assertEqual(saved["early_stopping"]["bad_epochs"],0)
            self.assertEqual(len((folder/"train_log.csv").read_text().splitlines()),2)
            model,_=run.load_trained(cfg,"metric_kd",torch.device("cpu"))
            self.assertFalse(model.training)
            cfg["learning_rate"]*=2
            with self.assertRaises(RuntimeError): run.load_trained(cfg,"metric_kd",torch.device("cpu"))

    def test_early_stop_patience_floor_and_small_cumulative_gains(self):
        cfg=config()
        state={"monitor_best_rmse":None,"bad_epochs":0,"stopped":False}
        for epoch in range(14):
            state=run.early_stop_update(state,1.,epoch,cfg)
            self.assertFalse(state["stopped"])
        state=run.early_stop_update(state,1.,14,cfg)
        self.assertTrue(state["stopped"])
        state={"monitor_best_rmse":1.,"bad_epochs":5,"stopped":False}
        state=run.early_stop_update(state,.9995,15,cfg)
        self.assertEqual(state["bad_epochs"],6)
        # Small gains accumulate relative to meaningful best, not preceding epoch.
        state=run.early_stop_update(state,.9988,16,cfg)
        self.assertEqual(state["bad_epochs"],0)
        self.assertFalse(state["stopped"])
        cfg["early_stopping"]=False
        for epoch in range(40): state=run.early_stop_update(state,1.,epoch,cfg)
        self.assertFalse(state["stopped"])

    def test_early_stop_full_trainer_checkpoint_and_resume(self):
        inputs,batch=fixture()
        class Loader(list): dataset=[0]
        cfg=config()
        cfg.update(epochs=6,early_stop_patience=2,early_stop_min_epochs=3,encoder_pretrained=False,
                   fused_adamw=False,amp="fp32",workers=0,log_every=1)
        original_validate=run.validate
        def constant_score(*args,**kwargs):
            report=original_validate(*args,**kwargs)
            report["final"]["all"]["rmse_m"]=1.
            return report
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            cfg.update(work=str(root/"work"),drive_runs=str(root/"drive"))
            write_json(root/"work/data_contract.json",{"fixture":True})
            with patch.object(run,"device_setup",return_value=torch.device("cpu")), \
                 patch.object(run,"data_loader",return_value=Loader([batch])), \
                 patch.object(torch.cuda,"get_device_name",return_value="CPU fixture"), \
                 patch.object(run,"validate",side_effect=constant_score):
                run.train(cfg,"metric_kd")
                folder=root/"drive"/cfg["run_name"]/"metric_kd"
                saved=torch.load(folder/"last.pth",weights_only=False)
                self.assertEqual(saved["epoch"],2)
                self.assertTrue(saved["early_stopping"]["stopped"])
                self.assertEqual(saved["early_stopping"]["bad_epochs"],2)
                run.train(cfg,"metric_kd")
            self.assertEqual(len((folder/"train_log.csv").read_text().splitlines()),4)
            status=json.loads((folder/"training_status.json").read_text())
            self.assertEqual(status["status"],"early_stopped")
            self.assertEqual(status["epochs_completed"],3)


if __name__=="__main__": unittest.main()
