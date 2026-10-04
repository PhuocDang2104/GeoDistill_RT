import inspect
import json
import tempfile
import unittest
from pathlib import Path
import torch
from model import AnchorFlowEdge, load_parent_state
from model_v3 import AnchorFlowEdge as V3
from model_v4 import AnchorFlowEdge as V4
from losses import objective
from run import sample_inputs, recipe_hash
from data import write_json


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_trained_checkpoint_migration_all_stages_noop(self):
        parent = torch.load(Path(__file__).with_name("init_v3_best.pth"), map_location="cpu", weights_only=False)
        source, target = V3().eval(), V4().eval()
        source.load_state_dict(parent["model"], strict=True)
        migration = load_parent_state(target, parent["model"])
        self.assertEqual(migration["parent_parameters_loaded"], 572017)
        self.assertEqual(migration["new_parameters"], 1285704)
        self.assertEqual(migration["old_keys_dropped"], 0)
        sample = sample_inputs(torch.device("cpu"), 64, 128, False)
        with torch.no_grad():
            old, new = source(*sample), target(*sample)
        for name in old:
            torch.testing.assert_close(old[name], new[name], atol=1e-5, rtol=1e-6)
        broken = dict(parent["model"])
        broken.pop(next(iter(broken)))
        with self.assertRaises(RuntimeError):
            load_parent_state(target, broken)

    def test_capacity_heads_gradients_amp_and_trunk_after_opening(self):
        torch.manual_seed(7)
        target = V4().train()
        parent = torch.load(Path(__file__).with_name("init_v3_best.pth"), map_location="cpu", weights_only=False)
        load_parent_state(target, parent["model"])
        target.freeze_encoder_bn()
        sample = sample_inputs(torch.device("cpu"), 64, 128, False)
        sample = tuple(x.repeat(2, *([1]*(x.ndim-1))) for x in sample)
        gt = torch.ones_like(sample[1])*25
        batch = {"rgb":sample[0],"sparse":sample[1],"mask":sample[2],"gt":gt,
                 "gt_mask":(torch.rand_like(gt)<.1).float(),"teacher":gt+2,"confidence":gt*0+.8}
        cfg = json.loads(Path(__file__).with_name("config.json").read_text())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            pred = target(*sample)
        loss, _ = objective(pred,batch,gt*0,0,cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in target.parameters()))
        for name in ("out16","out8","out4","out2"):
            head=getattr(target.capacity,name)
            self.assertGreater(float(head.weight.grad.abs().sum()),0,name)
            torch.nn.init.normal_(head.weight,std=.001)
        target.zero_grad(set_to_none=True)
        target(*sample)["D_full"].mean().backward()
        for name in ("in32","in16","in8","in4"):
            self.assertGreater(float(getattr(target.capacity,name)[0].weight.grad.abs().sum()),0,name)

    def test_no_teacher_gt_in_forward_and_empty_sparse(self):
        for constructor in (V3,V4):
            self.assertEqual(list(inspect.signature(constructor.forward).parameters),["self","rgb","sparse","mask","K"])
            m=constructor().eval()
            rgb,s,mask,K=sample_inputs(torch.device("cpu"),64,128,False)
            with torch.no_grad():
                p=m(rgb,s*0,mask*0,K)
            self.assertTrue(torch.isfinite(p["D_full"]).all())
            torch.testing.assert_close(p["D_full"],p["D1"],atol=0,rtol=0)

    def test_comparison_recipe_excludes_model_only(self):
        cfg=json.loads(Path(__file__).with_name("config.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            cfg["work"]=tmp
            write_json(Path(tmp)/"data_contract.json",{"fixture":True})
            control={**cfg,"model_name":"v3","architecture":"AnchorFlow-Edge-v3-RefineKD","run_name":"control"}
            self.assertEqual(recipe_hash(cfg),recipe_hash(control))
            control["kd_weight"]=.3
            self.assertNotEqual(recipe_hash(cfg),recipe_hash(control))

    def test_model_switch_counts(self):
        self.assertEqual(sum(p.numel() for p in AnchorFlowEdge(model_name="v3").parameters()),572017)
        self.assertEqual(sum(p.numel() for p in AnchorFlowEdge(model_name="v4").parameters()),1857721)


if __name__ == "__main__":
    unittest.main()
