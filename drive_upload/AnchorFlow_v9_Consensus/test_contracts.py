import ast
import csv
import hashlib
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
from model import AnchorFlowEdge, candidate_queries, MultiJetReadout
from model_v8 import AnchorFlowEdge as V8
from losses import objective, relative_structure
from losses_v8 import objective as v8_objective
from generate_relative import standardize, fuse_tta, MODEL_REVISION, CODE_REVISION
from relative_data import relative_arrays, extract_relative
from data import KITTIDataset, write_json, safe_extract
import run


def config(): return json.loads(Path(__file__).with_name("config.json").read_text())


def fixture():
    inputs=run.sample_inputs(torch.device("cpu"),64,128)
    rgb,sparse,mask,K=inputs
    h,w=mask.shape[-2:]
    relative=torch.linspace(-1,1,w)[None,None,None].expand(1,1,h,w).clone()
    batch={"rgb":rgb,"sparse":sparse,"mask":mask,"K":K,
           "gt":torch.full_like(mask,25),"gt_mask":(torch.rand_like(mask)<.15).float(),
           "teacher":torch.full_like(mask,22),"confidence":torch.full_like(mask,.8),
           "relative":relative,"relative_confidence":torch.ones_like(mask)}
    return inputs,batch


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_config_and_fresh_contract(self):
        c=config()
        self.assertEqual(c["amp"],"bf16")
        self.assertTrue(c["relative_enabled"] and c["teacher_enabled"])
        self.assertEqual(c["epochs"],30)
        self.assertNotIn("init_checkpoint",c)
        self.assertEqual(list(inspect.signature(AnchorFlowEdge.forward).parameters),["self","rgb","sparse","mask","K"])

    def test_candidate_origin_translation_and_phase_order(self):
        x=torch.arange(8)[None,None,None,:].float()
        y=torch.arange(4)[None,None,:,None].float()
        hxx,hxy,hyy=.0001,.00004,.00008
        v=.1+.002*x+.003*y+.5*hxx*x*x+hxy*x*y+.5*hyy*y*y
        z=torch.zeros_like(v)
        j=torch.cat((v,z+.002+hxx*x+hxy*y,z+.003+hxy*x+hyy*y,z+hxx,z+hxy,z+hyy),1)
        q,valid=candidate_queries(j)
        reference=torch.cat([.1+.002*(x+sx)+.003*(y+sy)+.5*hxx*(x+sx)**2+hxy*(x+sx)*(y+sy)+.5*hyy*(y+sy)**2
                             for sy,sx in ((-.25,-.25),(-.25,.25),(.25,-.25),(.25,.25))],1)
        error=(q-reference[:,None]).abs()*valid
        self.assertLess(float(error.max()),1e-7)
        self.assertEqual(float(valid[0,1,0,0,-1]),0)
        self.assertEqual(float(valid[0,2,0,0,0]),0)
        self.assertEqual(float(valid[0,3,0,-1,0]),0)

    def test_consensus_gradients_bf16_and_empty_sparse(self):
        inputs,batch=fixture()
        model=AnchorFlowEdge().train().to(memory_format=torch.channels_last)
        model.freeze_encoder_bn()
        with torch.autocast("cpu",dtype=torch.bfloat16): out=model(*inputs)
        loss,stats=objective(out,batch,torch.zeros_like(batch["mask"]),4,config())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.phase2.candidates.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.dynamics.reaction.weight.grad.abs().sum()),0)
        self.assertGreater(float(stats["weighted_relative"]),0)
        model.eval()
        rgb,s,m,K=inputs
        with torch.no_grad(): out=model(rgb,s*0,m*0,K)
        self.assertTrue(torch.isfinite(out["D_full"]).all())
        self.assertEqual(float(out["sensor_gate"].max()),0)

    def test_constant_surface_zero_disagreement(self):
        j=torch.zeros(1,6,4,8); j[:,0]=.05
        b=MultiJetReadout().eval()
        with torch.no_grad():
            out=b(j,torch.zeros(1,32,4,8),torch.zeros(1,32,4,8),
                  torch.full((1,1,8,16),20),torch.zeros(1,1,8,16),torch.zeros(1,1,8,16),torch.zeros(1,2,4,8))
        self.assertLess(float(out[4]["query_uncertainty_mean"]),1e-10)
        self.assertTrue(torch.allclose(out[1],torch.full_like(out[1],20),atol=1e-5))

    def test_relative_near_far_tta_and_positive_affine_invariance(self):
        depth=np.linspace(1,20,2048,dtype=np.float32).reshape(32,64)
        r,c=fuse_tta(depth,depth*3)
        self.assertGreater(float(r[0,0]),float(r[-1,-1]))
        self.assertGreater(float(c.min()),.999)
        a,_=standardize(depth); b,_=standardize(depth*7)
        self.assertTrue(np.allclose(a,b,atol=1e-5))
        relative=torch.from_numpy(a)[None,None]
        pred=torch.from_numpy(depth)[None,None]
        rgb=torch.rand(1,3,32,64)
        args=(torch.ones_like(pred),rgb,torch.zeros_like(pred))
        loss1=relative_structure(pred,relative,*args)
        loss2=relative_structure(pred,relative*3+2,*args)
        self.assertLess(float((loss1[0]-loss2[0]).abs()),1e-4)

    def test_relative_forbidden_and_missing_cache_fail(self):
        inputs,batch=fixture()
        model=AnchorFlowEdge().eval()
        with torch.no_grad(): out=model(*inputs)
        forbidden=torch.ones_like(batch["mask"])
        grad,ordinal,coverage=relative_structure(out["D2"],batch["relative"],batch["relative_confidence"],batch["rgb"],forbidden)
        self.assertEqual(float(grad+ordinal+coverage),0)
        del batch["relative"]
        with self.assertRaisesRegex(RuntimeError,"no fallback"):
            objective(out,batch,batch["mask"]*0,4,config())

    def test_relative_schema_and_train_only_access(self):
        class Fake:
            files=["R_T","C_T"]
            def __getitem__(self,key):
                if key=="C_T": return np.ones((352,1216),np.float32)
                return np.linspace(-1,1,352*1216,dtype=np.float32).reshape(352,1216)
        self.assertEqual(relative_arrays(Fake()).shape,(2,352,1216))
        for split in ("val","test"):
            with self.assertRaises(ValueError): KITTIDataset(config(),split,teacher=True)

    def test_rgb_flip_teacher_alignment(self):
        from generate_relative import fuse_tta
        depth=np.linspace(1,20,2048,dtype=np.float32).reshape(32,64)
        # Predict flipped RGB then flip prediction back; asymmetric ramp detects a missing unflip.
        r,correct=fuse_tta(depth,depth[:,::-1][:,::-1])
        _,wrong=fuse_tta(depth,depth[:,::-1])
        self.assertGreater(float(correct.mean()),float(wrong.mean()))

    def test_v8_control_exact_objective(self):
        inputs,batch=fixture()
        c=config(); c.update(relative_enabled=False,inverse_weight=0,model_name="v8_control")
        with torch.no_grad(): out=V8().eval()(*inputs)
        a,_=objective(out,batch,batch["mask"]*0,4,c)
        b,_=v8_objective(out,batch,batch["mask"]*0,4,c)
        self.assertEqual(float(a),float(b))
        self.assertIsInstance(run.make_model(c,torch.device("cpu"),False),V8)

    def test_fresh_train_resume_protocol_and_relative_logging(self):
        inputs,batch=fixture()
        class Loader(list): dataset=[0]
        cfg=config()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg.update(work=str(root/"work"),drive_runs=str(root/"drive"),epochs=1,workers=0,
                       encoder_pretrained=False,fused_adamw=False,amp="fp32",log_every=1)
            write_json(root/"work/data_contract.json",{"synthetic_fixture":True})
            with patch.object(run,"device_setup",return_value=torch.device("cpu")), \
                 patch.object(run,"data_loader",return_value=Loader([batch])), \
                 patch.object(torch.cuda,"get_device_name",return_value="CPU fixture"):
                run.train(cfg,"dual_teacher")
                run.train(cfg,"dual_teacher")
            folder=root/"drive"/cfg["run_name"]/"dual_teacher"
            saved=torch.load(folder/"last.pth",weights_only=False)
            self.assertEqual(saved["epoch"],0)
            self.assertEqual(saved["global_step"],1)
            with (folder/"train_log.csv").open() as file: rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),1)
            self.assertIn("loss_relative_gradient",rows[0])
            self.assertIn("val_tail_5m_pixel_fraction",rows[0])
            cfg["relative_weight"]*=2
            with self.assertRaises(RuntimeError): run.load_trained(cfg,"dual_teacher",torch.device("cpu"))

    def test_nonfinite_loss_aborts_before_backward_checkpoint(self):
        _,batch=fixture()
        class Loader(list): dataset=[0]
        cfg=config()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg.update(work=str(root/"work"),drive_runs=str(root/"drive"),epochs=1,
                       encoder_pretrained=False,fused_adamw=False,amp="fp32")
            write_json(root/"work/data_contract.json",{"fixture":True})
            with patch.object(run,"device_setup",return_value=torch.device("cpu")), \
                 patch.object(run,"data_loader",return_value=Loader([batch])), \
                 patch.object(torch.cuda,"get_device_name",return_value="CPU fixture"), \
                 patch.object(run,"objective",return_value=(torch.tensor(float("nan")),{})):
                with self.assertRaisesRegex(RuntimeError,"BEFORE backward"):
                    run.train(cfg,"dual_teacher")
            self.assertFalse((root/"drive"/cfg["run_name"]/"dual_teacher/last.pth").exists())

    def test_no_student_dependency_in_teacher_and_pinned_revisions(self):
        src=Path(__file__).with_name("generate_relative.py").read_text()
        self.assertNotIn("from model",src)
        self.assertEqual(len(MODEL_REVISION),40)
        self.assertEqual(len(CODE_REVISION),40)
        for name in ("model.py","losses.py","generate_relative.py","relative_data.py","run.py"):
            ast.parse(Path(__file__).with_name(name).read_text())

    def test_relative_archive_hashes_train_cache_and_val_audit_only(self):
        train=["2011_09_26_drive_0001_sync_image_0000000001_image_02",
               "2011_09_26_drive_0001_sync_image_0000000002_image_02"]
        val=["2011_09_28_drive_0002_sync_image_0000000003_image_03"]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); subset=root/"data/teacher_subset_2000"
            subset.mkdir(parents=True)
            selected=json.dumps({"train_ids":train,"val_ids":val}).encode()
            (subset/"selected_2000_ids.json").write_bytes(selected)
            content=io.BytesIO()
            np.savez_compressed(content,R_T=np.linspace(-1,1,2048,dtype=np.float32).reshape(32,64),
                                C_T=np.ones((32,64),np.float32))
            raw=content.getvalue()
            cfg=config(); cfg.update(drive_data=str(root/"data"),work=str(root/"work"))
            metadata={"selected_manifest_sha256":hashlib.sha256(selected).hexdigest(),
                      "train_ids":train,"val_ids":val,"representation":"standardized_relative_inverse_depth_near_high",
                      "model_id":cfg["relative_model_id"],"model_revision":cfg["relative_model_revision"],
                      "records":{sid:hashlib.sha256(raw).hexdigest() for sid in train+val}}
            def archive(meta):
                with tarfile.open(subset/cfg["relative_tar"],"w") as handle:
                    value=json.dumps(meta).encode(); info=tarfile.TarInfo("relative_manifest.json"); info.size=len(value)
                    handle.addfile(info,io.BytesIO(value))
                    for sid in train+val:
                        info=tarfile.TarInfo("relative/"+sid+".npz"); info.size=len(raw)
                        handle.addfile(info,io.BytesIO(raw))
            archive(metadata)
            with patch("relative_data.SIZE",(32,64)):
                report=extract_relative(cfg,train,val)
                self.assertEqual((report["records"],report["cached_train"],report["cached_val"]),(3,2,0))
                self.assertEqual(len(list((root/"work/relative_train").glob("*.npy"))),2)
                self.assertFalse((root/"work/relative_train"/(val[0]+".npy")).exists())
                metadata["records"][train[0]]="0"*64; archive(metadata)
                with self.assertRaisesRegex(RuntimeError,"checksum"):
                    extract_relative(cfg,train,val)

    def test_relative_dtype_and_subset_mismatch_rejected(self):
        class Fake:
            files=["R_T","C_T"]
            def __getitem__(self,key): return np.ones((352,1216),np.float64)
        with self.assertRaisesRegex(RuntimeError,"float32"):
            relative_arrays(Fake())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); subset=root/"teacher_subset_2000"; subset.mkdir()
            (subset/"selected_2000_ids.json").write_text("{}")
            cfg=config(); cfg.update(drive_data=str(root),work=str(root/"work"))
            with tarfile.open(subset/cfg["relative_tar"],"w") as handle:
                raw=json.dumps({"selected_manifest_sha256":"wrong"}).encode()
                info=tarfile.TarInfo("relative_manifest.json"); info.size=len(raw)
                handle.addfile(info,io.BytesIO(raw))
            with self.assertRaisesRegex(RuntimeError,"different subset"):
                extract_relative(cfg,[],[])

    def test_two_notebook_snapshots_are_clean_and_parse_without_ui_files(self):
        folder=Path(__file__).parent
        snapshots=sorted(folder.glob("0*_contract.json"))
        self.assertEqual(len(snapshots),2)
        for snapshot in snapshots:
            book=json.loads(snapshot.read_text(encoding="utf-8"))
            for cell in book["cells"]:
                if cell["cell_type"]=="code":
                    self.assertFalse(cell["outputs"])
                    self.assertIsNone(cell["execution_count"])
                    ast.parse("".join(cell["source"]))
            source="\n".join("".join(c["source"]) for c in book["cells"])
            self.assertIn("if name.endswith(\".ipynb\"): continue",source)
            self.assertIn("drive.mount",source)
            self.assertIn("GeoLift_Data",source)

    def test_early_stop_cumulative_gain_floor_and_raw_best_separation(self):
        c=config(); c.update(early_stop_min_epochs=4,early_stop_patience=2,early_stop_min_delta_m=.01)
        state={"monitor_best_rmse":None,"bad_epochs":0,"stopped":False}
        for epoch,rmse in enumerate((1.,.996,.992)):
            state=run.early_stop_update(state,rmse,epoch,c)
        self.assertFalse(state["stopped"])
        self.assertEqual(state["bad_epochs"],2)
        state=run.early_stop_update(state,.989,3,c)
        self.assertFalse(state["stopped"])
        self.assertEqual(state["bad_epochs"],0)
        for epoch in (4,5): state=run.early_stop_update(state,.99,epoch,c)
        self.assertTrue(state["stopped"])

    def test_nonfinite_gradient_aborts_before_optimizer_step(self):
        _,batch=fixture()
        class Loader(list): dataset=[0]
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx,value): return value
            @staticmethod
            def backward(ctx,gradient): return gradient*float("nan")
        def bad_objective(*args):
            loss,stats=objective(*args)
            return BadGradient.apply(loss),stats
        cfg=config()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg.update(work=str(root/"work"),drive_runs=str(root/"drive"),epochs=1,
                       encoder_pretrained=False,fused_adamw=False,amp="fp32")
            write_json(root/"work/data_contract.json",{"fixture":True})
            with patch.object(run,"device_setup",return_value=torch.device("cpu")), \
                 patch.object(run,"data_loader",return_value=Loader([batch])), \
                 patch.object(torch.cuda,"get_device_name",return_value="CPU fixture"), \
                 patch.object(run,"objective",side_effect=bad_objective), \
                 patch.object(torch.optim.AdamW,"step") as optimizer_step:
                with self.assertRaisesRegex(RuntimeError,"non-finite"):
                    run.train(cfg,"dual_teacher")
                optimizer_step.assert_not_called()
            self.assertFalse((root/"drive"/cfg["run_name"]/"dual_teacher/last.pth").exists())


if __name__=="__main__": unittest.main()
