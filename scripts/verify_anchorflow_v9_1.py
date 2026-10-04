"""CPU contract, real KITTI parent migration, BF16 backward and strict ORT checks."""
from pathlib import Path
import gc, hashlib, json, re, shutil, subprocess, sys, tempfile
import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"drive_upload/AnchorFlow_v9_1_MetricRefine"
AUDIT=ROOT/"results/anchorflow_v9_1_audit"
sys.path.insert(0,str(FOLDER))
import run
from data import KITTIDataset,write_json
from losses import objective
from metrics import Metrics

def dataset():
    ds=object.__new__(KITTIDataset)
    ds.root=ROOT/"data/teacher_subset_2000/kitti_bundle"
    ds.split,ds.teacher="val",False
    ds.rows=[r.split() for r in (ds.root/"splits/val_400.txt").read_text().splitlines()]
    return ds

def batch(ds,i):
    return {k:(v[None] if torch.is_tensor(v) else v) for k,v in ds[i].items()}

def parent_model(cfg):
    model=run.make_model(cfg,torch.device("cpu"),False)
    run.initialize_from_parent(model,cfg,check_data=False)
    return model.eval()

def strict_stage_export(model,ds):
    keys=("D0","D4_step1","D4_step2","D4","D2_base","D2_query","D2","D1_base","D1","D_full")
    class Stages(nn.Module):
        def __init__(self): super().__init__(); self.model=model
        def forward(self,rgb,sparse,mask,K):
            output=self.model(rgb,sparse,mask,K)
            return tuple(output[k] for k in keys)
    model=model.to(memory_format=torch.contiguous_format)
    wrapped=Stages().eval()
    samples=[]
    for seed in (0,1,7,42):
        torch.manual_seed(seed)
        sample=run.sample_inputs(torch.device("cpu"),channels_last=False)
        samples.append((f"seed{seed}_sparse",sample))
        rgb,s,m,K=sample; samples.append((f"seed{seed}_empty",(rgb,s*0,m*0,K)))
    for i in (0,199,399):
        b=batch(ds,i)
        samples.append((f"real_kitti_{i}",tuple(b[k].contiguous() for k in ("rgb","sparse","mask","K"))))
    # Export an opened new head against actual trained parent, not a constant zero branch.
    with torch.no_grad(): torch.nn.init.normal_(model.phase2.metric[-1].weight,std=.001)
    path=AUDIT/"parent_opened_metric_head.onnx"
    torch.onnx.export(wrapped,samples[0][1],str(path),input_names=["rgb","sparse","mask","K"],
                      output_names=list(keys),opset_version=17,dynamo=False)
    graph=onnx.load(path); onnx.checker.check_model(graph)
    operations=sorted({n.op_type for n in graph.graph.node})
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.log_severity_level=3
    session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
    report={}
    for label,sample in samples:
        with torch.no_grad(): expected=wrapped(*sample)
        actual=session.run(None,{k:v.numpy() for k,v in zip(("rgb","sparse","mask","K"),sample)})
        rows={}
        for key,a,e in zip(keys,actual,expected):
            err=float(np.max(np.abs(a-e.numpy())))
            assert np.isfinite(a).all() and np.allclose(a,e.numpy(),atol=.01,rtol=1e-4),(label,key,err)
            rows[key]=err
        report[label]=rows
        print("ORT all stages PASS",label,"max",max(rows.values()),flush=True)
    return {"opset":17,"all_stage_cases":report,"max_abs_m":max(max(r.values()) for r in report.values()),
            "atol_m":.01,"rtol":1e-4,"operations":operations,"onnxruntime_cpu_parity":True,
            "actual_v9_parent_loaded":True,"new_head_randomly_opened":True,"new_student_trained":False}

def main():
    torch.set_num_threads(4); AUDIT.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s",str(FOLDER),"-v"],
                          cwd=FOLDER,capture_output=True,text=True,encoding="utf-8")
    log=result.stdout+result.stderr; print(log,flush=True)
    if result.returncode: raise RuntimeError("Contract tests failed")
    with tempfile.TemporaryDirectory(prefix="anchorflow_v9_1_colab_") as directory:
        dest=Path(directory)
        for p in FOLDER.iterdir():
            if p.is_file() and p.suffix in (".py",".json",".pth"): shutil.copy2(p,dest/p.name)
        assert not list(dest.glob("*.ipynb"))
        staged=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s",str(dest),"-v"],
                              cwd=dest,capture_output=True,text=True,encoding="utf-8")
        if staged.returncode:
            print(staged.stdout+staged.stderr); raise RuntimeError("Colab layout tests failed")
    cfg=json.loads((FOLDER/"config.json").read_text()); ds=dataset(); model=parent_model(cfg)
    shapes=[]
    with torch.no_grad():
        for i in (0,199,399):
            b=batch(ds,i); out=model(*(b[k] for k in ("rgb","sparse","mask","K")))
            assert torch.isfinite(out["D_full"]).all(); shapes.append(list(out["D_full"].shape))
    b.update(teacher=torch.full_like(b["gt"],20),confidence=torch.full_like(b["gt"],.8),
             relative=torch.linspace(-1,1,1216)[None,None,None].expand(1,1,352,1216).clone(),
             relative_confidence=torch.ones_like(b["gt"]))
    model.train(); model.freeze_encoder_bn()
    with torch.autocast("cpu",dtype=torch.bfloat16): out=model(*(b[k] for k in ("rgb","sparse","mask","K")))
    loss,stats=objective(out,b,torch.zeros_like(b["mask"]),4.,cfg)
    loss.backward(); assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    gradients={k:float(p.grad.abs().sum()) for k,p in model.named_parameters()
               if k in ("dynamics.reaction.weight","phase2.candidates.weight","phase2.metric.4.weight")}
    assert all(x>0 for x in gradients.values())
    assert float(stats["relative_pair_coverage"])>0
    model.zero_grad(set_to_none=True)
    # T4 recipe is full FP32, not FP16 or BF16 emulation.
    out_fp32=model(*(b[k] for k in ("rgb","sparse","mask","K")))
    loss_fp32,_=objective(out_fp32,b,torch.zeros_like(b["mask"]),4.,{**cfg,"amp":"fp32"})
    loss_fp32.backward(); assert torch.isfinite(loss_fp32)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.phase2.metric[-1].weight.grad.abs().sum()>0
    del out_fp32
    complexity=run.count_operations(model.eval(),run.sample_inputs(torch.device("cpu")))
    del model,out,b; gc.collect()
    parity=strict_stage_export(parent_model(cfg),ds)
    # Initialization-only accuracy, no new learning. This measures minmod's migration impact.
    model=parent_model(cfg).to(memory_format=torch.channels_last); score=Metrics(torch.device("cpu"))
    with torch.no_grad():
        for i in range(400):
            b=batch(ds,i); out=model(*(b[k] for k in ("rgb","sparse","mask","K")))
            score.update(out["D_full"],b["gt"],b["gt_mask"],b["rgb"])
            if (i+1)%50==0: print("Initialization-only FP32 validation",i+1,"/400",flush=True)
    initial=score.report(); assert initial["all"]["pixels"]==25424992
    write_json(AUDIT/"parent_minmod_initial_metrics_cpu_fp32.json",initial)
    proof={"tests":int(re.search(r"Ran (\d+) tests",log).group(1)),"tests_returncode":0,"tests_log":log,
           "colab_copy_without_ipynb_tests_returncode":staged.returncode,"source_sha256":run.source_hash(),
           "torch":torch.__version__,"python":sys.version,"complexity_v9_1":complexity,
           "real_kitti_samples":[0,199,399],"output_shapes":shapes,"bf16_fixture_loss":float(loss.detach()),
           "real_kitti_bf16_gradients_finite":True,"head_gradient_l1":gradients,
           "real_kitti_fp32_gradients_finite":True,"fp32_fixture_loss":float(loss_fp32.detach()),
           "relative_pair_coverage_fixture":float(stats["relative_pair_coverage"]),
           "fixture_note":"Real KITTI RGB/sparse/GT; synthetic metric and relative teachers. Actual V9 parent loaded; no new optimizer steps. CPU BF16 backward is not GPU training.",
           "parent_loaded":True,"parent_checkpoint_sha256":cfg["init_checkpoint_sha256"],
           "trained_parent_opened_new_head_onnx":parity,
           "initialization_only_cpu_fp32_validation":initial,
           "new_training_accuracy_measured":False,"gpu_training_tested":False,"gpu_latency_measured":False,
           "teacher_generation_recipe_unchanged":(FOLDER/"generate_relative.py").read_bytes()==(ROOT/"drive_upload/AnchorFlow_v9_Consensus/generate_relative.py").read_bytes()}
    verified=[p for p in FOLDER.iterdir() if p.is_file() and
              (p.suffix in (".py",".pth") or p.name in ("config.json","requirements.txt","teacher_requirements.txt") or p.name.endswith("_contract.json"))]
    proof["verified_files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in verified}
    write_json(FOLDER/"local_verification.json",proof)
    print("VERIFIED",json.dumps({"parameters":complexity["total_parameters"],"ConvLinearMAC":complexity["total_conv_linear_macs"],
          "initial_RMSE_CPU_FP32":initial["all"]["rmse_m"],"ONNX_max_abs_m":parity["max_abs_m"]}),flush=True)

if __name__=="__main__": main()
