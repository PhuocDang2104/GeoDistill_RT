"""Reproducible CPU/Colab-layout/ONNX checks; no claim of trained V9 accuracy."""
from pathlib import Path
from unittest.mock import patch
import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import torch

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"drive_upload/AnchorFlow_v9_Consensus"


def teacher_adapter_check():
    """Check pinned official config and ALL tensor names/shapes, not tensor values."""
    from generate_relative import MODEL_ID, MODEL_REVISION, CODE_REVISION
    source=ROOT/"results/anchorflow_v9_audit/teacher_source"
    head=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    assert head==CODE_REVISION
    base=f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/"
    with urllib.request.urlopen(base+"config.json",timeout=45) as response:
        configuration=json.load(response)
    assert configuration["model_name"]=="da3mono-large"
    request=urllib.request.Request(base+"model.safetensors?header_audit=v9",headers={"Range":"bytes=0-524287"})
    with urllib.request.urlopen(request,timeout=45) as response:
        length=int.from_bytes(response.read(8),"little")
        assert 0<length<524280
        header=json.loads(response.read(length))
    tensors={k:v for k,v in header.items() if k!="__metadata__"}
    assert all(k.startswith("model.") for k in tensors)
    sys.path.insert(0,str(source/"src"))
    from depth_anything_3.cfg import create_object,load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    teacher=create_object(load_config(MODEL_REGISTRY["da3mono-large"])).eval()
    state=teacher.state_dict()
    assert set(state)=={k[len("model."):] for k in tensors}
    assert all(list(v.shape)==tensors["model."+k]["shape"] for k,v in state.items())
    with torch.no_grad():
        out=teacher(torch.rand(1,1,3,56,112),infer_gs=False,use_ray_pose=False)
    assert out["depth"].shape==(1,1,56,112) and torch.isfinite(out["depth"]).all()
    report={"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"code_revision":CODE_REVISION,
            "official_tensor_keys_and_shapes_matched":len(tensors),
            "official_parameters":sum(p.numel() for p in teacher.parameters()),
            "random_weight_official_network_forward_shape":list(out["depth"].shape),
            "pretrained_tensor_values_downloaded":False,"pretrained_teacher_accuracy_tested":False,
            "header_only_bytes":length+8,"full_teacher_generation_tested":False}
    del teacher,state,out; gc.collect()
    return report


def main():
    sys.path.insert(0,str(FOLDER))
    torch.set_num_threads(2)
    completed=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s",str(FOLDER),"-v"],
                             cwd=FOLDER,capture_output=True,text=True,encoding="utf-8")
    log=completed.stdout+completed.stderr
    print(log,flush=True)
    if completed.returncode: raise RuntimeError("Unit contracts failed")
    with tempfile.TemporaryDirectory(prefix="anchorflow_v9_colab_layout_") as temp:
        target=Path(temp)
        for p in FOLDER.iterdir():
            if p.is_file() and p.suffix in (".py",".json"): shutil.copy2(p,target/p.name)
        assert not list(target.glob("*.ipynb"))
        staged=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s",str(target),"-v"],
                              cwd=target,capture_output=True,text=True,encoding="utf-8")
        if staged.returncode:
            print(staged.stdout+staged.stderr,flush=True)
            raise RuntimeError("Colab staging without .ipynb failed")
    print("Both original and Colab-staged 17-test suites PASS",flush=True)
    from model import AnchorFlowEdge
    from data import KITTIDataset,write_json
    from losses import objective
    import run
    cfg=json.loads((FOLDER/"config.json").read_text())
    torch.manual_seed(42)
    model=AnchorFlowEdge().to(memory_format=torch.channels_last)
    ds=object.__new__(KITTIDataset)
    ds.root=ROOT/"data/teacher_subset_2000/kitti_bundle"
    ds.split,ds.teacher="val",False
    ds.rows=[r.split() for r in (ds.root/"splits/val_400.txt").read_text().splitlines()]
    shapes=[]
    for index in (0,199,399):
        sample=ds[index]
        batch={k:(v[None] if torch.is_tensor(v) else v) for k,v in sample.items()}
        model.eval()
        with torch.no_grad(): out=model(*(batch[k] for k in ("rgb","sparse","mask","K")))
        assert torch.isfinite(out["D_full"]).all()
        shapes.append(list(out["D_full"].shape))
    batch.update(teacher=torch.full_like(batch["gt"],20),confidence=torch.full_like(batch["gt"],.8),
                 relative=torch.linspace(-1,1,1216)[None,None,None].expand(1,1,352,1216).clone(),
                 relative_confidence=torch.ones_like(batch["gt"]))
    model.train(); model.freeze_encoder_bn()
    with torch.autocast("cpu",dtype=torch.bfloat16):
        out=model(*(batch[k] for k in ("rgb","sparse","mask","K")))
    loss,stats=objective(out,batch,torch.zeros_like(batch["mask"]),4.,cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    names=("dynamics.reaction.weight","dynamics.state.weight","dynamics.conductance.weight",
           "phase2.candidates.weight","phase2.blend.weight")
    gradients={k:float(p.grad.abs().sum()) for k,p in model.named_parameters() if k in names}
    assert all(x>0 for x in gradients.values())
    assert float(stats["relative_pair_coverage"])>0
    model.eval(); model.zero_grad(set_to_none=True)
    complexity=run.count_operations(model,run.sample_inputs(torch.device("cpu")))
    control_cfg={**cfg,"model_name":"v8_control","relative_enabled":False,"inverse_weight":0}
    control=run.make_model(control_cfg,torch.device("cpu"),False).eval()
    baseline=run.count_operations(control,run.sample_inputs(torch.device("cpu")))
    del control; gc.collect()
    # Open the fresh heads so export parity exercises nonconstant weights/metric correction.
    with torch.no_grad():
        torch.nn.init.normal_(model.phase2.candidates.weight,std=.01)
        torch.nn.init.normal_(model.dynamics.reaction.weight,std=.01)
        torch.nn.init.normal_(model.phase2.delta.weight,std=.003)
        torch.nn.init.normal_(model.detail1.delta.weight,std=.003)
    export_cfg={**cfg,"work":str(ROOT/"results/anchorflow_v9_audit/onnx_work"),
                "drive_runs":str(ROOT/"results/anchorflow_v9_audit"),"run_name":"opened_consensus_export"}
    with patch.object(run,"make_model",return_value=model):
        exported=run.export(export_cfg,"structural",untrained=True)
    print("Actual KITTI RGB/sensor/GT backward + opened-head ONNX PASS",flush=True)
    del model,out; gc.collect()
    adapter=teacher_adapter_check()
    report={"tests":int(re.search(r"Ran (\d+) tests",log).group(1)),"tests_returncode":0,"tests_log":log,
            "colab_copy_without_ipynb_tests_returncode":staged.returncode,"source_sha256":run.source_hash(),
            "python":sys.version,"torch":torch.__version__,"real_kitti_samples":[0,199,399],
            "output_shapes":shapes,"real_kitti_bf16_loss":float(loss.detach()),
            "real_kitti_bf16_gradients_finite":True,"head_gradient_l1":gradients,
            "relative_pair_coverage_fixture":float(stats["relative_pair_coverage"]),
            "weighted_relative_fixture":float(stats["weighted_relative"]),
            "fixture_note":"Real KITTI RGB/sparse/GT. Metric and relative teachers here are SYNTHETIC; not a training accuracy or teacher quality test.",
            "complexity_v9":complexity,"complexity_v8_control":baseline,
            "onnx_opened_consensus":exported,"teacher_adapter":adapter,
            "student_checkpoint_loaded":False,"gpu_training_tested":False,
            "gpu_latency_measured":False,"new_training_accuracy_measured":False}
    verified=[p for p in FOLDER.iterdir() if p.is_file() and
              (p.suffix==".py" or p.name in ("config.json","requirements.txt","teacher_requirements.txt")
               or p.name.endswith("_contract.json"))]
    report["verified_files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in verified}
    write_json(FOLDER/"local_verification.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="tests_log"},indent=2),flush=True)


if __name__=="__main__": main()
