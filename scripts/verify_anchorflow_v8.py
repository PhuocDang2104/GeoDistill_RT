"""CPU contracts + actual KITTI BF16 backward + opened-dynamics ONNX parity."""
from pathlib import Path
from unittest.mock import patch
import json
import re
import subprocess
import sys
import shutil
import tempfile
import torch


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"drive_upload/AnchorFlow_v8_Dynamics"
    sys.path.insert(0,str(folder))
    tests=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s",str(folder),"-v"],
                         cwd=folder,capture_output=True,text=True,encoding="utf-8")
    text=tests.stdout+tests.stderr
    print(text[-5000:],flush=True)
    if tests.returncode: raise RuntimeError("Contracts failed")
    # Reproduce the Colab staging layout exactly: no UI .ipynb in /content/code.
    # Do not delete or mutate any user data; use an isolated temporary directory.
    with tempfile.TemporaryDirectory(prefix="anchorflow_v8_colab_copy_") as temp:
        staged=Path(temp)
        for source in folder.iterdir():
            if source.is_file() and source.suffix in (".py",".json"):
                shutil.copy2(source,staged/source.name)
        assert not list(staged.glob("*.ipynb"))
        staged_tests=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s",str(staged),"-v"],
                                    cwd=staged,capture_output=True,text=True,encoding="utf-8")
        if staged_tests.returncode:
            print(staged_tests.stdout+staged_tests.stderr,flush=True)
            raise RuntimeError("Colab copy without .ipynb regression failed")
        print("COLAB STAGING WITHOUT .ipynb: ALL TESTS PASS",flush=True)
    from model import AnchorFlowEdge
    from data import KITTIDataset,write_json
    from losses import objective
    import run
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=AnchorFlowEdge().to(memory_format=torch.channels_last)
    ds=object.__new__(KITTIDataset)
    ds.root=root/"data/teacher_subset_2000/kitti_bundle"
    ds.split,ds.teacher="val",False
    ds.rows=[r.split() for r in (ds.root/"splits/val_400.txt").read_text().splitlines()]
    cfg=json.loads((folder/"config.json").read_text())
    shapes=[]
    for index in (0,199,399):
        sample=ds[index]
        batch={k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in sample.items()}
        model.eval()
        with torch.no_grad():
            out=model(*(batch[k] for k in ("rgb","sparse","mask","K")))
        assert torch.isfinite(out["D_full"]).all()
        shapes.append(list(out["D_full"].shape))
    # Actual RGB/sensor/GT; only this local test's KD target is synthetic.
    batch.update(teacher=torch.full_like(batch["gt"],20.),confidence=torch.full_like(batch["gt"],.8))
    model.train(); model.freeze_encoder_bn()
    with torch.autocast("cpu",dtype=torch.bfloat16):
        out=model(*(batch[k] for k in ("rgb","sparse","mask","K")))
    loss,stats=objective(out,batch,torch.zeros_like(batch["mask"]),1.,cfg)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    gradients={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if n in
               ("dynamics.reaction.weight","dynamics.state.weight","dynamics.conductance.weight","phase2.blend.weight")}
    assert all(v>0 for v in gradients.values())
    model.eval(); model.zero_grad(set_to_none=True)
    complexity=run.count_operations(model,run.sample_inputs(torch.device("cpu")))
    # Exercise nonzero forcing/metric heads; structural export test, NOT trained accuracy.
    with torch.no_grad():
        torch.nn.init.normal_(model.dynamics.reaction.weight,std=.015)
        torch.nn.init.normal_(model.phase2.delta.weight,std=.005)
        torch.nn.init.normal_(model.detail1.delta.weight,std=.003)
    cfg.update(work=str(root/"results/anchorflow_v8_audit/onnx_work"),
               drive_runs=str(root/"results/anchorflow_v8_audit"),run_name="opened_dynamics_export")
    with patch.object(run,"make_model",return_value=model):
        exported=run.export(cfg,"structural",untrained=True)
    report={"tests":int(re.search(r"Ran (\d+) tests",text).group(1)),"tests_returncode":0,"tests_log":text,
            "colab_copy_without_ipynb_tests_returncode":staged_tests.returncode,
            "source_sha256":run.source_hash(),"python":sys.version,"torch":torch.__version__,
            "scope":"CPU synthetic contracts, trainer resume, actual KITTI RGB/sparse/GT, BF16 gradients, FP32 ONNX",
            "real_kitti_samples":[0,199,399],"output_shapes":shapes,"real_kitti_bf16_loss":float(loss.detach()),
            "real_kitti_bf16_gradients_finite":True,"head_gradient_l1":gradients,
            "gradient_check_teacher":"Synthetic target; actual train-cache KD coverage is checked in Colab",
            "dynamics_diagnostics":{k:float(v) for k,v in stats.items() if k.startswith("dynamics_")},
            "complexity":complexity,"onnx_opened_dynamics":exported,
            "imagenet_download_tested_locally":False,"student_checkpoint_loaded":False,
            "cuda_fp16_training_tested":False,"gpu_latency_measured":False,"new_training_accuracy_measured":False}
    write_json(folder/"local_verification.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="tests_log"},indent=2),flush=True)


if __name__=="__main__": main()
