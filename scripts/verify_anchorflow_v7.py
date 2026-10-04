"""Actual V6 migration + real KITTI gradients + opened-jet ONNX validation (CPU)."""
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import torch


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"drive_upload/AnchorFlow_v7_Jet"
    sys.path.insert(0,str(folder))
    tests=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s",str(folder),"-p","test*.py","-v"],
                         cwd=folder,check=False,capture_output=True,text=True,encoding="utf-8")
    test_log=tests.stdout+tests.stderr
    print(test_log[-3500:],flush=True)
    if tests.returncode:
        raise RuntimeError("Contract tests failed; see the captured traceback above")
    from model import AnchorFlowEdge,load_parent_state
    from data import KITTIDataset,write_json,digest
    from losses import objective
    from run import count_operations,sample_inputs,export,source_hash
    import run
    torch.set_num_threads(2)
    torch.manual_seed(42)
    state=torch.load(folder/"init_v6_best.pth",map_location="cpu",weights_only=False)["model"]
    reference=AnchorFlowEdge(model_name="v6_reduced").eval()
    reference.load_state_dict(state,strict=True)
    model=AnchorFlowEdge().eval()
    migration=load_parent_state(model,state)
    ds=object.__new__(KITTIDataset)
    ds.root=root/"data/teacher_subset_2000/kitti_bundle"
    ds.split,ds.teacher="val",False
    ds.rows=[r.split() for r in (ds.root/"splits/val_400.txt").read_text().splitlines()]
    noop=[]
    for index in (0,199,399):
        row=ds[index]
        inputs=tuple(row[k].unsqueeze(0) for k in ("rgb","sparse","mask","K"))
        with torch.inference_mode():
            old,new=reference(*inputs),model(*inputs)
        noop.append(float((old["D_full"]-new["D_full"]).abs().max()))
        torch.testing.assert_close(old["D_full"],new["D_full"],atol=0,rtol=0)
    batch={k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in row.items()}
    batch.update(teacher=torch.full_like(batch["gt"],20),confidence=torch.full_like(batch["gt"],.8))
    cfg=json.loads((folder/"config.json").read_text())
    model.train(); model.freeze_encoder_bn()
    with torch.autocast("cpu",dtype=torch.bfloat16):
        pred=model(*(batch[k] for k in ("rgb","sparse","mask","K")))
    loss,stats=objective(pred,batch,torch.zeros_like(batch["mask"]),0,cfg)
    loss.backward()
    finite=bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not finite:
        raise RuntimeError("Real KITTI BF16 gradients failed")
    head_grads={name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if name in
                ("connection4.jet.weight","phase2.proposal_adapter.weight")}
    assert all(v>0 for v in head_grads.values())
    model.eval(); model.zero_grad(set_to_none=True)
    complexity=count_operations(model,sample_inputs(torch.device("cpu")))
    # Exercise nonzero jets with trained V6 CNN/phase/surface. This is export
    # validation, NOT a newly trained V7 accuracy claim.
    with torch.no_grad():
        torch.nn.init.normal_(model.connection4.jet.weight,std=.001)
        torch.nn.init.normal_(model.phase2.proposal_adapter.weight,std=.001)
    cfg.update(work=str(root/"results/anchorflow_v7_audit/onnx_work"),
               drive_runs=str(root/"results/anchorflow_v7_audit"),run_name="opened_jet_export",profile_runs=2,profile_warmup=1)
    with patch.object(run,"make_model",return_value=model):
        onnx=export(cfg,"structural",untrained=True)
    causal=json.loads((folder/"v6_causal_audit.json").read_text())
    assert causal["samples"]==400
    report={"tests":int(re.search(r"Ran (\d+) tests",test_log).group(1)),"tests_returncode":tests.returncode,
            "tests_log":test_log,"source_sha256":source_hash(),"python":sys.version,"torch":torch.__version__,
            "test_scope":"CPU contract/real-checkpoint/KITTI/BF16/ONNX (not CUDA)","migration":migration,
            "parent_checkpoint_sha256":digest(folder/"init_v6_best.pth"),"real_kitti_samples":[0,199,399],
            "full_size_noop_vs_reduced_v6_max_abs_m":noop,"real_kitti_bf16_loss":float(loss.detach()),
            "real_kitti_bf16_gradients_finite":finite,"new_head_gradient_l1":head_grads,
            "gradient_check_teacher":"Synthetic target; actual train-only cache coverage gate runs in Colab",
            "complexity":complexity,"onnx_nonzero_jet_parity":onnx,
            "causal_audit_400":{k:v["all"]["rmse_m"] for k,v in causal["metrics"].items()},
            "gpu_latency_measured":False,"v7_new_training_accuracy_measured":False,
            "full_v6_zero_head_equality_claimed":False}
    write_json(folder/"local_verification.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="tests_log"},indent=2),flush=True)


if __name__=="__main__":
    main()
