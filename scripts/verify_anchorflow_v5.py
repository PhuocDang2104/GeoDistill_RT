"""Local verification only: actual v3 weights + opened v5 heads, CPU BF16/ONNX."""
import json
import sys
import subprocess
import re
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/"drive_upload/AnchorFlow_v5_Piecewise"
    sys.path.insert(0,str(folder))
    tests = subprocess.run([sys.executable,"-m","unittest","discover","-s",str(folder),"-p","test*.py","-v"],
                           cwd=folder,check=True,capture_output=True,text=True,encoding="utf-8")
    test_log = tests.stdout+tests.stderr
    print(test_log,flush=True)
    test_count = int(re.search(r"Ran (\d+) tests",test_log).group(1))
    from model import AnchorFlowEdge,load_parent_state
    from model_v3 import AnchorFlowEdge as V3
    from data import KITTIDataset,write_json
    from run import count_operations,sample_inputs,export,source_hash
    from losses import objective
    import run
    torch.set_num_threads(2)
    torch.manual_seed(42)
    state = torch.load(folder/"init_v3_best.pth",map_location="cpu",weights_only=False)["model"]
    model = AnchorFlowEdge().eval()
    migration = load_parent_state(model,state)
    reference = V3().eval()
    reference.load_state_dict(state,strict=True)
    ds = object.__new__(KITTIDataset)
    ds.root = root/"data/teacher_subset_2000/kitti_bundle"
    ds.split,ds.teacher = "val",False
    ds.rows = [r.split() for r in (ds.root/"splits/val_400.txt").read_text().splitlines()]
    noop = []
    for index in (0,199,399):
        row = ds[index]
        inputs = tuple(row[k].unsqueeze(0) for k in ("rgb","sparse","mask","K"))
        with torch.inference_mode():
            old,new = reference(*inputs),model(*inputs)
        noop.append(float((old["D_full"]-new["D_full"]).abs().max()))
        torch.testing.assert_close(new["D_full"],old["D_full"],atol=0,rtol=0)
    config = json.loads((folder/"config.json").read_text())
    batch = {k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in row.items()}
    # Synthetic teacher for this gradient-only check, NOT a teacher coverage audit.
    batch.update(teacher=torch.full_like(batch["gt"],20),confidence=torch.full_like(batch["gt"],.8))
    model.train()
    model.freeze_encoder_bn()
    with torch.autocast("cpu",dtype=torch.bfloat16):
        prediction = model(*(batch[k] for k in ("rgb","sparse","mask","K")))
    loss,stats = objective(prediction,batch,torch.zeros_like(batch["mask"]),0,config)
    loss.backward()
    finite = bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not finite:
        raise RuntimeError("Real KITTI BF16 gradients failed")
    model.eval()
    model.zero_grad(set_to_none=True)
    complexity = count_operations(model,sample_inputs(torch.device("cpu")))
    # Exercise ONNX parity with NON-ZERO correction, transport and barrier heads.
    with torch.no_grad():
        model.surface.amplitude.fill_(.15)
        for head in (model.surface.slopes,model.surface.reaction,model.surface.barriers,model.surface.conductance):
            torch.nn.init.normal_(head.weight,std=.001)
    config.update(work=str(root/"results/anchorflow_v5_audit/onnx_work"),
                  drive_runs=str(root/"results/anchorflow_v5_audit"),run_name="opened_heads_structural",profile_runs=2,profile_warmup=1)
    with patch.object(run,"make_model",return_value=model):
        onnx = export(config,"structural",untrained=True)
    report = {"tests":test_count,"tests_returncode":tests.returncode,"tests_log":test_log,
              "source_sha256":source_hash(),"torch":torch.__version__,"python":sys.version,
              "test_scope":"CPU fixtures and BF16 (not CUDA)","migration":migration,
              "real_kitti_samples":[0,199,399],"full_size_noop_max_abs_m":noop,
              "real_kitti_bf16_loss":float(loss.detach()),"real_kitti_bf16_gradients_finite":finite,
              "gradient_check_teacher":"synthetic metric target; actual teacher cache gate runs in Colab prepare/smoke",
              "complexity":complexity,"onnx_nonzero_heads_parity":onnx,
              "v5_accuracy_trained":False,"gpu_latency_measured":False}
    write_json(folder/"local_verification.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
