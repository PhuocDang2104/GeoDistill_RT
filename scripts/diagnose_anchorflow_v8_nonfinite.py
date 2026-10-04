"""Observe a resumed V8 train without changing its source/config/checkpoint protocol.

Stops before backward on the FIRST bad objective; records tensor/module failures
and compares that identical pre-forward state/batch under FP16/BF16/FP32.
Does not skip bad batches, sanitize predictions, change AMP, or rewrite checkpoints.
Run in Colab alongside the ORIGINAL frozen run.py and resolved_config.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime, timezone
import torch


def summary(tensor):
    x=tensor.detach().float()
    finite=torch.isfinite(x)
    values=x[finite]
    return {"shape":list(x.shape),"dtype":str(tensor.dtype),
            "nonfinite_count":int((~finite).sum()),
            "finite_min":float(values.min()) if values.numel() else None,
            "finite_max":float(values.max()) if values.numel() else None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--code",type=Path,default=Path("/content/anchorflow_v8_code"))
    parser.add_argument("--config",type=Path)
    parser.add_argument("--variant",choices=("metric_kd","gt_only"),default="metric_kd")
    parser.add_argument("--save-replay",action="store_true",help="Also save roughly 60+ MB failing batch/model artifact on Colab SSD")
    args=parser.parse_args()
    sys.path.insert(0,str(args.code.resolve()))
    import run
    cfg=json.loads((args.config or args.code/"resolved_config.json").read_text(encoding="utf-8"))
    cfg["teacher_enabled"]=args.variant=="metric_kd"
    original_make,original_objective=run.make_model,run.objective
    active={"epoch":None,"batch":0}

    def observed_make(*a,**kw):
        model=original_make(*a,**kw)
        active["model"]=model
        def before(root,inputs):
            active["flags"]=[]
            # BN statistics can be contaminated by a bad forward; retain the
            # actual state BEFORE it, not the poisoned buffers afterward.
            active["buffers"]={n:b.detach().clone() for n,b in root.named_buffers()}
        model.register_forward_pre_hook(before)
        def leaf(name):
            def after(module,inputs,output):
                if torch.is_tensor(output):
                    active["flags"].append((name,torch.isfinite(output.detach()).all()))
            return after
        for name,module in model.named_modules():
            if not list(module.children()): module.register_forward_hook(leaf(name))
        return model

    def observed_objective(pred,batch,holdout,progress,config):
        loss,stats=original_objective(pred,batch,holdout,progress,config)
        epoch=int(progress)
        if active["epoch"]!=epoch: active.update(epoch=epoch,batch=0)
        active["batch"]+=1
        keys=list(stats)
        good=torch.isfinite(torch.stack([stats[k].float() for k in keys]))
        if bool(good.all()) and bool(torch.isfinite(loss)):
            return loss,stats
        model=active["model"]
        state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
        state.update({n:t.detach().cpu().clone() for n,t in active["buffers"].items() if n in state})
        bad_stats=[k for k,ok in zip(keys,good.cpu().tolist()) if not ok]
        bad_modules=[n for n,ok in active["flags"] if not bool(ok)]
        report={"epoch":epoch,"batch":active["batch"],"epoch_progress":progress,
                "amp":config["amp"],"source_sha256":run.source_hash(),
                "sample_ids":batch.get("sid"),"bad_loss_or_stat_names":bad_stats,
                "nonfinite_module_outputs_in_execution_order":bad_modules,
                "inputs":{n:summary(t) for n,t in batch.items() if torch.is_tensor(t)},
                "outputs":{n:summary(t) for n,t in pred.items() if torch.is_tensor(t)},
                "nonfinite_preforward_state_keys":[n for n,t in state.items() if t.is_floating_point() and not bool(torch.isfinite(t).all())],
                "same_batch_precision_probes":{},"precision_probe_device":str(batch["rgb"].device)}
        input_mask=batch["mask"]*(1-holdout)
        device=batch["rgb"].device
        for precision in ("fp16","bf16","fp32"):
            try:
                probe=original_make(config,device,pretrained=False)
                probe.load_state_dict(state,strict=True)
                probe.train()
                if config["freeze_encoder_bn"]: probe.freeze_encoder_bn()
                with torch.no_grad(),run.autocast({**config,"amp":precision},device):
                    output=probe(batch["rgb"],batch["sparse"],input_mask,batch["K"])
                value,items=original_objective(output,batch,holdout,progress,config)
                v=float(value)
                report["same_batch_precision_probes"][precision]={
                    "loss":v if math.isfinite(v) else str(v),
                    "bad_stats":[n for n,t in items.items() if not bool(torch.isfinite(t).all())],
                    "bad_outputs":[n for n,t in output.items() if not bool(torch.isfinite(t).all())]}
                del probe,output,items,value
            except Exception as error:
                report["same_batch_precision_probes"][precision]={"error":repr(error)}
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder=Path(config["work"])/"nonfinite_debug"/stamp
        folder.mkdir(parents=True,exist_ok=True)
        local=folder/"nonfinite_report.json"
        run.write_json(local,report)
        _,drive=run.directories(config,args.variant)
        destination=drive/"nonfinite_debug"/stamp/"nonfinite_report.json"
        run.copy_atomic(local,destination)
        if args.save_replay:
            torch.save({"model_before_forward":state,
                        "batch":{n:(t.detach().cpu() if torch.is_tensor(t) else t) for n,t in batch.items()},
                        "input_mask":input_mask.detach().cpu(),"holdout":holdout.detach().cpu(),
                        "config":config,"epoch_progress":progress},folder/"failure_replay.pth")
        print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
        print("REPORT ON DRIVE:",destination,flush=True)
        print("COLAB ARTIFACTS:",folder,flush=True)
        raise RuntimeError("First nonfinite objective captured BEFORE backward. No NaN masking/batch skipping/checkpoint overwrite. Send nonfinite_report.json.")

    run.make_model=observed_make
    run.objective=observed_objective
    print("Diagnostic observer: same frozen model/loss/AMP/protocol; per-layer finite checks add overhead.",flush=True)
    run.train(cfg,args.variant)


if __name__=="__main__": main()
