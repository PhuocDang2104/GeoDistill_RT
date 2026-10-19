"""All-400 accuracy from TensorRT+CUDA Graph, not a renamed PyTorch metric."""
import csv
import gc
import json
from pathlib import Path
import time
import zipfile
import cv2
import numpy as np
import torch
import experiment as E
from data import digest,write_json
from metrics import Metrics
from boundaries import BoundaryMetrics
from deployment import typed_deployment,DepthOnly,export_onnx,build_engine,TRTRunner,TorchGraph,parity


def timing(runner,images,warmup,iterations):
    for i in range(warmup):runner(images[i%len(images)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu,wall,enqueue=[],[],[]
    # Serialized per-frame latency, not overlapped throughput. No profiler attached.
    for i in range(iterations):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start=time.perf_counter()
        begin.record()
        runner(images[i%len(images)])
        submitted=time.perf_counter()
        end.record()
        end.synchronize()
        finished=time.perf_counter()
        gpu.append(begin.elapsed_time(end))
        enqueue.append(1000*(submitted-start))
        wall.append(1000*(finished-start))
    def summary(values):
        return {"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95)),
                "mean_ms":float(np.mean(values)),"fps_from_median":1000/float(np.median(values))}
    free,total=torch.cuda.mem_get_info()
    return {"gpu_stream":summary(gpu),"host_enqueue":summary(enqueue),
            "synchronized_wall":summary(wall),"warmup":warmup,"iterations":iterations,
            "distinct_images":len(images),"peak_pytorch_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
            "cuda_free_mib_after":free/2**20,"cuda_total_mib":total/2**20,
            "memory_scope":"Torch peak excludes direct TensorRT allocations; process holds all compared backends and preloaded inputs, NOT isolated per-model footprint",
            "scope":"GPU-resident inputs; includes D2D staging; excludes decode/preprocess/H2D/metrics/build/capture"}


@torch.no_grad()
def test_predictions(runner,cfg,variant,root):
    from run import loader,move
    previews=root/"test_preview"
    previews.mkdir(parents=True,exist_ok=True)
    archive=root/"kitti_test_predictions.zip"
    records=[]
    # Same deterministic scene indices for all arms, fixed 0..120m color range.
    picks={i*100 for i in range(10)}
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zipped:
        for index,cpu in enumerate(loader(cfg,"test")):
            batch=move(cpu,torch.device("cuda"),cfg)
            depth=runner(batch)[0,0].cpu().numpy()
            if not np.isfinite(depth).all():raise RuntimeError("Nonfinite test depth")
            image=np.clip(np.round(depth*256),0,65535).astype(np.uint16)
            ok,encoded=cv2.imencode(".png",image)
            if not ok:raise RuntimeError("PNG encoding failed")
            sid=cpu["sid"][0]
            zipped.writestr(sid+".png",encoded.tobytes())
            if index in picks:
                color=cv2.applyColorMap(np.clip(depth/120*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
                rgb=np.clip(cpu["rgb"][0].numpy().transpose(1,2,0)*255,0,255).astype(np.uint8)[...,::-1]
                cv2.imwrite(str(previews/f"{index:04d}_depth.png"),color)
                pair=np.concatenate((rgb,color),axis=1)
                cv2.imwrite(str(previews/f"{index:04d}_rgb_depth.png"),pair)
                records.append({"index":index,"sample_id":sid,"source_backend":"TensorRT CUDA Graph"})
    if index+1!=1000:raise RuntimeError("Expected all 1000 anonymous test frames")
    write_json(previews/"manifest.json",{"images":records,"color_min_m":0,"color_max_m":120,
        "colormap":"TURBO; global linear metric depth, no per-image normalization","public_gt":False})


@torch.no_grad()
def benchmark(cfg,variant):
    from run import setup,load_checkpoint,loader,move,run_dir
    device=setup(cfg)
    root=run_dir(cfg,variant)
    model,ckpt,checkpoint_path=load_checkpoint(cfg,variant,device)
    status=json.loads((root/"training_status.json").read_text())
    if status["completed_epochs"]!=cfg["epochs"]:raise RuntimeError("Train the entire registered epoch budget before final benchmark")
    model.set_diagnostics(False)
    native=DepthOnly(model,native_amp=True).eval()
    images=[]
    for i,batch in enumerate(loader(cfg,"val")):
        if i<cfg["benchmark"]["timed_images"]:
            moved=move(batch,device,cfg)
            images.append({k:moved[k].contiguous() for k in E.INPUTS})
    sample=images[0]
    typed=typed_deployment(model).to(device).eval()
    export_path=root/"deployment.onnx"
    export_onnx(typed,sample,export_path,cfg["benchmark"]["export_opset"])
    engine_path=root/"deployment.plan"
    build=build_engine(export_path,engine_path,cfg["benchmark"]["trt_workspace_gib"],cfg["tf32"])
    trt=TRTRunner(engine_path,sample,capture=False)
    graph=TRTRunner(engine_path,sample,capture=True)
    torch_graph=TorchGraph(native,sample)
    # Input freshness: A -> B -> A and sparse/K perturbations, with independent eager reference.
    B={k:v.clone() for k,v in sample.items()}
    B["rgb"]=(B["rgb"]*.7+.1).clamp(0,1)
    B["sparse"]=torch.zeros_like(B["sparse"]);B["mask"]=torch.zeros_like(B["mask"])
    B["K"][:,0,0]*=1.1
    replay_checks=[]
    for tag,batch in (("A",sample),("B",B),("A_again",sample)):
        expected=typed(*(batch[k] for k in E.INPUTS))
        actual=graph(batch).clone()
        check=parity(actual,expected,cfg["benchmark"])
        replay_checks.append(dict(tag=tag,**check))
        native_expected=native(*(batch[k] for k in E.INPUTS))
        torch_check=parity(torch_graph(batch),native_expected,cfg["benchmark"])
        replay_checks.append(dict(tag="pytorch_graph_"+tag,**torch_check))
        trt_check=parity(trt(batch),actual,cfg["benchmark"])
        replay_checks.append(dict(tag="trt_eager_vs_graph_"+tag,**trt_check))
    write_json(root/"replay_parity.json",replay_checks)
    if not all(x["passed"] for x in replay_checks):
        raise RuntimeError("TRT graph A/B/A parity failed; report saved, no fallback")
    scores={name:Metrics(device) for name in ("pytorch_native","pytorch_typed","trt_graph")}
    boundary=BoundaryMetrics(device)
    maximum_mean,maximum_pixel=0.,0.
    failed_frames=[]
    per_image=[]
    for index,cpu in enumerate(loader(cfg,"val")):
        batch=move(cpu,device,cfg)
        native_depth=native(*(batch[k] for k in E.INPUTS))
        typed_depth=typed(*(batch[k] for k in E.INPUTS))
        actual=graph(batch).clone()
        check=parity(actual,typed_depth,cfg["benchmark"])
        maximum_mean=max(maximum_mean,check["mean_abs_m"])
        maximum_pixel=max(maximum_pixel,check["max_abs_m"])
        if not check["passed"]:failed_frames.append(index)
        for name,pred in (("pytorch_native",native_depth),("pytorch_typed",typed_depth),("trt_graph",actual)):
            scores[name].update(pred,batch["gt"],batch["gt_mask"],batch["rgb"])
        boundary.update(actual,batch["gt"],batch["gt_mask"])
        valid=batch["gt_mask"]>0
        count=int(valid.sum())
        error=(actual-batch["gt"]).square()
        per_image.append({"sample_id":cpu["sid"][0],"pixels":count,
                          "sse_m2":float((error*valid).sum(dtype=torch.float64))})
        if (index+1)%100==0:print(f"{variant} TensorRT graph accuracy {index+1}/400",flush=True)
    reports={k:v.report() for k,v in scores.items()}
    for report in reports.values():
        if report["all"]["pixels"]!=25424992:raise RuntimeError("Validation protocol changed")
    a,b=reports["pytorch_native"]["all"],reports["trt_graph"]["all"]
    rmse_drift=abs(a["rmse_m"]-b["rmse_m"])
    inverse_drift=abs(a["irmse_km_inv"]-b["irmse_km_inv"])
    passed=not failed_frames and rmse_drift<=cfg["benchmark"]["rmse_drift_m"] and (
        inverse_drift<=cfg["benchmark"]["irmse_drift_km_inv"])
    report={"variant":variant,"checkpoint_epoch":ckpt["state"]["selection"]["policy_epoch"],
        "architecture":{"coarse_nfe":4,"fine_nfe":0,"fine_module":None},
        "upir_input_coefficients":model.detail1.pir.body[0].weight[:,40,0,0].detach().cpu().tolist(),
        "training_epochs":cfg["epochs"],
        "inference_precision":"explicit_FP32_TF32_OFF",
        "protocol_note":"UPIR predeclared 6cm max gate; matches historical diagnostic6cm, NOT original5cm parity",
        "checkpoint_sha256":digest(checkpoint_path),"onnx_sha256":digest(export_path),
        "engine_sha256":digest(engine_path),"source_sha256":E.source_hash(),
        "initial_shared_sha256":ckpt["state"]["initial_shared_sha256"],
        "parameters":sum(p.numel() for p in model.parameters()),"build":build,
        "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,
        "qualification_passed":False,"accuracy_qualification_passed":passed,"parity_gates":cfg["benchmark"],
        "parity":{"failed_frames":failed_frames,"worst_mean_m":maximum_mean,"worst_pixel_m":maximum_pixel,
                  "final_rmse_drift_m":rmse_drift,"final_irmse_drift_km_inv":inverse_drift},
        **reports,"trt_graph_boundary":boundary.report(),"per_image":per_image}
    write_json(root/"trt_benchmark.json",report)
    if not passed:
        raise RuntimeError("Export/precision/engine failed frozen parity gates; inspect trt_benchmark.json")
    report["timing"]={}
    for name,runner in (("pytorch_eager",lambda b:native(*(b[k] for k in E.INPUTS))),
                         ("pytorch_graph",torch_graph),("trt_eager",trt),("trt_graph",graph)):
        report["timing"][name]=timing(runner,images,cfg["benchmark"]["warmup"],cfg["benchmark"]["iterations"])
    test_predictions(graph,cfg,variant,root)
    report["qualification_passed"]=True
    write_json(root/"trt_benchmark.json",report)
    print(json.dumps({"variant":variant,"all":report["trt_graph"]["all"],
                      "timing":report["timing"]["trt_graph"]},indent=2))
