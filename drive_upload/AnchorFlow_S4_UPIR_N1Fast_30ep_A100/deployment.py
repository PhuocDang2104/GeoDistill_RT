"""Explicit FP32 export and TensorRT CUDA graph runtime, no plugins."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import experiment as E


def typed_deployment(model,fast=True):
    """New model copy. Trained checkpoint/weights/BN state remain untouched."""
    # Clear transient telemetry: backward smoke leaves non-leaf tensors here.
    # These caches are not parameters/buffers or inputs to the next prediction.
    for child in model.modules():
        if hasattr(child,"last_diagnostics"):
            child.last_diagnostics={}
    model=copy.deepcopy(model).eval()
    model.set_diagnostics(False)
    # D uses the FP32 recovery protocol from the completed A/B/C runs. No mixed
    # BF16 Conv export and no manual BN folding; weights/checkpoint untouched.
    model=model.float()
    if fast:
        from fast_ops import enable_fast_ops
        enable_fast_ops(model)
    return DepthOnly(model,native_amp=False).eval()


class DepthOnly(nn.Module):
    def __init__(self,model,native_amp=False):
        super().__init__()
        self.model,self.native_amp=model,native_amp

    def forward(self,rgb,sparse,mask,K):
        with E.amp(rgb.device,self.native_amp):
            return self.model(rgb,sparse,mask,K)["D_full"]


def export_onnx(module,sample,path,opset=22,input_names=None):
    import onnx
    module.eval()
    if opset<22:
        raise ValueError("Standard ONNX Conv requires opset22+ for BF16 type validity")
    with torch.no_grad():
        # Same legacy-export + official opset conversion as historical recovery.
        names=tuple(input_names or E.INPUTS)
        torch.onnx.export(module,tuple(sample[k] for k in names),str(path),
            input_names=list(names),output_names=["D_full"],opset_version=20,
            dynamic_axes=None,dynamo=False,do_constant_folding=True)
    module.eval()
    if any(child.training for child in module.modules()):
        raise RuntimeError('Export must not switch any deployment module/BN to train')
    # Legacy exporter symbols stop at20. Use the OFFICIAL schema converter,
    # then full type/shape checking, rather than shipping BF16 Conv under opset18.
    graph=onnx.version_converter.convert_version(onnx.load(str(path)),opset)
    onnx.checker.check_model(graph,full_check=True)
    onnx.save(graph,str(path))


def build_engine(onnx_path,engine_path,workspace_gib=4,tf32=False,optimization_level=5):
    import tensorrt as trt
    major=int(trt.__version__.split(".")[0])
    if major!=10:
        raise RuntimeError("Qualified API target is TensorRT 10.x; use notebook pinned requirements")
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("TensorRT ONNX parser:\n"+"\n".join(str(parser.get_error(i))
                           for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    if optimization_level not in (3,5):raise ValueError('Registered builder levels3/5 only')
    config.builder_optimization_level=optimization_level
    config.max_aux_streams=0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,int(workspace_gib*2**30))
    if tf32:config.set_flag(trt.BuilderFlag.TF32)
    else:config.clear_flag(trt.BuilderFlag.TF32)
    # Strongly typed: no FP16/BF16 flags or weak layer precision overrides.
    plan=builder.build_serialized_network(network,config)
    if plan is None:raise RuntimeError("TensorRT engine build failed; no PyTorch fallback")
    Path(engine_path).write_bytes(bytes(plan))
    runtime=trt.Runtime(logger)
    engine=runtime.deserialize_cuda_engine(plan)
    inspector=engine.create_engine_inspector()
    Path(engine_path).with_suffix(".layers.json").write_text(
        inspector.get_engine_information(trt.LayerInformationFormat.JSON),encoding="utf-8")
    return {"tensorrt":trt.__version__,"strongly_typed":True,"tf32":tf32,
            "precision":"explicit FP32 entire engine, TF32 OFF; no BF16 Conv",
            "workspace_gib":workspace_gib,"builder_optimization_level":optimization_level,"max_aux_streams":0}


class TRTRunner:
    """Fixed FP32 input/output buffers. H2D excluded; D2D staging INCLUDED."""
    def __init__(self,path,sample,capture=False,input_names=None):
        import tensorrt as trt
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError("Cannot deserialize TensorRT engine")
        self.context=self.engine.create_execution_context()
        self.stream=torch.cuda.current_stream()
        self.input_names=tuple(input_names or E.INPUTS)
        self.inputs={k:sample[k].float().contiguous().clone() for k in self.input_names}
        expected=set(self.input_names)|{"D_full"}
        actual={self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if actual!=expected:raise RuntimeError(f"TensorRT IO mismatch: {actual}")
        for name,value in self.inputs.items():
            if self.engine.get_tensor_dtype(name)!=trt.float32:
                raise RuntimeError("Deployment IO must remain FP32")
            self.context.set_tensor_address(name,value.data_ptr())
        shape=tuple(self.context.get_tensor_shape("D_full"))
        if shape!=(1,1,352,1216):raise RuntimeError(f"Unexpected output shape {shape}")
        self.output=torch.empty(shape,device="cuda",dtype=torch.float32)
        self.context.set_tensor_address("D_full",self.output.data_ptr())
        self.graph=None
        for _ in range(3):self._enqueue()
        torch.cuda.synchronize()
        if capture:
            side=torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):self._enqueue()
            torch.cuda.current_stream().wait_stream(side)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=side):
                self._enqueue()
            torch.cuda.current_stream().wait_stream(side)

    def _enqueue(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def __call__(self,batch):
        for name in self.input_names:self.inputs[name].copy_(batch[name],non_blocking=True)
        if self.graph is not None:self.graph.replay()
        else:self._enqueue()
        return self.output


_BaseTRTRunner=TRTRunner
class TRTRunner(_BaseTRTRunner):
    """Same ordering/staging as historical FP32 recovery: dedicated non-default stream."""
    def __init__(self,*args,**kwargs):
        caller=torch.cuda.current_stream()
        self.work_stream=torch.cuda.Stream()
        self.work_stream.wait_stream(caller)
        with torch.cuda.stream(self.work_stream):super().__init__(*args,**kwargs)
        caller.wait_stream(self.work_stream)
    def __call__(self,batch):
        caller=torch.cuda.current_stream();self.work_stream.wait_stream(caller)
        with torch.cuda.stream(self.work_stream):value=super().__call__(batch)
        caller.wait_stream(self.work_stream)
        return value

class TorchGraph:
    def __init__(self,module,sample):
        self.module=module
        self.inputs={k:sample[k].clone() for k in E.INPUTS}
        side=torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):module(*(self.inputs[k] for k in E.INPUTS))
        torch.cuda.current_stream().wait_stream(side)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=side):
            self.output=module(*(self.inputs[k] for k in E.INPUTS))
        torch.cuda.current_stream().wait_stream(side)

    def __call__(self,batch):
        for name in E.INPUTS:self.inputs[name].copy_(batch[name],non_blocking=True)
        self.graph.replay()
        return self.output


def parity(actual,expected,cfg):
    error=(actual.float()-expected.float()).abs()
    mean,maximum=float(error.mean()),float(error.max())
    passed=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()) and (
        mean<=cfg["pixel_mean_atol_m"] and maximum<=cfg["pixel_max_atol_m"])
    return {"mean_abs_m":mean,"max_abs_m":maximum,"passed":passed}
