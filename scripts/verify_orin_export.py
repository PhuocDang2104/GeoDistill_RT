"""CPU structural export check, NOT a trained accuracy or Orin runtime claim."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'drive_upload/AnchorFlow_S4_UPIR_NoRelative_30ep_A100'
sys.path.insert(0, str(CODE))
import experiment as E
import torch
import onnx
from deployment import typed_deployment


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    cfg = json.loads((CODE/'config.json').read_text())
    model = E.make_model(cfg, 'G_s4_upir_norel').eval()
    before = state_hash(model)
    module = typed_deployment(model)
    spec = importlib.util.spec_from_file_location('orin_export_helper', ROOT/'production/AnchorFlow_S4_UPIR_Orin/export_assets.py')
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rgb = torch.rand(1, 3, 352, 1216)
    mask = (torch.rand(1, 1, 352, 1216) > .95).float()
    values = (rgb, mask*20, mask, torch.tensor([[[700.,0.,607.],[0.,700.,175.],[0.,0.,1.]]]))
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='anchorflow_onnx17_') as tmp, torch.inference_mode():
        original = module(*values)
        deployment_before = state_hash(module)
        path = Path(tmp)/'model.onnx'
        helper.export_onnx_eval(module, values, path, E.INPUTS)
        graph = onnx.load(str(path));onnx.checker.check_model(graph, full_check=True)
        assert graph.opset_import[0].version == 17
        assert list(x.name for x in graph.graph.input) == list(E.INPUTS)
        assert all(t.data_type != onnx.TensorProto.BFLOAT16 for t in graph.graph.initializer)
        assert state_hash(model) == before
        assert state_hash(module) == deployment_before
        assert all(not child.training for child in module.modules())
        torch.testing.assert_close(module(*values), original, atol=0, rtol=0)
        result = {'device': 'CPU', 'passed': True, 'parameters': sum(p.numel() for p in model.parameters()),
            'input_shape': list(rgb.shape), 'output_shape': list(original.shape), 'opset': 17,
            'schema_full_check': True, 'trained_weights_used': False, 'weights_unchanged': True,
            'production_helper_eval_bn_regression': True,
            'numerical_onnx_parity': 'Not run: target TensorRT versus exported real FP32 references is mandatory',
            'orin_runtime_tested': False, 'seconds': time.perf_counter()-start}
    (ROOT/'production/AnchorFlow_S4_UPIR_Orin/local_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
