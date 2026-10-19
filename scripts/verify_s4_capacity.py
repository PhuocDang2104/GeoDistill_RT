"""Portable CPU contracts + full-resolution ONNX Runtime numerical checks."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'drive_upload/AnchorFlow_S4_UPIR_DecoderCapacity_30ep_A100'


def main():
    result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'discover',
        '-s', str(CODE), '-p', 'test_contracts.py', '-v'], capture_output=True, text=True,
        encoding='utf-8', cwd=ROOT)
    print(result.stdout+result.stderr)
    if result.returncode: raise SystemExit(result.returncode)
    sys.path.insert(0, str(CODE))
    import numpy as np
    import torch
    import onnxruntime as ort
    import experiment as E
    from deployment import typed_deployment, export_onnx
    torch.set_num_threads(2)
    cfg = json.loads((CODE/'config.json').read_text())
    exports = {}
    for variant in E.VARIANTS:
        model = E.make_model(cfg, variant).eval()
        # Exercise active new lanes and U-PIR, not only their zero-initial states.
        with torch.no_grad():
            model.detail1.pir.body[0].weight[:,40:].fill_(.1)
            model.detail1.pir.head.weight.normal_(std=.01)
            for path in E.TARGETS[variant]:
                block = model.decoder.get_submodule(path)
                channels = block.body[0][0].in_channels
                block.body[2][0].weight[:,2*channels:].normal_(std=.001)
        module = typed_deployment(model)
        rgb = torch.rand(1,3,352,1216)
        mask = (torch.rand(1,1,352,1216)>.95).float()
        values = (rgb,mask*20,mask,torch.tensor([[[700.,0.,607.],[0.,700.,175.],[0.,0.,1.]]]))
        with tempfile.TemporaryDirectory(prefix='s4_capacity_onnx_') as tmp, torch.no_grad():
            path = Path(tmp)/'model.onnx'
            expected = module(*values).numpy()
            export_onnx(module,dict(zip(E.INPUTS,values)),path)
            options = ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            actual = session.run(None,{k:v.numpy() for k,v in zip(E.INPUTS,values)})[0]
            error = np.abs(actual-expected)
            mean, maximum = float(error.mean()), float(error.max())
            if not np.isfinite(actual).all() or mean > .001 or maximum > .06:
                raise RuntimeError(f'CPU ONNX parity failed {variant}: {mean}/{maximum}')
            exports[variant] = {'mean_abs_m':mean,'max_abs_m':maximum,'passed':True,
                'shape':list(actual.shape),'opset':22,'new_lanes_and_upir_active':True}
    report = {'date':'2026-10-19','device':'CPU','passed':True,
        'contract_tests':int(re.search(r'Ran (\d+) tests',result.stderr)[1]),
        'onnxruntime':ort.__version__,'full_resolution_onnx_numerical_parity':exports,
        'not_verified':['A100 training','TensorRT parser/build/capture','trained accuracy','A100/Orin runtime'],
        'baseline_G_untouched':True,'relative_teacher_modules_packaged':False}
    (CODE/'local_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    # Refresh the manifest after the verification report is complete.
    from package_s4_capacity import package
    package()


if __name__=='__main__': main()
