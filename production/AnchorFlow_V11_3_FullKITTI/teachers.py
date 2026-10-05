"""Pinned official teacher networks. No use of the repo's legacy DMD3C wrapper."""
import sys
import types
import subprocess
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils import identity, sha256, numerical_environment
from full_data import SIZE

DMD_CODE = '2af8f0fad6ebe32d1f3aea9330acb0db17c88cec'
DMD_WEIGHTS_REV = '2128fd6e5d86282566767676397e419ac00e01b1'
DMD_WEIGHTS_SHA = 'e57228e1a4157c784bc98dd1a6156f4c8b8bd5db6698d917a4433b491f95e4fd'
MONO_CODE = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
MONO_REV = 'f465978e618db8cc79c83b8bbf24964857db1875'
METRIC_REV = '4010e39f3634a45bc60553321fb49fb760bd594e'


def native_cuda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Teacher flow requires a native BF16 CUDA GPU, not T4 BF16 emulation')


def teacher_recipe(cfg, role):
    return {'schema': 1, 'role': role, 'shape': list(SIZE), 'cache_dtype': 'float32',
            'numerical_environment': numerical_environment(),
            'generator_sha256': sha256(Path(__file__)),
            'extension_compat_sha256': sha256(Path(__file__).with_name('extension_compat.py')) if role=='metric' else None,
            'provider': ('Sharpiless/DMD3Cpp' if role == 'metric' else 'depth-anything/DA3MONO-LARGE'),
            'code_revision': DMD_CODE if role == 'metric' else MONO_CODE,
            'weights_revision': DMD_WEIGHTS_REV if role == 'metric' else MONO_REV,
            'weights_sha256': DMD_WEIGHTS_SHA if role == 'metric' else None,
            'internal_metric_foundation_revision': METRIC_REV if role == 'metric' else None,
            'relative_long_side': cfg['relative_long_side'] if role == 'relative' else None,
            'depth_convention': 'metres' if role == 'metric' else 'standardized_relative_inverse_depth_near_high',
            'inputs': 'RGB,S,K; train GT used ONLY for heuristic confidence' if role == 'metric' else 'RGB only',
            'confidence': 'local GT agreement; unsupported=.5; heuristic, not calibrated' if role == 'metric'
                          else 'horizontal-flip standardized inverse agreement, non-sky; heuristic',
            'student_preprocess': 'full-frame352x1216; RGB linear/depth nearest; scaled K'}


def load_da3(model_id, revision, code_root, preset, cache):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    sys.path.insert(0, str(code_root))
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    location = Path(snapshot_download(model_id, revision=revision, cache_dir=cache,
                                     allow_patterns=['config.json', 'model.safetensors']))
    import json
    if json.loads((location/'config.json').read_text())['model_name'] != preset:
        raise RuntimeError('Foundation preset mismatch')
    net = create_object(load_config(MODEL_REGISTRY[preset]))
    payload = load_file(str(location/'model.safetensors'))
    if not payload or not all(k.startswith('model.') for k in payload):
        raise RuntimeError('Official safetensors key convention changed')
    net.load_state_dict({k[6:]: v for k, v in payload.items()}, strict=True)
    return net.eval().cuda()


def verify_checkout(root, revision):
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:
        raise RuntimeError(f'Teacher code revision mismatch: {root}')
    subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD'],check=True)


def metric_confidence(depth, gt, valid):
    valid = valid.astype(np.float32)
    quality = np.exp(-np.abs(depth-gt)/(.5+.05*np.maximum(gt, .1))).astype(np.float32)*valid
    weight = cv2.boxFilter(valid, -1, (15, 15), normalize=False)
    numerator = cv2.boxFilter(quality, -1, (15, 15), normalize=False)
    c = np.where(weight > 0, numerator/np.maximum(weight, 1), .5)
    return np.where((depth > .1)&(depth < 120), c, 0).astype(np.float32)


class MetricTeacher:
    def __init__(self, cfg):
        native_cuda()
        root = Path(cfg['third_party_root']) / 'DMD3Cpp'
        verify_checkout(root,DMD_CODE)
        # Headless facade avoids optional exporter packages. Network, weights,
        # preprocessing and modified feature-return implementation remain official.
        # DMD3Cpp keeps DA3 in a Python list, outside its completion state_dict.
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root/'exts'))
        sys.path.insert(0,str(Path(cfg['third_party_root'])/'BpOps_build_torch210'))
        foundation = load_da3('depth-anything/DA3METRIC-LARGE', METRIC_REV, root,
                              'da3metric-large', cfg['hf_cache'])
        class Facade:
            device = torch.device('cuda')
            @classmethod
            def from_pretrained(cls, name):
                if name != 'depth-anything/DA3METRIC-LARGE':
                    raise RuntimeError('Unexpected internal metric foundation')
                return cls()
            def requires_grad_(self, value):
                foundation.requires_grad_(value); return self
            def to(self, device):
                foundation.to(device); self.device = torch.device(device); return self
            def _run_model_forward(self, image, *args):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return foundation(image, *args)
        api = types.ModuleType('depth_anything_3.api'); api.DepthAnything3 = Facade
        sys.modules['depth_anything_3.api'] = api
        import BpOps  # compiled for this exact image/torch/CUDA architecture
        from models import Pre_MF_Post_Residual_Norm_fast
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download('Liangyingping/DMD3Cpp-checkpoints', 'result_ema.pth',
                          repo_type='dataset', revision=DMD_WEIGHTS_REV, cache_dir=cfg['hf_cache']))
        if sha256(checkpoint) != DMD_WEIGHTS_SHA:
            raise RuntimeError('DMD3Cpp completion checkpoint SHA mismatch')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        for key in ('state_dict', 'model', 'net', 'net_ema'):
            if isinstance(state, dict) and key in state:
                state = state[key]; break
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.net = Pre_MF_Post_Residual_Norm_fast()
        self.net.load_state_dict(state, strict=True)  # No silent partial teacher load.
        self.net.eval().cuda()
        self.mean = torch.tensor([90.9950,96.2278,94.3213], device='cuda')[None,:,None,None]
        self.std = torch.tensor([79.2382,80.5267,82.1483], device='cuda')[None,:,None,None]

    @torch.inference_mode()
    def predict(self, sample):
        image = sample['rgb'][None].cuda()*255
        sparse = sample['sparse'][None].cuda(); k = sample['K'][None].cuda()
        # Completion/BpOps FP32; internal official DA3 BF16. Do not autocast BpOps.
        output = self.net((image-self.mean)/self.std, sparse, k)
        d = output[-1][0,0].float().cpu().numpy()
        if d.shape != SIZE or not np.isfinite(d).all():
            raise RuntimeError('Nonfinite/invalid official metric teacher output')
        d = np.where((d > .1)&(d < 120), d, 0).astype(np.float32)
        c = metric_confidence(d, sample['gt'][0].numpy(), sample['gt_mask'][0].numpy())
        return np.stack((d,c))


def standardize(depth, sky):
    valid = np.isfinite(depth)&(depth > 1e-6)
    if sky is not None:
        valid &= ~sky
    if valid.sum() < 1024:
        raise RuntimeError('Relative model has insufficient non-sky support')
    inv = np.zeros_like(depth, np.float32); inv[valid] = 1/depth[valid]
    std = inv[valid].std()
    if std < 1e-8:
        raise RuntimeError('Constant inverse depth')
    return np.where(valid, (inv-inv[valid].mean())/std, 0).astype(np.float32), valid


class RelativeTeacher:
    def __init__(self, cfg):
        native_cuda(); self.cfg = cfg
        root = Path(cfg['third_party_root']) / 'Depth-Anything-3' / 'src'
        verify_checkout(root.parent,MONO_CODE)
        self.net = load_da3('depth-anything/DA3MONO-LARGE', MONO_REV, root, 'da3mono-large', cfg['hf_cache'])

    @torch.inference_mode()
    def once(self, rgb):
        h,w = rgb.shape[:2]; scale = self.cfg['relative_long_side']/max(h,w)
        size = (max(14,round(h*scale/14)*14),max(14,round(w*scale/14)*14))
        x = torch.from_numpy(cv2.resize(rgb,size[::-1],interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy()).cuda().float()/255
        x = ((x-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None])[None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out = self.net(x,infer_gs=False,use_ray_pose=False)
        d = F.interpolate(out['depth'].float().reshape(1,1,*size),size=SIZE,mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        sky = out.get('sky')
        if sky is not None:
            sky = F.interpolate(sky.float().reshape(1,1,*size),size=SIZE,mode='bilinear',align_corners=False)[0,0].cpu().numpy() >= .3
        return d,sky

    def predict(self,rgb):
        d,sky = self.once(rgb); f,fs = self.once(rgb[:,::-1].copy())
        a,ma = standardize(d,sky); b,mb = standardize(f[:,::-1],None if fs is None else fs[:,::-1])
        valid = ma&mb
        return np.stack((np.where(valid,(a+b)*.5,0).astype(np.float32),
                         (np.exp(-np.abs(a-b))*valid).astype(np.float32)))


def build_teacher(cfg,role):
    return MetricTeacher(cfg) if role=='metric' else RelativeTeacher(cfg)
