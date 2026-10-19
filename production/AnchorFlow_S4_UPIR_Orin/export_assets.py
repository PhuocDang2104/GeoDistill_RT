"""Run on the ORIGINAL Colab runtime, outside frozen CODE. No training.

Exports FP32 opset17 for JetPack6/TRT10, plus real validation inputs/references.
Never edits the checkpoint, training source, config or historical benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', type=Path, default=Path('/content/anchorflow_s4_upir_norel_code'))
    p.add_argument('--config', type=Path, required=True, help='Original resolved runtime config JSON')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, choices=(2, 100, 400), default=100)
    p.add_argument('--audit-upir', action='store_true', help='Full400 same-checkpoint cue-removal audit')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new output directory; do not overwrite previous evidence')
    if a.code.resolve() == a.output.resolve() or a.code.resolve() in a.output.resolve().parents:
        raise ValueError('Output must be OUTSIDE frozen training CODE')
    sys.path.insert(0, str(a.code.resolve()))
    import experiment as E
    import run
    import torch
    import numpy as np
    import onnx
    from deployment import typed_deployment
    from metrics import Metrics
    from boundaries import BoundaryMetrics
    cfg = json.loads(a.config.read_text())
    device = run.setup(cfg)
    variant = 'G_s4_upir_norel'
    # Retain strict source/config/data checks; never replace source hash in ckpt.
    model, ckpt, checkpoint = run.load_checkpoint(cfg, variant, device)
    module = typed_deployment(model)
    a.output.mkdir(parents=True)
    (a.output/'samples').mkdir()
    files, audit = {}, None
    if a.audit_upir:
        import copy
        removed = copy.deepcopy(module)
        with torch.no_grad():
            removed.model.detail1.pir.body[0].weight[:, 40:].zero_()
        metrics = [Metrics(device), Metrics(device)]
        boundaries = [BoundaryMetrics(device), BoundaryMetrics(device)]
        output_difference_sum, output_pixels = 0., 0
    with torch.inference_mode():
        for i, batch in enumerate(run.loader(cfg, 'val')):
            if i >= a.frames and not a.audit_upir:
                break
            batch = run.move(batch, device, cfg)
            values = tuple(batch[k].float().contiguous() for k in E.INPUTS)
            pred = module(*values)
            if i == 0:
                target = a.output/'model_fp32_op17.onnx'
                torch.onnx.export(module, values, str(target), input_names=list(E.INPUTS),
                    output_names=['D_full'], opset_version=17, dynamo=False,
                    dynamic_axes=None, do_constant_folding=True)
                onnx.checker.check_model(onnx.load(str(target)), full_check=True)
                files[target.name] = sha(target)
            if i < a.frames:
                arrays = {k: v.cpu().numpy() for k, v in zip(E.INPUTS, values)}
                arrays['reference'] = pred.float().cpu().numpy()
                arrays['gt'] = batch['gt'].float().cpu().numpy()
                arrays['gt_mask'] = batch['gt_mask'].float().cpu().numpy()
                path = a.output/'samples'/f'{i:04d}.npz'
                np.savez_compressed(path, **arrays)
                files[path.relative_to(a.output).as_posix()] = sha(path)
            if a.audit_upir:
                control = removed(*values)
                for j, prediction in enumerate((pred, control)):
                    metrics[j].update(prediction, batch['gt'], batch['gt_mask'], batch['rgb'])
                    boundaries[j].update(prediction, batch['gt'], batch['gt_mask'])
                output_difference_sum += float((pred-control).abs().sum())
                output_pixels += pred.numel()
            if (i+1) % 20 == 0:
                print(f'Export/audit {i+1}/400', flush=True)
    if a.audit_upir:
        audit = {'original': metrics[0].report(), 'cue_removed': metrics[1].report(),
            'original_boundary': boundaries[0].report(), 'cue_removed_boundary': boundaries[1].report(),
            'mean_output_change_m': output_difference_sum/output_pixels,
            'interpretation': 'Same trained checkpoint, extra U-column zeroed in COPY only. '
                'Measures reliance at inference, not benefit versus retraining no-UPIR.'}
        (a.output/'upir_intervention.json').write_text(json.dumps(audit, indent=2))
        files['upir_intervention.json'] = sha(a.output/'upir_intervention.json')
    manifest = {'schema': 1, 'variant': variant, 'checkpoint_sha256': sha(checkpoint),
        'source_sha256': ckpt['contract']['source_sha256'], 'training_contract': ckpt['contract'],
        'selection': ckpt['state']['selection'], 'frames': a.frames,
        'precision': 'FP32; TF32 OFF; new opset17 export, NOT A100 opset22 engine',
        'input_contract': 'B1 NCHW float32 RGB[0,1], sparse meters, binary mask, adjusted K[1,3,3]',
        'sample_order': 'Original validation DataLoader, shuffle=False; filenames are ordered indices',
        'files': files}
    (a.output/'io_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Copy this asset folder to Orin:', a.output)
    print('Checkpoint and teachers are NOT required on Orin. ONNX needs Orin parity qualification.')


if __name__ == '__main__':
    main()
