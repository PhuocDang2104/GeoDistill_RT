"""Inference-only, same-checkpoint solver-cap audit; never warm-starts training."""
import csv
import json
import time
from pathlib import Path
import torch
from data import digest,write_json

ORIGINAL_SOURCE='2d01818cbbb93b873c9ef5d619a2e9ac3c7fe8c81857c35a3ac93c184156ac3d'
CASES=(('legacy_cap025','bosh3',.25,True),
       ('cap025_dense','bosh3',.25,False),
       ('cap05_dense','bosh3',.5,False),
       ('cap1_dense','bosh3',1.,False),
       ('fixed_midpoint8','midpoint',.25,False))


@torch.inference_mode()
def audit_model(model,checkpoint,config,variant,folder,checkpoint_sha):
    import run
    device=next(model.parameters()).device
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    records=[];reference=None
    for label,method,cap,forced in CASES:
        node=model.dynamics
        node.method,node.first_step,node.max_step=method,.25,cap
        node.rtol,node.atol,node.fixed_steps=.01,.001,4
        node.force_observation_steps=forced
        node.max_nfe,node.max_num_steps=193,64
        start=time.perf_counter()
        result=run.validate(model,run.data_loader(config,'val'),config,device)
        assert result['final']['all']['pixels']==25424992,'GT support changed'
        model.set_diagnostics(False)
        # Warm identical B1 production path; wall timing excludes metric/I/O/H2D.
        inputs=run.sample_inputs(device,channels_last=config['channels_last'])
        for _ in range(config.get('profile_warmup',30)):
            with run.autocast(config,device):model(*inputs)
        if device.type=='cuda':torch.cuda.synchronize()
        timing=run.profile_real(model,config,device,False)
        result.update(checkpoint_epoch=checkpoint['epoch'],checkpoint_sha256=checkpoint_sha,
                      solver_label=label,first_step=.25,max_step=cap,
                      force_observation_steps=forced,teacher_at_inference=False,
                      samples=400,real_scene_profile=timing,
                      fine_node_enabled=model.fine_node is not None,
                      audit_seconds_including_evaluation_metrics=time.perf_counter()-start)
        write_json(folder/f'val_metrics_{label}.json',result)
        score=result['final']['all']
        if reference is None:reference=score
        records.append({'solver':label,'epoch':checkpoint['epoch'],'max_step':cap,
            'forced_half_endpoint':forced,'RMSE_m':score['rmse_m'],
            'iRMSE_km_inv':score['irmse_km_inv'],'MAE_m':score['mae_m'],
            'delta_RMSE_m_vs_legacy025':score['rmse_m']-reference['rmse_m'],
            'delta_iRMSE_vs_legacy025':score['irmse_km_inv']-reference['irmse_km_inv'],
            'nfe_mean':result['integration']['nfe_mean'],'nfe_p95':result['integration']['nfe_p95'],
            'rejects_mean':result['integration']['rejected_steps_mean'],
            'median_ms':timing['wall_median_ms'],'p95_ms':timing['wall_p95_ms'],
            'profile_nfe_mean':timing['solver']['nfe_mean']})
        records[-1].update(fine_nfe=result['fine_integration']['nfe'],
                           total_field_nfe_mean=result['total_field_nfe_mean'],
                           D2_pre_fine_RMSE_m=result['stage_native_gt_metrics']['D2_pre_fine']['rmse_m'],
                           D2_RMSE_m=result['stage_native_gt_metrics']['D2']['rmse_m'])
        print(json.dumps(records[-1]),flush=True)
    with (folder/'solver_comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(folder/'solver_comparison.json',{'same_checkpoint':True,'checkpoint_sha256':checkpoint_sha,
        'checkpoint_epoch':checkpoint['epoch'],'rows':records,'gt_pixel_support':25424992,
        'time_points':[0,.5,1.],'first_step_fixed':.25,'rtol':.01,'atol':.001,
        'no_posthoc_training_config_change':True,'hardware':run.runtime_environment(),
        'interpretation':'Cap and observation endpoint policy are named separately. Dense half observation is NOT a forced step. No guaranteed 13->10->7; acceptance/rejects are measured.'})
    # Caller does not subsequently use this model for training/resume.
    return records


def baseline_audit(config,variant,path):
    import run
    path=Path(path)
    expected=config['reference_checkpoint_sha256']
    if digest(path)!=expected:raise RuntimeError('Original V11 reference checkpoint SHA mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    original=checkpoint['protocol']
    if original['source_sha256']!=ORIGINAL_SOURCE:raise RuntimeError('Not the audited original V11 checkpoint')
    if original['data']['manifest_sha256']!=config['expected_subset_sha256']:
        raise RuntimeError('V11/subset protocol mismatch')
    for key in ('encoder','flow_steps','phase_context_enabled','amp','channels_last'):
        if original['config'][key]!=config[key]:raise RuntimeError(f'Reference input/runtime differs: {key}')
    if checkpoint['epoch']!=29:raise RuntimeError('Expected V11 primary epoch29 reference')
    device=run.device_setup(config,require_gpu=True)
    # Original V11 has no fine-node weights. Do NOT silently random-initialize
    # extra heads for a same-checkpoint original-model audit.
    model=run.make_model({**config,'fine_node_enabled':False},device,False).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    _,drive=run.directories(config,variant)
    return audit_model(model,checkpoint,config,variant,drive/'original_v11_solver_audit',expected)


def trained_solver_audit(config,variant):
    import run
    device=run.device_setup(config,require_gpu=True)
    model,checkpoint=run.load_trained(config,variant,device)
    _,drive=run.directories(config,variant)
    names={'rmse':'best.pth','inverse':'best_inverse.pth','joint':'best_joint.pth','policy':'best_policy.pth'}
    checkpoint_sha=digest(drive/names[config.get('checkpoint_selection','policy')])
    return audit_model(model,checkpoint,config,variant,drive/'v11_2_solver_audit',checkpoint_sha)
