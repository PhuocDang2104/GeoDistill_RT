"""Never silently relabel historical latency as a new same-session measurement."""
import csv
import json
from pathlib import Path
import torch
import experiment as E
from data import digest,write_json

HISTORY_TAG='benchmark_trt_fp32_diagnostic_6cm_v2'
ARMS=('A_s4_rk2','B_s4_rescnn','C_s4_phasecnn','D_s4_nofine','E_s4_jpir')
IGNORED_RECIPE={'experiment','run_name','drive_data','drive_runs','work','workers',
                'prefetch_factor','fine_mode','historical_run','historical_d_run','benchmark',
                'jpir_uncertainty','epochs','historical_e_run','upir_uncertainty_scale','acceptance',
                'relative_enabled','relative_weight'}

def recipe(cfg):return {k:v for k,v in cfg.items() if k not in IGNORED_RECIPE}

def metric_data_contract(data):
    # Only relative cache metadata is absent by design. Do NOT relax the
    # train/val split, original sensor/GT, metric teacher content or other gates.
    return {k:v for k,v in data.items() if k not in
            ('relative_teacher','relative_val_used','min_non_gt_relative_pixels')}

def arm_paths(cfg,variant):
    if variant=='D_s4_nofine':
        root=Path(cfg['historical_d_run'])/variant
        return root,root
    if variant=='E_s4_jpir':
        root=Path(cfg['historical_e_run'])/variant
        return root,root
    root=Path(cfg['historical_run'])/variant
    return root,root/HISTORY_TAG

def acceptance(row,cfg):
    t=cfg['acceptance']
    checks={'rmse_not_worse_than_D':row['rmse_m']<=t['rmse_m'],
            'irmse_not_worse_than_D':row['irmse_km_inv']<=t['irmse_km_inv'],
            'wall_median_within_target':row['median_ms']<=t['median_ms']}
    return dict(checks,all_targets_met=all(checks.values()),limits=t,
        interpretation='Historical D targets, not same-session speed proof or accuracy guarantee')

def report_row(report):
    all_=report['trt_graph']['all']
    wall=report['timing']['trt_graph']['synchronized_wall']
    return dict(variant=report['variant'],checkpoint_epoch=report['checkpoint_epoch'],
        rmse_m=all_['rmse_m'],irmse_km_inv=all_['irmse_km_inv'],mae_m=all_['mae_m'],
        boundary3_rmse_m=report['trt_graph_boundary']['bands']['3']['rmse_m'],
        median_ms=wall['median_ms'],p95_ms=wall['p95_ms'],parameters=report['parameters'],
        pixel_max_gate_m=report['parity_gates']['pixel_max_atol_m'],
        training='subset1600_fresh'+str(report.get('training_epochs',30)))

def compare_history(cfg):
    from run import run_dir,contract
    root=Path(cfg['drive_runs'])/cfg['run_name']
    d=json.loads((run_dir(cfg,E.VARIANTS[0])/'trt_benchmark.json').read_text())
    if not d['qualification_passed']:raise RuntimeError('D inference not qualified')
    ckpt_path=run_dir(cfg,E.VARIANTS[0])/'best_policy.pth'
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    if ckpt['contract']!=contract(cfg,E.VARIANTS[0]):raise RuntimeError('D checkpoint protocol changed')
    if digest(ckpt_path)!=d['checkpoint_sha256']:raise RuntimeError('D checkpoint/report mismatch')
    own_data=ckpt['contract']['data']
    own_init=ckpt['state']['initial_shared_sha256']
    reference=json.loads((E.ROOT/'historical_reference.json').read_text())
    rows=[];warnings=[]
    history=Path(cfg['historical_run'])
    for variant in ARMS:
        arm_root,report_root=arm_paths(cfg,variant)
        path=report_root/'trt_benchmark.json'
        if path.is_file():
            report=json.loads(path.read_text())
            old=torch.load(arm_root/'best_policy.pth',map_location='cpu',weights_only=False)
            if not report['qualification_passed']:raise RuntimeError(f'Unqualified history: {variant}')
            precision=(report.get('recovery',{}).get('precision') or report.get('inference_precision'))
            if precision!='explicit_FP32_TF32_OFF':raise RuntimeError('Historical precision differs')
            if metric_data_contract(old['contract']['data'])!=metric_data_contract(own_data):raise RuntimeError('Historical split/metric-teacher/data contract differs')
            if recipe(old['contract']['config'])!=recipe(cfg):raise RuntimeError('Historical training recipe differs beyond declared epoch budget/architecture')
            expected_epochs=26 if variant=='E_s4_jpir' else 30
            if old['contract']['config']['epochs']!=expected_epochs:raise RuntimeError('Unexpected historical training budget')
            if old['state']['initial_shared_sha256']!=own_init or report['initial_shared_sha256']!=own_init:
                raise RuntimeError('Pretrained common initial weights differ; not matched ablation')
            if digest(arm_root/'best_policy.pth')!=report['checkpoint_sha256']:
                raise RuntimeError('Historical report/checkpoint hash mismatch')
            for key in ('warmup','iterations','timed_images','pixel_mean_atol_m','pixel_max_atol_m','rmse_drift_m','irmse_drift_km_inv'):
                if report['parity_gates'][key]!=cfg['benchmark'][key]:raise RuntimeError(f'Historical benchmark gate differs: {key}')
            row=report_row(report)
            same_relative=(old['contract']['config'].get('relative_enabled')==cfg['relative_enabled'] and
                           old['contract']['config'].get('relative_weight')==cfg['relative_weight'])
            row.update(evidence='Split/metric teacher/shared-init verified; relative-KD treatment differs',matched_training_verified=expected_epochs==cfg['epochs'] and same_relative,shared_data_init_verified=True)
            for key in ('gpu','torch','cuda'):
                if report.get(key)!=d.get(key):warnings.append(f'{variant}: {key} differs from D; historical timing is not environment-matched')
            del old
        else:
            row=dict(next(r for r in reference['rows'] if r['variant']==variant))
            row.update(evidence='user-log snapshot; live Drive report missing',matched_training_verified=False,shared_data_init_verified=False)
            warnings.append(f'{variant}: numeric snapshot only; initialization/data not independently verified here')
        rows.append(row)
    own_row=report_row(d)
    own_row.update(evidence='new UPIR NoRelative Fresh30 full400 TRT graph measurement',matched_training_verified=True,shared_data_init_verified=True)
    rows.append(own_row)
    msg=dict(next(r for r in reference['rows'] if r['variant']=='MSG_CHN_pretrained'))
    msg.update(evidence='user-log snapshot; official pretrained',matched_training_verified=False,shared_data_init_verified=False)
    rows.append(msg)
    a=rows[0]
    for row in rows:
        row['relative_kd']='off' if row['variant']==E.VARIANTS[0] else ('not_comparable' if row['variant']=='MSG_CHN_pretrained' else 'on')
        row.update(fps=1000/row['median_ms'],
            latency_reduction_vs_A_pct=100*(1-row['median_ms']/a['median_ms']),
            rmse_improvement_vs_A_pct=100*(1-row['rmse_m']/a['rmse_m']),
            irmse_improvement_vs_A_pct=100*(1-row['irmse_km_inv']/a['irmse_km_inv']))
    fields=list(rows[0])
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    target=acceptance(own_row,cfg)
    write_json(root/'acceptance.json',target)
    write_json(root/'comparison.json',{'rows':rows,'warnings':warnings,'acceptance':target,
        'historical_snapshot_sha256':digest(E.ROOT/'historical_reference.json'),
        'note':'UPIR NoRelative fresh30: architecture identical to dual-teacher UPIR but relative loss/cache removed. Old A-E have relative KD ON, so NOT identical-loss comparisons; E also26vs30. Split/metric-teacher/shared-init checks remain enforced. Historical timing not same-session; MSG not training-matched. 6cm parity unchanged. No accuracy/runtime improvement assumed.'})
    try:
        import cv2
        import numpy as np
        previews=root/'comparison_previews';previews.mkdir(exist_ok=True)
        for index in range(0,1000,100):
            own=run_dir(cfg,E.VARIANTS[0])/'test_preview'
            pair=cv2.imread(str(own/f'{index:04d}_rgb_depth.png'))
            panels=[('RGB',pair[:,:1216])]
            # Original outputs use the same fixed scene indices; verify sample IDs.
            new_manifest=json.loads((own/'manifest.json').read_text())['images']
            new_id=next(x['sample_id'] for x in new_manifest if x['index']==index)
            for variant in ARMS:
                folder=arm_paths(cfg,variant)[1]/'test_preview'
                manifest=folder/'manifest.json'
                if not manifest.is_file():continue
                old_images=json.loads(manifest.read_text())['images']
                old_id=next(x['sample_id'] for x in old_images if x['index']==index)
                if old_id!=new_id:raise RuntimeError('Preview scene ID mismatch')
                image=cv2.imread(str(folder/f'{index:04d}_depth.png'))
                if image is not None:panels.append((variant,image))
            panels.append(('G_s4_upir_norel',pair[:,1216:]))
            named=[]
            for label,image in panels:
                bar=np.full((32,image.shape[1],3),245,np.uint8)
                cv2.putText(bar,label,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1)
                named.append(np.concatenate([bar,image]))
            cv2.imwrite(str(previews/f'{index:04d}_comparison.png'),np.concatenate(named))
    except ImportError:pass
    print(json.dumps(rows,indent=2))
    print('Comparison:',root/'comparison.csv')
