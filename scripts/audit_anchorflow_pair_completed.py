"""Audit the completed paired benchmark; export SMALL evidence, never weights/data.

Archive Python is inspected as bytes only, not imported/executed. The original
~1GB ZIP and prediction ZIPs stay local. Derived numbers preserve same-checkpoint
metric pairs; inference timing is isolated, training time is under contention.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/Compare_V10_1_V11_Fresh40_bf16_pair01-20261004T151154Z-1-001.zip'
FOLDERS={'V10_1':ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric',
         'V11':ROOT/'drive_upload/AnchorFlow_v11_NODE'}
PIXELS=25424992

def sha_file(path):
    with Path(path).open('rb') as f:return sha_stream(f)

def sha_stream(handle):
    h=hashlib.sha256()
    for block in iter(lambda:handle.read(2**20),b''):h.update(block)
    return h.hexdigest()

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,default=DEFAULT)
    ap.add_argument('--output',type=Path,default=ROOT/'results/benchmarks/v10_1_v11_pair01')
    args=ap.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Export must stay inside this repository')
    with ZipFile(args.archive) as z:
        roots={n.split('/')[0] for n in z.namelist() if n.endswith('/comparison_summary.json')}
        if len(roots)!=1:raise RuntimeError('Expected exactly one benchmark root')
        prefix=roots.pop()+'/'
        def raw(name):return z.read(prefix+name)
        def js(name):return json.loads(raw(name))
        def rows(name):return list(csv.DictReader(io.StringIO(raw(name).decode('utf-8'))))
        sources={};checkpoints={};metrics={};logs={};profiles={};budgets={}
        # Verify frozen archive files and exact equality to the local release.
        for label,folder in FOLDERS.items():
            stem='source_bundle/'+label+'/'
            manifest=js(stem+'bundle_manifest.json')
            for name,expected in manifest['files'].items():
                if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Unsafe frozen filename')
                if name.endswith('.ipynb'):continue # Mutable Colab UI was deliberately not frozen.
                value=raw(stem+name)
                if hashlib.sha256(value).hexdigest()!=expected:raise RuntimeError(f'Archive source checksum mismatch: {label}/{name}')
                if (folder/name).read_bytes()!=value:raise RuntimeError(f'Local release differs from trained source: {label}/{name}')
            sources[label]=manifest['source_sha256']
            base=label+'/dual_teacher/'
            metrics[label]=js(base+'val_metrics.json');logs[label]=rows(base+'train_log.csv')
            profiles[label]=js(base+'profile.json');budgets[label]=js(base+'training_status.json')
            if metrics[label]['final']['all']['pixels']!=PIXELS:raise RuntimeError('GT support changed')
            if metrics[label]['teacher_at_inference'] or metrics[label]['samples']!=400:raise RuntimeError('Invalid evaluation')
            if [int(r['epoch']) for r in logs[label]]!=list(range(34)):raise RuntimeError('Expected 34 completed epochs')
            if int(logs[label][-1]['global_step'])!=13600:raise RuntimeError('Training budget changed')
            if budgets[label]['status']!='early_stopped':raise RuntimeError('Unexpected training completion')
            if js(base+'run_manifest.json')['protocol']['source_sha256']!=sources[label]:raise RuntimeError('Run source lock changed')
            for action in ('evaluate','profile','test','evaluate_inverse','evaluate_joint'):
                record=js(base+action+'_artifact_record.json');key=base+record['checkpoint']
                if key not in checkpoints:
                    with z.open(prefix+key) as handle:checkpoints[key]=sha_stream(handle)
                if checkpoints[key]!=record['checkpoint_sha256']:raise RuntimeError('Checkpoint provenance changed')
            for selection in ('rmse','inverse','joint'):
                r=js(base+('val_metrics.json' if selection=='rmse' else f'val_metrics_{selection}.json'))
                epoch=r['checkpoint_epoch'];log=logs[label][epoch]
                if abs(float(log['val_rmse_m'])-r['final']['all']['rmse_m'])>1e-9:raise RuntimeError('Isolated RMSE differs from epoch log')
                if abs(float(log['val_irmse_km_inv'])-r['final']['all']['irmse_km_inv'])>1e-9:raise RuntimeError('Inverse checkpoint pair differs')
        for action in ('train','evaluate','test','profile'):
            r=js(action+'_pair_status.json')
            if r['status']!='completed' or any(p['returncode'] for p in r['results']):raise RuntimeError('Incomplete subprocess stage')
            if action!='train':
                r=sorted(r['results'],key=lambda p:p['started_unix'])
                if r[1]['started_unix']<r[0]['finished_unix']:raise RuntimeError('Final benchmark stages overlapped')
        execution_names=[n for n in z.namelist() if n.startswith(prefix+'execution_') and n.endswith('.json')]
        if len(execution_names)!=1:raise RuntimeError('Expected one execution inventory')
        execution=json.loads(z.read(execution_names[0]))
        if hashlib.sha256(raw('source_bundle/anchorflow_pair_benchmark.py')).hexdigest()!=execution['coordinator_sha256']:
            raise RuntimeError('Coordinator source does not match recorded execution')
        contract=js('data_contract.json')
        if (contract['train'],contract['val'],contract['test'])!=(1600,400,1000):raise RuntimeError('Split changed')
        if contract['validation_teacher_used'] or contract['relative_val_used']:raise RuntimeError('Teacher leakage')
        if contract['teacher_report']['cached_val'] or contract['relative_teacher']['cached_val']:raise RuntimeError('Validation teacher cache exists')
        for key in ('device','torch','precision','channels_last','runs','shape','batch','compiled_total'):
            if profiles['V10_1'][key]!=profiles['V11'][key]:raise RuntimeError('Profiler protocol mismatch: '+key)
        va,vb=(metrics[k]['final']['all'] for k in ('V10_1','V11'))
        fixed=js('V11/dual_teacher/val_metrics_fixed_midpoint8.json')
        fp=js('V11/dual_teacher/profile_fixed_midpoint8.json')
        tighter=js('V11/dual_teacher/val_metrics_adaptive_tighter_reference.json')
        for r in (fixed,tighter):
            if r['checkpoint_epoch']!=metrics['V11']['checkpoint_epoch'] or r['final']['all']['pixels']!=PIXELS:
                raise RuntimeError('Solver checkpoint/support mismatch')
        audit_record=js('V11/dual_teacher/solver_audit_artifact_record.json')
        if audit_record['checkpoint_sha256']!=checkpoints['V11/dual_teacher/best.pth']:raise RuntimeError('Solver weights mismatch')
        real_a=profiles['V10_1']['real_scene_profile'];real_b=profiles['V11']['real_scene_profile'];real_f=fp['real_scene_profile']
        solver=metrics['V11']['integration']
        shares={name:v['squared_error_fraction_global'] for name,v in metrics['V11']['final'].items() if name!='all'}
        weighted={}
        for label in FOLDERS:
            row=logs[label][metrics[label]['checkpoint_epoch']];total=float(row['loss_total'])
            weighted[label]={k.removeprefix('loss_weighted_'):{'weighted_value':float(v),'fraction_total':float(v)/total}
                for k,v in row.items() if k.startswith('loss_weighted_')}
        near=metrics['V11']['near_inverse_diagnostics'];tail=vb['error_tail']['10']
        report={'archive':args.archive.name,'archive_sha256':sha_file(args.archive),'audit_pass':True,
            'source_release_matches_trained_snapshots':sources,'checked_checkpoint_sha256':checkpoints,
            'train':1600,'val':400,'anonymous_test':1000,'valid_GT_pixels':PIXELS,'seed':42,
            'completed_epochs_per_model':34,'optimizer_updates_per_model':13600,
            'RMSE_V10_1_m':va['rmse_m'],'RMSE_V11_m':vb['rmse_m'],
            'V11_RMSE_gain_mm':1000*(va['rmse_m']-vb['rmse_m']),
            'V11_RMSE_gain_percent':100*(1-vb['rmse_m']/va['rmse_m']),
            'V11_metric_SSE_reduction_percent':100*(1-(vb['rmse_m']/va['rmse_m'])**2),
            'V11_iRMSE_change_km_inv':vb['irmse_km_inv']-va['irmse_km_inv'],
            'V11_adaptive_wall_latency_ratio':real_b['wall_median_ms']/real_a['wall_median_ms'],
            'V11_static_wall_latency_reduction_percent':100*(1-real_f['wall_median_ms']/real_b['wall_median_ms']),
            'V11_static_RMSE_minus_adaptive_m':fixed['final']['all']['rmse_m']-vb['rmse_m'],
            'default_and_tighter_metrics_exactly_equal':tighter['final']==metrics['V11']['final'],
            'validation_solver':solver,
            'all_epochs_train_NFE_means':[float(r['loss_dynamics_nfe']) for r in logs['V11']],
            'all_epochs_validation_NFE_means':[float(r['val_executed_nfe']) for r in logs['V11']],
            'all_epochs_train_rejected_step_means':[float(r['loss_dynamics_rejected_steps']) for r in logs['V11']],
            'GT_range_SSE_fractions':shares,
            'GT_boundary_3px_pixel_fraction':metrics['V11']['gt_boundary']['bands']['3']['pixels']/PIXELS,
            'GT_boundary_3px_SSE_fraction':metrics['V11']['gt_boundary']['bands']['3']['sse_fraction_global'],
            'tail_gt10m_pixel_fraction':tail['pixel_fraction'],'tail_gt10m_SSE_fraction':tail['sse_fraction'],
            'tail_gt10m_required_SSE_reduction_for_RMSE_0_9_all_else_equal':(vb['rmse_m']**2-.9**2)/tail['mse_contribution_m2'],
            'near_0_5_pixel_fraction':near['0-5']['pixels']/PIXELS,
            'near_0_5_inverse_SSE_fraction':near['0-5']['inverse_sse']/(vb['irmse_km_inv']**2*PIXELS),
            'weighted_loss_budgets_at_best_epoch':weighted,
            'interpretation_limits':['single seed, shared validation for selection, no independent test GT',
                'different chart/solver/horizon and fresh initialization order; not solver-only causality',
                'stage GT supports differ across scales; same-scale comparisons only',
                'training durations under contention are not isolated speed',
                'solver tolerances bound chart numerical error, not final metric depth error']}
        # Export only tabular/JSON evidence and original overview PNG. No weights,
        # predictions, personal paths or duplicated frozen code enter Git here.
        selected=[]
        for name in z.namelist():
            if not name.startswith(prefix):continue
            relative=name[len(prefix):]
            if '/' not in relative and relative.startswith('comparison_') and relative.endswith(('.csv','.json','.png')):selected.append(relative)
            if relative in ('COMPARISON.md','benchmark_recipe.json','gpu_before_isolated_profiles.json'):selected.append(relative)
        for label in FOLDERS:
            for name in ('train_log.csv','training_status.json','resolved_config.json','val_metrics.json',
                         'val_metrics_inverse.json','val_metrics_joint.json','profile.json','selection_metrics.json'):
                selected.append(label+'/dual_teacher/'+name)
        selected.extend('V11/dual_teacher/'+name for name in ('solver_comparison.json','profile_fixed_midpoint8.json',
                        'val_metrics_fixed_midpoint8.json','val_metrics_adaptive_tighter_reference.json'))
        selected.append(execution_names[0][len(prefix):])
        for name in selected:
            target=out/'artifacts'/name;value=raw(name)
            if target.exists() and target.read_bytes()!=value:raise RuntimeError('Existing audit evidence differs: '+str(target))
        for name in selected:
            target=out/'artifacts'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(raw(name))
        dump(out/'audit_summary.json',report)
        dump(out/'artifact_manifest.json',{name:hashlib.sha256(raw(name)).hexdigest() for name in selected})
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print('SMALL AUDIT OUTPUT:',out)

if __name__=='__main__':main()
