"""Read actual completed reports; never infer metrics from training loss."""
import json
import zipfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/anchorflow_v9_1_completed_audit'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'results/dual_teacher-20261004T054209Z-1-001.zip'
    with zipfile.ZipFile(path) as archive:
        for item in archive.namelist():
            p=Path(item)
            if p.parent.as_posix()=='dual_teacher' and (p.suffix in ('.json','.csv','.jsonl','.log') or p.name=='best.pth'):
                (OUT/p.name).write_bytes(archive.read(item))
    versions={label:ROOT/'results'/folder for label,folder in (
        ('V8','anchorflow_v8_completed_audit'),('V9','anchorflow_v9_completed_audit'),('V9.1','anchorflow_v9_1_completed_audit'))}
    reports={k:json.loads((v/'val_metrics.json').read_text()) for k,v in versions.items()}
    profiles={k:json.loads((v/'profile.json').read_text()) for k,v in versions.items()}
    rows=[]
    def row(name,values): rows.append({'metric':name,**values})
    for key in ('rmse_m','mae_m','irmse_km_inv','imae_km_inv','abs_rel','delta1','pixels'):
        row(key,{k:r['final']['all'][key] for k,r in reports.items()})
    for key in ('0-20','20-40','40-60','60-80','80-120','edge','non_edge'):
        row('RMSE '+key,{k:r['final'][key]['rmse_m'] for k,r in reports.items()})
    for band in ('1','3','5'):
        row('boundary '+band,{k:r['gt_boundary']['bands'][band]['rmse_m'] for k,r in reports.items()})
    for threshold in ('2','5','10','20'):
        for key in ('sse_m2','pixel_fraction'):
            row('tail>'+threshold+' '+key,{k:r['final']['all']['error_tail'][threshold][key] for k,r in reports.items()})
    for key in ('total_parameters','total_conv_linear_macs','median_ms','p95_ms','peak_cuda_allocated_mib'):
        row(key,{k:r[key] for k,r in profiles.items()})
    frame=pd.DataFrame(rows)
    frame['V9.1_vs_V9_percent']=100*(frame['V9.1']/frame['V9']-1)
    frame['V9.1_vs_V8_percent']=100*(frame['V9.1']/frame['V8']-1)
    frame.to_csv(OUT/'comparison_v8_v9_v9_1.csv',index=False)
    stage_rows=[]
    for name in reports['V9.1']['stage_native_gt_metrics']:
        stage_rows.append({'stage':name,**{k:r['stage_native_gt_metrics'].get(name,{}).get('rmse_m') for k,r in reports.items()}})
    pd.DataFrame(stage_rows).to_csv(OUT/'stage_comparison.csv',index=False)
    log=pd.read_csv(OUT/'train_log.csv')
    best=log.loc[log.val_rmse_m.idxmin()]
    budget=best[[c for c in log if c.startswith('loss_weighted_')]]
    budget.to_frame('weighted_value').assign(fraction=budget/best.loss_total).to_csv(OUT/'best_loss_budget.csv')
    initial=json.loads((OUT/'initial_val_metrics.json').read_text())
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    summary={'source_archive':str(path),'best_epoch':reports['V9.1']['checkpoint_epoch'],
             'initial_rmse':initial['final']['all']['rmse_m'],
             'best_rmse':reports['V9.1']['final']['all']['rmse_m'],
             'epochs_completed':len(log),'config':manifest['protocol']['config'],
             'source_sha256':manifest['protocol']['source_sha256'],
             'runtime_backend':profiles['V9.1'].get('runtime_backend'),
             'historical_runtime_not_backend_locked':True,
             'fresh_comparison_not_established':True,
             'required_SSE_reduction_to_0_9':1-(.9/reports['V9.1']['final']['all']['rmse_m'])**2,
             'quarter_stage2_vs_stage3_final_monotonicity_not_proven':True}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(frame.to_string(index=False)); print(pd.DataFrame(stage_rows).to_string(index=False))
    print('epoch log:',log[[c for c in ('epoch','val_rmse_m','val_mae_m','loss_total','loss_rmse','lr') if c in log]].to_string(index=False))
    print('summary:',json.dumps({k:v for k,v in summary.items() if k not in ('config','runtime_backend')},indent=2))
    for name in ('export_report.json','training_status.json'):
        data=json.loads((OUT/name).read_text())
        if name=='export_report.json': data={k:v for k,v in data.items() if k!='parity_cases'}
        print(name,json.dumps(data,indent=2))

if __name__=='__main__': main()
