"""Extract small actual run artifacts and compare policies/history without inference claims."""
import json
import zipfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/anchorflow_v10_completed_audit'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'results/dual_teacher-20261004T093136Z-1-001.zip'
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            p=Path(name)
            if p.parent.as_posix()=='dual_teacher' and (p.suffix in ('.json','.csv','.jsonl','.log') or p.name=='best.pth'):
                (OUT/p.name).write_bytes(archive.read(name))
    read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
    current=read('val_metrics.json');policies=read('policy_audit.json')['reports']
    history={name:json.loads((ROOT/'results'/folder/'val_metrics.json').read_text()) for name,folder in (
        ('V8','anchorflow_v8_completed_audit'),('V9','anchorflow_v9_completed_audit'),('V9.1','anchorflow_v9_1_completed_audit'))}
    history.update({'V10':current,**{'V10 '+k:v for k,v in policies.items()}})
    rows=[]
    for name,r in history.items():
        a=r['final']['all'];row={'run':name,**{k:a[k] for k in ('rmse_m','mae_m','irmse_km_inv','imae_km_inv','abs_rel','delta1','pixels')}}
        row.update({f'RMSE_{band}':r['final'][band]['rmse_m'] for band in ('0-20','20-40','40-60','60-80','80-120','edge','non_edge')})
        row['boundary3_rmse']=r['gt_boundary']['bands']['3']['rmse_m']
        row.update({f'tail>{t}_{key}':a['error_tail'][str(t)][key] for t in (2,5,10,20) for key in ('pixel_fraction','sse_m2','sse_fraction')})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'comparison_history_policies.csv',index=False)
    log=pd.read_csv(OUT/'train_log.csv');best=log.loc[log.val_rmse_m.idxmin()]
    budget=best[[c for c in log if c.startswith('loss_weighted_')]]
    budget.to_frame('weighted_value').assign(fraction=budget/best.loss_total).to_csv(OUT/'best_loss_budget.csv')
    stages=pd.DataFrame({name:{k:v['rmse_m'] for k,v in r['stage_native_gt_metrics'].items()} for name,r in history.items()})
    stages.to_csv(OUT/'stage_history_policies.csv')
    protocol=read('run_manifest.json')['protocol']
    print(pd.DataFrame(rows).iloc[:,:17].to_string(index=False))
    print('STAGES',stages.to_string());print('BUDGET',budget.to_string())
    print('INTEGRATION',json.dumps(current['integration'],indent=2))
    print('QUERY',json.dumps(current['query_diagnostics'],indent=2))
    names=['epoch','val_rmse_m','val_irmse_km_inv','val_selected_nfe','val_terminal_time',
           'loss_rmse','loss_inverse','loss_controller_bce','loss_stop2_accuracy','loss_stop2_target_fraction',
           'loss_h1','loss_h2','loss_h3','loss_h4','val_h1_mean','val_h2_mean']
    print('LOG',log[[n for n in names if n in log]].to_string(index=False))
    summary={'best_epoch':current['checkpoint_epoch'],'completed_epochs':len(log),
             'best_rmse':current['final']['all']['rmse_m'],'best_irmse':current['final']['all']['irmse_km_inv'],
             'required_sse_reduction_to_0_9':1-(.9/current['final']['all']['rmse_m'])**2,
             'required_inverse_sse_reduction_to_3_2':1-(3.2/current['final']['all']['irmse_km_inv'])**2,
             'data_manifest':protocol['data']['manifest_sha256'],'source_sha256':protocol['source_sha256'],
             'initialization':protocol['config']['initialization'],'integration':current['integration'],
             'query':current['query_diagnostics'],'historical_runs_not_equal_budget':True}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
