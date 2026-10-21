"""Record user-supplied Q1 console evidence; never invent missing private hashes."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'benchmarks/q1_n1fast_2026_10_21'
BUNDLE=ROOT/'drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100'


def objects(text):
    decoder=json.JSONDecoder();result=[];position=0
    while position<len(text):
        start=text.find('{',position)
        if start<0:break
        try:value,end=decoder.raw_decode(text,start)
        except json.JSONDecodeError:position=start+1;continue
        result.append(value);position=end
    return result


def summarize():
    raw=(EVIDENCE/'user_console.txt').read_bytes()
    blocks=objects(raw.decode('utf-8-sig'))
    measurement=next(b for b in blocks if b.get('variant')=='Q1_s4_n1_fast' and 'metrics' in b)
    comparison=next(b for b in blocks if 'rows' in b and 'selection' in b)
    rows=comparison['rows'];new=next(r for r in rows if r['variant']=='Q1_s4_n1_fast' and r['backend']=='trt_fast')
    original=next(r for r in rows if r['variant']=='Q1_s4_n1_fast' and r['backend']=='trt_original')
    old=next(r for r in rows if r['variant']=='N1_s4_m3_f512' and r['backend']=='trt_fast')
    assert len(rows)==4 and len({r['benchmark_session_id'] for r in rows})==1
    assert measurement['metrics']['pixels']==25424992
    assert new['parameters']==582888 and new['rmse_m']<.98 and new['irmse_km_inv']<3.2 and new['mean_ms']<=3.9
    assert measurement['targets']['reverse_repeat_mean_ms']<=3.9
    for key in ('rmse_m','irmse_km_inv','mae_m'):assert measurement['metrics'][key]==new[key]
    attribution={
        'same_Q1_checkpoint_wall_reduction_pct':100*(original['mean_ms']-new['mean_ms'])/original['mean_ms'],
        'same_Q1_checkpoint_rmse_drift_m':new['rmse_m']-original['rmse_m'],
        'same_Q1_checkpoint_irmse_drift_km_inv':new['irmse_km_inv']-original['irmse_km_inv'],
        'fresh_Q1_vs_old_N1_fast_rmse_improvement_pct':100*(old['rmse_m']-new['rmse_m'])/old['rmse_m'],
        'quality_attribution':'Different trained weights; NOT evidence that deployment rewrites improve accuracy',
        'speed_attribution':'Combined operator rewrites + builderlevel3->5; both use CUDA Graph/auxstreams0',
    }
    evidence={'evidence_level':'user_supplied_console_snapshot','user_console_sha256':hashlib.sha256(raw).hexdigest(),
        'benchmark_session_id':new['benchmark_session_id'],'measurement':measurement,'comparison':comparison,
        'attribution':attribution,'independent_A100_rerun_by_local_agent':False,
        'full_GPU_report_available_locally':False,'trained_checkpoint_available_locally':False,
        'missing':['actual_checkpoint_sha256','actual_ONNX_engine_sha256','selected_checkpoint_epoch','full400_per_image_report'],
        'not_established':['full_KITTI_training_or_test','multi_seed_confirmation','Orin_runtime','isolated_fastop_or_Graph_speedup']}
    with (EVIDENCE/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (EVIDENCE/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    adoption={'baseline_id':'Q1_s4_n1_fast','adopted_date':'2026-10-21','adoption_authority':'explicit_user_request',
        'status':'current_subset_development_baseline','architecture_parent':'N1_s4_m3_f512',
        'neural_parameters':582888,'coarse_NFE':4,'fine_NFE':0,'new_neural_parameters_vs_N1':0,
        'selection':new,'reverse_repeat_mean_ms':measurement['targets']['reverse_repeat_mean_ms'],
        'frozen_bundle':'drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100',
        'frozen_bundle_manifest_sha256':hashlib.sha256((BUNDLE/'bundle_manifest.json').read_bytes()).hexdigest(),
        'checkpoint_Drive_path':'GeoLift_RT_Runs/S4_UPIR_N1Fast_Fresh30_pair01/Q1_s4_n1_fast/best_policy.pth',
        'checkpoint_sha256':None,'checkpoint_epoch_0based':None,
        'evidence':'benchmarks/q1_n1fast_2026_10_21/evidence.json',
        'full_KITTI_evaluated':False,'Orin_measured':False,
        'historical_candidate_only_report_preserved':True,'theory':'docs/AnchorFlow_Q1_Theory.md'}
    (ROOT/'production/AnchorFlow_S4_Q1_Fast/baseline.json').write_text(json.dumps(adoption,indent=2),encoding='utf-8')
    print(json.dumps({'baseline':adoption['baseline_id'],'targets_passed':True,**attribution},indent=2))


if __name__=='__main__':summarize()
