"""Original vs fast FP32 engines on IDENTICAL trained weights and actual400 GT."""
import csv
import json
from pathlib import Path
import torch
import experiment as E
from data import digest,write_json
from metrics import Metrics
from boundaries import BoundaryMetrics
from deployment import DepthOnly,typed_deployment,export_onnx,build_engine,TRTRunner,TorchGraph,parity
from trt_benchmark import timing,test_predictions


def target_checks(score,wall,cfg):
    values={'rmse_m':float(score['rmse_m']),'irmse_km_inv':float(score['irmse_km_inv']),
            'mean_wall_ms':float(wall['mean_ms']),'parameters':582888}
    target=cfg['experiment_targets']
    checks={'rmse_lt_0_98':values['rmse_m']<target['rmse_m_strict_max'],
            'irmse_lt_3_2':values['irmse_km_inv']<target['irmse_km_inv_strict_max'],
            'mean_wall_le_3_9ms':values['mean_wall_ms']<=target['mean_wall_ms_max'],
            'parameters_exact':values['parameters']==target['parameters']}
    return {'actual':values,'checks':checks,'all_passed':all(checks.values()),
            'note':'Strict unrounded quality thresholds; <=3.9ms cap also accepts faster models. No guarantee.'}


@torch.no_grad()
def benchmark(cfg,reference=False):
    from run import setup,load_checkpoint,loader,move,run_dir
    from history import load_reference,recipe,comparable_data
    device=setup(cfg);session_root=Path(cfg['drive_runs'])/cfg['run_name']
    session=json.loads((session_root/'benchmark_session.json').read_text())['session_id']
    if reference:
        model,ckpt,path=load_reference(cfg,device);root=session_root/'preflight_N1'
        label=E.REFERENCE_VARIANT
    else:
        label=E.VARIANTS[0];model,ckpt,path=load_checkpoint(cfg,label,device);root=run_dir(cfg,label)
        status=json.loads((root/'training_status.json').read_text())
        if status['completed_epochs']!=30:raise RuntimeError('Complete all30 epochs before benchmark')
        if recipe(ckpt['contract']['config'])!=recipe(cfg):raise RuntimeError('Training recipe differs')
        current_data=json.loads((Path(cfg['work'])/'data_contract.json').read_text())
        if comparable_data(ckpt['contract']['data'])!=comparable_data(current_data):raise RuntimeError('Training/benchmark data differs')
    # best_policy may be saved BEFORE epoch30; completed budget is audited via
    # training_status/report, not incorrectly inferred from best checkpoint rows.
    if not ckpt['state']['rows']:raise RuntimeError('Missing trained checkpoint history')
    root.mkdir(parents=True,exist_ok=True)
    model.set_diagnostics(False)
    native=DepthOnly(model,native_amp=True).eval()
    original=typed_deployment(model,fast=False).to(device)
    fast=typed_deployment(model,fast=True).to(device)
    images=[]
    for index,cpu in enumerate(loader(cfg,'val')):
        if index>=100:break
        batch=move(cpu,device,cfg);images.append({k:batch[k].contiguous() for k in E.INPUTS})
    if len(images)!=100:raise RuntimeError('Need100 distinct resident timing images')
    sample=images[0];builds={};runners={};eager={};artifacts={}
    for kind,module,level in (('original',original,3),('fast',fast,5)):
        onnx_path=root/(kind+'.onnx');engine_path=root/(kind+'.plan')
        export_onnx(module,sample,onnx_path,cfg['benchmark']['export_opset'])
        builds[kind]=build_engine(onnx_path,engine_path,cfg['benchmark']['trt_workspace_gib'],False,level)
        runners[kind]=TRTRunner(engine_path,sample,capture=True)
        eager[kind]=TRTRunner(engine_path,sample,capture=False)
        artifacts[kind]={'onnx_sha256':digest(onnx_path),'engine_sha256':digest(engine_path)}
    native_graph=TorchGraph(native,sample)
    changed={k:v.clone() for k,v in sample.items()}
    changed['rgb']=(changed['rgb']*.7+.1).clamp(0,1)
    changed['mask'].zero_();changed['sparse'].zero_();changed['K'][:,0,0]*=1.1
    checks=[]
    for tag,batch in (('A',sample),('B',changed),('A_again',sample)):
        expected=original(*(batch[k] for k in E.INPUTS))
        for kind,runner in runners.items():
            actual=runner(batch).clone()
            checks.append(dict(tag=tag,backend=kind,**parity(actual,expected,cfg['benchmark'])))
            checks.append(dict(tag=tag,backend=kind+'_eager_vs_graph',**parity(eager[kind](batch),actual,cfg['benchmark'])))
        checks.append(dict(tag=tag,backend='fast_torch_vs_original',**parity(fast(*(batch[k] for k in E.INPUTS)),expected,cfg['benchmark'])))
        checks.append(dict(tag=tag,backend='native_graph',**parity(native_graph(batch),native(*(batch[k] for k in E.INPUTS)),cfg['benchmark'])))
    write_json(root/'replay_parity.json',checks)
    if not all(r['passed'] for r in checks):raise RuntimeError('Input freshness/export gate failed; no fallback')
    names=('native_bf16','original_fp32','fast_fp32','trt_original','trt_fast')
    scores={k:Metrics(device) for k in names}
    boundary={k:BoundaryMetrics(device) for k in ('trt_original','trt_fast')}
    failed=[];per_image=[];worst={k:{'mean_abs_m':0.,'max_abs_m':0.} for k in ('fast_fp32','trt_original','trt_fast')}
    for index,cpu in enumerate(loader(cfg,'val')):
        batch=move(cpu,device,cfg);args=tuple(batch[k] for k in E.INPUTS)
        predictions={'native_bf16':native(*args),'original_fp32':original(*args),
                     'fast_fp32':fast(*args),'trt_original':runners['original'](batch).clone(),
                     'trt_fast':runners['fast'](batch).clone()}
        frame={'sample_id':cpu['sid'][0],'pixels':int((batch['gt_mask']>0).sum())}
        for key,pred in predictions.items():
            scores[key].update(pred,batch['gt'],batch['gt_mask'],batch['rgb'])
            if key in boundary:
                boundary[key].update(pred,batch['gt'],batch['gt_mask'])
                frame[key+'_sse_m2']=float(((pred-batch['gt']).square()*(batch['gt_mask']>0)).sum(dtype=torch.float64))
            if key in worst:
                check=parity(pred,predictions['original_fp32'],cfg['benchmark'])
                if not check['passed']:failed.append({'index':index,'sample_id':frame['sample_id'],'backend':key,**check})
                for metric in worst[key]:worst[key][metric]=max(worst[key][metric],check[metric])
        per_image.append(frame)
        if (index+1)%100==0:print(f'{label} original/fast TensorRT graph accuracy {index+1}/400',flush=True)
    if len(per_image)!=400 or len({f['sample_id'] for f in per_image})!=400:raise RuntimeError('Need400 unique frames')
    reports={k:v.report() for k,v in scores.items()}
    if any(r['all']['pixels']!=25424992 for r in reports.values()):raise RuntimeError('GT support changed')
    drift={}
    for key in ('trt_original','trt_fast'):
        a,b=reports['native_bf16']['all'],reports[key]['all']
        rmse=abs(a['rmse_m']-b['rmse_m']);inverse=abs(a['irmse_km_inv']-b['irmse_km_inv'])
        drift[key]={'rmse_m':rmse,'irmse_km_inv':inverse,
                    'passed':rmse<=cfg['benchmark']['rmse_drift_m'] and inverse<=cfg['benchmark']['irmse_drift_km_inv']}
    report={'variant':label,'phase':'old_checkpoint_preflight' if reference else 'fresh30_final',
        'qualification_passed':False,'source_sha256':E.source_hash(),'benchmark_session_id':session,
        'training_contract':ckpt['contract'],'checkpoint_sha256':digest(path),
        'checkpoint_epoch_0based':ckpt['state']['selection']['policy_epoch'],
        'training_epochs':30,'initial_shared_sha256':ckpt['state']['initial_shared_sha256'],
        'parameters':sum(p.numel() for p in model.parameters()),'architecture':E.architecture(model),
        'inference_precision':'explicit_FP32_TF32_OFF','build':builds,'artifacts':artifacts,
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
        'metrics':reports,'boundary':{k:v.report() for k,v in boundary.items()},'per_image':per_image,
        'parity':{'failed_frames':failed,'worst':worst,'native_metric_drift':drift},
        'parity_gates':cfg['benchmark'],'reference_provenance':getattr(model,'_reference_provenance',None),
        'fastpath_note':'Same NN/checkpoint, algebraic deployment rewrites NOT bit-exact FP32; all checks use ORIGINAL FP32 reference.'}
    write_json(root/'trt_benchmark.json',report)
    if failed or not all(r['passed'] for r in drift.values()):raise RuntimeError('Full400 original-reference parity failed; report saved, no fallback')
    backends={'pytorch_eager':lambda b:native(*(b[k] for k in E.INPUTS)),
              'pytorch_graph':native_graph,'trt_original_eager':eager['original'],
              'trt_original':runners['original'],'trt_fast_eager':eager['fast'],'trt_fast':runners['fast']}
    report['timing']={key:timing(runner,images,100,300) for key,runner in backends.items()}
    # Repeat both engine timings in reversed order. Never choose the fastest repetition.
    report['timing_repeat_reverse']={key:timing(backends[key],images,100,300) for key in ('trt_fast','trt_original')}
    report['targets']=target_checks(reports['trt_fast']['all'],report['timing']['trt_fast']['synchronized_wall'],cfg)
    report['targets']['reverse_repeat_mean_ms']=report['timing_repeat_reverse']['trt_fast']['synchronized_wall']['mean_ms']
    report['targets']['reverse_repeat_within_3_9ms']=report['targets']['reverse_repeat_mean_ms']<=3.9
    report['targets']['stable_speed_target_passed']=report['targets']['checks']['mean_wall_le_3_9ms'] and report['targets']['reverse_repeat_within_3_9ms']
    if not reference:test_predictions(runners['fast'],cfg,label,root)
    report['qualification_passed']=True
    write_json(root/'trt_benchmark.json',report)
    print(json.dumps({'variant':label,'metrics':reports['trt_fast']['all'],'targets':report['targets'],
                      'original_wall':report['timing']['trt_original']['synchronized_wall']},indent=2),flush=True)
    return report


def rows(report):
    result=[]
    for name in ('trt_original','trt_fast'):
        score=report['metrics'][name]['all'];wall=report['timing'][name]['synchronized_wall']
        result.append({'variant':report['variant'],'backend':name,'parameters':report['parameters'],
            'rmse_m':score['rmse_m'],'irmse_km_inv':score['irmse_km_inv'],'mae_m':score['mae_m'],
            'boundary3_rmse_m':report['boundary'][name]['bands']['3']['rmse_m'],
            'mean_ms':wall['mean_ms'],'median_ms':wall['median_ms'],'p95_ms':wall['p95_ms'],
            'fps_from_mean':1000/wall['mean_ms'],'benchmark_session_id':report['benchmark_session_id']})
    return result


def compare(cfg):
    from history import audit_history,recipe,comparable_data
    root=Path(cfg['drive_runs'])/cfg['run_name'];proof=audit_history(cfg)
    ref=json.loads((root/'preflight_N1/trt_benchmark.json').read_text())
    new=json.loads((root/E.VARIANTS[0]/'trt_benchmark.json').read_text())
    session=json.loads((root/'benchmark_session.json').read_text())['session_id']
    for report in (ref,new):
        if not report['qualification_passed'] or report['benchmark_session_id']!=session or report['source_sha256']!=E.source_hash():raise RuntimeError('Unqualified/stale report/source')
        if report['parameters']!=582888 or report['initial_shared_sha256']!=proof['initial_shared_sha256']:raise RuntimeError('Weights/architecture initialization differs')
        if report['training_epochs']!=30 or report['inference_precision']!='explicit_FP32_TF32_OFF':raise RuntimeError('Training/precision differs')
        if report['parity_gates']!=cfg['benchmark']:raise RuntimeError('Numerical/timing gates differ')
        if any(report[k]!=ref[k] for k in ('gpu','torch','cuda','build')):raise RuntimeError('GPU/software/engine policy differs')
        if recipe(report['training_contract']['config'])!=recipe(cfg):raise RuntimeError('Training recipe differs')
        if comparable_data(report['training_contract']['data'])!=comparable_data(ref['training_contract']['data']):raise RuntimeError('Data differs')
        if len(report['per_image'])!=400 or len({f['sample_id'] for f in report['per_image']})!=400 or sum(f['pixels'] for f in report['per_image'])!=25424992:raise RuntimeError('Wrong validation support')
        for backend in ('trt_original','trt_fast'):
            t=report['timing'][backend]
            if (t['warmup'],t['iterations'],t['distinct_images'])!=(100,300,100):raise RuntimeError('Timing protocol differs')
    if ref['checkpoint_sha256']!=proof['checkpoint_sha256']:raise RuntimeError('Old checkpoint changed')
    if [(f['sample_id'],f['pixels']) for f in ref['per_image']]!=[(f['sample_id'],f['pixels']) for f in new['per_image']]:raise RuntimeError('Validation IDs/order/support differs')
    values=rows(ref)+rows(new)
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    checks=target_checks(new['metrics']['trt_fast']['all'],new['timing']['trt_fast']['synchronized_wall'],cfg)
    stable=new['timing_repeat_reverse']['trt_fast']['synchronized_wall']['mean_ms']<=3.9
    choice={'selected_variant':E.VARIANTS[0] if checks['all_passed'] and stable else None,
            'targets':checks,'reverse_repeat_within_3_9ms':stable,
            'status':'subset_development_candidate_only','full_KITTI_baseline_promoted':False,
            'next':'Confirm another seed and lock independent full-KITTI validation; no automatic promotion.'}
    write_json(root/'comparison.json',{'rows':values,'selection':choice})
    write_json(root/'baseline_candidate.json',choice)
    print(json.dumps({'rows':values,'selection':choice},indent=2),flush=True)
    return choice
