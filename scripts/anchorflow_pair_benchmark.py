"""Independent process orchestration and matched reporting; no model imports here.

Copied verbatim into the self-contained Colab notebook. Existing sealed model
bundles are never patched. Training can overlap; final evaluation/profiling cannot.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

LABELS=('V10_1','V11')
ARCH_KEYS={'architecture','model_name','parent_source_sha256','step_size','learned_step_size',
           'step_min','step_max','step_init_logit'}
RUNTIME_KEYS={'run_name','drive_runs','work','drive_data'}
SHARED_SOURCES=('core.py','support.py','data.py','relative_data.py','relative_loss.py',
                'loss_helpers.py','metrics.py','boundaries.py')

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)

def verify_bundle(folder):
    folder=Path(folder);manifest=read_json(folder/'bundle_manifest.json')
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or '/' in name or '\\' in name or name in ('.','..'):raise ValueError(f'Unsafe bundle filename: {name}')
        if name.endswith('.ipynb'):continue # Mutable Colab UI only; immutable snapshot remains checked.
        p=folder/name
        if not p.is_file() or digest(p)!=expected:raise RuntimeError(f'Bundle changed: {folder.name}/{name}')
    return manifest

def stage_bundle(folder,destination):
    manifest=verify_bundle(folder);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for name in [*manifest['files'],'bundle_manifest.json']:
        if name.endswith('.ipynb'):continue
        source,target=Path(folder)/name,destination/name
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
    return manifest

def matched_recipe(configs):
    a,b=(configs[label] for label in LABELS)
    ignored=ARCH_KEYS|RUNTIME_KEYS|{k for k in set(a)|set(b) if k.startswith('ode_')}
    differences={k:{'V10_1':a.get(k),'V11':b.get(k)} for k in (set(a)|set(b))-ignored
                 if a.get(k)!=b.get(k)}
    if differences:raise RuntimeError('Unmatched training recipe: '+json.dumps(differences,sort_keys=True))
    for cfg in (a,b):
        if cfg['init_checkpoint'] is not None or cfg['epochs']!=40:raise ValueError('Both experiments must be fresh student, max40')
        if not cfg['encoder_pretrained']:raise ValueError('Both experiments require the same ImageNet RGB initialization policy')
        if cfg['amp'] not in ('bf16','fp32'):raise ValueError('Common native BF16 or FP32 only, no FP16')
        if cfg['flow_steps']!=2 or not cfg['teacher_enabled'] or not cfg['relative_enabled']:
            raise ValueError('Two-stage dual-teacher contract required')
        if not cfg['early_stopping']:raise ValueError('Keep early stopping enabled for both')
    if a['model_name']!='v10_lite' or not a.get('learned_step_size'):
        raise ValueError('Upload the CURRENT learned-h V10.1 LiteMetric bundle, not the old fixed-h one')
    if b['model_name']!='v11_node' or b['ode_method']!='bosh3' or b['ode_terminal_time']!=1.:
        raise ValueError('V11 must use its default adaptive solver to T=1')
    return {'matched_non_architecture_settings':True,'shared_config':{k:a[k] for k in sorted(a) if k not in ignored},
            'different_state_chart_and_solver':True,'same_seed_does_not_force_identical_all_student_weights':True}

def freeze_snapshot(snapshot,code,config):
    """Preflight ALL targets before writing; never overwrite a frozen experiment."""
    snapshot,code=Path(snapshot),Path(code)
    manifest=verify_bundle(code)
    content={name:(code/name).read_bytes() for name in manifest['files'] if not name.endswith('.ipynb')}
    content['bundle_manifest.json']=(code/'bundle_manifest.json').read_bytes()
    content['resolved_config.json']=json.dumps(config,indent=2,ensure_ascii=False).encode('utf-8')
    for name,value in content.items():
        p=snapshot/name
        if p.exists() and p.read_bytes()!=value:raise RuntimeError(f'Frozen recipe changed: {p}; choose a new BENCH_TAG')
    snapshot.mkdir(parents=True,exist_ok=True)
    for name,value in content.items():
        p=snapshot/name
        if not p.exists():p.write_bytes(value)
    (code/'resolved_config.json').write_bytes(content['resolved_config.json'])

def visible_gpu_token(index):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:return str(index)
    tokens=[x.strip() for x in visible.split(',') if x.strip()]
    if not 0<=index<len(tokens):raise ValueError(f'GPU index {index} outside CUDA_VISIBLE_DEVICES={visible!r}')
    return tokens[index]

def process_environment(gpu,code,cpu_threads=4):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONPATH=str(code),
               OMP_NUM_THREADS=str(cpu_threads),MKL_NUM_THREADS=str(cpu_threads),
               OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    return env

@dataclass
class Command:
    label:str
    action:str
    args:list
    cwd:Path
    local_log:Path
    drive_log:Path
    env:dict|None=None

class ProcessGroupError(RuntimeError):
    def __init__(self,message,results=()):
        super().__init__(message);self.results=list(results)

def _mirror(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix(destination.suffix+'.partial')
    shutil.copy2(source,tmp);tmp.replace(destination)

def _terminate(process):
    if process.poll() is not None:return
    try:
        if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
        else:process.terminate()
    except ProcessLookupError:return # Child exited between poll and signal.
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()
        except ProcessLookupError:pass
        process.wait(timeout=10)

def run_processes(commands,parallel=True,status_path=None,mirror_seconds=30):
    """Launch independent interpreters, multiplex logs, preserve a surviving peer.

    A failed model does not kill the other model. Report failure AFTER peers finish.
    User interrupt terminates only the process groups launched by this invocation.
    Console tails mirror to Drive periodically even while a child is quiet.
    """
    commands=list(commands)
    if not parallel and len(commands)>1:
        results=[];failures=[]
        for c in commands:
            try:results.extend(run_processes([c],True,None,mirror_seconds))
            except ProcessGroupError as exc:results.extend(exc.results);failures.append(str(exc))
        if status_path:atomic_json(status_path,{'status':'failed' if failures else 'completed','results':results,'failures':failures})
        if failures:raise ProcessGroupError('\n'.join(failures),results)
        return results
    messages=queue.Queue();active=[];results=[]
    def reader(index,pipe):
        try:
            for line in iter(pipe.readline,''):messages.put((index,line))
        finally:pipe.close();messages.put((index,None))
    def publish(status):
        if status_path:atomic_json(status_path,{'status':status,'parallel':parallel,
            'active':[{'label':r['command'].label,'pid':r['process'].pid} for r in active if not r['done']],
            'results':results})
    try:
        for c in commands:
            Path(c.local_log).parent.mkdir(parents=True,exist_ok=True)
            handle=Path(c.local_log).open('w',encoding='utf-8')
            try:
                process=subprocess.Popen(c.args,cwd=c.cwd,env=c.env,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,
                    start_new_session=os.name=='posix')
            except BaseException:handle.close();raise
            index=len(active)
            thread=threading.Thread(target=reader,args=(index,process.stdout),daemon=True);thread.start()
            active.append({'command':c,'process':process,'log':handle,'thread':thread,'eof':False,
                           'done':False,'start':time.time(),'last_mirror':0.})
            print(f'[{c.label}/{c.action}] START pid={process.pid}',flush=True)
        publish('running')
        while not all(r['done'] for r in active):
            try:
                index,line=messages.get(timeout=.25);r=active[index]
                if line is None:r['eof']=True
                else:
                    r['log'].write(line);r['log'].flush()
                    print(f'[{r["command"].label}/{r["command"].action}] {line}',end='',flush=True)
            except queue.Empty:pass
            now=time.time()
            for r in active:
                if r['done']:continue
                c=r['command'];code=r['process'].poll()
                if now-r['last_mirror']>=mirror_seconds or (code is not None and r['eof']):
                    r['log'].flush();_mirror(c.local_log,c.drive_log);r['last_mirror']=now
                if code is not None and r['eof']:
                    r['done']=True;r['log'].close();r['thread'].join(timeout=2)
                    result={'label':c.label,'action':c.action,'pid':r['process'].pid,'returncode':code,
                            'started_unix':r['start'],'finished_unix':now,'wall_seconds':now-r['start'],
                            'console':str(c.drive_log)}
                    results.append(result);publish('running')
                    print(f'[{c.label}/{c.action}] EXIT={code}; console={c.drive_log}',flush=True)
        failures=[r for r in results if r['returncode']]
        publish('failed' if failures else 'completed')
        if failures:raise ProcessGroupError('One or more subprocesses failed; peers were allowed to finish:\n'+
            '\n'.join(f'{r["label"]}/{r["action"]}: exit{r["returncode"]}; {r["console"]}' for r in failures),results)
        return results
    except (KeyboardInterrupt,SystemExit):
        for r in active:
            _terminate(r['process'])
            if not r['log'].closed:r['log'].flush();r['log'].close()
            _mirror(r['command'].local_log,r['command'].drive_log)
        publish('interrupted');raise
    except BaseException as exc:
        # Operational orchestration failures (launch/copy), not ordinary child
        # exit failures, must not leave our own orphan GPU workers behind.
        for r in active:
            if not r['done']:_terminate(r['process'])
            if not r['log'].closed:r['log'].close()
        if not isinstance(exc,ProcessGroupError):
            if status_path:atomic_json(status_path,{'status':'orchestration_failed','error':repr(exc),
                'results':results,'children_terminated':True})
        raise

def model_command(job,action,gpu=None,extra=(),suffix=''):
    label=job['label'];code=Path(job['code']);run_dir=Path(job['run_dir'])
    filename=f'{action}{suffix}_console.log'
    return Command(label,action,[sys.executable,'-u',str(code/'run.py'),action,'--config',str(code/'resolved_config.json'),
        '--variant',job['variant'],*extra],code,code/filename,run_dir/filename,
        process_environment(job['gpu'] if gpu is None else gpu,code,job.get('cpu_threads',4)))

def run_stage(jobs,action,parallel=False,status_path=None,gpu=None,extra=(),suffix=''):
    jobs=list(jobs);records=[]
    # The sealed runners do not put checkpoint identity in every profiler JSON.
    # Freeze it around the subprocess instead, so accuracy and timing cannot be
    # accidentally joined after another run has replaced best.pth.
    if action in ('evaluate','test','profile','solver_audit','export') and '--untrained' not in extra:
        selection='rmse'
        if '--checkpoint-selection' in extra:selection=extra[extra.index('--checkpoint-selection')+1]
        for j in jobs:
            path=Path(j['run_dir'])/('best.pth' if selection=='rmse' else f'best_{selection}.pth')
            records.append((j,path,{'action':action,'selection':selection,'checkpoint':path.name,
                'checkpoint_sha256':digest(path),'resolved_config_sha256':digest(Path(j['code'])/'resolved_config.json'),
                'bundle_manifest_sha256':digest(Path(j['code'])/'bundle_manifest.json'),
                'GPU_visible_token':str(j['gpu'] if gpu is None else gpu)}))
    result=run_processes([model_command(j,action,gpu,extra,suffix) for j in jobs],parallel,status_path)
    for j,path,record in records:
        if digest(path)!=record['checkpoint_sha256']:raise RuntimeError(f'Checkpoint changed during {action}: {path}')
        atomic_json(Path(j['run_dir'])/f'{action}{suffix}_artifact_record.json',{**record,'completed':True})
    return result

def verify_data_contract(contract,config):
    if (contract['train'],contract['val'],contract['test'])!=(1600,400,1000):raise RuntimeError('Split counts changed')
    if contract['manifest_sha256']!=config['expected_subset_sha256']:raise RuntimeError('Subset changed')
    if contract['size']!=[352,1216] or contract['depth_scale']!=256:raise RuntimeError('Depth/image protocol changed')
    if contract['validation_teacher_used']:raise RuntimeError('Validation teacher leakage')
    if config['teacher_enabled']:
        r=contract['teacher_report']
        if r['cached_train']!=1600 or r['cached_val']!=0:raise RuntimeError('Metric cache must be train-only1600')
        if r['selected_teacher_content_sha256']!='06bd17a84faa5020692873ee598498486d2030f588b7429321fa0a70d9ebb7a6':
            raise RuntimeError('Metric teacher content changed')
    if config['relative_enabled']:
        r=contract['relative_teacher']
        if r['cached_train']!=1600 or r['cached_val']!=0 or contract['relative_val_used']:raise RuntimeError('Relative cache leakage')
        if r['selected_content_sha256']!='dd3b49a4474560fedbfc78504dd584c0234d790c65a356c2e74b6bb82759e86a':
            raise RuntimeError('Relative teacher content changed')

def gpu_snapshot():
    try:
        p=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=10)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:return {'unavailable':str(e)}

def _csv_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def _accuracy(label,selection,report):
    score=report['final'];a=score['all']
    return {'model':label,'selection':selection,'epoch':report.get('checkpoint_epoch',report.get('epoch')),
        'RMSE_m':a['rmse_m'],'MAE_m':a['mae_m'],'iRMSE_km_inv':a['irmse_km_inv'],
        'iMAE_km_inv':a['imae_km_inv'],'AbsRel':a['abs_rel'],'delta1':a['delta1'],'pixels':a['pixels'],
        **{f'RMSE_{k}_m':score[k]['rmse_m'] for k in ('0-20','20-40','40-60','60-80','80-120','edge')},
        'GT_boundary3_m':report['gt_boundary']['bands']['3']['rmse_m'],
        'joint_target_score':max(a['rmse_m']/.9,a['irmse_km_inv']/3.2),
        'both_targets':a['rmse_m']<.9 and a['irmse_km_inv']<3.2}

def _write_csv(path,rows):
    if not rows:return
    names=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)

def compare_reports(jobs,root,expected_pixels=25424992):
    """Accuracy from isolated final reevaluation, runtime from isolated real100.

    Both model checkpoint selections stay explicit; never combine metrics from
    different checkpoints. Parallel training epoch duration is NOT isolated speed.
    """
    jobs=list(jobs)
    if tuple(j['label'] for j in jobs)!=LABELS:raise ValueError('Expected ordered V10_1/V11 pair')
    matched_recipe({j['label']:j['config'] for j in jobs})
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    primary=[];selections=[];efficiency=[];budget=[];main_profiles=[];curves={};solver_rows=[]
    stages=[];tails=[];boundary_rows=[];policies=[];provenance=[]
    for job in jobs:
        label=job['label'];path=Path(job['run_dir']);cfg=job['config']
        report=read_json(path/'val_metrics.json')
        if report['samples']!=400 or report['teacher_at_inference'] or report['checkpoint_selection']!='rmse':
            raise RuntimeError(f'{label}: invalid evaluation protocol')
        evaluation=read_json(path/'evaluate_artifact_record.json');profile_record=read_json(path/'profile_artifact_record.json')
        for key in ('checkpoint_sha256','resolved_config_sha256','bundle_manifest_sha256','GPU_visible_token'):
            if evaluation[key]!=profile_record[key]:raise RuntimeError(f'{label}: evaluation/profile provenance mismatch: {key}')
        if not evaluation['completed'] or not profile_record['completed']:raise RuntimeError('Incomplete artifact stage')
        if digest(path/'best.pth')!=profile_record['checkpoint_sha256']:raise RuntimeError('Best checkpoint changed since profiling')
        provenance.append({'model':label,'checkpoint_epoch':report['checkpoint_epoch'],**profile_record})
        if report['final']['all']['pixels']!=expected_pixels:raise RuntimeError(f'{label}: GT support changed')
        primary.append(_accuracy(label,'rmse',report))
        for selection in ('rmse','inverse','joint'):
            file=path/('val_metrics.json' if selection=='rmse' else f'val_metrics_{selection}.json')
            if file.is_file():
                r=read_json(file)
                if r['final']['all']['pixels']!=expected_pixels:raise RuntimeError('Selection GT support changed')
                evidence=read_json(path/('evaluate_artifact_record.json' if selection=='rmse' else f'evaluate_{selection}_artifact_record.json'))
                checkpoint=path/('best.pth' if selection=='rmse' else f'best_{selection}.pth')
                if digest(checkpoint)!=evidence['checkpoint_sha256']:raise RuntimeError('Selection checkpoint changed since evaluation')
                if evidence['resolved_config_sha256']!=evaluation['resolved_config_sha256']:
                    raise RuntimeError('Selection config changed')
                selections.append(_accuracy(label,selection,r))
        rows=_csv_rows(path/'train_log.csv')
        if not rows:raise RuntimeError(f'No trained epochs for {label}')
        epochs=[int(r['epoch']) for r in rows]
        if epochs!=list(range(max(epochs)+1)):raise RuntimeError(f'{label}: duplicate/missing epoch logs')
        curves[label]=rows
        status=read_json(path/'training_status.json')
        if status['status'] not in ('max_epochs_completed','early_stopped') or status['epochs_completed']!=len(rows):
            raise RuntimeError(f'{label}: training status incomplete; rerun train cell to resume, then reevaluate')
        budget.append({'model':label,'epochs_logged':len(rows),'last_epoch_logged':max(epochs),
            'optimizer_updates':int(rows[-1]['global_step']),'batch_size':cfg['batch_size'],
            'accumulation':cfg['accumulation'],'effective_batch':cfg['batch_size']*cfg['accumulation'],
            'epoch_seconds_mean_UNDER_TRAIN_CONTENTION':sum(float(r['epoch_seconds']) for r in rows)/len(rows),
            'train_seconds_total_UNDER_TRAIN_CONTENTION':sum(float(r['train_seconds']) for r in rows),
            'training_status':status['status']})
        p=read_json(path/'profile.json');main_profiles.append(p)
        real=p['real_scene_profile']
        if real.get('samples')!=100 or real.get('indices')!='i*13 mod400':raise RuntimeError('Unmatched real-scene profile')
        if p['untrained_weights'] or p['batch']!=1 or p['shape']!=[352,1216]:raise RuntimeError('Invalid inference profile')
        nfe=real.get('solver',{}).get('nfe_mean',real.get('executed_nfe'))
        efficiency.append({'model':label,'graph':'primary adaptive RK3(2)' if label=='V11' else 'primary learned-h Euler2',
            'parameters':p['total_parameters'],'synthetic_conv_linear_GMAC':p['total_conv_linear_macs']/1e9,
            'real100_wall_median_ms':real['wall_median_ms'],'real100_wall_P95_ms':real['wall_p95_ms'],
            'real100_actual_NFE_mean':nfe,'peak_torch_allocated_MiB':p['peak_cuda_allocated_mib'],
            'GPU':p['device'],'CNN_precision':p['precision'],'torch':p['torch'],
            'RMSE_m':report['final']['all']['rmse_m'],'iRMSE_km_inv':report['final']['all']['irmse_km_inv'],
            'latency_measured_without_peer_training':True})
        if label=='V11':
            solver=path/'solver_comparison.json'
            if solver.is_file():solver_rows=read_json(solver)
            fixed_p,fixed_r=path/'profile_fixed_midpoint8.json',path/'val_metrics_fixed_midpoint8.json'
            if fixed_p.is_file() and fixed_r.is_file():
                pp,rr=read_json(fixed_p),read_json(fixed_r);real=pp['real_scene_profile']
                if rr['checkpoint_epoch']!=report['checkpoint_epoch']:raise RuntimeError('Static candidate used a different checkpoint')
                audit=read_json(path/'solver_audit_artifact_record.json')
                if audit['checkpoint_sha256']!=profile_record['checkpoint_sha256']:raise RuntimeError('Static solver audit used different weights')
                if rr['final']['all']['pixels']!=expected_pixels:raise RuntimeError('Static solver GT support changed')
                for key in ('device','torch','precision','channels_last','runs','shape','batch','compiled_total'):
                    if pp[key]!=p[key]:raise RuntimeError(f'Static profile protocol mismatch: {key}')
                if real['samples']!=100 or real['indices']!='i*13 mod400':raise RuntimeError('Static real-scene profile changed')
                efficiency.append({'model':'V11','graph':'static midpoint8 (different numerical solver)',
                    'parameters':pp['total_parameters'],'synthetic_conv_linear_GMAC':pp['total_conv_linear_macs']/1e9,
                    'real100_wall_median_ms':real['wall_median_ms'],'real100_wall_P95_ms':real['wall_p95_ms'],
                    'real100_actual_NFE_mean':real['solver']['nfe_mean'],'peak_torch_allocated_MiB':pp['peak_cuda_allocated_mib'],
                    'GPU':pp['device'],'CNN_precision':pp['precision'],'torch':pp['torch'],
                    'RMSE_m':rr['final']['all']['rmse_m'],'iRMSE_km_inv':rr['final']['all']['irmse_km_inv'],
                    'latency_measured_without_peer_training':True})
        stages.extend({'model':label,'stage':name,**score} for name,score in report['stage_native_gt_metrics'].items())
        tails.extend({'model':label,'absolute_error_threshold_m':threshold,**score}
                     for threshold,score in report['final']['all']['error_tail'].items())
        boundary_rows.extend({'model':label,'GT_boundary_radius_px':radius,**{k:v for k,v in score.items() if k!='bad_pixel_rates'}}
                             for radius,score in report['gt_boundary']['bands'].items())
        policies.extend({'model':label,'policy':name,**{k:v for k,v in report[name]['all'].items() if k!='error_tail'}}
                        for name in ('final','pre_anchor','legacy_hard_anchor'))
    for key in ('device','torch','precision','channels_last','runs','shape','batch','compiled_total'):
        if main_profiles[0][key]!=main_profiles[1][key]:raise RuntimeError(f'Profile hardware/protocol mismatch: {key}')
    for key in ('cudnn_enabled','cudnn_benchmark','cudnn_deterministic','cudnn_allow_tf32','matmul_allow_tf32','cuda_build','cudnn_version'):
        if main_profiles[0]['runtime_backend'][key]!=main_profiles[1]['runtime_backend'][key]:
            raise RuntimeError(f'Profile backend mismatch: {key}')
    common_epochs=min(len(r) for r in curves.values());common=[]
    for label,rows in curves.items():
        r=min(rows[:common_epochs],key=lambda r:float(r['val_rmse_m']))
        common.append({'model':label,'common_completed_epochs':common_epochs,'selected_epoch':int(r['epoch']),
                       'RMSE_m':float(r['val_rmse_m']),'iRMSE_km_inv_SAME_EPOCH':float(r['val_irmse_km_inv']),
                       'source':'epoch validation log, not isolated reevaluation; checkpoint may not have been retained'})
    a,b=primary
    delta={'V11_minus_V10_1_RMSE_m':b['RMSE_m']-a['RMSE_m'],
           'V11_minus_V10_1_iRMSE_km_inv':b['iRMSE_km_inv']-a['iRMSE_km_inv'],
           'RMSE_improvement_percent':100*(a['RMSE_m']-b['RMSE_m'])/a['RMSE_m'],
           'metric_SSE_reduction_percent':100*(1-(b['RMSE_m']/a['RMSE_m'])**2),
           'real100_latency_V11_over_V10_1':efficiency[1]['real100_wall_median_ms']/efficiency[0]['real100_wall_median_ms']}
    for name,rows in [('comparison_accuracy.csv',primary),('comparison_checkpoint_selections.csv',selections),
                      ('comparison_efficiency.csv',efficiency),('comparison_training_budget.csv',budget),
                      ('comparison_common_epoch_budget.csv',common),('comparison_v11_solvers.csv',solver_rows),
                      ('comparison_stages.csv',stages),('comparison_tails.csv',tails),
                      ('comparison_GT_boundaries.csv',boundary_rows),('comparison_sensor_policies.csv',policies)]:
        _write_csv(root/name,rows)
    atomic_json(root/'comparison_summary.json',{'primary':primary,'selections':selections,'efficiency':efficiency,
        'training_budget':budget,'common_epoch_budget':common,'deltas':delta,
        'artifact_provenance':provenance,
        'interpretation':'Same data/loss/hyperparameter policy, single seed. Different chart/solver, student initialization order and stopping epochs; not a pure solver ablation. Parallel epoch duration is not isolated model speed.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for label,rows in curves.items():
        x=[int(r['epoch'])+1 for r in rows]
        for ax,key,title in [(axes[0,0],'val_rmse_m','Global validation RMSE (m)'),
                              (axes[0,1],'val_irmse_km_inv','Validation iRMSE (km^-1)'),
                              (axes[1,0],'val_native_D4_rmse_m','Native quarter-grid RMSE (m)')]:
            ax.plot(x,[float(r[key]) for r in rows],label=label);ax.set_title(title)
    axes[0,0].axhline(.9,color='red',ls='--',alpha=.5);axes[0,1].axhline(3.2,color='red',ls='--',alpha=.5)
    for r in efficiency:
        axes[1,1].scatter(r['real100_wall_median_ms'],r['RMSE_m'],s=70)
        axes[1,1].annotate(r['model']+' '+('static8' if 'static' in r['graph'] else 'main'),
                         (r['real100_wall_median_ms'],r['RMSE_m']),xytext=(4,6),textcoords='offset points')
    axes[1,1].set_xlabel('Isolated real100 wall median (ms)');axes[1,1].set_ylabel('RMSE (m)')
    axes[1,1].set_title('Same-checkpoint accuracy / latency')
    for ax in axes.flat:ax.grid(alpha=.25)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.legend();ax.set_xlabel('Completed epochs')
    fig.tight_layout();fig.savefig(root/'comparison_dashboard.png',dpi=160);plt.close(fig)
    lines=['# V10.1 / V11 benchmark','',
        '| Model | Best RMSE m | iRMSE same checkpoint | Best epoch |',
        '|---|---:|---:|---:|',
        *[f'| {r["model"]} | {r["RMSE_m"]:.6f} | {r["iRMSE_km_inv"]:.6f} | {r["epoch"]} |' for r in primary],
        '',f'V11 RMSE improvement: {delta["RMSE_improvement_percent"]:.2f}%; negative means worse.',
        f'V11/main real100 wall latency ratio: {delta["real100_latency_V11_over_V10_1"]:.3f}x.',
        '',f'Common logged budget: {common_epochs}epochs; see comparison_common_epoch_budget.csv.',
        '', 'Training was scheduled in independent subprocesses; epoch durations under overlap are not isolated model speed.',
        'Final evaluation/profile ran sequentially on the same GPU. Anonymous1000 has no public GT.',
        'V11 static8 is a different numerical solver, evaluated separately. Single seed, not pure solver-only causal evidence.']
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return read_json(root/'comparison_summary.json')
