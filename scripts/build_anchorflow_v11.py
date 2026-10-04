"""Mechanically isolate the latest LiteMetric backbone/readout/runtime for V11."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'drive_upload/AnchorFlow_v10_1_LiteMetric'
DEST = ROOT / 'drive_upload/AnchorFlow_v11_NODE'

def replace(text, old, new):
    if old not in text:
        raise RuntimeError(f'Migration anchor missing: {old[:100]}')
    return text.replace(old, new)

def main():
    DEST.mkdir(exist_ok=True)
    names = ['core.py','support.py','model.py','model_base.py','losses.py','relative_loss.py',
             'loss_helpers.py','data.py','relative_data.py','metrics.py','boundaries.py','run.py',
             'requirements.txt','v8_val_metrics.json','v9_val_metrics.json','v9_1_val_metrics.json','v10_val_metrics.json']
    for name in names:
        if (DEST/name).exists():
            raise RuntimeError(f'Will not overwrite {DEST/name}')
        shutil.copy2(SOURCE/name,DEST/name)
    # Geometry helpers and neural layers stay identical; new mathematical code is
    # maintained directly in geometry.py/ode_solver.py, not generated here.
    old = (SOURCE/'geometry.py').read_text(encoding='utf-8')
    prefix = old[:old.index('class FixedJetDynamics')]
    (DEST/'geometry_primitives.py').write_text(prefix,encoding='utf-8')
    for name in ('model.py','model_base.py'):
        text = (DEST/name).read_text(encoding='utf-8').replace('v10_lite','v11_node')
        text = text.replace('FixedJetDynamics','JetNODE')
        (DEST/name).write_text(text,encoding='utf-8')
    cfg = json.loads((SOURCE/'config.json').read_text())
    for name in ('step_size','learned_step_size','step_min','step_max','step_init_logit','parent_source_sha256'):
        cfg.pop(name,None)
    cfg.update(architecture='AnchorFlow-V11-JetNODE',model_name='v11_node',
        work='/content/anchorflow_v11_work',run_name='AnchorFlow_v11_NODE_Fresh40_ES',
        flow_steps=2,ode_method='bosh3',ode_rtol=.01,ode_atol=.001,
        ode_first_step=.25,ode_max_step=.25,ode_max_nfe=193,ode_max_num_steps=64,
        ode_terminal_time=1.,ode_fixed_steps=4,ode_field_precision='fp32',
        ode_norm='max_sample_channel_rms',ode_library='torchdiffeq==0.2.5',
        initialization='fresh_student_imagenet_rgb_only',init_checkpoint=None)
    (DEST/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    req = (DEST/'requirements.txt').read_text() + '\ntorchdiffeq==0.2.5\n'
    (DEST/'requirements.txt').write_text(req,encoding='utf-8')
    text = (DEST/'run.py').read_text(encoding='utf-8')
    start,end = text.index('def make_model('),text.index('def move_batch(')
    text=text[:start]+'''def make_model(config,device,pretrained):
    model=AnchorFlowEdge(pretrained=pretrained,flow_steps=2,encoder=config['encoder'],
                        model_name='v11_node',phase_context_enabled=config['phase_context_enabled']).to(device)
    model.dynamics.configure(config)
    if config['channels_last']:model=model.to(memory_format=torch.channels_last)
    return model

def initialize_from_parent(model,config,check_data=True):
    if config.get('init_checkpoint'):raise ValueError('V11 is fresh-only; no V8/V9/V10 weight migration')
    return {'loaded':False,'parent_epoch':None,'parent_completed_epochs':0}


'''+text[end:]
    text=replace(text,'"geometry.py","support.py"','"geometry.py","geometry_primitives.py","ode_solver.py","support.py"')
    text=replace(text,'"phase_context.","dynamics.step_head."','"phase_context.",')
    text=text.replace('v10_lite','v11_node')
    text=replace(text,"    step_sums=torch.zeros(config['flow_steps']+1,dtype=torch.float64,device=device)","    solver_records=[]")
    line="        step_sums+=torch.stack([output[f'dynamics_step_mean_{k}'] for k in range(1,config['flow_steps']+1)]+[output['dynamics_terminal_time_mean']]).double()*n"
    text=replace(text,line,"        solver_records.append(dict(model.dynamics.last_solver_report))")
    start=text.index('            "integration":{"executed_nfe":')
    end=text.index('\n\n\ndef learning_rate',start)
    text=text[:start]+'''            "integration":solver_summary(solver_records,model.dynamics.method)}


def solver_summary(records,method):
    def values(key):return [r[key] for r in records]
    nfe=values('nfe');accepted=values('accepted_steps');rejected=values('rejected_steps')
    return {'method':method,'library':'torchdiffeq==0.2.5' if method=='bosh3' else 'explicit fixed midpoint',
        'nfe_mean':float(np.mean(nfe)),'nfe_p50':float(np.percentile(nfe,50)),
        'nfe_p95':float(np.percentile(nfe,95)),'nfe_max':max(nfe),
        'nfe_histogram':{str(k):nfe.count(k) for k in sorted(set(nfe))},
        'accepted_steps_mean':float(np.mean(accepted)),'rejected_steps_mean':float(np.mean(rejected)),
        'accepted_h_mean':float(np.mean(values('accepted_h_mean'))),
        'terminal_time_mean':float(np.mean(values('terminal_time'))),
        'terminal_time_max_error':max(abs(t-1.) for t in values('terminal_time')),
        'learned_step_size':False,'learned_stopping':False,'fixed_terminal_time':1.,
        'nfe_includes_rejected_trials':True,'batch_step_policy':'one grid; worst sample/channel normalized RMS'}
'''+text[end:]
    text=replace(text,'"val_selected_nfe":config["flow_steps"],"val_executed_nfe":config["flow_steps"],',
                 '"val_executed_nfe":validation["integration"]["nfe_mean"],"val_nfe_p95":validation["integration"]["nfe_p95"],')
    text=replace(text,"**{f'val_h{k+1}_mean':v for k,v in enumerate(validation['integration']['step_means'])},",
                 "'val_accepted_h_mean':validation['integration']['accepted_h_mean'],'val_rejected_steps':validation['integration']['rejected_steps_mean'],")
    text=replace(text,"'student_checkpoint_loaded':parent['loaded'],'executed_nfe':config['flow_steps'],",
                 "'student_checkpoint_loaded':False,'solver':model.dynamics.last_solver_report,")
    # Fixed-time samples are not accepted-step numbers (adaptive grids differ).
    text=text.replace('All available D4_steps share the quarter-grid target.','D4_step1/2 are samples at t=0.5/1.0, NOT solver-step numbers, and share the quarter-grid target.')
    text=replace(text,"ds=KITTIDataset(config,'val',teacher=False);times=[]","ds=KITTIDataset(config,'val',teacher=False);times=[];solver_records=[]")
    text=replace(text,"        times.append((time.perf_counter()-started)*1000)","        times.append((time.perf_counter()-started)*1000)\n        solver_records.append(dict(model.dynamics.last_solver_report))")
    text=replace(text,"'wall_p95_ms':float(np.percentile(times,95)),'executed_nfe':config['flow_steps'],",
                 "'wall_p95_ms':float(np.percentile(times,95)),'solver':solver_summary(solver_records,model.dynamics.method),")
    text=text.replace('fixed two/three shared jet Euler steps','solver-controlled NODE to fixed T=1')
    # Adaptive loops cannot be faithfully traced into a static ONNX graph.
    text=replace(text,'    model = model.to(memory_format=torch.contiguous_format)',
                 "    model.dynamics.method='midpoint' # Explicit DIFFERENT solver; never trace adaptive Python branches.\n    model = model.to(memory_format=torch.contiguous_format)")
    text=replace(text,"    report.update(executed_nfe=config[\"flow_steps\"],learned_step_size=config['learned_step_size'],step_bounds=[config['step_min'],config['step_max']],adaptive_stopping=False)",
                 "    report.update(executed_nfe=2*config['ode_fixed_steps'],solver='fixed_midpoint',terminal_time=1.,\n                  learned_step_size=False,adaptive_export=False,parity_scope='ONNX versus fixed-midpoint PyTorch, NOT adaptive bosh3')")
    text=replace(text,"('prepare','smoke','train','evaluate','test','profile','export')","('prepare','smoke','train','evaluate','test','profile','export','solver_audit')")
    start=text.index("    if config['model_name']!='v11_node'")
    end=text.index("    if config['amp']",start)
    text=text[:start]+'''    if config['model_name']!='v11_node' or config['flow_steps']!=2:raise ValueError('V11 model/sample contract violated')
    if config.get('init_checkpoint') or config['ode_terminal_time']!=1.:raise ValueError('Fresh-only V11 requires fixed T=1')
    if config['compile']:raise ValueError('Adaptive torchdiffeq is eager; do not claim torch.compile/static graph')
'''+text[end:]
    text=replace(text,'if __name__ == "__main__":','''@torch.inference_mode()
def solver_audit(config,variant):
    device=device_setup(config);model,ckpt=load_trained(config,variant,device)
    _,drive=directories(config,variant)
    rows=[]
    for label,method,rtol,atol in [('adaptive_train_solver','bosh3',config['ode_rtol'],config['ode_atol']),
                                  ('fixed_midpoint8','midpoint',config['ode_rtol'],config['ode_atol']),
                                  ('adaptive_tighter_reference','bosh3',.002,.0002)]:
        model.dynamics.method,model.dynamics.rtol,model.dynamics.atol=method,rtol,atol
        if label=='adaptive_tighter_reference':model.dynamics.max_nfe=769
        start=time.perf_counter()
        report=validate(model,data_loader(config,'val'),config,device)
        report.update(checkpoint_epoch=ckpt['epoch'],solver_label=label,rtol=rtol,atol=atol,
                      validation_wall_seconds_including_metrics=time.perf_counter()-start)
        write_json(drive/f'val_metrics_{label}.json',report)
        rows.append({'solver':label,'epoch':ckpt['epoch'],**report['final']['all'],**report['integration']})
    write_json(drive/'solver_comparison.json',rows)
    print(json.dumps(rows,indent=2),flush=True)


if __name__ == "__main__":''')
    (DEST/'run.py').write_text(text,encoding='utf-8')
    print('Scaffold:',DEST)

if __name__=='__main__':main()
