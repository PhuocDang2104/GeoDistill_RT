"""Function sources spliced into the self-contained runner by its build script.

This is a build-time template, not an importable alternate training entry point.
"""

def make_model(config,device,pretrained):
    model=AnchorFlowEdge(pretrained=pretrained,flow_steps=config['flow_steps'],encoder=config['encoder'],
                        phase_context_enabled=config['phase_context_enabled']).to(device)
    if config['channels_last']:model=model.to(memory_format=torch.channels_last)
    return model


def initialize_from_parent(model,config,check_data=True):
    if not config.get('init_checkpoint'):
        return {'loaded':False,'parent_epoch':None,'parent_completed_epochs':0}
    path=Path(config['init_checkpoint'])
    if not path.is_file():raise FileNotFoundError(f'V10 best.pth missing: {path}; use fresh40 or specify the real Drive checkpoint')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('protocol',{}).get('source_sha256')!=config['parent_source_sha256']:
        raise RuntimeError('Warm start requires the completed, sealed original V10 source')
    if p['epoch']!=23:raise RuntimeError('Default migration is pinned to completed V10 best epoch23; do not silently initialize from last.pth')
    old_data=p['protocol']['data']
    if check_data:
        new_data=json.loads((Path(config['work'])/'data_contract.json').read_text())
        for key in ('manifest_sha256','train','val','test','size','depth_scale','train_split_sha256','val_split_sha256'):
            if key in old_data and old_data.get(key)!=new_data.get(key):raise RuntimeError(f'Parent data protocol differs: {key}')
        for name in ('teacher_report','relative_teacher'):
            a,b=old_data.get(name,{}),new_data.get(name,{})
            for key in ('content_sha256','recipe_sha256','manifest_sha256'):
                if config.get('relative_enabled',False) or name=='teacher_report':
                    if key in a and a.get(key)!=b.get(key):raise RuntimeError(f'Parent teacher differs: {name}/{key}')
    result=model.load_state_dict(p['model'],strict=False)
    if any(not k.startswith('phase_context.') for k in result.missing_keys):raise RuntimeError(f'Missing legacy weights: {result.missing_keys}')
    if any(not k.startswith('dynamics.controller.') for k in result.unexpected_keys):raise RuntimeError(f'Unexpected parent keys: {result.unexpected_keys}')
    # Last projection must remain exactly zero for model-only migration.
    if not torch.equal(model.phase_context[-1].weight,torch.zeros_like(model.phase_context[-1].weight)):
        raise RuntimeError('New context contribution must initialize as no-op')
    print('V10 epoch23 MODEL-only loaded; removed controller, new context no-op; optimizer reset',flush=True)
    return {'loaded':True,'parent_epoch':p['epoch'],'parent_completed_epochs':p['epoch']+1,
            'parent_run_completed_epochs':32,'checkpoint_sha256':digest(path),
            'removed_keys':result.unexpected_keys,'new_keys':result.missing_keys,
            'model_only':True,'initial_output_exactly_identical':False,
            'note':'Fixed h=1/3 approximates learned saturated h. Measure initial validation before training.'}


def smoke(config):
    device=device_setup(config);seed_everything(config['seed'])
    model=make_model(config,device,pretrained=config['encoder_pretrained'] and not config.get('init_checkpoint'))
    parent=initialize_from_parent(model,config);model.train()
    if config['freeze_encoder_bn']:model.freeze_encoder_bn()
    batch=move_batch(next(iter(data_loader(config,'train',batch_size=1))),config,device)
    mask,holdout=input_with_holdout(batch,config['holdout_rate'])
    with smoke_failure_observer(model,config),autocast(config,device):output=model(batch['rgb'],batch['sparse'],mask,batch['K'])
    loss,stats=objective(output,batch,holdout,1,config)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke objective')
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite smoke gradients')
    if not torch.equal(output['D_hard'][mask.bool()],batch['sparse'][mask.bool()]):raise RuntimeError('Hard diagnostic failed')
    for step in range(1,config['flow_steps']+1):
        if not float(stats[f'dynamics_state_change_{step}'])>0:raise RuntimeError('Feedback state did not evolve')
    if config['teacher_enabled'] and not float(stats['kd_coverage'])>0:raise RuntimeError('Empty metric KD coverage')
    if config.get('relative_enabled',False) and not float(stats['relative_pair_coverage'])>0:raise RuntimeError('Empty relative KD coverage')
    reaction=model.dynamics.reaction.weight.grad
    if reaction is None or not reaction.abs().sum()>0:raise RuntimeError('Reaction field has no gradient')
    new=model.phase_context[-1].weight.grad
    if config['phase_context_enabled'] and parent['loaded'] and (new is None or not new.abs().sum()>0):raise RuntimeError('Warm-start context has no learning signal')
    report={'passed':True,'loss':float(loss.detach()),'parameters':sum(p.numel() for p in model.parameters()),
       'device':str(device),'teacher_at_inference':False,'parent_initialization':parent,
       'student_checkpoint_loaded':parent['loaded'],'executed_nfe':config['flow_steps'],
       'reaction_gradient_l1':float(reaction.abs().sum()),'phase_context_gradient_l1':float(new.abs().sum()) if new is not None else 0.,
       'kd_coverage':float(stats['kd_coverage']),'relative_pair_coverage':float(stats['relative_pair_coverage']),
       'inverse_rmse_km_inv':float(stats['inverse_rmse_km_inv']),
       'cold_zero_head_note':'Fresh phase/detail heads are zero-init; context gradient opens after those old heads first update.'}
    write_json(Path(config['work'])/'smoke_report.json',report);print(json.dumps(report,indent=2),flush=True)


def profile_real(model,config,device,untrained):
    if untrained:return {'measured':False,'reason':'Real-scene benchmark requires trained model and prepared validation'}
    ds=KITTIDataset(config,'val',teacher=False);times=[]
    for i in range(100):
        b=move_batch({k:v[None] if torch.is_tensor(v) else v for k,v in ds[(i*13)%400].items()},config,device)
        if device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        with autocast(config,device):model(*(b[k] for k in ('rgb','sparse','mask','K')))
        if device.type=='cuda':torch.cuda.synchronize()
        times.append((time.perf_counter()-started)*1000)
    return {'samples':100,'indices':'i*13 mod400','wall_median_ms':float(np.median(times)),
            'wall_p95_ms':float(np.percentile(times,95)),'executed_nfe':config['flow_steps'],
            'includes_host_launch_and_finish_sync':True,'excludes_io_and_h2d':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','smoke','train','evaluate','test','profile','export'))
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--variant',choices=('dual_teacher','metric_kd','gt_only'),default='dual_teacher')
    parser.add_argument('--untrained',action='store_true')
    parser.add_argument('--checkpoint-selection',choices=('rmse','inverse','joint'),default=None)
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    config['teacher_enabled']=args.variant!='gt_only';config['relative_enabled']=args.variant=='dual_teacher'
    if args.checkpoint_selection:config['checkpoint_selection']=args.checkpoint_selection
    if config['model_name']!='v10_lite' or config['flow_steps'] not in (2,3) or config['step_size']!=1/3:raise ValueError('Fixed-graph V10.1 model contract violated')
    if config['amp'] not in ('bf16','fp32'):raise ValueError('Native BF16 or explicit FP32 only')
    if not 1<=config['epochs']<=40 or config['accumulation']<1:raise ValueError('Training budget outside1..40')
    if config['early_stop_patience']<1 or config['early_stop_min_epochs']<1 or config['early_stop_min_delta_m']<0:raise ValueError('Invalid early stopping')
    if Path(config['run_name']).name!=config['run_name'] or config['run_name'] in ('','.','..'):raise ValueError('Invalid run_name')
    if args.command in ('profile','export'):globals()[args.command](config,args.variant,args.untrained)
    elif args.command in ('prepare','smoke'):globals()[args.command](config)
    else:globals()[args.command](config,args.variant)
