"""Single-GPU server training with deterministic consumed-batch resume and SIGTERM commit."""
import csv
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from full_data import FullKITTI, ResumableBatches, SIZE
from utils import atomic_json, identity, read_json, sha256, source_identity, numerical_environment
from server_runtime.recovery import (capture_rng, restore_rng, commit_checkpoint,
                                     load_checkpoint, alias_checkpoint, reconcile_logs)


def loader(cfg, split, epoch=0, cursor=0, teachers=False, batch_size=None):
    from adapter import worker_setup
    ds = FullKITTI(cfg, split, teachers)
    tc = cfg['training']; workers = tc['workers']
    options = {'prefetch_factor': tc['prefetch_factor'], 'persistent_workers': True,
               'worker_init_fn': worker_setup} if workers else {}
    # Loader/worker seeds never consume the augmentation/model global RNG.
    options['generator'] = torch.Generator().manual_seed(tc['seed']+epoch)
    if split == 'train':
        sampler = ResumableBatches(len(ds), batch_size or tc['batch_size'], tc['seed'], epoch, cursor)
        return DataLoader(ds,batch_sampler=sampler,num_workers=workers,pin_memory=True,**options)
    return DataLoader(ds,batch_size=batch_size or 1,shuffle=False,num_workers=workers,pin_memory=True,**options)


def contract(cfg):
    tc = dict(cfg['training'])
    for key in ('workers','prefetch_factor','log_every','checkpoint_steps','checkpoint_seconds'):
        tc.pop(key,None)
    from adapter import model_config
    mc=model_config(cfg)
    # Include effective frozen model/loss defaults, not just the override keys.
    for key in ('workers','prefetch_factor','log_every','checkpoint_steps','checkpoint_seconds',
                'drive_data','drive_runs','work','run_name'):
        mc.pop(key,None)
    return {'architecture':'V11.3 unchanged', 'training':tc,'effective_model_objective':mc,
            'numerical_environment':numerical_environment(),
            'source_sha256':source_identity(), 'data_gate':read_json(Path(cfg['work'])/'data_gate.json')}


def setup(cfg):
    from adapter import precision
    if not torch.cuda.is_available():
        raise RuntimeError('Training/smoke requires GPU; CPU path is tests only')
    device = torch.device('cuda')
    with precision(cfg,device):
        # Exercise native CUDA kernels before loading teachers/student.
        x=torch.randn(16,16,device=device); (x@x).sum().item()
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=cfg['training']['cudnn_benchmark']
    torch.backends.cudnn.enabled=True
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    seed=cfg['training']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return device


def smoke(cfg):
    from adapter import ModelAdapter,model_config,move_batch,input_with_holdout
    device=setup(cfg); model=ModelAdapter(cfg,device,pretrained=False)
    batch=move_batch(next(iter(loader(cfg,'train',teachers=True,batch_size=cfg['training']['batch_size']))),model_config(cfg),device)
    model.network.train(); model.network.freeze_encoder_bn()
    mask,holdout=input_with_holdout(batch,cfg['training']['holdout_rate'])
    out=model.forward(batch,mask); loss,stats=model.loss(out,batch,holdout,4.)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite real-data smoke loss')
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.network.parameters()):
        raise RuntimeError('Nonfinite real-data smoke gradients')
    if not float(stats['kd_coverage'])>0 or not float(stats['relative_pair_coverage'])>0:
        raise RuntimeError('Empty teacher loss coverage')
    if model.network.fine_node.last_solver_report['nfe']!=2:
        raise RuntimeError('Fine NODE must execute exactly two NFE')
    report={'passed':True,'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
            'loss':float(loss.detach()),'batch':batch['rgb'].shape[0],'parameters':sum(p.numel() for p in model.network.parameters()),
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'coarse_solver':model.network.dynamics.last_solver_report,
            'fine_solver':model.network.fine_node.last_solver_report,
            'full_objective_backward':True,'smoke_optimizer_step':False}
    atomic_json(Path(cfg['run_dir'])/'smoke_report.json',report)
    print(json.dumps(report,indent=2),flush=True)


def finish_aliases(root,payload):
    for name in payload.get('alias_updates',[]):
        alias_checkpoint(root/'checkpoints'/'last.pth',root/'checkpoints'/name)


def train(cfg,stop):
    from adapter import ModelAdapter,model_config,move_batch,augment,input_with_holdout,learning_rate,validate,update_policy_best,early_stop_update
    device=setup(cfg); tc=cfg['training']; mc=model_config(cfg)
    root=Path(cfg['run_dir']); root.mkdir(parents=True,exist_ok=True)
    last=root/'checkpoints'/'last.pth'; recipe=contract(cfg)
    model=ModelAdapter(cfg,device,pretrained=not last.exists())
    encoder_ids={id(p) for p in model.network.encoder.parameters()}
    groups=[{'params':list(model.network.encoder.parameters()),'lr_scale':tc['encoder_lr_ratio']},
            {'params':[p for p in model.network.parameters() if id(p) not in encoder_ids],'lr_scale':1.}]
    optimizer=torch.optim.AdamW(groups,lr=tc['learning_rate'],weight_decay=tc['weight_decay'],fused=tc['fused_adamw'])
    state={'epoch':0,'cursor':0,'global_step':0,'best_rmse':float('inf'),'best_inverse':float('inf'),
           'selection':{},'stopping':{'bad_epochs':0,'stopped':False},'totals':{},'seen':0}
    if last.exists():
        payload=load_checkpoint(last)
        if payload['contract']!=recipe:
            raise RuntimeError('Resume contract changed (source/config/data/teacher/batch). New run required.')
        model.network.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        state=payload['state'];restore_rng(payload['rng']);finish_aliases(root,payload)
        reconcile_logs(root,state['epoch']-1)
        print(f"Resume epoch={state['epoch']} consumed_batches={state['cursor']} global_step={state['global_step']}",flush=True)
    atomic_json(root/'resolved_config.json',cfg)
    atomic_json(root/'run_manifest.json',{'contract':recipe,'torch':str(torch.__version__),
         'gpu':torch.cuda.get_device_name(0),'initialization':'fresh student, ImageNet RGB encoder only',
         'model_parameters':sum(p.numel() for p in model.network.parameters()),
         'full_training_images':len(FullKITTI(cfg,'train')),'validation_images':len(FullKITTI(cfg,'val')),
         'teacher_inputs_at_inference':False,'single_gpu':True,'midbatch_resume':True})
    last_save=time.monotonic()
    def save(aliases=()):
        nonlocal last_save
        payload={'epoch':state['epoch'],'model':model.network.state_dict(),'optimizer':optimizer.state_dict(),
                 'state':state.copy(),'rng':capture_rng(),'contract':recipe,'alias_updates':list(aliases),
                 'progress':{'global_step':state['global_step'],'consumed_batches':state['cursor']}}
        commit_checkpoint(last,payload);finish_aliases(root,payload);last_save=time.monotonic()
        # Keep snapshots referenced by last and ALL metric-selection pointers.
        keep={read_json(p)['file'] for p in (root/'checkpoints').glob('*.json')}
        for p in (root/'checkpoints'/'snapshots').glob('*.pth'):
            if 'snapshots/'+p.name not in keep:
                p.unlink();p.with_suffix('.json').unlink(missing_ok=True)
    n_batches=math.ceil(len(FullKITTI(cfg,'train'))/tc['batch_size'])
    total_updates=n_batches*tc['epochs'];warmup=n_batches*tc['warmup_epochs']
    val_loader=loader(cfg,'val')
    if state['stopping']['stopped'] or state['epoch']>=tc['epochs']:
        return True
    while state['epoch']<tc['epochs']:
        epoch=state['epoch'];start_cursor=state['cursor'];started=time.monotonic()
        train_loader=loader(cfg,'train',epoch,start_cursor,teachers=True)
        model.network.train();model.network.freeze_encoder_bn();optimizer.zero_grad(set_to_none=True)
        totals={k:torch.tensor(v,device=device) for k,v in state['totals'].items()}
        for batch_index,batch in enumerate(train_loader,start_cursor):
            batch=augment(move_batch(batch,mc,device),mc)
            mask,holdout=input_with_holdout(batch,tc['holdout_rate'])
            output=model.forward(batch,mask)
            loss,stats=model.loss(output,batch,holdout,epoch+batch_index/n_batches)
            if not bool(torch.isfinite(loss)):
                atomic_json(root/'failure.json',{'epoch':epoch,'batch':batch_index,'sample_ids':batch['sid'],
                                                'reason':'Nonfinite loss BEFORE backward; no NaN masking or skip'})
                raise RuntimeError('Nonfinite training loss; last COMMITTED checkpoint preserved')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.network.parameters(),tc['gradient_clip'],error_if_nonfinite=True)
            factor=learning_rate(state['global_step'],total_updates,warmup,tc['min_lr_ratio'])
            for group in optimizer.param_groups:
                group['lr']=tc['learning_rate']*group['lr_scale']*factor
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            n=batch['rgb'].shape[0]
            for k,v in stats.items():
                totals[k]=totals.get(k,torch.zeros((),device=device))+v.float()*n
            state.update(cursor=batch_index+1,global_step=state['global_step']+1,seen=state['seen']+n)
            stopping=stop.requested or (Path(cfg['work'])/'STOP').exists()
            due=(state['global_step']%tc['checkpoint_steps']==0 or time.monotonic()-last_save>=tc['checkpoint_seconds'])
            if due or stopping:
                keys=list(totals);vals=torch.stack([totals[k] for k in keys]).cpu().tolist()
                state['totals']=dict(zip(keys,vals));save()
                atomic_json(root/'training_status.json',{'status':'paused' if stopping else 'training',
                            'epoch':epoch,'consumed_batches':state['cursor'],'global_step':state['global_step']})
            if (batch_index+1)%tc['log_every']==0:
                print(f"epoch={epoch} batch={batch_index+1}/{n_batches} loss={float(loss):.5f} lr={optimizer.param_groups[-1]['lr']:.3g}",flush=True)
            if stopping:
                return False
        del train_loader
        # Interruption during validation replays validation only, not the trained epoch.
        state['totals']={k:float(v) for k,v in totals.items()};save()
        model.network.eval();report=validate(model.network,val_loader,mc,device)
        all_score=report['final']['all'];rmse=all_score['rmse_m'];inverse=all_score['irmse_km_inv']
        if not math.isfinite(rmse) or not math.isfinite(inverse):
            raise RuntimeError('Nonfinite validation; no checkpoint selection')
        aliases=[]
        if rmse<state['best_rmse']:
            state['best_rmse']=rmse;aliases.append('best.pth')
        if inverse<state['best_inverse']:
            state['best_inverse']=inverse;aliases.append('best_inverse.pth')
        joint=max(rmse/mc['rmse_target'],inverse/mc['inverse_target'])
        if joint<state['selection'].get('joint_best',float('inf')):
            state['selection']['joint_best']=joint;aliases.append('best_joint.pth')
        state['selection'],improved=update_policy_best(state['selection'],rmse,inverse,epoch,mc)
        if improved:
            aliases.append('best_policy.pth')
        state['stopping']=early_stop_update(state['stopping'],rmse,epoch,mc,inverse)
        row={'epoch':epoch,'global_step':state['global_step'],'train_images':state['seen'],
             'val_rmse_m':rmse,'val_irmse_km_inv':inverse,'epoch_seconds_this_session':time.monotonic()-started,
             **{k:v/state['seen'] for k,v in state['totals'].items()}}
        for label,score in report['final'].items():
            if isinstance(score,dict) and 'rmse_m' in score:
                row['val_'+label+'_rmse_m']=score['rmse_m']
        for path in (root/'train_log.csv',):
            exists=path.exists()
            with path.open('a',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if not exists:writer.writeheader()
                writer.writerow(row);f.flush();os.fsync(f.fileno())
        with (root/'train_log.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        atomic_json(root/'last_val_metrics.json',{'checkpoint_epoch':epoch,**report})
        for name in aliases:
            atomic_json(root/(name.replace('.pth','_val_metrics.json')),{'checkpoint_epoch':epoch,**report})
        state.update(epoch=epoch+1,cursor=0,totals={},seen=0);save(aliases)
        atomic_json(root/'training_status.json',{'status':'early_stopped' if state['stopping']['stopped'] else 'training',
                   'epochs_completed':state['epoch'],'best_rmse_m':state['best_rmse'],
                   'best_inverse_km_inv':state['best_inverse'],'selection':state['selection'],'early_stopping':state['stopping']})
        print(f'epoch={epoch} RMSE={rmse:.6f} m iRMSE={inverse:.6f} km^-1',flush=True)
        if state['stopping']['stopped']:
            return True
    atomic_json(root/'training_complete.json',{'epochs_completed':state['epoch'],'selection':state['selection']})
    return True


@torch.inference_mode()
def evaluate_test(cfg):
    from adapter import ModelAdapter,model_config,move_batch,validate
    device=setup(cfg);root=Path(cfg['run_dir']);mc=model_config(cfg)
    payload=load_checkpoint(root/'checkpoints'/'best_policy.pth')
    if payload['contract']!=contract(cfg):
        raise RuntimeError('Inference checkpoint/source/data protocol mismatch')
    model=ModelAdapter(cfg,device);model.network.load_state_dict(payload['model'],strict=True);model.network.eval()
    report=validate(model.network,loader(cfg,'val'),mc,device)
    atomic_json(root/'val_metrics_policy.json',report)
    model.network.set_diagnostics(False)
    preview=root/'test_preview';preview.mkdir(exist_ok=True)
    tmp=root/'kitti_test_predictions.zip.partial'
    indices=set(np.linspace(0,len(FullKITTI(cfg,'test'))-1,10,dtype=int).tolist())
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
        for i,batch in enumerate(loader(cfg,'test')):
            gpu=move_batch(batch,mc,device);output=model.forward(gpu)['D_full'][0,0].float().cpu().numpy()
            if output.shape!=SIZE or not np.isfinite(output).all() or output.min()<=0:
                raise RuntimeError('Invalid test depth')
            ok,png=cv2.imencode('.png',np.clip(np.rint(output*256),1,65535).astype(np.uint16))
            if not ok:raise RuntimeError('PNG encode failure')
            z.writestr(batch['sid'][0]+'.png',png.tobytes())
            if i in indices:
                rgb=(batch['rgb'][0].permute(1,2,0).numpy()*255).round().astype(np.uint8)[:,:,::-1].copy()
                # Fixed metric scale across every example: near warm/far cool.
                color=cv2.applyColorMap(np.rint((1-np.clip(output/80,0,1))*255).astype(np.uint8),cv2.COLORMAP_TURBO)
                canvas=np.concatenate((rgb,color),axis=0)
                cv2.putText(canvas,'RGB / V11.3 depth | TURBO: 0 m warm -> 80 m cool (clipped)',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                if not cv2.imwrite(str(preview/(batch['sid'][0]+'.png')),canvas):raise RuntimeError('Preview write failed')
            if (i+1)%100==0:print(f'Test {i+1}/1000',flush=True)
    expected = len(FullKITTI(cfg,'test'))
    with zipfile.ZipFile(tmp) as z:
        if len(z.namelist()) != expected or len(set(z.namelist())) != expected:
            raise RuntimeError('Anonymous test ZIP must contain exactly one depth PNG per test ID')
    tmp.replace(root/'kitti_test_predictions.zip')
    atomic_json(root/'final_summary.json',{'checkpoint_selection':'best_policy',
          'checkpoint_epoch':payload['state']['epoch']-1,'validation':report['final'],
          'test_count':len(FullKITTI(cfg,'test')),'test_has_public_gt':False,'preview_count':len(indices),
          'preview_scale_m':[0,80],'prediction_units':'uint16 round(depth_m*256)'})
