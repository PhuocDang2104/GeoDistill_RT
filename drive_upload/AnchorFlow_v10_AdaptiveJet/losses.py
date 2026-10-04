"""Keep selected V9.1 loss; replace its dynamics auxiliary, add stop supervision."""
import torch
from torch.nn import functional as F
from losses_baseline import objective as baseline_objective
from loss_helpers import pooled,mean_masked,huber


def stop_targets(pred,batch,margin=.003):
    target,support=pooled(batch['gt'].float(),batch['gt_mask'].float(),pred['D4'].shape[-2:])
    valid=(support>0).float(); count=valid.flatten(1).sum(1)
    quality=[]
    for k in (1,2,3,4):
        error=huber(pred[f'D4_step{k}'].float()-target)
        quality.append((error*valid).flatten(1).sum(1)/count.clamp_min(1))
    quality=torch.stack(quality,1).detach()
    benefit=(quality[:,1:3]-quality[:,2:4])/quality[:,1:3].clamp_min(1e-6)
    labels=(benefit<=margin).float()
    return labels,benefit,(count>0).float()


def objective(pred,batch,holdout_mask,epoch_progress,config):
    policy=config['integration_policy']
    # Fixed3 control is the exact original V9.1 objective and auxiliary weighting.
    legacy={**config,'model_name':'v9_metric_refine'}
    if policy!='fixed3': legacy['dynamics_aux_weight']=0
    total,stats=baseline_objective(pred,batch,holdout_mask,epoch_progress,legacy)
    zero=pred['D_full'].sum()*0
    traj=controller=compute=zero
    if policy!='fixed3':
        target,support=pooled(batch['gt'].float(),batch['gt_mask'].float(),pred['D4'].shape[-2:])
        weights=config.get('trajectory_weights',[.15,.25,.30,.30])
        if len(weights)!=4 or abs(sum(weights)-1)>1e-6: raise ValueError('Trajectory weights must sum to one')
        for k,w in enumerate(weights,1): traj=traj+w*mean_masked(huber(pred[f'D4_step{k}']-target),(support>0).float())
        total=total+config.get('dynamics_aux_weight',.1)*traj
        stats.update(dynamics_aux=traj.detach(),weighted_dynamics_aux=(config.get('dynamics_aux_weight',.1)*traj).detach())
    if policy=='adaptive':
        labels,benefit,valid=stop_targets(pred,batch,config.get('benefit_margin',.003))
        logits=pred['adaptive_stop_logits'][:,1:3]
        bce=F.binary_cross_entropy_with_logits(logits.float(),labels,reduction='none')
        controller=(bce*valid[:,None]).sum()/(2*valid.sum().clamp_min(1))
        p=logits.sigmoid(); expected=2+(1-p[:,0])+(1-p[:,0])*(1-p[:,1])
        compute=expected.mean()
        total=total+config.get('controller_weight',.02)*controller+config.get('compute_weight',0)*compute
        for idx,k in enumerate((2,3)):
            stats[f'stop{k}_target_fraction']=labels[:,idx].mean()
            stats[f'stop{k}_accuracy']=(((p[:,idx]>=config['stop_threshold'])==labels[:,idx].bool()).float()*valid).sum()/valid.sum().clamp_min(1)
            stats[f'benefit{k}_mean']=benefit[:,idx].mean()
        stats['expected_nfe']=compute.detach()
    stats.update(total=total.detach(),controller_bce=controller.detach(),
        weighted_controller=(config.get('controller_weight',.02)*controller).detach(),
        weighted_compute=(config.get('compute_weight',0)*compute).detach(),
        selected_nfe=pred['adaptive_selected_nfe'].float().mean(),
        executed_nfe=pred['adaptive_executed_nfe'].float().mean(),
        terminal_time=pred['adaptive_terminal_time'].mean())
    for k in range(4): stats[f'h{k+1}']=pred['adaptive_h'][:,k].mean()
    return total,{k:v.detach() for k,v in stats.items()}
