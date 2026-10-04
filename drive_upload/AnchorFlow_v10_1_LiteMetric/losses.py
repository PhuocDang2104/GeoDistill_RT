"""GT-first metric/inverse objective. No teacher/GT enters inference.

All losses use FP32. Direct inverse RMSE is measured in km^-1, matches the
evaluation quantity, and uses a scratch-safe ramp rather than depth clipping.
"""
import torch
from torch.nn import functional as F
from loss_helpers import pooled,mean_masked,huber,pair_gradient,teacher_weights,range_rmse
from boundaries import boundary_mask,band_mask,balanced_barrier_loss
from relative_loss import relative_structure


def smooth_rmse(error,valid,epsilon=1e-6):
    return (mean_masked(error.float().square(),valid.float())+epsilon).sqrt()-epsilon**.5


def sensor_target(batch):
    gt,sparse=batch['gt'].float(),batch['sparse'].float()
    quality=torch.exp(-(sparse-gt).abs()/(.25+.01*gt.clamp_min(.1)))
    return torch.where(batch['gt_mask']>0,quality,torch.ones_like(quality))


def objective(pred,batch,holdout_mask,epoch_progress,config):
    gt,valid=batch['gt'].float(),batch['gt_mask'].float()
    final,raw=pred['D_full'].float(),pred['D1'].float()
    zero=final.sum()*0
    metric=zero;parts={}
    for name,w in (('D16',.025),('D8',.05),('D4',.15),('D2',.30),('D1',.50),('D_full',1.)):
        target,support=pooled(gt,valid,pred[name].shape[-2:])
        term=mean_masked(huber(pred[name].float()-target),(support>0).float())
        parts['gt_'+name]=term;metric=metric+w*term
    rmse=smooth_rmse(final-gt,valid)
    balanced,active_bins=range_rmse(final,gt,valid,config.get('range_min_pixels',64))
    quality=sensor_target(batch);observed=batch['mask']*(1-holdout_mask)
    sparse=(huber(raw-batch['sparse'])*observed*quality).sum()/observed.sum().clamp_min(1)
    holdout=(huber(raw-batch['sparse'])*holdout_mask*quality).sum()/holdout_mask.sum().clamp_min(1)
    trust_mask=observed*valid;trust_weights=1+3*(1-quality)
    trust=(F.binary_cross_entropy_with_logits(pred['sensor_logits'].float(),quality,reduction='none')*
           trust_mask*trust_weights).sum()/(trust_mask*trust_weights).sum().clamp_min(1)
    kd,kd_edge,coverage,mean_conf=zero,zero,zero,zero
    fraction=min(1.,max(0.,float(epoch_progress))/max(1,config['epochs']))
    kd_weight=config['kd_weight']*(1-.5*fraction) if config['teacher_enabled'] else 0.
    if config['teacher_enabled']:
        teacher,confidence,forbidden,eligible=teacher_weights(batch,config['kd_conf_min'])
        coverage=eligible.float().mean();mean_conf=confidence.sum()/eligible.sum().clamp_min(1)
        for name,w in (('D4',.25),('D2',.50),('D1',.25)):
            depth=pred[name].float();target,weight=pooled(teacher,confidence,depth.shape[-2:])
            factor=gt.shape[-1]//depth.shape[-1]
            blocked=F.max_pool2d(forbidden,factor,factor) if factor>1 else forbidden
            weight=weight*(blocked==0)
            term=huber(depth-target)+.2*huber(depth.clamp_min(.1).log()-target.clamp_min(.1).log(),.1)
            kd=kd+w*(term*weight).sum()/(weight>0).sum().clamp_min(1)
            if name=='D2' and config.get('kd_edge_weight',0)>0:
                kd_edge=pair_gradient(depth.clamp_min(.1).log(),target.clamp_min(.1).log(),weight)
    log,edge=zero,zero
    if config.get('log_weight',0)>0:
        log=mean_masked(huber(final.clamp_min(.1).log()-gt.clamp_min(.1).log(),.1),valid)
    if config.get('edge_weight',0)>0:
        edge=pair_gradient(final.clamp_min(.1).log(),gt.clamp_min(.1).log(),valid)
    boundary,barrier,barrier_support,barrier_positive=zero,zero,zero,zero
    if config.get('boundary_weight',0)>0:
        band=band_mask(boundary_mask(gt,valid),3).float()*valid
        boundary=smooth_rmse(final-gt,band)
    if config.get('barrier_weight',0)>0:
        barrier,barrier_support,barrier_positive=balanced_barrier_loss(pred['surface_barrier_logits'],gt,valid)
    excess=F.relu((final-gt).abs()-config.get('tail_threshold_m',2.))
    tail=mean_masked(2*huber(excess,config.get('tail_huber_delta_m',10.)),valid)
    tail_ramp=min(1.,max(0.,float(epoch_progress))/max(1e-6,config.get('tail_warmup_epochs',2.)))
    target4,support4=pooled(gt,valid,pred['D4'].shape[-2:]);aux=zero
    for step in range(1,config['flow_steps']+1):
        aux=aux+mean_masked(huber(pred[f'D4_step{step}']-target4),(support4>0).float())/config['flow_steps']
    # Evaluation-bound depth is ALREADY part of the model. No new prediction
    # clamp is introduced to artificially improve the reported inverse metric.
    inv_rmse=smooth_rmse(1000*(final.reciprocal()-gt.clamp_min(.1).reciprocal()),valid)
    inv_ramp=min(1.,max(0.,float(epoch_progress))/max(1e-6,config.get('inverse_warmup_epochs',4.)))
    inverse=zero
    if config.get('inverse_weight',0)>0: # Reference-objective control only.
        for name,w in (('D4',.25),('D2',.5),('D_full',1.)):
            target,support=pooled(gt,valid,pred[name].shape[-2:])
            inverse=inverse+w*mean_masked(huber(100*(pred[name].float().reciprocal()-target.clamp_min(.1).reciprocal()),1.),(support>0).float())
    relative,relative_coverage=zero,zero
    if config.get('relative_enabled',False):
        if not {'relative','relative_confidence'}.issubset(batch):raise RuntimeError('Audited relative cache required; no fallback')
        forbidden=((batch['gt_mask']>0)|(batch['mask']>0)).float()
        relative,_,relative_coverage=relative_structure(pred['D2'],batch['relative'],batch['relative_confidence'],batch['rgb'],forbidden)
    rel_ramp=min(1.,max(0.,float(epoch_progress))/max(1e-6,config.get('relative_warmup_epochs',3.)))
    weighted={'metric':metric,'rmse':config['rmse_weight']*rmse,'range':config['range_weight']*balanced,
      'sparse':.02*sparse,'holdout':.05*holdout,'trust':.02*trust,'metric_kd':kd_weight*kd,
      'boundary':config.get('boundary_weight',0)*boundary,'barrier':config.get('barrier_weight',0)*barrier,
      'tail':config.get('tail_weight',.25)*tail_ramp*tail,'dynamics_aux':config.get('dynamics_aux_weight',.1)*aux,
      'inverse_rmse':config.get('inverse_rmse_weight',.04)*inv_ramp*inv_rmse,
      'inverse':config.get('inverse_weight',0)*inverse,
      'relative':config.get('relative_weight',.015)*rel_ramp*relative,
      'log':config.get('log_weight',0)*log,'edge':config.get('edge_weight',0)*edge,
      'teacher_edge':config.get('kd_edge_weight',0)*kd_edge}
    total=sum(weighted.values())
    stats={**parts,'total':total,'metric':metric,'rmse':rmse,'range':balanced,'range_active_bins':active_bins,
      'sparse':sparse,'holdout':holdout,'trust':trust,'metric_kd':kd,'kd_coverage':coverage,
      'kd_mean_confidence':mean_conf,'lambda_kd':zero.detach()+kd_weight,
      'sensor_gate_mean':pred['sensor_gate'].sum()/observed.sum().clamp_min(1),
      'abs_delta4_mean':pred['delta4'].abs().mean(),'abs_delta1_mean':pred['delta1'].abs().mean(),
      'boundary_rmse':boundary,'barrier':barrier,'barrier_support':barrier_support,'barrier_positive_fraction':barrier_positive,
      'tail':tail,'tail_ramp':zero.detach()+tail_ramp,'dynamics_aux':aux,'inverse_rmse_km_inv':inv_rmse,
      'inverse_ramp':zero.detach()+inv_ramp,'inverse':inverse,'relative_gradient':relative,
      'relative_pair_coverage':relative_coverage,'relative_ramp':zero.detach()+rel_ramp,
      **{k:v for k,v in pred.items() if k.startswith('dynamics_')},
      **{f'weighted_{k}':v for k,v in weighted.items()}}
    for name in ('query_uncertainty_mean','query_gate_mean','query_center_weight_mean','query_entropy_mean',
                 'innovation_abs_mean_m','innovation_support_fraction','innovation_head_raw_abs','metric_head_saturation_fraction'):
        stats[name]=pred.get(name,zero)
    return total,{k:v.detach() for k,v in stats.items()}
