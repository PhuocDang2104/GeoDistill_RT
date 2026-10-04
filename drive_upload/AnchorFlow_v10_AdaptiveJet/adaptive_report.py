"""Small device-side diagnostics; transfer only accumulated scalar statistics."""
import numpy as np
import torch
from loss_helpers import pooled
from losses import stop_targets


class AdaptiveMetrics:
    def __init__(self,device,config):
        self.config=config
        self.sums=torch.zeros(5,7,dtype=torch.float64,device=device)
        self.records=[]

    @torch.no_grad()
    def update(self,pred,batch):
        target,support=pooled(batch['gt'].float(),batch['gt_mask'].float(),pred['D4'].shape[-2:])
        valid=support>0
        for k in range(5):
            if k==4 and self.config['integration_policy']=='fixed3': continue
            depth=pred[f'D4_step{k}'].float(); error=depth-target
            inverse=1000*(depth.clamp_min(1e-6).reciprocal()-target.clamp_min(1e-6).reciprocal())
            tail=valid&(error.abs()>5)
            self.sums[k]+=torch.stack((valid.sum(),(error.square()*valid).sum(dtype=torch.float64),
                (error.abs()*valid).sum(dtype=torch.float64),(inverse.square()*valid).sum(dtype=torch.float64),
                (valid&(error.abs()>2)).sum(),tail.sum(),(error.square()*tail).sum(dtype=torch.float64)))
        labels,benefit,valid_samples=stop_targets(pred,batch,self.config['benefit_margin'])
        p=pred['adaptive_stop_logits'][:,1:3].sigmoid()
        decisions=(p>=self.config['stop_threshold']).float()
        self.records.append(torch.cat((pred['adaptive_h'],p,
            pred['adaptive_selected_nfe'][:,None],pred['adaptive_executed_nfe'][:,None],
            pred['adaptive_terminal_time'][:,None],(decisions==labels).float(),labels,benefit,
            valid_samples[:,None]),1).detach())

    def report(self):
        if not self.records: return {}
        rows=torch.cat(self.records).float().cpu().numpy()
        n=rows[:,6]; executed=rows[:,7]
        result={'samples':len(rows),'selected_nfe_mean':float(n.mean()),
                'selected_nfe_p50':float(np.median(n)),'selected_nfe_p95':float(np.percentile(n,95)),
                'executed_nfe_mean':float(executed.mean()),'terminal_time_mean':float(rows[:,8].mean()),
                'terminal_time_min':float(rows[:,8].min()),'terminal_time_max':float(rows[:,8].max()),
                'exit_fraction':{str(k):float((n==k).mean()) for k in (2,3,4)},
                'h_mean_all_calls':[float(x) for x in rows[:,:4].mean(0)],
                'h_std_all_calls':[float(x) for x in rows[:,:4].std(0)],
                'stop_probability_mean':[float(x) for x in rows[:,4:6].mean(0)],
                'expected_nfe_mean':float((2+1-rows[:,4]+(1-rows[:,4])*(1-rows[:,5])).mean()),
                'stop_accuracy':{},'stop_target_fraction':{},'future_benefit_mean':{},
                'quarter_trajectory':{},
                'compute_note':'Validation executes all calls; selected NFE is a policy statistic, NOT measured savings.'}
        eligible=rows[:,15]>0
        for idx,k in enumerate((2,3)):
            if self.config['integration_policy']=='adaptive':
                result['stop_accuracy'][str(k)]=float(rows[eligible,9+idx].mean()) if eligible.any() else None
                result['stop_target_fraction'][str(k)]=float(rows[eligible,11+idx].mean()) if eligible.any() else None
            result['future_benefit_mean'][str(k)]=float(rows[eligible,13+idx].mean()) if eligible.any() else None
        for k,(count,sse,sae,ise,tail2,tail5,tail_sse) in enumerate(self.sums.cpu().tolist()):
            result['quarter_trajectory']['D0' if k==0 else f'D4_step{k}']={
                'pixels':int(count),'rmse_m':(sse/count)**.5 if count else None,
                'mae_m':sae/count if count else None,'irmse_km_inv':(ise/count)**.5 if count else None,
                'tail_gt2_fraction':tail2/count if count else None,'tail_gt5_fraction':tail5/count if count else None,
                'tail_gt5_sse_share':tail_sse/sse if sse else None,'tail_gt5_sse_m2':tail_sse}
        return result
