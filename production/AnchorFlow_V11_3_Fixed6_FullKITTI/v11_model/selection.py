"""Predeclared dual-metric selection and patience; no post-hoc oracle policy."""
import math


def policy_key(rmse, inverse, cfg):
    if not math.isfinite(rmse) or not math.isfinite(inverse):
        raise RuntimeError('Nonfinite validation selection metrics; checkpoint not saved')
    if inverse <= cfg.get('inverse_target',3.2):
        return (0,rmse,inverse)  # Once feasible, minimize RMSE.
    return (1,max(rmse/cfg.get('rmse_target',.9),inverse/cfg.get('inverse_target',3.2)),rmse)


def update_policy_best(selection, rmse, inverse, epoch, cfg):
    result=dict(selection)
    key=policy_key(rmse,inverse,cfg)
    previous=result.get('policy_best_key')
    improved=previous is None or key<tuple(previous)
    if improved:
        result.update(policy_best_key=list(key),policy_epoch=epoch,
                      policy_rmse_m=rmse,policy_irmse_km_inv=inverse,
                      policy_inverse_feasible=key[0]==0)
    return result,improved


def early_stop_update(state, rmse, epoch, cfg, inverse=None):
    state=dict(state)
    if cfg.get('early_stop_mode','dual_policy')=='rmse':
        improved=state.get('monitor_best_rmse') is None or rmse<state['monitor_best_rmse']-cfg['early_stop_min_delta_m']
        if improved:state.update(monitor_best_rmse=rmse,bad_epochs=0)
        else:state['bad_epochs']+=1
    else:
        if inverse is None:raise ValueError('dual_policy early stop requires iRMSE')
        key=policy_key(rmse,inverse,cfg);previous=state.get('monitor_policy_key')
        meaningful=(previous is None or key[0]<previous[0])
        if previous is not None and key[0]==previous[0]:
            delta=cfg['early_stop_min_delta_m'] if key[0]==0 else cfg.get('early_stop_min_delta_joint',.001)
            meaningful=key[1]<previous[1]-delta
        if meaningful:
            state.update(monitor_policy_key=list(key),monitor_best_rmse=rmse,
                         monitor_best_irmse=inverse,bad_epochs=0)
        else:state['bad_epochs']+=1
    state['stopped']=bool(cfg['early_stopping'] and epoch+1>=cfg['early_stop_min_epochs']
                          and state['bad_epochs']>=cfg['early_stop_patience'])
    return state
