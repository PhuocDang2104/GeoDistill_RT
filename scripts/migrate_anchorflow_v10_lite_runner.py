"""Mechanical runner migration: retain stable I/O/RNG/resume, remove adaptive machinery."""
import ast
from pathlib import Path
from scaffold_anchorflow_v10_lite import OLD,OUT,ROOT,selected

def replace_function(body,name,value):
    node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=body.splitlines(True);start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
    return ''.join(lines[:start])+value+'\n'+''.join(lines[node.end_lineno:])

def change(body,old,new):
    if old not in body:raise RuntimeError('Migration source changed: '+old[:100])
    return body.replace(old,new)

def main():
    if (OUT/'run.py').exists():raise RuntimeError('Runner already migrated; edit with apply_patch')
    body=(OLD/'run.py').read_text(encoding='utf-8')
    body=change(body,'from adaptive_report import AdaptiveMetrics\n','')
    body=change(body,'("model.py", "model_baseline.py", "adaptive_report.py", "model_v9_reference.py", "model_v8.py", "support.py", "boundaries.py", "core.py", "losses.py", "losses_baseline.py", "losses_v8.py", "loss_helpers.py", "data.py", "relative_data.py", "metrics.py", "run.py")',
        '("model.py","model_base.py","geometry.py","support.py","boundaries.py","core.py","losses.py","relative_loss.py","loss_helpers.py","data.py","relative_data.py","metrics.py","run.py")')
    body=change(body,'"log_every", "profile_warmup", "profile_runs", "compile", "init_checkpoint"}',
        '"log_every", "profile_warmup", "profile_runs", "compile", "init_checkpoint", "checkpoint_selection"}')
    template=ROOT/'scripts/anchorflow_v10_lite_runtime_template.py'
    for name in ('make_model','initialize_from_parent','smoke','main'):
        body=replace_function(body,name,selected(template,(name,)))
    body=replace_function(body,'profile_policies',selected(template,('profile_real',)))
    body=replace_function(body,'policy_audit','')
    body=change(body,'drive / "best.pth", map_location="cpu"','drive / {"rmse":"best.pth","inverse":"best_inverse.pth","joint":"best_joint.pth"}[config.get("checkpoint_selection","rmse")], map_location="cpu"')
    body=change(body,'def validate(model, loader, config, device, mode="masked_adaptive"):', 'def validate(model, loader, config, device):')
    body=change(body,'    # Four-call batched rollout, hard state selection: matches conditional B1 output.\n    model.dynamics.mode = mode\n    adaptive = AdaptiveMetrics(device,config)\n','    model.set_diagnostics(True)\n')
    body=change(body,'"D4_step1", "D4_step2", "D4_step3", "D4_step4",','*[f"D4_step{k}" for k in range(1,config["flow_steps"]+1)],')
    body=change(body,'        adaptive.update(output,batch)\n','')
    body=change(body,'            if name=="D4_step4" and config["integration_policy"]=="fixed3": continue\n','')
    body=change(body,'{name:(None if name=="D4_step4" and config["integration_policy"]=="fixed3" else value)','{name:value')
    body=change(body,'if config["model_name"] == "v10_adaptive_jet" else {}','if config["model_name"] == "v10_lite" else {}')
    body=change(body,'"integration":adaptive.report()',
        '"integration":{"executed_nfe":config["flow_steps"],"selected_nfe":config["flow_steps"],"step_size":1/3,"terminal_time":config["flow_steps"]/3,"adaptive":False}')
    body=change(body,'"Native stage vs valid-area-mean GT. D0 and D4_step1/2/3/4 share the quarter-grid target.',
        '"Native stage vs valid-area-mean GT. All available D4_steps share the quarter-grid target.')
    body=change(body,'parent={"parent_epoch":23 if config.get("init_checkpoint") else None,\n            "parent_completed_epochs":30 if config.get("init_checkpoint") else 0,"loaded":bool(config.get("init_checkpoint"))}',
        'parent={"parent_epoch":None,"parent_completed_epochs":0,"loaded":False}')
    body=change(body,'if (name.startswith("phase2.metric.") or name.startswith("dynamics.controller."))','if name.startswith("phase_context.")')
    body=change(body,'        restore_rng(checkpoint, generator)','        parent=checkpoint.get("parent_initialization",parent)\n        selection=checkpoint["metric_selection"]\n        restore_rng(checkpoint, generator)')
    body=change(body,'    stopping = {"monitor_best_rmse": None, "bad_epochs": 0, "stopped": False}',
        '    stopping = {"monitor_best_rmse": None, "bad_epochs": 0, "stopped": False}\n    selection={"inverse_best":float("inf"),"joint_best":float("inf")}')
    body=change(body,'        payload["early_stopping"] = stopping\n        torch.save(payload, local / "best.pth")',
        '''        selection={"inverse_best":initial["final"]["all"]["irmse_km_inv"],
                   "joint_best":max(best/.9,initial["final"]["all"]["irmse_km_inv"]/3.2)}
        payload["early_stopping"]=stopping;payload["metric_selection"]=selection;payload["parent_initialization"]=parent
        torch.save(payload,local/"best.pth")
        for name in ("inverse","joint"):
            torch.save(payload,local/f"best_{name}.pth")
            copy_atomic(local/f"best_{name}.pth",drive/f"best_{name}.pth")
            write_json(drive/f"best_{name}_val_metrics.json",{"epoch":-1,**initial})''')
    body=change(body,'"V10: fresh ImageNet RGB only; shared geometric field with bounded learned step sizes and learned stopping. Historical V8/V9/V9.1 are NOT matched-budget causal ablations."',
        '"V10.1: fixed feedback field, context-guided final phase lift and inverse objective. Historical runs and warm starts are NOT matched-budget causal ablations."')
    start=body.index('               "val_selected_nfe":');end=body.index('               "dynamics_parameter_norm":',start)
    body=body[:start]+'''               "val_selected_nfe":config["flow_steps"],"val_executed_nfe":config["flow_steps"],
               "val_terminal_time":config["flow_steps"]/3,
'''+body[end:]
    body=change(body,'for k in (1,2,3,4)},','for k in range(1,config["flow_steps"]+1)},')
    body=change(body,'        improved = rmse < best','''        inv=score["irmse_km_inv"];joint=max(rmse/.9,inv/3.2)
        inverse_improved=inv<selection["inverse_best"];joint_improved=joint<selection["joint_best"]
        selection={"inverse_best":min(inv,selection["inverse_best"]),"joint_best":min(joint,selection["joint_best"])}
        improved = rmse < best''')
    body=change(body,'        payload["early_stopping"] = stopping\n        torch.save(payload, local / "last.pth.partial")',
        '        payload["early_stopping"]=stopping;payload["metric_selection"]=selection;payload["parent_initialization"]=parent\n        torch.save(payload, local / "last.pth.partial")')
    body=change(body,'        copy_atomic(local / "last.pth", drive / "last.pth")',
        '''        for label,flag in (("inverse",inverse_improved),("joint",joint_improved)):
            if flag:
                shutil.copy2(local/"last.pth",local/f"best_{label}.pth")
                copy_atomic(local/f"best_{label}.pth",drive/f"best_{label}.pth")
                write_json(drive/f"best_{label}_val_metrics.json",{"epoch":epoch,**validation})
        write_json(drive/"selection_metrics.json",{**selection,"joint_score_definition":"max(RMSE/0.9,iRMSE/3.2)",
            "default_checkpoint":"best.pth (minimum RMSE)","targets_achieved_together":joint<1})
        copy_atomic(local / "last.pth", drive / "last.pth")''')
    body=change(body,'    if config["integration_policy"]=="adaptive": model.dynamics.mode="adaptive_batch1"','    model.set_diagnostics(False)')
    body=change(body,'    measured = torch.compile(Deploy(model))','    model.set_diagnostics(False)\n    measured = torch.compile(Deploy(model))')
    body=change(body,'"RGB encoder + sparse pyramid + F32 context + decoder + budgeted 2-4 shared jet Euler steps (masked4 validation computes four) + phase query/detail + learned sensor fusion"',
        '"RGB encoder + sparse pyramid + F32 context + decoder + fixed two/three shared jet Euler steps + five-jet query + context-guided phase lift/detail + soft sensor fusion"')
    body=change(body,'if config["model_name"] == "v10_adaptive_jet" else "original V8 center-jet readout"','if config["model_name"] == "v10_lite" else "unknown"')
    body=change(body,'report["policy_profiles"]=profile_policies(model,config,device,untrained)','report["real_scene_profile"]=profile_real(model,config,device,untrained)')
    body=change(body,'def _export_graph(config, variant, untrained=False, graph_mode="masked_adaptive"):','def _export_graph(config, variant, untrained=False):')
    body=change(body,'    # Deterministic four-call graph selecting the exact same exit state. NO NFE saving.\n    model.dynamics.mode=graph_mode','    model.set_diagnostics(False)')
    body=change(body,'("anchorflow_static4_fp32.onnx" if graph_mode=="static4" else "anchorflow_edge_fp32.onnx")','"anchorflow_edge_fp32.onnx"')
    start=body.index('    report.update(integration_policy=');end=body.index('    print("ONNX checker',start)
    body=body[:start]+'''    report.update(executed_nfe=config["flow_steps"],step_size=1/3,adaptive=False)
    write_json(drive/"export_report.json",report)
'''+body[end:]
    body=replace_function(body,'export','def export(config,variant,untrained=False):\n    return _export_graph(config,variant,untrained)\n')
    # Export checks real first/middle/last in addition to six synthetic cases.
    body=change(body,'        batch = next(iter(data_loader(config,"val",batch_size=1)))\n        cases.append(("real_kitti_val_first",tuple(batch[k].contiguous() for k in ("rgb","sparse","mask","K"))))',
        '''        ds=KITTIDataset(config,"val",teacher=False)
        for index in (0,199,399):
            b=ds[index]
            cases.append((f"real_kitti_val_{index}",tuple(b[k][None].contiguous() for k in ("rgb","sparse","mask","K"))))''')
    ast.parse(body)
    (OUT/'run.py').write_text(body,encoding='utf-8')
    print('Migrated runner:',len(body.splitlines()),'lines')

if __name__=='__main__':main()
