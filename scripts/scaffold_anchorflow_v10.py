"""Mechanical isolated copy of the selected V9.1 technical baseline."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'drive_upload/AnchorFlow_v9_1_MetricRefine'
OUT=ROOT/'drive_upload/AnchorFlow_v10_AdaptiveJet'

def main():
    if (OUT/'run.py').exists():
        raise RuntimeError('V10 is already assembled; scaffolding would overwrite its integration workflow. Use verify/package scripts, not scaffold again.')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('model_v9_reference.py','model_v8.py','core.py','support.py','data.py','relative_data.py',
                 'metrics.py','boundaries.py','losses_v8.py','loss_helpers.py','run.py',
                 'generate_relative.py','requirements.txt','teacher_requirements.txt'):
        shutil.copy2(BASE/name,OUT/name)
    shutil.copy2(BASE/'model.py',OUT/'model_baseline.py')
    shutil.copy2(BASE/'losses.py',OUT/'losses_baseline.py')
    for label,folder in (('v8','anchorflow_v8_completed_audit'),('v9','anchorflow_v9_completed_audit'),('v9_1','anchorflow_v9_1_completed_audit')):
        for name in ('val_metrics.json','profile.json'):
            shutil.copy2(ROOT/'results'/folder/name,OUT/(label+'_'+name))
    cfg=json.loads((BASE/'config.json').read_text())
    cfg.update(architecture='AnchorFlow-V10-BudgetedAdaptiveJet',model_name='v10_adaptive_jet',
               initialization='fresh_student_imagenet_rgb_only',init_checkpoint=None,
               epochs=40,early_stop_min_epochs=20,early_stop_patience=8,
               learning_rate=3e-4,encoder_lr_ratio=.5,new_head_lr_ratio=1.,
               run_name='AnchorFlow_v10_AdaptiveJet_Fresh40_ES',work='/content/anchorflow_v10_work',
               integration_policy='adaptive',step_min=1/6,step_max=1/3,
               stop_threshold=.5,benefit_margin=.003,controller_weight=.02,
               compute_weight=0.,train_exit_exploration=.2,
               trajectory_weights=[.15,.25,.30,.30])
    for name in ('init_checkpoint_sha256','expected_parent_source_sha256'):
        cfg.pop(name,None)
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    run=(OUT/'run.py').read_text()
    run=run.replace('"model.py", "model_v9_reference.py"','"model.py", "model_baseline.py", "adaptive_report.py", "model_v9_reference.py"')
    run=run.replace('"losses.py", "losses_v8.py"','"losses.py", "losses_baseline.py", "losses_v8.py"')
    start=run.index('def make_model('); end=run.index('\ndef move_batch(',start)
    run=run[:start]+'''def make_model(config, device, pretrained):
    model = AnchorFlowEdge(pretrained=pretrained, flow_steps=3, encoder=config["encoder"],
                           model_name="v10_adaptive_jet", limited=config.get("limited_jet_init",True),
                           policy=config["integration_policy"], step_min=config["step_min"],
                           step_max=config["step_max"], stop_threshold=config["stop_threshold"],
                           exploration=config.get("train_exit_exploration",.2)).to(device)
    if config["channels_last"]: model = model.to(memory_format=torch.channels_last)
    return model


def initialize_from_parent(model,config,check_data=True):
    if config.get("init_checkpoint"):
        raise ValueError("V10 is fresh ImageNet-RGB initialization only; no V8/V9 student migration")
    return {"loaded":False,"parent_epoch":None,"parent_completed_epochs":0}

''' +run[end:]
    run=run.replace('"v9_metric_refine"','"v10_adaptive_jet"')
    run=run.replace('"D4_step3", "D4",','"D4_step3", "D4_step4", "D4",')
    run=run.replace('for k in (1,2,3)},','for k in (1,2,3,4)},')
    run=run.replace('name.startswith("phase2.metric.")','(name.startswith("phase2.metric.") or name.startswith("dynamics.controller."))')
    run=run.replace('V9.1: minmod initialization + sparse-innovation metric head, weak structural KD, robust excess tail. Default warm start V9 MODEL only, new optimizer/schedule. Historical comparison is NOT an equal-budget architecture ablation.',
                    'V10: fresh ImageNet RGB only; shared geometric field with bounded learned step sizes and learned stopping. Historical V8/V9/V9.1 are NOT matched-budget causal ablations.')
    run=run.replace('stage2/3','stage2/3/4').replace('D4_step1/2/3 share','D4_step1/2/3/4 share')
    run=run.replace('three shared jet Euler steps','budgeted 2-4 shared jet Euler steps (masked4 validation computes four)')
    (OUT/'run.py').write_text(run,encoding='utf-8')
    print(OUT)

if __name__=='__main__': main()
