"""Mechanical extraction/copy of audited primitives. Never overwrite old bundles."""
import ast
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'drive_upload/AnchorFlow_v10_AdaptiveJet'
OUT=ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric'

def selected(path,names):
    body=path.read_text(encoding='utf-8'); lines=body.splitlines(True)
    found={node.name:node for node in ast.parse(body).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    def source(node):
        start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
        return ''.join(lines[start:node.end_lineno])
    return '\n\n\n'.join(source(found[n]) for n in names)+'\n'

def write(name,body):
    (OUT/name).write_text(body,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'run.py').exists():raise RuntimeError('Scaffold already exists; edit with apply_patch, do not re-scaffold')
    for name in ('data.py','relative_data.py','loss_helpers.py','metrics.py','boundaries.py','requirements.txt'):
        shutil.copy2(OLD/name,OUT/name)
    header='from __future__ import annotations\nimport math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'
    write('core.py','"""Only live convolution, sparse and phase primitives; checkpoint keys preserved."""\n'+header+
        selected(OLD/'core.py',('ConvBN','LiteBlock','SparsePyramid','Fusion','PyramidDecoder','PhaseUpsample')))
    geometry=header+'from core import ConvBN,LiteBlock\n\n'+selected(OLD/'model_v8.py',('adjacent','translate','edge_weights'))
    geometry+=selected(OLD/'model_baseline.py',('minmod','limited_derivatives'))
    geometry+=selected(OLD/'model_v9_reference.py',('candidate_queries',))
    jet=selected(OLD/'model_v8.py',('JetDynamics',)).replace('class JetDynamics','class FixedJetDynamics')
    jet=jet.replace('if steps != 3:', 'if steps not in (2,3):').replace('Canonical V8 fixes three Euler steps; dt and schema depend on it','Fixed graph supports 2 or 3 steps, h=1/3')
    jet=jet.replace('dt = 1/self.steps','dt = 1/3').replace('self.steps, self.feedback = steps, feedback','self.steps, self.feedback = steps, feedback\n        self.diagnostics = True')
    # Replace ONLY base-jet initializer with the continuous minmod implementation.
    tree=ast.parse(jet);cls=tree.body[0];method=next(n for n in cls.body if getattr(n,'name',None)=='base_jet');lines=jet.splitlines(True)
    start=min([method.lineno]+[d.lineno for d in method.decorator_list])-1
    init='    def base_jet(self,depth):\n        v=depth.float().clamp(.1,120).reciprocal()\n        gx,gy=limited_derivatives(v.detach())\n        hxx,hxy_a=limited_derivatives(gx)\n        hxy_b,hyy=limited_derivatives(gy)\n        return self.project(torch.cat((v,gx,gy,hxx,.5*(hxy_a+hxy_b),hyy),1))\n'
    jet=''.join(lines[:start])+init+''.join(lines[method.end_lineno:])
    # Diagnostics are useful in training/evaluation but not part of deploy math.
    start=jet.index('            diagnostics.update(');end=jet.index('            j = next_j',start)
    jet=jet[:start]+'            if self.diagnostics:\n'+''.join('    '+line for line in jet[start:end].splitlines(True))+jet[end:]
    start=jet.index('        diagnostics["delta4"]');end=jet.index('        return j, context, diagnostics',start)
    jet=jet[:start]+'        if self.diagnostics:\n'+''.join('    '+line for line in jet[start:end].splitlines(True))+jet[end:]
    jet=jet.replace('Projected explicit Euler, dt=1/3, three steps.','Projected explicit Euler, dt=1/3, two/three fixed steps.')
    write('geometry.py',geometry+'\n'+jet)
    write('support.py',header+selected(OLD/'support.py',('Context32','PhaseDetailAndTrust')))
    # Build one live base graph, rather than a chain of legacy model subclasses.
    readout=selected(OLD/'model_baseline.py',('sparse_innovation','InnovationReadout'))
    readout=readout.replace('class InnovationReadout(MultiJetReadout)','class InnovationReadout(nn.Module)')
    readout=readout.replace('        super().__init__()','''        super().__init__()
        self.diagnostics=True
        self.body=nn.Sequential(ConvBN(80,24),LiteBlock(24))
        self.blend=nn.Conv2d(24,4,1);self.delta=nn.Conv2d(24,4,1)
        self.candidates=nn.Conv2d(24,20,1)
        self.log_uncertainty_strength=nn.Parameter(torch.full((4,),math.log(math.expm1(.5))))
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.blend.weight);nn.init.constant_(self.blend.bias,-1.4)
        nn.init.zeros_(self.candidates.weight);nn.init.zeros_(self.candidates.bias)
        with torch.no_grad():self.candidates.bias[:4].fill_(1.5)''')
    readout=readout.replace('        diagnostics={','        diagnostics={}\n        if self.diagnostics:\n            diagnostics={')
    # Dict continuation indentation is legal within braces.
    base=selected(OLD/'model_v8.py',('AnchorFlowEdge','Deploy'))
    node=ast.parse(base).body[0];method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='forward');lines=base.splitlines(True)
    canonical=selected(OLD/'model_baseline.py',('AnchorFlowEdge',));n=ast.parse(canonical).body[0]
    f=next(x for x in n.body if getattr(x,'name',None)=='forward');f_lines=canonical.splitlines(True)
    base=''.join(lines[:method.lineno-1])+''.join(f_lines[f.lineno-1:f.end_lineno])+''.join(lines[method.end_lineno:])
    base=base.replace('flow_steps=3,','flow_steps=2,').replace('model_name="v8_dynamics"','model_name="v10_lite"')
    base=base.replace('if model_name not in ("v8_dynamics","v8_frozen_feedback"):', 'if model_name != "v10_lite":')
    base=base.replace('JetDynamics(flow_steps,feedback=model_name=="v8_dynamics")','FixedJetDynamics(flow_steps,feedback=True)')
    base=base.replace('JetPhaseReadout()','InnovationReadout()')
    base=base.replace('self.detail1(rgb,sparse,mask,d1_base,p2,g2)','self.detail1(rgb,sparse,mask,d1_base,p2,g2,p4)')
    write('model_base.py',header+'from core import ConvBN,LiteBlock,SparsePyramid,PyramidDecoder,PhaseUpsample\nfrom support import Context32,PhaseDetailAndTrust\nfrom geometry import FixedJetDynamics,candidate_queries\n\n'+readout+'\n'+base)
    cfg=json.loads((OLD/'config.json').read_text())
    for key in ('integration_policy','step_min','step_max','stop_threshold','benefit_margin','controller_weight','compute_weight','train_exit_exploration','trajectory_weights','limited_jet_init','robust_mse_delta'):
        cfg.pop(key,None)
    cfg.update(architecture='AnchorFlow-V10.1-LiteMetric',model_name='v10_lite',flow_steps=2,step_size=1/3,
        work='/content/anchorflow_v10_1_work',run_name='AnchorFlow_v10_1_LiteMetric_Fresh40_ES',
        phase_context_enabled=True,inverse_weight=0.,inverse_rmse_weight=.04,inverse_warmup_epochs=4,
        rmse_weight=.6,log_weight=0.,edge_weight=0.,kd_edge_weight=0.,robust_mse_weight=0.,
        checkpoint_selection='rmse',parent_source_sha256='64be55bb46aae74208b245d362004bb686ba4b8fdc4b431696b9337ddbf1073c',
        expected_subset_sha256='f0f482799db63b1e89f4ef21be0aff71cc409bac8bde20259192f3bc1cd40ddb')
    write('config.json',json.dumps(cfg,indent=2)+'\n')
    fine={**cfg,'epochs':20,'run_name':'AnchorFlow_v10_1_LiteMetric_FromV10Best20_ES',
          'learning_rate':.0001,'encoder_lr_ratio':.25,'new_head_lr_ratio':2.,
          'early_stop_min_epochs':8,'early_stop_patience':5,'initialization':'v10_best_model_only',
          'init_checkpoint':'/content/drive/MyDrive/GeoLift_RT_Runs/AnchorFlow_v10_adaptive_Fresh40_ES_bf16/dual_teacher/best.pth'}
    write('finetune20_config.json',json.dumps(fine,indent=2)+'\n')
    control={**cfg,'phase_context_enabled':False,'inverse_weight':.01,'inverse_rmse_weight':0.,
             'rmse_weight':.4,'log_weight':.2,'edge_weight':.05,'kd_edge_weight':.01,
             'run_name':'AnchorFlow_v10_1_Fixed2_ReferenceObjective40_ES'}
    write('fixed2_control_config.json',json.dumps(control,indent=2)+'\n')
    for name in ('v8_val_metrics.json','v9_val_metrics.json','v9_1_val_metrics.json'):
        shutil.copy2(OLD/name,OUT/name)
    shutil.copy2(ROOT/'results/anchorflow_v10_completed_audit/val_metrics.json',OUT/'v10_val_metrics.json')
    shutil.copy2(ROOT/'results/anchorflow_v10_completed_audit/comparison_history_policies.csv',OUT/'historical_comparison.csv')
    # The runner is migrated separately from template; scaffold does not create it.
    print(OUT)

if __name__=='__main__':main()
