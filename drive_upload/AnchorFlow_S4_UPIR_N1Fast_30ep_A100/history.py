"""Read-only N1 provenance. Never modify old sources or execute an old optimizer."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import torch
import experiment as E
from data import digest,write_json

PARENT_VARIANT='N1_s4_m3_f512'
HISTORICAL_VARIANTS=('K1_d4_fusion768','M0_s4_trimlow','M1_s4_fuse256',
                     'M2_s4_trimlow_fuse768',PARENT_VARIANT,'MSG32_scratch28')
METADATA={'experiment','capacity_design','run_name','reference_run','msgchn','experiment_targets',
          'drive_data','drive_runs','work','workers','prefetch_factor','deployment_optimization'}


def recipe(cfg):return {k:v for k,v in cfg.items() if k not in METADATA}


def comparable_data(data):
    result=copy.deepcopy(data)
    result.get('teacher_report',{}).get('signature',{}).pop('archive_mtime_ns',None)
    return result


def canonical_ast_dump(dump):
    # Python3.12 type_params / Python3.13 omitted empty fields, no eval().
    def convert(node):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            fields={}
            for keyword in node.keywords:
                value=keyword.value
                if isinstance(value,ast.List) and not value.elts:continue
                if isinstance(value,ast.Constant) and value.value is None:continue
                fields[keyword.arg]=convert(value)
            return [node.func.id,[convert(arg) for arg in node.args],fields]
        if isinstance(node,ast.List):return [convert(item) for item in node.elts]
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant):return -node.operand.value
        raise ValueError('Unsupported AST dump format')
    return convert(ast.parse(dump,mode='eval').body)


def verify_frozen_source(root):
    source=Path(root)/'frozen_bundle'
    manifest=json.loads((source/'bundle_manifest.json').read_text())['files']
    expected={n:v for n,v in manifest.items() if n.endswith('.py')}
    actual={p.relative_to(source).as_posix() for p in source.rglob('*.py') if '__pycache__' not in p.parts}
    if actual!=set(expected):raise RuntimeError('Old frozen Python source set differs; do not add upload helpers there')
    checksum=hashlib.sha256()
    for name in sorted(expected):
        path=source/name
        if digest(path)!=expected[name]:raise RuntimeError('Old frozen source hash differs: '+name)
        checksum.update(str(path.relative_to(source)).encode());checksum.update(path.read_bytes())
    policy=json.loads((E.ROOT/'evidence/parent_policy.json').read_text())
    for name,sha in policy['critical_python_sha256'].items():
        if digest(source/name)!=sha:raise RuntimeError('Old N1 architecture/objective differs: '+name)
        if name!='experiment.py' and (source/name).read_bytes()!=(E.ROOT/name).read_bytes():raise RuntimeError('Core model/loss differs: '+name)
    functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((source/'run.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    mismatch=[n for n,v in policy['training_function_ast'].items() if n not in functions or canonical_ast_dump(functions[n])!=canonical_ast_dump(v)]
    if mismatch:raise RuntimeError('Old N1 training functions differ: '+', '.join(mismatch))
    return {'source_sha256':checksum.hexdigest(),'critical_architecture_training_verified':True}


def validate_parent(report,checkpoint,status,cfg,data,source):
    old=report['training_contract']
    if report['variant']!=PARENT_VARIANT or old['variant']!=PARENT_VARIANT:raise RuntimeError('Wrong N1 parent variant')
    if old!=checkpoint['contract'] or old['source_sha256']!=source['source_sha256']:raise RuntimeError('N1 report/checkpoint/source contract differs')
    if recipe(old['config'])!=recipe(cfg) or comparable_data(old['data'])!=comparable_data(data):raise RuntimeError('N1 recipe or train/val data differs')
    if not report['qualification_passed'] or status['completed_epochs']!=30 or report['training_epochs']!=30:raise RuntimeError('Need completed qualified N1 fresh30')
    if report['parameters']!=582888 or report['inference_precision']!='explicit_FP32_TF32_OFF':raise RuntimeError('N1 parameter/precision differs')
    if report['initial_shared_sha256']!=checkpoint['state']['initial_shared_sha256']:raise RuntimeError('N1 initial hash/report differs')
    images=report['per_image']
    if len(images)!=400 or len({r['sample_id'] for r in images})!=400 or sum(r['pixels'] for r in images)!=25424992:raise RuntimeError('N1 validation support differs')
    return True


def audit_history(cfg):
    root=Path(cfg['reference_run']);folder=root/PARENT_VARIANT
    output=Path(cfg['drive_runs'])/cfg['run_name']
    if output.resolve()==root.resolve():raise ValueError('New outputs must not overwrite old N1 run')
    needed=[folder/n for n in ('best_policy.pth','training_status.json','trt_benchmark.json','train_log.csv')]+[root/'frozen_bundle/bundle_manifest.json']
    missing=[str(p) for p in needed if not p.is_file()]
    if missing:raise FileNotFoundError('Need previous N1 evidence:\n'+'\n'.join(missing))
    source=verify_frozen_source(root)
    checkpoint=torch.load(folder/'best_policy.pth',map_location='cpu',weights_only=False)
    report=json.loads((folder/'trt_benchmark.json').read_text())
    status=json.loads((folder/'training_status.json').read_text())
    data=json.loads((Path(cfg['work'])/'data_contract.json').read_text())
    validate_parent(report,checkpoint,status,cfg,data,source)
    if digest(folder/'best_policy.pth')!=report['checkpoint_sha256']:raise RuntimeError('N1 checkpoint SHA differs')
    # Fresh ImageNet start only. Never warm-start new training from trained N1.
    start=E.make_model(cfg,E.VARIANTS[0],pretrained=True)
    initial=E.common_weight_hash(start)
    if initial!=report['initial_shared_sha256']:raise RuntimeError('Fresh ImageNet/shared initialization differs from N1')
    proof={'verified':True,'reference_run':str(root),'parent_variant':PARENT_VARIANT,
      'checkpoint_sha256':report['checkpoint_sha256'],'original_source':source,
      'initial_shared_sha256':initial,'new_source_sha256':E.source_hash(),
      'history_report_sha256':digest(folder/'trt_benchmark.json'),
      'warm_start':False,'old_training_or_inference_executed':False,
      'ignored_data_metadata':['teacher_report.signature.archive_mtime_ns']}
    output.mkdir(parents=True,exist_ok=True)
    proof['full_initial_weight_sha256']=full_weight_hash(start)
    write_json(output/'history_provenance.json',proof)
    return proof


def full_weight_hash(model):
    sha=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        sha.update(name.encode());sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def load_reference(cfg,device='cpu'):
    """Validate old source/config/data/weights, then strictly load without editing old files."""
    proof=audit_history(cfg)
    path=Path(cfg['reference_run'])/PARENT_VARIANT/'best_policy.pth'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=E.make_model(cfg,E.VARIANTS[0],device,pretrained=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    model._reference_provenance=proof
    return model.eval(),checkpoint,path
