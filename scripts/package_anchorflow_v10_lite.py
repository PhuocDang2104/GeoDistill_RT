"""Seal verified V10.1 code/notebook/report. No data, weights, caches or old code."""
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    proof=json.loads((FOLDER/'local_verification.json').read_text())
    assert proof['tests']==30 and proof['tests_returncode']==proof['isolated_staged_tests_returncode']==0
    assert proof['runner_resume_integration']['passed'] and proof['runner_resume_integration']['uninterrupted_vs_resume_model_bitwise_equal']
    assert proof['complexity']['total_parameters']==582945
    assert proof['complexity']['total_conv_linear_macs']==3286924608
    assert proof['onnx']['onnxruntime_cpu_parity'] and len(proof['onnx']['cases'])==9
    assert proof['onnx_learned_step_open']['passed'] and proof['onnx_learned_step_open']['sample_dependent_h_verified']
    assert proof['real_kitti_backward']['fp32']['finite'] and proof['real_kitti_backward']['bf16']['finite']
    assert proof['pruned_parent_initial_validation_cpu_fp32']['all']['pixels']==25424992
    for name,expected in proof['verified_files'].items():assert sha(FOLDER/name)==expected,('proof stale',name)
    cfg=json.loads((FOLDER/'config.json').read_text())
    assert cfg['epochs']==40 and cfg['init_checkpoint'] is None and cfg['flow_steps']==2 and cfg['step_size']==1/3
    assert cfg['learned_step_size'] and cfg['step_min']==1/6 and cfg['step_max']==1/3
    assert cfg['teacher_enabled'] and cfg['relative_enabled'] and cfg['inverse_rmse_weight']==.04
    names=[p.name for p in FOLDER.iterdir() if p.is_file() and p.name!='bundle_manifest.json']
    assert not any(Path(n).suffix in ('.tar','.pth','.onnx','.png','.npy','.npz','.pt','.pyc','.zip') for n in names)
    for name in names:
        path=FOLDER/name
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
        if path.suffix=='.md':
            body=path.read_text(encoding='utf-8')
            assert body.count('~~~')%2==0 and sum(line.strip()=='$$' for line in body.splitlines())%2==0
            assert '\\operatorname' not in body and '\\left{' not in body
            for target in re.findall(r'\]\(([^)]+)\)',body):
                if not target.startswith(('http','/','#')):assert (FOLDER/target).is_file(),(name,target)
        if path.suffix=='.ipynb':
            book=nbformat.read(path,as_version=4);nbformat.validate(book)
            assert json.loads(path.read_text())==json.loads(path.with_name(path.stem+'_contract.json').read_text())
            for cell in book.cells:
                if cell.cell_type=='code':ast.parse(cell.source);assert not cell.outputs and cell.execution_count is None
    assert len(list(FOLDER.glob('*.ipynb')))==1
    manifest={'name':FOLDER.name,'default_model':'v10_lite','default_variant':'dual_teacher',
              'default_epochs':40,'default_flow_steps':2,'learned_step_size':True,'adaptive_stopping':False,'initialization':'fresh_student_imagenet_rgb_only',
              'optional_model_only_finetune_epochs':20,'contains_student_checkpoint':False,
              'contains_data':False,'contains_teacher_weights':False,'source_sha256':proof['source_sha256'],
              'precision_policy':'native_bf16_else_explicit_fp32','no_fp16_fallback':True,
              'files':{n:sha(FOLDER/n) for n in sorted(names)},
              'checksum_policy':'Skip mutable .ipynb UI only; immutable _contract.json/source/config remain strict.'}
    (FOLDER/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    # Sealed V8/V9/V10 must match. Existing V9.1 GPU troubleshooting edits
    # predate this work and its old manifest; preserve, never revert/reseal it.
    old_audit={}
    for frozen in ('AnchorFlow_v8_Dynamics','AnchorFlow_v9_Consensus','AnchorFlow_v9_1_MetricRefine','AnchorFlow_v10_AdaptiveJet'):
        folder=ROOT/'drive_upload'/frozen;old=json.loads((folder/'bundle_manifest.json').read_text())
        mismatches=[name for name,expected in old['files'].items() if not name.endswith('.ipynb') and sha(folder/name)!=expected]
        if frozen!='AnchorFlow_v9_1_MetricRefine':assert not mismatches,('historical bundle changed',frozen,mismatches)
        old_audit[frozen]={'manifest_mismatches':mismatches,'modified_by_this_task':False}
    proof['historical_checksum_audit']=old_audit
    proof['historical_checksum_note']='V9.1 has existing troubleshooting edits after its seal; this task does not modify, revert or reseal them.'
    (FOLDER/'local_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    manifest['files']['local_verification.json']=sha(FOLDER/'local_verification.json')
    (FOLDER/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    output=FOLDER.with_suffix('.zip')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in sorted(names)+['bundle_manifest.json']:archive.write(FOLDER/name,FOLDER.name+'/'+name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name,expected in manifest['files'].items():assert hashlib.sha256(archive.read(FOLDER.name+'/'+name)).hexdigest()==expected
    print(json.dumps({'folder':str(FOLDER),'zip':str(output),'bytes':output.stat().st_size,'sha256':sha(output),
                     'files':len(names)+1,'source_sha256':manifest['source_sha256'],'historical_checksum_audit':old_audit},indent=2))

if __name__=='__main__':main()
