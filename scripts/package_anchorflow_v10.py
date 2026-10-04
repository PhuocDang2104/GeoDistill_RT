"""Seal verified V10-only sources and two clean notebooks; no datasets/weights."""
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'drive_upload/AnchorFlow_v10_AdaptiveJet'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    # Explicit matched-control configuration, separate from adaptive default.
    canonical=json.loads((FOLDER/'config.json').read_text())
    baseline={**canonical,'integration_policy':'fixed3','run_name':'AnchorFlow_v10_fixed3_Fresh40_ES'}
    (FOLDER/'baseline_fixed3_config.json').write_text(json.dumps(baseline,indent=2)+'\n',encoding='utf-8')
    audit=ROOT/'results/anchorflow_v9_1_completed_audit'
    for name in ('comparison_v8_v9_v9_1.csv','stage_comparison.csv','best_loss_budget.csv'):
        shutil.copy2(audit/name,FOLDER/name)
    names=[p.name for p in FOLDER.iterdir() if p.is_file() and p.name!='bundle_manifest.json']
    assert not any(Path(n).suffix in ('.pth','.pt','.npz','.npy','.tar','.png','.onnx') for n in names)
    proof=json.loads((FOLDER/'local_verification.json').read_text())
    assert proof['tests']==22 and proof['tests_returncode']==0 and proof['colab_staged_tests_returncode']==0
    assert proof['real_kitti_backward']['fp32']['finite'] and proof['real_kitti_backward']['bf16']['finite']
    assert proof['complexity_by_policy']['adaptive']['total_parameters']==582350
    assert proof['complexity_by_policy']['adaptive']['total_conv_linear_macs']==3368789312
    for mode in ('masked_adaptive','static4'):
        assert proof['onnx']['graphs'][mode]['onnxruntime_cpu_parity']
        assert len(proof['onnx']['graphs'][mode]['cases'])==9
    for name,digest in proof['verified_files'].items(): assert sha(FOLDER/name)==digest,('stale proof',name)
    cfg=json.loads((FOLDER/'config.json').read_text())
    assert cfg['epochs']==40 and cfg['init_checkpoint'] is None and cfg['integration_policy']=='adaptive'
    assert cfg['step_min']==1/6 and cfg['step_max']==1/3
    assert cfg['teacher_enabled'] and cfg['relative_enabled'] and cfg['amp']=='bf16'
    assert cfg['encoder_pretrained'] and cfg['compute_weight']==0
    assert (cfg['early_stop_min_epochs'],cfg['early_stop_patience'])==(20,8)
    for name in names:
        p=FOLDER/name
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        if p.suffix=='.md':
            body=p.read_text(encoding='utf-8')
            assert body.count('~~~')%2==0
            assert sum(x.strip()=='$$' for x in body.splitlines())%2==0
            assert '\\operatorname' not in body and '\\left{' not in body
        if p.suffix=='.ipynb':
            book=nbformat.read(p,as_version=4);nbformat.validate(book)
            assert json.loads(p.read_text())==json.loads(p.with_name(p.stem+'_contract.json').read_text())
            for cell in book.cells:
                if cell.cell_type=='code':
                    ast.parse(cell.source);assert not cell.outputs and cell.execution_count is None
    assert len(list(FOLDER.glob('*.ipynb')))==2
    manifest={'name':FOLDER.name,'default_model':'v10_adaptive_jet','default_variant':'dual_teacher',
        'default_policy':'adaptive','default_epochs':40,'initialization':'fresh_student_imagenet_rgb_only',
        'contains_student_checkpoint':False,'contains_teacher_weights':False,'contains_data':False,
        'notebooks':2,'source_sha256':proof['source_sha256'],
        'precision_policy':'native_bf16_else_explicit_fp32','no_fp16_fallback':True,
        'files':{name:sha(FOLDER/name) for name in sorted(names)},
        'checksum_policy':'Mutable .ipynb UI skipped at runtime; immutable _contract.json and all source/config verified.'}
    (FOLDER/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    # Validate frozen V8/V9 bundles were not changed by this task.
    for frozen in ('AnchorFlow_v8_Dynamics','AnchorFlow_v9_Consensus'):
        old=ROOT/'drive_upload'/frozen
        original=json.loads((old/'bundle_manifest.json').read_text())
        for name,digest in original['files'].items():
            if not name.endswith('.ipynb'): assert sha(old/name)==digest,('historical bundle changed',frozen,name)
    output=FOLDER.with_suffix('.zip')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in sorted(names)+['bundle_manifest.json']:archive.write(FOLDER/name,FOLDER.name+'/'+name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name,digest in manifest['files'].items():assert hashlib.sha256(archive.read(FOLDER.name+'/'+name)).hexdigest()==digest
    print(json.dumps({'folder':str(FOLDER),'zip':str(output),'bytes':output.stat().st_size,'sha256':sha(output),
        'files':len(names)+1,'notebooks':2,'tests':proof['tests'],'source_sha256':proof['source_sha256']},indent=2))

if __name__=='__main__':main()
