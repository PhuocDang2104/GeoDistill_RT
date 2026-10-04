"""Seal verified code-only Drive package; preserve every historical bundle."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'drive_upload/AnchorFlow_v11_NODE'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    proof=json.loads((FOLDER/'local_verification.json').read_text())
    assert proof['tests_returncode']==0 and proof['isolated_staged_tests_returncode']==0
    assert proof['tests']==32 and proof['runner_resume_integration']['passed']
    assert proof['runner_resume_integration']['uninterrupted_vs_resume_model_bitwise_equal']
    assert proof['onnx']['checker'] and proof['onnx']['onnxruntime_cpu_parity']
    assert not proof['onnx']['adaptive_solver_exported']
    assert all(v['finite'] for v in proof['real_kitti_full_resolution_backward'].values())
    assert proof['adaptive_complexity']['total_parameters']==582912
    names=('model.py','model_base.py','geometry.py','geometry_primitives.py','ode_solver.py','support.py',
           'boundaries.py','core.py','losses.py','relative_loss.py','loss_helpers.py','data.py',
           'relative_data.py','metrics.py','run.py')
    source=hashlib.sha256()
    for name in names:source.update((FOLDER/name).read_bytes())
    assert source.hexdigest()==proof['source_sha256'],'Source changed since actual verification'
    for name,expected in proof['verified_files'].items():
        assert sha(FOLDER/name)==expected,f'Verified file changed: {name}'
    cfg=json.loads((FOLDER/'config.json').read_text())
    assert cfg['epochs']==40 and cfg['init_checkpoint'] is None and cfg['ode_terminal_time']==1.
    assert cfg['ode_method']=='bosh3'
    files=sorted(p for p in FOLDER.iterdir() if p.is_file() and p.name!='bundle_manifest.json')
    assert not any(p.suffix in ('.pth','.tar','.npy','.onnx','.zip') for p in files)
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        if p.suffix=='.ipynb':
            book=nbformat.read(p,as_version=4);nbformat.validate(book)
            snapshot=nbformat.read(p.with_name(p.stem+'_contract.json'),as_version=4)
            assert book['cells']==snapshot['cells']
            for cell in book.cells:
                if cell.cell_type=='code':ast.parse(cell.source)
    # Check latest inherited data/metric/readout sources remain byte-identical.
    unchanged=('core.py','support.py','data.py','relative_data.py','relative_loss.py','loss_helpers.py','metrics.py','boundaries.py')
    inherited={name:sha(FOLDER/name)==sha(ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric'/name) for name in unchanged}
    assert all(inherited.values())
    historical={}
    for folder in ('AnchorFlow_v8_Dynamics','AnchorFlow_v9_Consensus','AnchorFlow_v10_AdaptiveJet','AnchorFlow_v10_1_LiteMetric'):
        path=ROOT/'drive_upload'/folder
        manifest=json.loads((path/'bundle_manifest.json').read_text())
        mismatches=[name for name,digest in manifest['files'].items() if not (path/name).is_file() or sha(path/name)!=digest]
        historical[folder]={'unchanged_sealed_sources':not mismatches,'mismatches':mismatches}
        assert not mismatches,(folder,mismatches)
    manifest={'architecture':'AnchorFlow-V11-JetNODE','fresh_epochs_max':40,'source_sha256':source.hexdigest(),
        'files':{p.name:sha(p) for p in files},'historical_bundles_preserved':historical,
        'unchanged_litemetric_primitives':inherited,
        'notebook_ui_checksum_policy':'Skip mutable .ipynb UI at runtime; verify immutable _contract.json and ALL source/config',
        'adaptive_solver':'torchdiffeq==0.2.5/bosh3','terminal_time':1.,
        'gpu_training_accuracy_or_latency_verified':False}
    (FOLDER/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    target=ROOT/'drive_upload/AnchorFlow_v11_NODE.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in [*files,FOLDER/'bundle_manifest.json']:z.write(p,f'{FOLDER.name}/{p.name}')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(f'{FOLDER.name}/{p.name}')).hexdigest()==manifest['files'][p.name]
    print(json.dumps({'zip':str(target),'bytes':target.stat().st_size,'zip_sha256':sha(target),
                      'source_sha256':source.hexdigest(),'root_files':len(files)+1},indent=2))

if __name__=='__main__':main()
