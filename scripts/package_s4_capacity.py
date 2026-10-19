"""Build a self-contained decoder-capacity Drive bundle; never edit frozen G."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'drive_upload/AnchorFlow_S4_UPIR_DecoderCapacity_30ep_A100'


def notebook():
    cells = [nbf.v4.new_markdown_cell('''# S4-U-PIR: decoder capacity · A100 · fresh30 ×3

Upload WHOLE **AnchorFlow_S4_UPIR_DecoderCapacity_30ep_A100** folder to MyDrive.
Select **A100**, Run all. G0 = current G architecture; H = wider D16/D8;
I = H + wider D4. Same seed42, ImageNet encoder,1600/400,30epochs, same losses.
**Metric teacher ON, relative teacher completely OFF/not packaged.**

Training is reliable eager BF16 + FP32 geometry/loss, NOT TensorRT training.
Inference accuracy AND latency use TensorRT FP32 + CUDA Graph on all400 val.
Run all3 for matched evidence; do not replace G0 with an old-session timing.
Resume reruns the same cells and uses each arm's last.pth. No early stop.
Do not upload Python helpers into frozen CODE; source/config/data checks stay strict.
'''), nbf.v4.new_code_cell('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
from IPython.display import display

BUNDLE = Path('/content/drive/MyDrive/AnchorFlow_S4_UPIR_DecoderCapacity_30ep_A100')
CODE = Path('/content/anchorflow_s4_capacity_code')
RUN_TAG = 'S4_UPIR_DecoderCapacity_Fresh30_pair01'  # keep SAME for resume
RUN = Path('/content/drive/MyDrive/GeoLift_RT_Runs')/RUN_TAG
assert (BUNDLE/'bundle_manifest.json').is_file(), 'Upload the WHOLE bundle folder'
RUN.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)
CONFIG = RUN/'runtime_config.json'  # OUTSIDE frozen CODE
VARIANTS = ('G0_s4_upir', 'H_s4_dec16_8', 'I_s4_dec16_8_4')
print('Run folder:', RUN)
'''), nbf.v4.new_code_cell('''manifest = json.loads((BUNDLE/'bundle_manifest.json').read_text())
for name, expected in manifest['files'].items():
    if name.endswith('.ipynb'): continue  # Colab edits metadata/outputs
    source = BUNDLE/name
    assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==expected, name
for name in [*manifest['files'], 'bundle_manifest.json']:
    if name.endswith('.ipynb'): continue
    source, target = BUNDLE/name, CODE/name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes()==source.read_bytes(), f'Local source changed: {name}; use clean CODE/new run'
    else: shutil.copy2(source, target)
expected_py = {name for name in manifest['files'] if name.endswith('.py')}
actual_py = {p.relative_to(CODE).as_posix() for p in CODE.rglob('*.py') if '__pycache__' not in p.parts}
assert actual_py==expected_py, f'Extra/missing Python files: {actual_py^expected_py}'
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(CODE/'requirements.txt')],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(CODE/'requirements-trt.txt')],check=True)
import torch, tensorrt
assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0), 'Select A100 runtime'
assert torch.cuda.is_bf16_supported(including_emulation=False), 'Native BF16 required'
assert tensorrt.__version__.startswith('10.13.3'), tensorrt.__version__
print(torch.__version__, torch.cuda.get_device_name(0), 'TensorRT', tensorrt.__version__)
test = subprocess.run([sys.executable,'-u','-m','unittest','discover','-s',str(CODE),
    '-p','test_contracts.py','-v'],cwd='/content',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(test.stdout)
(RUN/'contracts.log').write_text(test.stdout)
assert test.returncode==0, 'Contracts failed; inspect full log above, do not skip tests'
display(json.loads((CODE/'model_profiles.json').read_text()))
'''), nbf.v4.new_markdown_cell('''## Data, smoke and frozen source snapshot

Uses GeoLift_Data/teacher_subset_2000: selected_2000_ids.json,
kitti_trainval_2000.tar, metric_coarse_train_2000.tar. No DA3/relative archive.
Test1000 uses GeoLift_Data/test_1000/kitti_test_1000.tar or the canonical download.
Reuses an existing metric-only G SSD cache when available; the data gates still run.
Smoke checks real backward + untrained TRT parser/build/capture before long training.
'''), nbf.v4.new_code_cell('''cfg = json.loads((CODE/'config.json').read_text())
cfg['run_name'] = RUN_TAG
old_work = Path('/content/anchorflow_s4_upir_norel_work')
if (old_work/'data_contract.json').is_file() and (old_work/'metric_train/teacher_report.json').is_file():
    cfg['work'] = str(old_work)  # identical metric-only data, no teacher changes
assert cfg['relative_enabled'] is False and cfg['relative_weight']==0
WORK = Path(cfg['work'])
CONFIG.write_text(json.dumps(cfg,indent=2))

def command(action, variant=None):
    label = f'{action}_{variant}' if variant else action
    argv = [sys.executable,'-u',str(CODE/'run.py'),action,'--config',str(CONFIG)]
    if variant: argv += ['--variant',variant]
    logfile = RUN/f'{label}.log'
    with logfile.open('a') as handle:
        proc = subprocess.Popen(argv,cwd=str(CODE),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='');handle.write(line);handle.flush()
        rc = proc.wait()
    if rc: raise RuntimeError(f'{label} failed; inspect {logfile}. No silent backend fallback.')

ready = all((WORK/name).is_file() for name in (
    'bundle_ready.json','metric_train/teacher_report.json','test_ready.json'))
required_gib = 4 if ready else 30
assert shutil.disk_usage('/content').free > required_gib*2**30, f'Need >{required_gib} GiB SSD'
command('prepare')
# Keep exact sources on Drive for future recovery if Colab/notebook is lost.
snapshot = RUN/'frozen_bundle'
for name in [*manifest['files'],'bundle_manifest.json']:
    if name.endswith('.ipynb'): continue
    target = snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): assert target.read_bytes()==(BUNDLE/name).read_bytes(), f'Snapshot changed: {name}'
    else: shutil.copy2(BUNDLE/name,target)
shutil.copy2(WORK/'data_contract.json',RUN/'data_contract.json')
for variant in VARIANTS: command('smoke',variant)
smokes = [json.loads((RUN/v/'smoke_report.json').read_text()) for v in VARIANTS]
assert all(s['passed'] and s['coarse']['nfe']==4 and s['fine']['nfe']==0 for s in smokes)
assert len({s['shared_initial_sha256'] for s in smokes})==1, 'Retained initial slices differ'
display(smokes)
'''), nbf.v4.new_markdown_cell('''## Fresh30 per arm / strict epoch-boundary resume

Each arm uses30epochs; best_policy selection stays the same iRMSE-feasible policy.
Old G first reached iRMSE<=3.2 at epoch24 and selected epoch27, so truncating to
26 would miss its best epoch. Same recipe/budget isolates capacity.
Re-running this cell resumes each last.pth, including optimizer/loader/RNG state;
already completed arms do no further optimizer steps. Do not change RUN_TAG,
source, seed or recipe to resume. No mid-epoch optimizer checkpoint guarantee.
'''), nbf.v4.new_code_cell('''for variant in VARIANTS:
    command('train',variant)
display([json.loads((RUN/v/'training_status.json').read_text()) for v in VARIANTS])
'''), nbf.v4.new_code_cell('''import pandas as pd
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,3,figsize=(15,4))
for variant in VARIANTS:
    frame = pd.read_csv(RUN/variant/'train_log.csv')
    assert len(frame)==30 and (frame['loss_weighted_relative']==0).all()
    for ax, metric in zip(axes,('val_rmse_m','val_irmse_km_inv','val_boundary3_rmse_m')):
        ax.plot(frame['epoch']+1,frame[metric],label=variant)
        ax.set_title(metric);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
axes[0].legend();plt.tight_layout();plt.show()
'''), nbf.v4.new_markdown_cell('''## SAME-session TensorRT FP32 + CUDA Graph benchmark

Full400 validation accuracy, GT boundaries/range/error tails. Same fixed numerical
gates; no fallback or post-hoc relaxation. Runtime B1,100 distinct GPU-resident
images, warmup100/iterations300, TF32 OFF, wall includes D2D staging but excludes
decode/preprocessing/H2D/build/capture. This is not end-to-end or Orin latency.
1000 anonymous test predictions +10 full-resolution preview pairs per arm.
G0 is the matched baseline; old G snapshot is context only. Compare does not
promote a new baseline automatically when a candidate loses accuracy/runtime.
'''), nbf.v4.new_code_cell('''for variant in VARIANTS: command('benchmark',variant)
command('compare')
import pandas as pd
display(pd.read_csv(RUN/'comparison.csv'))
display(json.loads((RUN/'paired_rmse_bootstrap.json').read_text()))
print('Comparison:',RUN/'comparison.csv')
print('Full-resolution comparisons:',RUN/'comparison_previews')
print('Sources for recovery:',RUN/'frozen_bundle')
''')]
    for cell in cells:
        if cell.cell_type == 'code': ast.parse(cell.source)
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU'})
    nbf.validate(nb)
    nbf.write(nb, BUNDLE/'Train_S4_UPIR_DecoderCapacity_Fresh30_A100.ipynb')


def profiles():
    sys.path.insert(0, str(BUNDLE))
    import torch
    import experiment as E
    torch.set_num_threads(2)
    cfg = json.loads((BUNDLE/'config.json').read_text())
    values = {v: E.architecture(E.make_model(cfg, v)) for v in E.VARIANTS}
    (BUNDLE/'model_profiles.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
    rows = list(csv.DictReader((BUNDLE/'evidence/G_train_log.csv').open()))
    selected = rows[27]
    evidence = {'source': 'Completed historical G CSV; not a new capacity training result',
        'epochs': len(rows), 'best_native_epoch': min(rows, key=lambda r: float(r['val_rmse_m']))['epoch'],
        'selected_loss_fractions': {k: float(v)/float(selected['loss_total']) for k,v in selected.items()
            if k.startswith('loss_weighted_')},
        'relative_all_zero': all(float(r['loss_weighted_relative']) == 0 for r in rows),
        'parameters_by_module_G': {name: sum(p.numel() for p in m.parameters())
            for name,m in E.make_model(cfg,E.VARIANTS[0]).named_children()},
        'upir_intervention': {'source': 'Rounded user-reported corrected eval-mode audit, not original JSON',
            'original': {'rmse_m': .990151,'irmse_km_inv':3.172073,'mae_m':.258984},
            'zero_U_input': {'rmse_m':.990889,'irmse_km_inv':3.191544,'mae_m':.261689},
            'mean_prediction_change_m':.03668375799530431,
            'interpretation':'Same-checkpoint reliance, not retrained architecture effect'}}
    (BUNDLE/'evidence/capacity_analysis.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(values, indent=2))


def package():
    notebook();profiles()
    files = {p.relative_to(BUNDLE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(BUNDLE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
        and p.suffix != '.pyc' and p.name != 'bundle_manifest.json'}
    (BUNDLE/'bundle_manifest.json').write_text(json.dumps({'schema':1,'files':files},indent=2),encoding='utf-8')
    with zipfile.ZipFile(BUNDLE.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*files,'bundle_manifest.json']: archive.write(BUNDLE/name,BUNDLE.name+'/'+name)
    print('ZIP:',BUNDLE.with_suffix('.zip'))


if __name__ == '__main__': package()
