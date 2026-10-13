"""CPU qualification and provenance for Fixed6; no GPU/teacher speed claims."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'production/AnchorFlow_V11_3_Fixed6_FullKITTI'

def main():
    started = time.monotonic()
    sys.path.insert(0,str(BUNDLE))
    from utils import atomic_json,sha256,source_identity
    manifest = json.loads((BUNDLE/'bundle_manifest.json').read_text(encoding='utf8'))
    parent = ROOT/'production/AnchorFlow_V11_3_FullKITTI'
    assert sha256(parent/'bundle_manifest.json') == manifest['original_bundle_manifest_sha256']
    for name, checksum in manifest['preserved_original_file_sha256'].items():
        assert sha256(BUNDLE/name) == sha256(parent/name) == checksum, name
    pyfiles = list(BUNDLE.rglob('*.py'))
    for path in pyfiles: ast.parse(path.read_text(encoding='utf8'),filename=str(path))
    integrity = subprocess.run([sys.executable,'-X','utf8','pipeline.py','verify'],
        cwd=BUNDLE,capture_output=True,text=True,encoding='utf8',check=True)
    tests = subprocess.run([sys.executable,'-X','utf8','-m','unittest','test_production','-v'],
        cwd=BUNDLE,capture_output=True,text=True,encoding='utf8')
    print(tests.stdout+tests.stderr)
    if tests.returncode: raise RuntimeError('Fixed6 CPU contracts failed')
    count = int(re.search(r'Ran (\d+) tests',tests.stderr).group(1))
    dry = subprocess.run([sys.executable,'launch.py','--storage-root',str(ROOT/'fixed6-dry-only'),
        '--accept-kitti-license','--cuda-arch','12.0','--dry-run'],
        cwd=BUNDLE,capture_output=True,text=True,encoding='utf8',check=True)
    assert not (ROOT/'fixed6-dry-only').exists()
    compose = 'not checked: Docker CLI unavailable'
    if shutil.which('docker'):
        subprocess.run(['docker','compose','-f',str(BUNDLE/'compose.yaml'),'config','--quiet'],check=True)
        compose = 'configuration PASS; image not built'
    bash = shutil.which('bash')
    if sys.platform == 'win32' and Path('C:/Program Files/Git/bin/bash.exe').is_file():
        bash = 'C:/Program Files/Git/bin/bash.exe'
    shell = 'not checked: bash unavailable'
    if bash:
        subprocess.run([bash,'-n',str(BUNDLE/'run_full_flow.sh')],check=True)
        shell = 'syntax PASS'
    report = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'passed_cpu_contracts':True,'tests_passed':count,'seconds':time.monotonic()-started,
        'python':sys.version,'parsed_python_files':len(pyfiles),'source_sha256':source_identity(),
        'manifest_integrity':integrity.stdout.strip(),'preserved_parent_files':len(manifest['preserved_original_file_sha256']),
        'model_parameters':584845,'coarse_method':'classic_rk3_two_half_steps',
        'coarse_nfe':6,'fine_nfe':2,'total_field_nfe':8,'horizon':1.,
        'default_batch':2,'default_amp':'bf16','default_cuda_arch':'12.0',
        'dry_run':dry.stdout.strip(),'docker_compose':compose,'bash':shell,
        'gpu_teacher_smoke_tested':False,'gpu_student_smoke_tested':False,
        'docker_image_built':False,'full_training_run':False,
        'limitations':['No 5060 Ti VRAM/latency measurement',
            'Real teacher/student GPU gates required before full training',
            'Fixed RK3 and B2 change training protocol; not matched adaptive B4 ablation',
            'CPU resume evidence is not bitwise CUDA reproducibility']}
    atomic_json(BUNDLE/'verification.json',report)
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
