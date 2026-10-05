"""Small local qualification report. Does not download model weights or KITTI."""
import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'production'/'AnchorFlow_V11_3_FullKITTI'


def main():
    sys.path.insert(0,str(BUNDLE))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-upstream',action='store_true',help='Fetch only the tiny pinned CUDA source; never weights')
    parser.add_argument('--standalone',action='store_true',help='Check committed model checksums without local research assets')
    args=parser.parse_args()
    started=time.monotonic()
    pyfiles=list(BUNDLE.rglob('*.py'))
    for path in pyfiles:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    parent=ROOT/'drive_upload'/'AnchorFlow_v11_3_PhaseInnovation'
    manifest=json.loads((BUNDLE/'bundle_manifest.json').read_text(encoding='utf-8'))
    original_available=parent.is_dir() and not args.standalone
    frozen=[]
    for path in (BUNDLE/'v11_model').glob('*.py'):
        reference=(parent/path.name).read_bytes() if original_available else None
        matches=(path.read_bytes()==reference if original_available else
                 hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files']['v11_model/'+path.name])
        if not matches:
            raise RuntimeError(f'Frozen model source modified: {path.name}')
        frozen.append(path.name)
    for name in ('__init__.py','common.py','recovery.py'):
        if (BUNDLE/'server_runtime'/name).read_text(encoding='utf-8')!=(ROOT/'src'/'runtime'/name).read_text(encoding='utf-8'):
            raise RuntimeError(f'S3 runtime primitive modified: {name}')
    integrity=subprocess.run([sys.executable,'-X','utf8','pipeline.py','verify'],cwd=BUNDLE,
                             capture_output=True,text=True,encoding='utf-8',check=True)
    tests=subprocess.run([sys.executable,'-X','utf8','-m','unittest','test_production','-v'],cwd=BUNDLE,
                         capture_output=True,text=True,encoding='utf-8')
    print(tests.stdout+tests.stderr)
    if tests.returncode:
        raise RuntimeError('Production contracts failed; do not publish PASS evidence')
    count=int(re.search(r'Ran (\d+) tests',tests.stderr).group(1))
    compose='not checked (Docker CLI unavailable)'
    if shutil.which('docker'):
        subprocess.run(['docker','compose','-f',str(BUNDLE/'compose.yaml'),'config','--quiet'],check=True)
        compose='configuration validation PASS; image not built'
    upstream=None
    if args.check_upstream:
        from extension_compat import KERNEL_SHA,modernize_text
        from teachers import DMD_CODE
        url=f'https://raw.githubusercontent.com/Sharpiless/DMD3Cpp/{DMD_CODE}/exts/bp_cuda_kernel.cu'
        with urllib.request.urlopen(url,timeout=30) as response:source=response.read().decode('utf-8')
        patched=modernize_text(source)
        upstream={'url':url,'original_sha256':KERNEL_SHA,
                  'mechanical_build_copy_sha256':hashlib.sha256(patched.encode()).hexdigest(),
                  'type_sites_replaced':3,'data_sites_replaced':8,
                  'remaining_deprecated_api_sites':patched.count('.type()')+patched.count('.data<'),
                  'cuda_compilation_tested':False}
    from utils import atomic_json,source_identity
    report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'passed_cpu_contracts':True,
            'tests_passed':count,'seconds':time.monotonic()-started,
            'python':sys.version,'parsed_python_files':len(pyfiles),
            'model_source_byte_identical':True,'model_files':sorted(frozen),
            'model_reference':'local research bundle' if original_available else 'committed standalone manifest',
            's3_atomic_runtime_source_unchanged':True,'source_sha256':source_identity(),
            'manifest_integrity':integrity.stdout.strip(),'docker_compose':compose,
            'docker_base':next(line[5:] for line in (BUNDLE/'Dockerfile').read_text().splitlines() if line.startswith('FROM ')),
            'optional_official_cuda_source_audit':upstream,
            'gpu_teacher_smoke_tested':False,'gpu_student_smoke_tested':False,
            'docker_image_built':False,'full_kitti_downloaded':False,'full_training_run':False,
            'limitations':['CPU toy-resume test is not a bitwise CUDA guarantee',
                           'GPU probes for both teachers and real-batch student backward are mandatory on member',
                           'No measured teacher minimum VRAM or full-data metric/throughput claim']}
    atomic_json(BUNDLE/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='model_files'},indent=2))


if __name__=='__main__':main()
