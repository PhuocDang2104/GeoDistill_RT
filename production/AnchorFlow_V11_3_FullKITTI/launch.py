"""Pull-and-run host launcher. Standard-library Python only; no host pip setup."""
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from utils import freeze, read_json

PACKAGE=Path(__file__).resolve().parent


def cuda_arch(explicit=None):
    value=explicit or os.environ.get('CUDA_ARCH_LIST')
    if value:
        if not re.fullmatch(r'\d+\.\d+(?:\+PTX)?(?:;\d+\.\d+(?:\+PTX)?)*',value):
            raise ValueError('Invalid --cuda-arch; examples: 8.0, 8.9, 12.0')
        return value
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('nvidia-smi unavailable. Specify --cuda-arch matching the GPU; see README.')
    result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=compute_cap','--format=csv,noheader'],
                          capture_output=True,text=True,check=True)
    value=result.stdout.strip()
    if not re.fullmatch(r'\d+\.\d+',value) or float(value)<8:
        raise RuntimeError('GPU 0 must support native BF16 (SM >= 80); no silent CPU/T4 fallback')
    return value


def launch_plan(args):
    cfg=read_json(PACKAGE/'config.json')
    if not args.accept_kitti_license and not cfg.get('accept_kitti_license'):
        raise ValueError('Read KITTI terms, then explicitly pass --accept-kitti-license: '
                         'https://www.cvlibs.net/datasets/kitti/raw_data.php')
    cfg['accept_kitti_license']=True
    # Host paths only: model/data config inside the container stays /data,/runs,/cache.
    base=Path(args.storage_root).expanduser().resolve() if args.storage_root else None
    env=os.environ.copy()
    for key,folder in (('DATA_DIR','data'),('RUNS_DIR','runs'),('CACHE_DIR','cache')):
        path=base/folder if base else Path(env.get(key,str(PACKAGE/('server-'+folder)))).expanduser().resolve()
        env[key]=str(path)
    config_dir=base/'config' if base else PACKAGE/'server-config'
    config_file=config_dir/'v11_3_full.json'
    env['CONFIG_FILE']=str(config_file)
    env['CUDA_ARCH_LIST']=cuda_arch(args.cuda_arch)
    return cfg,env,config_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-root',help='Persistent SSD root; creates data/runs/cache/config subdirectories')
    parser.add_argument('--accept-kitti-license',action='store_true',help='Explicit acknowledgement after reading KITTI terms')
    parser.add_argument('--cuda-arch',help='Optional override; otherwise detects GPU 0 compute capability')
    parser.add_argument('--dry-run',action='store_true',help='Print the plan without writing files, downloading or starting Docker')
    args=parser.parse_args()
    cfg,env,config_file=launch_plan(args)
    print(json.dumps({'data':env['DATA_DIR'],'runs':env['RUNS_DIR'],'cache':env['CACHE_DIR'],
                      'runtime_config':str(config_file),'cuda_arch':env['CUDA_ARCH_LIST'],
                      'initialization':cfg['initialization'],'epochs':cfg['training']['epochs']},indent=2),flush=True)
    if args.dry_run:
        print('Dry-run only: no files/containers/data changed.');return
    if not shutil.which('docker'):
        raise RuntimeError('Install Docker Compose + NVIDIA Container Toolkit before launching')
    for key in ('DATA_DIR','RUNS_DIR','CACHE_DIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    freeze(config_file,cfg)  # Never edit tracked config or change an existing running recipe.
    subprocess.run(['docker','compose','config','--quiet'],cwd=PACKAGE,env=env,check=True)
    subprocess.run(['docker','compose','build'],cwd=PACKAGE,env=env,check=True)
    subprocess.run(['docker','compose','up','-d'],cwd=PACKAGE,env=env,check=True)
    print('Detached job started; closing SSH does not stop it.',flush=True)
    container=subprocess.check_output(['docker','compose','ps','--all','-q','trainer'],cwd=PACKAGE,env=env,text=True).strip()
    print('Logs: docker logs -f '+container,flush=True)
    print('Graceful stop: docker stop -t 600 '+container,flush=True)
    print('Results: '+str(Path(env['RUNS_DIR'])/'v11_3_full_fresh40'),flush=True)
    print('Rerun the SAME launcher command to resume. Do not git pull over a running source.',flush=True)


if __name__=='__main__':main()
