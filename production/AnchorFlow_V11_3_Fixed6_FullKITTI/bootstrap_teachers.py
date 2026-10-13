"""Install pinned teacher source + build BpOps in the SAME torch/CUDA environment."""
import argparse
import subprocess
import sys
from pathlib import Path
from utils import read_json
from teachers import DMD_CODE, MONO_CODE


def clone(url,revision,target):
    if not target.exists():
        target.mkdir(parents=True)
        subprocess.run(['git','init',str(target)],check=True)
        subprocess.run(['git','-C',str(target),'remote','add','origin',url],check=True)
    if not (target/'.git').is_dir():raise RuntimeError(f'Not a Git checkout: {target}')
    dirty=subprocess.check_output(['git','-C',str(target),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:raise RuntimeError(f'Teacher checkout has local modifications: {target}; no reset/overwrite')
    subprocess.run(['git','-C',str(target),'fetch','--depth','1','origin',revision],check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',revision],check=True)
    if subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()!=revision:
        raise RuntimeError('Teacher revision mismatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    cfg=read_json(p.parse_args().config);root=Path(cfg['third_party_root'])
    clone('https://github.com/Sharpiless/DMD3Cpp.git',DMD_CODE,root/'DMD3Cpp')
    clone('https://github.com/ByteDance-Seed/Depth-Anything-3.git',MONO_CODE,root/'Depth-Anything-3')
    # Host installations need CUDA toolkit (nvcc), C++ compiler, ninja, matching CUDA torch.
    from extension_compat import prepare_build
    build=prepare_build(root)
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=build,check=True)
    print('Pinned teacher source and BpOps ready. Real GPU smoke still required.',flush=True)


if __name__=='__main__':main()
