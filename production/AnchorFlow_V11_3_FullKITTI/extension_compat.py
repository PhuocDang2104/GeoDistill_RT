"""Mechanical tensor-API compatibility, NOT a change to BpOps math/kernel loops."""
import hashlib
import shutil
from pathlib import Path
from utils import atomic_json, sha256

KERNEL_SHA='ff6dd04d68f242188ad065c8e1253aad11cf50c42e79f2e95a2d4e69c3b4b47b'


def modernize_text(text):
    if hashlib.sha256(text.encode()).hexdigest()!=KERNEL_SHA:
        raise RuntimeError('Official BpOps kernel differs from pinned source; refusing automatic patch')
    if text.count('.type()')!=3 or text.count('.data<')!=8:
        raise RuntimeError('Unexpected number of deprecated tensor API sites')
    return text.replace('.type()', '.scalar_type()').replace('.data<', '.data_ptr<')


def prepare_build(root):
    source=Path(root)/'DMD3Cpp'/'exts'
    target=Path(root)/'BpOps_build_torch210';target.mkdir(parents=True,exist_ok=True)
    for name in ('setup.py','bp_cuda.cpp','bp_cuda.h','bp_cuda_kernel.cu'):
        shutil.copy2(source/name,target/name)
    # Bulk mechanical API rewrite in an isolated build copy. Official Git checkout stays clean.
    kernel=target/'bp_cuda_kernel.cu'
    text=kernel.read_text(encoding='utf-8')
    kernel.write_text(modernize_text(text),encoding='utf-8',newline='\n')
    atomic_json(target/'compatibility.json',{'original_kernel_sha256':KERNEL_SHA,
                 'compiled_kernel_sha256':sha256(kernel),
                 'changes':['3 Tensor.type() -> scalar_type()', '8 Tensor.data<T>() -> data_ptr<T>()'],
                 'kernel_math_unchanged':True})
    return target
