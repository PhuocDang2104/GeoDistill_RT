"""One entrypoint: official data -> two teachers -> full audit -> fresh V11.3 -> results."""
import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from utils import atomic_json, freeze, read_json, sha256, source_identity
from server_runtime.common import lock
from server_runtime.recovery import StopRequest


def verify_bundle():
    root=Path(__file__).parent
    manifest=read_json(root/'bundle_manifest.json')
    for name,checksum in manifest['files'].items():
        if name=='config.json':
            continue  # User fills paths/license; resolved recipe is frozen separately.
        if sha256(root/name)!=checksum:
            raise RuntimeError(f'Bundle checksum mismatch: {name}. Repackage source, no bypass.')


def doctor(cfg):
    import platform
    import shutil
    import torch
    report={'python':platform.python_version(),'torch':str(torch.__version__),'cuda_build':torch.version.cuda,
            'cuda_available':torch.cuda.is_available(),'source_sha256':source_identity(),
            'cuda_gpu_count':torch.cuda.device_count(),'single_gpu_flow':True}
    for name in ('dataset_root','teacher_root','work','run_dir','hf_cache'):
        path=Path(cfg[name]);path.mkdir(parents=True,exist_ok=True)
        report[name]={'path':str(path),'free_gib':shutil.disk_usage(path).free/2**30}
    if torch.cuda.is_available():
        report.update(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)))
        x=torch.randn(32,32,device='cuda');(x@x).sum().item()
    atomic_json(Path(cfg['work'])/'hardware.json',report)
    print(__import__('json').dumps(report,indent=2),flush=True)
    if not torch.cuda.is_available():raise RuntimeError('GPU not available; no automatic CPU training')
    if cfg['training']['amp']=='bf16' and torch.cuda.get_device_capability(0)[0]<8:
        raise RuntimeError('Native BF16 GPU required; T4/SM75 is not supported by this BF16 recipe')
    if cfg['training']['accumulation']!=1:
        raise ValueError('Production release supports accumulation=1 only; batch never silently changes')
    dataset=Path(cfg['dataset_root']);teachers=Path(cfg['teacher_root'])
    # Fail BEFORE transferring KITTI, not after wasting hours of download.
    if cfg.get('download_dataset') and not (Path(cfg['work'])/'dataset_contract.json').exists():
        count=cfg['expected_counts']['train']
        teacher_gib=count*4*352*1216*4/2**30+cfg.get('teacher_reserve_gib',20)
        common_fs=dataset.stat().st_dev==teachers.stat().st_dev
        # After acquisition has begun, some dataset bytes already occupy the SSD.
        # Do not reserve the full KITTI budget twice when recovering that phase.
        started=(dataset/'.downloads'/'data_depth_annotated.zip.complete.json').exists()
        needed=teacher_gib+(cfg.get('dataset_download_min_free_gib',120) if common_fs and not started else 0)
        if shutil.disk_usage(teachers).free<needed*2**30:
            raise RuntimeError(f'Before download: teacher SSD requires {needed:.1f} GiB free '
                               '(includes KITTI reserve when on same filesystem). Recommend >=1 TB SSD.')
    from adapter import ModelAdapter
    with torch.inference_mode():
        net=ModelAdapter(cfg,torch.device('cuda'),pretrained=False);net.network.eval()
        rgb=torch.rand(1,3,352,1216,device='cuda');mask=torch.zeros(1,1,352,1216,device='cuda')
        k=torch.tensor([[[700.,0.,608.],[0.,700.,176.],[0.,0.,1.]]],device='cuda')
        out=net.forward({'rgb':rgb,'sparse':mask,'mask':mask,'K':k})['D_full']
        if not torch.isfinite(out).all():raise RuntimeError('Student GPU forward is nonfinite before download')
        report['native_bf16_student_forward']=True
        atomic_json(Path(cfg['work'])/'hardware.json',report)
    return report


def child(action,config_path,stop,cfg):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),action,'--config',str(config_path)]
    p=subprocess.Popen(command)
    signalled=False
    while p.poll() is None:
        if stop.requested and not signalled:
            import signal
            if os.name=='nt':p.terminate()
            else:p.send_signal(signal.SIGTERM)
            signalled=True
        time.sleep(.5)
    if p.returncode==75 or stop.requested:
        return False
    if p.returncode:
        raise RuntimeError(f'Pipeline phase {action} failed (exit {p.returncode}). See preceding traceback. Train NOT started.')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('all','doctor','download','index','metric_smoke','relative_smoke','metric','relative','audit','smoke','train','evaluate_test','verify'))
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    args=parser.parse_args();verify_bundle()
    if args.action=='verify':
        print('Bundle integrity PASS');return
    cfg=read_json(args.config)
    if cfg.get('initialization')!='fresh_imagenet_encoder':
        raise ValueError('This requested production recipe is fresh student + ImageNet encoder. No subset optimizer resume.')
    work=Path(cfg['work']);work.mkdir(parents=True,exist_ok=True)
    gpu_actions = {'doctor','metric_smoke','relative_smoke','metric','relative','smoke','train','evaluate_test'}
    gpu_lock = lock(work/'.gpu.lock') if args.action in gpu_actions else contextlib.nullcontext()
    with lock(work/('.pipeline.lock' if args.action=='all' else '.'+args.action+'.lock')),gpu_lock,StopRequest() as stop:
        if args.action=='all':
            freeze(work/'pipeline_recipe.json',{'config':cfg,'source_sha256':source_identity()})
            with lock(work/'.gpu.lock'):
                doctor(cfg)
            for phase in ('download','index','metric_smoke','relative_smoke','metric','relative','audit','smoke','train','evaluate_test'):
                if (work/'STOP').exists() or stop.requested:
                    raise SystemExit(75)
                atomic_json(work/'pipeline_status.json',{'phase':phase,'status':'running','time':time.time()})
                print('\nPIPELINE PHASE:',phase,flush=True)
                if not child(phase,args.config.resolve(),stop,cfg):
                    atomic_json(work/'pipeline_status.json',{'phase':phase,'status':'paused'})
                    raise SystemExit(75)
            atomic_json(work/'pipeline_status.json',{'status':'complete','results':cfg['run_dir']})
        elif args.action=='doctor':doctor(cfg)
        elif args.action=='download':
            from acquire import acquire
            acquire(cfg)
        elif args.action=='index':
            from full_data import make_index
            make_index(cfg)
        elif args.action in ('metric_smoke','relative_smoke'):
            from teacher_cache import teacher_smoke
            teacher_smoke(cfg,args.action.removesuffix('_smoke'))
        elif args.action in ('metric','relative'):
            from full_data import load_index
            from teacher_cache import generate,storage_gate
            storage_gate(cfg,load_index(cfg,'train'))
            if not generate(cfg,args.action,stop):raise SystemExit(75)
        elif args.action=='audit':
            from teacher_cache import audit
            audit(cfg)
        else:
            gate=read_json(work/'data_gate.json')
            if not gate['passed']:raise RuntimeError('Full teacher/data gate required before GPU student execution')
            from trainer import smoke,train,evaluate_test
            if args.action=='train':
                if not train(cfg,stop):raise SystemExit(75)
            else:globals_fn={'smoke':smoke,'evaluate_test':evaluate_test}[args.action];globals_fn(cfg)


if __name__=='__main__':
    main()
