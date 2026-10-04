"""Two isolated REAL train runners concurrently; synthetic CPU QA, no GPU claims.

No sealed model source/config is edited. Each child simulates one interruption
after epoch0, then resumes epoch1 using the runner's actual checkpoint path.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
FOLDERS={'V10_1':ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric',
         'V11':ROOT/'drive_upload/AnchorFlow_v11_NODE'}

def child(label,root):
    from unittest.mock import patch
    import torch
    from torch.utils.data import Dataset,DataLoader
    folder=FOLDERS[label];sys.path.insert(0,str(folder))
    import run
    from test_contracts import batch
    torch.set_num_threads(2)
    started=time.time()
    class Toy(Dataset):
        def __init__(self):self.b=batch()
        def __len__(self):return 2
        def __getitem__(self,index):return {k:v[0].clone() for k,v in self.b.items()}
    def loader(c,split,generator=None,batch_size=None):
        return DataLoader(Toy(),batch_size=2,shuffle=split=='train',generator=generator)
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    config.update(work=str(root/'work'),drive_runs=str(root/'drive'),run_name=label,
        epochs=2,batch_size=2,workers=0,fused_adamw=False,amp='fp32',encoder_pretrained=False,
        log_every=1,early_stop_min_epochs=20,checkpoint_selection='rmse')
    path=root/'drive'/label/'dual_teacher'
    with patch.object(run,'device_setup',return_value=torch.device('cpu')), \
         patch.object(run,'data_loader',side_effect=loader), \
         patch('torch.cuda.get_device_name',return_value='CPU QA only, NOT GPU'):
        original=run.validate;calls=[0]
        def interrupted_validation(*args,**kwargs):
            calls[0]+=1
            if calls[0]==3:raise RuntimeError('Simulated QA disconnect before epoch1 save')
            return original(*args,**kwargs)
        with patch.object(run,'validate',side_effect=interrupted_validation):
            try:run.train(config,'dual_teacher')
            except RuntimeError as exc:
                assert str(exc)=='Simulated QA disconnect before epoch1 save',exc
            else:raise AssertionError('QA interruption not triggered')
        saved=torch.load(path/'last.pth',map_location='cpu',weights_only=False)
        assert saved['epoch']==0
        run.train(config,'dual_teacher')
        checkpoint=torch.load(path/'last.pth',map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==1 and checkpoint['global_step']==2
        before={k:t.clone() for k,t in checkpoint['model'].items()}
        run.train(config,'dual_teacher') # Completed run: no further optimizer updates.
        again=torch.load(path/'last.pth',map_location='cpu',weights_only=False)
        assert all(torch.equal(t,again['model'][k]) for k,t in before.items())
    rows=list(csv.DictReader((path/'train_log.csv').open(encoding='utf-8')))
    assert [int(r['epoch']) for r in rows]==[0,1]
    assert all(float(r['val_executed_nfe'])>= (13 if label=='V11' else 2) for r in rows)
    status=json.loads((path/'training_status.json').read_text())
    assert status['status']=='max_epochs_completed' and status['epochs_completed']==2
    for name in ('best.pth','best_inverse.pth','best_joint.pth','initial_val_metrics.json','train_log.jsonl'):
        assert (path/name).is_file(),name
    proof={'label':label,'passed':True,'device':'CPU; GPU guard and label mocked',
        'dataset':'synthetic64x128, 2 train/val samples; NOT KITTI benchmark',
        'optimizer_updates':2,'epochs_logged':[0,1],'interruption_resume_verified':True,
        'completed_resume_is_noop':True,'started_unix':started,'finished_unix':time.time(),
        'sources_sha256':run.source_hash(),'actual_NFE_logged':[float(r['val_executed_nfe']) for r in rows]}
    (root/f'{label}_proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',choices=tuple(FOLDERS));parser.add_argument('--root',type=Path)
    args=parser.parse_args()
    if args.child:return child(args.child,args.root)
    import anchorflow_pair_benchmark as bench
    with tempfile.TemporaryDirectory(prefix='anchorflow_pair_cpu_qa_') as temp:
        root=Path(temp);work=root/'work';work.mkdir()
        bench.atomic_json(work/'data_contract.json',{'train':1600,'val':400,'test':1000,'synthetic_cpu_qa_only':True})
        commands=[bench.Command(label,'synthetic-runner-qa',
            [sys.executable,'-u',str(Path(__file__).resolve()),'--child',label,'--root',str(root)],folder,
            root/f'{label}.log',root/'logs'/f'{label}.log',bench.process_environment('',folder,2))
            for label,folder in FOLDERS.items()]
        processes=bench.run_processes(commands,parallel=True,status_path=root/'process_status.json',mirror_seconds=1)
        proofs=[bench.read_json(root/f'{label}_proof.json') for label in FOLDERS]
        overlap=max(r['started_unix'] for r in proofs)<min(r['finished_unix'] for r in proofs)
        assert overlap,'Actual training runner intervals must overlap'
        for c in commands:assert c.local_log.read_bytes()==c.drive_log.read_bytes()
        report={'passed':True,'actual_train_runner_intervals_overlap':overlap,'isolated_module_imports':True,
                'shared_work_and_separate_outputs':True,'gpu_training_tested':False,'GPU_metrics_measured':False,
                'children':proofs,'processes':processes,'notebook_cells':16,
                'limitation':'Synthetic CPU QA, not runtime/VRAM/accuracy evidence on Colab GPU.'}
    output=ROOT/'results/anchorflow_pair_local_verification.json'
    bench.atomic_json(output,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
