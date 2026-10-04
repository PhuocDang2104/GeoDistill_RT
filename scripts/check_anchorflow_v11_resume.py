"""Actual runner interruption/resume QA on synthetic CPU data, not training results."""
import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'drive_upload/AnchorFlow_v11_NODE'
sys.path.insert(0,str(FOLDER))
import run
from test_contracts import batch

class Toy(Dataset):
    def __init__(self):self.b=batch()
    def __len__(self):return 2
    def __getitem__(self,index):return {k:v[0].clone() for k,v in self.b.items()}

def main():
    torch.set_num_threads(2)
    cfg=json.loads((FOLDER/'config.json').read_text())
    with tempfile.TemporaryDirectory(prefix='anchorflow_v11_resume_') as temp:
        root=Path(temp);work=root/'work';work.mkdir()
        (work/'data_contract.json').write_text(json.dumps({'train':1600,'val':400,'test':1000,'synthetic_cpu_qa_only':True}))
        cfg.update(work=str(work),drive_runs=str(root/'drive'),epochs=2,batch_size=2,workers=0,
            fused_adamw=False,amp='fp32',encoder_pretrained=False,log_every=1,early_stop_min_epochs=20)
        def loader(c,split,generator=None,batch_size=None):
            return DataLoader(Toy(),batch_size=2,shuffle=split=='train',generator=generator)
        def mocks():
            return (patch.object(run,'device_setup',return_value=torch.device('cpu')),
                    patch.object(run,'data_loader',side_effect=loader),
                    patch('torch.cuda.get_device_name',return_value='CPU QA mock, NOT GPU'))
        for name,interrupted in (('full',False),('resumed',True)):
            config={**cfg,'run_name':name};a,b,c=mocks()
            with a,b,c:
                if interrupted:
                    original=run.validate;calls=[0]
                    def interrupt(*args,**kwargs):
                        calls[0]+=1
                        if calls[0]==3:raise RuntimeError('QA interruption before epoch1 save')
                        return original(*args,**kwargs)
                    with patch.object(run,'validate',side_effect=interrupt):
                        try:run.train(config,'dual_teacher')
                        except RuntimeError as exc:assert str(exc)=='QA interruption before epoch1 save',exc
                        else:raise AssertionError('Interruption not triggered')
                    saved=torch.load(root/'drive/resumed/dual_teacher/last.pth',map_location='cpu',weights_only=False)
                    assert saved['epoch']==0
                run.train(config,'dual_teacher')
        runs={}
        for name in ('full','resumed'):
            path=root/'drive'/name/'dual_teacher'
            for file in ('last.pth','best.pth','best_inverse.pth','best_joint.pth',
                         'best_val_metrics.json','best_inverse_val_metrics.json','best_joint_val_metrics.json',
                         'selection_metrics.json','training_status.json'):
                assert (path/file).is_file(),(name,file)
            runs[name]=torch.load(path/'last.pth',map_location='cpu',weights_only=False)
            rows=list(csv.DictReader((path/'train_log.csv').open()))
            assert [int(r['epoch']) for r in rows]==[0,1]
            assert all(float(r['val_terminal_time'])==1. for r in rows)
            assert all(float(r['val_executed_nfe'])>=13 for r in rows)
        a,b=runs['full'],runs['resumed']
        assert a['global_step']==b['global_step']==2
        assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
        assert a['metric_selection']==b['metric_selection']
        assert torch.equal(a['rng_torch'],b['rng_torch']) and torch.equal(a['rng_loader'],b['rng_loader'])
        proof={'passed':True,'device':'CPU, GPU guard/label mocked','data':'synthetic64x128',
               'uninterrupted_vs_resume_model_bitwise_equal':True,'rng_equal':True,'selection_equal':True,
               'optimizer_updates':2,'epochs':[0,1],'fixed_terminal_time':1.,
               'actual_adaptive_nfe_logged':True,'gpu_training_measured':False}
    path=FOLDER/'local_verification.json';previous=json.loads(path.read_text())
    previous['runner_resume_integration']=proof;run.write_json(path,previous)
    print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
