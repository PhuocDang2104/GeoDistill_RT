"""Read-only V10 checkpoint pixel audit, CPU FP32; not a training result."""
import heapq
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'drive_upload/AnchorFlow_v10_AdaptiveJet'
OUT = ROOT / 'results/anchorflow_v10_completed_audit'
sys.path.insert(0, str(FOLDER))
from model import AnchorFlowEdge
from data import KITTIDataset, write_json


def main():
    torch.set_num_threads(4)
    model = AnchorFlowEdge().eval()
    model.load_state_dict(torch.load(OUT/'best.pth', map_location='cpu', weights_only=False)['model'])
    ds = object.__new__(KITTIDataset)
    ds.root = ROOT/'data/teacher_subset_2000/kitti_bundle'
    ds.split, ds.teacher = 'val', False
    ds.rows = [r.split() for r in (ds.root/'splits/val_400.txt').read_text().splitlines()]
    sums, top = {}, []
    saturation = {'D1': [0,0], 'D2': [0,0]}
    with torch.inference_mode():
        for i in range(len(ds)):
            b = {k: v[None] if torch.is_tensor(v) else v for k,v in ds[i].items()}
            o = model(b['rgb'], b['sparse'], b['mask'], b['K'])
            gt, m = b['gt'].numpy()[0,0], b['gt_mask'].numpy()[0,0] > .5
            m &= (gt > .1) & (gt < 120)
            observed = b['mask'].numpy()[0,0] > .5
            groups = {'all':m, 'observed':m & observed, 'unobserved':m & ~observed}
            for lo,hi in ((0,5),(5,10),(10,20),(20,40),(40,60),(60,80),(80,120)):
                groups[f'{lo}-{hi}'] = m & (gt>=lo) & (gt<hi)
            for stage in ('D1_base','D1','D_full','D_hard'):
                p = o[stage].numpy()[0,0]
                e = p.astype(np.float64)-gt
                inv = 1000/np.maximum(p.astype(np.float64),.1)-1000/np.maximum(gt,.1)
                for label, mask in groups.items():
                    key = stage+'/'+label
                    t = sums.setdefault(key, [0,0.,0.,0.,0.,0,0.])
                    t[0] += int(mask.sum()); t[1] += float(np.square(e[mask]).sum())
                    t[2] += float(np.square(inv[mask]).sum()); t[3] += float(np.abs(e[mask]).sum())
                    t[4] += float(np.abs(inv[mask]).sum())
                    tail = mask & (np.abs(e)>5)
                    t[5] += int(tail.sum()); t[6] += float(np.square(e[tail]).sum())
                if stage == 'D_full':
                    indices=np.flatnonzero(m)
                    worst=indices[np.argsort(np.square(inv).ravel()[indices])[-10:]]
                    for index in worst:
                        y,x=divmod(int(index),gt.shape[1]); score=float(inv[y,x]**2)
                        row={'sample_index':i,'sample_id':str(b.get('sid',i)),'u':x,'v':y,
                             'gt':float(gt[y,x]),'observed':bool(observed[y,x]),
                             **{s:float(o[s][0,0,y,x]) for s in ('D1_base','D1','D_full','sensor_gate')}}
                        heapq.heappush(top,(score,i,index,row))
                        if len(top)>100:heapq.heappop(top)
            for stage,delta,base in (('D1','delta1','D1_base'),('D2','phase2_delta','D2_base')):
                bound = (.5+.05*o[base]) if stage=='D1' else (1+.05*o[base])
                ratio=(o[delta].abs()/bound).numpy()
                saturation[stage][0]+=int((ratio>.9).sum());saturation[stage][1]+=ratio.size
            if (i+1)%50==0:
                print(f'V10 pixel audit {i+1}/400',flush=True)
                write_json(OUT/'pixel_audit_progress.json',{'completed':i+1})
    result={'precision':'CPU FP32 re-evaluation, not original GPU BF16', 'samples':len(ds),
            'saturation_fraction':{k:n/max(d,1) for k,(n,d) in saturation.items()}, 'metrics':{}}
    for key,(n,se,ise,ae,iae,tailn,tailse) in sums.items():
        total=sums[key.split('/')[0]+'/all']
        result['metrics'][key]={'pixels':n,'rmse_m':(se/max(n,1))**.5,'irmse_km_inv':(ise/max(n,1))**.5,
          'mae_m':ae/max(n,1),'imae_km_inv':iae/max(n,1),'metric_sse_share':se/max(total[1],1e-12),
          'inverse_sse_share':ise/max(total[2],1e-12),'tail_gt5_fraction':tailn/max(n,1),
          'tail_gt5_sse_share':tailse/max(se,1e-12)}
    result['top100_inverse_outliers']=[row for *_,row in sorted(top,reverse=True)]
    write_json(OUT/'pixel_audit.json',result)
    print(json.dumps({k:v for k,v in result['metrics'].items() if k.startswith('D_full/')},indent=2))

if __name__=='__main__':main()
