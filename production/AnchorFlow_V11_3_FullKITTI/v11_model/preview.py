"""Actual final KITTI predictions -> ten native-resolution RGB/depth PNG pairs."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def render(config,variant):
    import run
    from data import KITTIDataset,write_json
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import colormaps,font_manager
    local,drive=run.directories(config,variant)
    report=json.loads((drive/'test_report.json').read_text())
    if report['checkpoint_selection']!=config.get('checkpoint_selection','policy'):
        raise RuntimeError('Preview/test checkpoint selection mismatch')
    if report['source_sha256']!=run.source_hash():raise RuntimeError('Preview/test source mismatch')
    folder=local/'test_preview';folder.mkdir(parents=True,exist_ok=True)
    for name in ('depth_color','pairs'):(folder/name).mkdir(exist_ok=True)
    font_path=font_manager.findfont('DejaVu Sans')
    def font(size):return ImageFont.truetype(font_path,size)
    def colored(depth):
        rgba=colormaps['turbo_r'](np.clip(depth/120.,0,1))
        return Image.fromarray(np.round(255*rgba[...,:3]).astype(np.uint8))
    ids=[f'{i:010d}' for i in np.linspace(0,999,10,dtype=int)]
    dataset=KITTIDataset(config,'test',teacher=False)
    indices={row[0]:i for i,row in enumerate(dataset.rows)}
    records=[];rows=[]
    with ZipFile(drive/'kitti_test_predictions.zip') as predictions:
        names=predictions.namelist()
        if len(names)!=1000 or len(set(names))!=1000:raise RuntimeError('Expected anonymous1000 ZIP')
        for sid in ids:
            raw=predictions.read(sid+'.png')
            with Image.open(io.BytesIO(raw)) as image:encoded=np.asarray(image)
            if encoded.dtype!=np.uint16 or encoded.shape!=(352,1216):raise RuntimeError('Wrong KITTI PNG encoding')
            depth=encoded.astype(np.float32)/256.
            if not np.isfinite(depth).all() or not (depth>0).all():raise RuntimeError('Invalid prediction')
            sample=dataset[indices[sid]]
            rgb=np.round(sample['rgb'].permute(1,2,0).numpy()*255).astype(np.uint8)
            rgb=Image.fromarray(rgb);depth_rgb=colored(depth)
            depth_rgb.save(folder/'depth_color'/f'{sid}.png')
            row=Image.new('RGB',(2444,440),'white');draw=ImageDraw.Draw(row)
            draw.text((16,10),f'RGB | KITTI {sid}',fill='black',font=font(24))
            draw.text((1244,10),f'V11_3 final depth | {sid} | epoch {report["checkpoint_epoch"]}',fill='black',font=font(24))
            row.paste(rgb,(0,50));row.paste(depth_rgb,(1228,50))
            draw.text((16,408),'Actual final student PNG /256 metres. No smoothing/per-image normalization. Anonymous test has no public GT.',fill='black',font=font(19))
            def legend(width):
                legend=Image.new('RGB',(width,70),'white');draw=ImageDraw.Draw(legend)
                draw.text((18,0),'Shared 0-120 m | red near / blue-purple far',fill='black',font=font(20))
                gradient=np.tile(np.linspace(0,120,width-40,dtype=np.float32),(15,1))
                legend.paste(colored(gradient),(20,28))
                for tick in range(0,121,20):
                    draw.text((20+(width-41)*tick/120,48),str(tick),anchor='mt',fill='black',font=font(16))
                return legend
            pair=Image.new('RGB',(row.width,510),'white');pair.paste(row,(0,0));pair.paste(legend(row.width),(0,440))
            pair.save(folder/'pairs'/f'{sid}_rgb_v11_3.png')
            rows.append(row)
            records.append({'sample_id':sid,'prediction_sha256':hashlib.sha256(raw).hexdigest(),
                            'min_m':float(depth.min()),'max_m':float(depth.max())})
    for page in range(2):
        sheet=Image.new('RGB',(2444,2330),'white')
        ImageDraw.Draw(sheet).text((20,12),f'Actual KITTI test | RGB / V11_3 | page {page+1}/2',fill='black',font=font(26))
        for i,row in enumerate(rows[5*page:5*page+5]):sheet.paste(row,(0,60+440*i))
        sheet.paste(legend(2444),(0,2260));sheet.save(folder/f'rgb_v11_3_sheet_{page+1}.png')
    write_json(folder/'preview_manifest.json',{'samples':records,'checkpoint_epoch':report['checkpoint_epoch'],
        'checkpoint_selection':report['checkpoint_selection'],'source_sha256':report['source_sha256'],
        'encoding':'Actual uint16 /256 metres','scale_m':[0,120],'colormap':'turbo_r',
        'native_panel':[1216,352],'selection':'Ten evenly spaced IDs; not error-based cherry-picking',
        'test_rmse_m':None,'no_public_test_gt':True,'no_smoothing_or_per_image_normalization':True})
    for source in folder.rglob('*'):
        if source.is_file():run.copy_atomic(source,drive/'test_preview'/source.relative_to(folder))
    print('10 colored PNGs / 10 RGB pairs / two 5-scene sheets:',drive/'test_preview',flush=True)
