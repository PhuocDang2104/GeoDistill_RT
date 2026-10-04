"""One clean, upload-ready Colab notebook; fresh40/control40/optional fine20."""
import ast
import textwrap
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric'

def md(s):return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s):return nb.v4.new_code_cell(textwrap.dedent(s).strip())

def main():
    cells=[md('''
    # AnchorFlow V10.1 LiteMetric — train / val / test

    Upload nguyên folder **AnchorFlow_v10_1_LiteMetric** vào MyDrive; không upload lại data.
    Mặc định **fresh epoch0, max40 + early stop**, ImageNet RGB encoder only.
    Cùng 1.600 train / 400 validation / 1.000 anonymous test như V10.
    Hai bước cố định, nhưng h vẫn learned theo sample/state (1/6..1/3, fresh init0.25).
    Bỏ stop controller/rollout thừa; context-guided final phase lift + loss trực tiếp iRMSE.
    Mục tiêu RMSE<0.9 và iRMSE<3.2 **chưa được chứng minh**. GPU accuracy/runtime phải đo sau train.
    '''),code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import hashlib,json,os,shutil,subprocess,sys
    BUNDLE=Path('/content/drive/MyDrive/AnchorFlow_v10_1_LiteMetric')
    DRIVE_DATA=Path('/content/drive/MyDrive/GeoLift_Data')
    DRIVE_RUNS=Path('/content/drive/MyDrive/GeoLift_RT_Runs')
    CODE=Path('/content/anchorflow_v10_1_code')
    WORK=Path('/content/anchorflow_v10_1_work')
    TRAIN_MODE='fresh40' # fresh40 | finetune20 | control40
    VARIANT='dual_teacher' # Same teachers as V10; metric_kd skips relative TAR, gt_only disables KD
    AMP_REQUEST='auto' # native BF16 GPU -> bf16; T4/V100 -> explicit fp32
    BATCH_SIZE=4
    WORKERS=2
    RUN_TAG='_learnedH2' # Isolated from previous fixed-h bundle; same tag resumes this recipe only
    PARENT_RUN='AnchorFlow_v10_adaptive_Fresh40_ES_bf16'
    PARENT_VARIANT='dual_teacher'
    RUN_EXPORT=True
    assert BUNDLE.is_dir(),f'Upload extracted folder: {BUNDLE}'
    assert TRAIN_MODE in ('fresh40','finetune20','control40')
    assert VARIANT in ('dual_teacher','metric_kd','gt_only')
    assert shutil.disk_usage('/content').free/2**30>25,'Need >25GiB local Colab SSD; does not use your PC SSD'
    '''),md('''
    ## Verify bundle, install dependencies, run contracts

    Checksum bỏ qua .ipynb UI vì Colab thay metadata/output. Code/config/_contract.json vẫn strict.
    Không cài lại Torch/CUDA. Test chạy subprocess, không giữ model GPU trong notebook parent.
    '''),code('''
    manifest=json.loads((BUNDLE/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,sha in manifest['files'].items():
        if name.endswith('.ipynb'):continue
        source=BUNDLE/name
        assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==sha,f'Bundle changed: {name}'
    CODE.mkdir(parents=True,exist_ok=True)
    for name in manifest['files']:
        if name.endswith('.ipynb'):continue
        source,dest=BUNDLE/name,CODE/name
        if source.resolve()!=dest.resolve():shutil.copy2(source,dest)
    shutil.copy2(BUNDLE/'bundle_manifest.json',CODE/'bundle_manifest.json')
    os.chdir(CODE)
    subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(CODE/'requirements.txt')],check=True)
    import torch
    assert torch.cuda.is_available(),'Select GPU runtime in Colab'
    native_bf16=torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported(including_emulation=False)
    assert AMP_REQUEST in ('auto','bf16','fp32')
    if AMP_REQUEST=='bf16':assert native_bf16,'No native BF16: choose fp32, not FP16'
    ACTUAL_AMP=('bf16' if native_bf16 else 'fp32') if AMP_REQUEST=='auto' else AMP_REQUEST
    print(torch.__version__,torch.cuda.get_device_name(0),'precision:',ACTUAL_AMP)
    tested=subprocess.run([sys.executable,'-u','-m','unittest','discover','-s',str(CODE),'-v'],cwd=CODE,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    print(tested.stdout+tested.stderr)
    assert tested.returncode==0,'Contract test failed; read full traceback above'
    '''),md('''
    ## Resolve isolated configuration

    fresh40 là comparison chính. finetune20 dùng V10 best epoch23 MODEL-only, optimizer mới;
    không phải equal-budget comparison. control40 bỏ context bypass, giữ các hệ số loss V10 cũ,
    nhưng vẫn cố định2 bước và bỏ controller; không phải exact adaptive-V10 reproduction.
    Không thay WIDTH/K/backbone/depth bounds. Cả2 teacher chỉ đọc ở training split.
    '''),code('''
    base={'fresh40':'config.json','finetune20':'finetune20_config.json','control40':'fixed2_control_config.json'}[TRAIN_MODE]
    cfg=json.loads((CODE/base).read_text(encoding='utf-8'))
    cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(DRIVE_RUNS),work=str(WORK),batch_size=BATCH_SIZE,
        workers=WORKERS,amp=ACTUAL_AMP,amp_request=AMP_REQUEST,
        teacher_enabled=VARIANT!='gt_only',relative_enabled=VARIANT=='dual_teacher')
    cfg['run_name']=f'AnchorFlow_v10_1_{TRAIN_MODE}_{ACTUAL_AMP}'+RUN_TAG
    if TRAIN_MODE=='finetune20':
        parent=DRIVE_RUNS/PARENT_RUN/PARENT_VARIANT/'best.pth'
        assert parent.is_file(),f'Missing {parent}; correct PARENT_RUN or use fresh40'
        cfg['init_checkpoint']=str(parent)
        cfg['init_checkpoint_sha256']=hashlib.sha256(parent.read_bytes()).hexdigest()
    else:cfg['init_checkpoint']=None
    CONFIG=CODE/'resolved_config.json'
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    RUN_ROOT=DRIVE_RUNS/cfg['run_name'];RUN_DIR=RUN_ROOT/VARIANT
    SUBSET=DRIVE_DATA/'teacher_subset_2000'
    required=['selected_2000_ids.json','kitti_trainval_2000.tar']
    if cfg['teacher_enabled']:required.append(cfg['metric_tar'])
    if cfg['relative_enabled']:required.append(cfg['relative_tar'])
    for name in required:assert (SUBSET/name).is_file(),f'Missing {SUBSET/name}'
    snapshot=RUN_ROOT/'source_bundle'
    names=[n for n in manifest['files'] if not n.endswith('.ipynb')]+['resolved_config.json','bundle_manifest.json']
    # Validate ALL existing snapshots before writing any new files.
    for name in names:
        dest=snapshot/name
        if dest.exists():assert dest.read_bytes()==(CODE/name).read_bytes(),f'Frozen recipe changed: {name}; change RUN_TAG'
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in names:
        src,dest=CODE/name,snapshot/name
        if not dest.exists() and src.resolve()!=dest.resolve():shutil.copy2(src,dest)
    print(json.dumps(cfg,indent=2));print('Output:',RUN_DIR)

    def command(action,extra=()):
        current=json.loads(CONFIG.read_text(encoding='utf-8'))
        target=Path(current['drive_runs'])/current['run_name']/VARIANT
        target.mkdir(parents=True,exist_ok=True)
        logfile=CODE/(action+'_console.log')
        args=[sys.executable,'-u',str(CODE/'run.py'),action,'--config',str(CONFIG),'--variant',VARIANT,*extra]
        with logfile.open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(args,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace')
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            status=proc.wait()
        shutil.copy2(logfile,target/logfile.name)
        diagnostic=Path(current['work'])/'gpu_failure_report.json'
        if diagnostic.is_file():shutil.copy2(diagnostic,target/(action+'_gpu_failure_report.json'))
        if status:raise RuntimeError(f'{action} failed; traceback above, Drive console: {target/logfile.name}')
    '''),md('''
    ## Data gate → real paired RGB / sparse / GT / teacher preview → smoke

    Đọc các TAR đang có trên GeoLift_Data, extract/cache trên /content để train nhanh.
    Nếu test TAR chưa có, chuẩn bị official KITTI test như workflow cũ.
    Source mới không cần geometry_fused/DSINE; relative TAR dùng đúng cache DA3 đã sinh.
    '''),code('''
    command('prepare')
    contract=json.loads((WORK/'data_contract.json').read_text(encoding='utf-8'))
    assert (contract['train'],contract['val'],contract['test'])==(1600,400,1000)
    assert contract['manifest_sha256']==cfg['expected_subset_sha256'] and not contract['validation_teacher_used']
    if cfg['relative_enabled']:
        assert contract['relative_teacher']['cached_train']==1600 and contract['relative_teacher']['cached_val']==0
    shutil.copy2(WORK/'data_contract.json',RUN_ROOT/'data_contract.json')
    sys.path.insert(0,str(CODE))
    import importlib,data
    importlib.reload(data)
    import matplotlib.pyplot as plt
    sample=data.KITTIDataset(cfg,'train',teacher=cfg['teacher_enabled'])[0]
    val=data.KITTIDataset(cfg,'val',teacher=False)[0]
    assert not {'teacher','confidence','relative','relative_confidence'}.intersection(val)
    keys=['rgb','sparse','gt']+(['teacher'] if cfg['teacher_enabled'] else [])+(['relative'] if cfg['relative_enabled'] else [])
    fig,axes=plt.subplots(1,len(keys),figsize=(4*len(keys),4))
    for ax,key in zip(axes,keys):
        if key=='rgb':ax.imshow(sample[key].permute(1,2,0))
        else:ax.imshow(sample[key][0],cmap='turbo_r',vmin=-2 if key=='relative' else 0,vmax=2 if key=='relative' else 80)
        ax.set_title(key);ax.axis('off')
    plt.tight_layout();plt.show();fig.savefig(RUN_ROOT/'data_preview.png',dpi=120)
    del sample,val
    import gc
    gc.collect();torch.cuda.empty_cache()
    command('smoke')
    print(json.loads((WORK/'smoke_report.json').read_text()))
    '''),md('''
    ## Train / resume

    Max40 epochs fresh; early stop min20/patience8/min_delta0.001m. Fine20: min8/patience5.
    Model train end-to-end từ đầu, inverse-RMSE loss ramp4epoch; không curriculum module.
    Bị ngắt: chạy lại notebook cùng TRAIN_MODE/VARIANT/RUN_TAG, tự resume last.pth + optimizer/RNG.
    Mỗi epoch đồng bộ CSV/JSONL/checkpoints Drive. Không skip NaN hoặc đổi precision giữa run.
    '''),code('''
    command('train')
    print(json.loads((RUN_DIR/'training_status.json').read_text()))
    print('Logs:',RUN_DIR/'train_log.csv','Best RMSE:',RUN_DIR/'best.pth')
    '''),md('''
    ## Evaluate global / far / near-inverse / boundary / tails

    best.pth luôn chọn minimum RMSE. best_inverse.pth chọn minimum iRMSE.
    best_joint.pth chọn minimum max(RMSE/0.9, iRMSE/3.2); đạt cả2 nếu score<1.
    Báo rõ từng checkpoint, không ghép RMSE của epoch này với iRMSE của epoch khác.
    '''),code('''
    command('evaluate')
    import pandas as pd
    best=json.loads((RUN_DIR/'val_metrics.json').read_text())
    assert best['final']['all']['pixels']==25424992,'GT support/protocol changed'
    display(pd.DataFrame(best['final']).T)
    display(pd.DataFrame(best['stage_native_gt_metrics']).T)
    display(pd.DataFrame(best['near_inverse_diagnostics']).T)
    display(pd.DataFrame(best['gt_boundary']['bands']).T)
    display(pd.DataFrame(best['final']['all']['error_tail']).T)
    selections=[]
    for label,filename in [('RMSE','best_val_metrics.json'),('inverse','best_inverse_val_metrics.json'),('joint','best_joint_val_metrics.json')]:
        r=json.loads((RUN_DIR/filename).read_text());s=r['final']['all']
        selections.append({'selection':label,'epoch':r['epoch'],'RMSE m':s['rmse_m'],'iRMSE km^-1':s['irmse_km_inv'],
                           'both_targets':s['rmse_m']<.9 and s['irmse_km_inv']<3.2})
    selection_table=pd.DataFrame(selections);display(selection_table)
    selection_table.to_csv(RUN_DIR/'checkpoint_selection_metrics.csv',index=False)
    print('Best RMSE epoch:',best['checkpoint_epoch'])
    print('Learned steps:',best['integration'])
    '''),code('''
    log=pd.read_csv(RUN_DIR/'train_log.csv')
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    axes[0].plot(log.epoch,log.val_rmse_m);axes[0].axhline(.9,color='red',ls='--');axes[0].set_title('RMSE m')
    axes[1].plot(log.epoch,log.val_irmse_km_inv);axes[1].axhline(3.2,color='red',ls='--');axes[1].set_title('iRMSE km^-1')
    for k in range(1,cfg['flow_steps']+1):axes[2].plot(log.epoch,log[f'val_native_D4_step{k}_rmse_m'],label=f'step{k}')
    axes[2].set_title('Native quarter-grid RMSE');axes[2].legend()
    for ax in axes:ax.grid(alpha=.3);ax.set_xlabel('Epoch')
    plt.tight_layout();plt.show();fig.savefig(RUN_DIR/'learning_curves.png',dpi=150)
    log[['epoch']+[k for k in log if k.startswith('loss_weighted_')]].to_csv(RUN_DIR/'loss_budget.csv',index=False)
    display(log[['epoch','val_h1_mean','val_h2_mean','val_terminal_time']].tail(10))
    def record(label,report):
        s=report['final'];a=s['all']
        return {'model':label,'RMSE m':a['rmse_m'],'MAE m':a['mae_m'],'iRMSE km^-1':a['irmse_km_inv'],
            **{f'RMSE {k}':s[k]['rmse_m'] for k in ('0-20','20-40','40-60','60-80','80-120','edge')},
            'GT boundary3':report['gt_boundary']['bands']['3']['rmse_m']}
    rows=[record(label,json.loads((CODE/name).read_text())) for label,name in [('V8','v8_val_metrics.json'),
          ('V9','v9_val_metrics.json'),('V9.1','v9_1_val_metrics.json'),('V10','v10_val_metrics.json')]]
    rows.append(record('V10.1 '+TRAIN_MODE,best));comparison=pd.DataFrame(rows);display(comparison)
    comparison.to_csv(RUN_DIR/'historical_comparison.csv',index=False)
    print('Historical runs differ in budget/init. Architecture + objective changed together, not causal ablation.')
    '''),md('''
    ## Test1000 → hardware profile → ONNX parity

    Dùng best.pth (minimum RMSE) thống nhất; anonymous test không có GT công khai.
    Profiler bỏ telemetry reductions, đo cùng graph fixed2 như ONNX/test, batch1.
    Báo synthetic CUDA median/P95, peak VRAM, separate components và real100scene wall median/P95.
    ONNX static1×352×1216: check9cases synthetic/empty/real; chưa phải TensorRT deployment benchmark.
    '''),code('''
    command('test')
    command('profile')
    profile=json.loads((RUN_DIR/'profile.json').read_text())
    print('Parameters:',profile['total_parameters'],'Conv/Linear MAC:',profile['total_conv_linear_macs'])
    print('GPU median/P95:',profile['median_ms'],profile['p95_ms'],'Peak MiB:',profile['peak_cuda_allocated_mib'])
    print('Real scenes:',profile['real_scene_profile'])
    display(pd.DataFrame({'median_ms':profile['eager_component_median_ms']}))
    if RUN_EXPORT:command('export')
    print('DONE:',RUN_DIR,'Predictions:',RUN_DIR/'kitti_test_predictions.zip')
    ''')]
    name='Train_V10_1_LiteMetric_TAR2000.ipynb'
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
         'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':name}})
    for cell in cells:
        if cell.cell_type=='code':ast.parse(cell.source)
    nb.validate(book);nb.write(book,OUT/name)
    (OUT/(Path(name).stem+'_contract.json')).write_text(nb.writes(book),encoding='utf-8')
    print('Notebook ready:',name,len(cells),'cells')

if __name__=='__main__':main()
