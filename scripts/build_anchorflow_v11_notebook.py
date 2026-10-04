"""One fresh40 Colab entry point; immutable source snapshot, mutable notebook UI."""
import ast
import textwrap
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'drive_upload/AnchorFlow_v11_NODE'
def md(s):return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s):return nb.v4.new_code_cell(textwrap.dedent(s).strip())

def main():
    cells=[md('''
    # AnchorFlow V11 — Jet NODE · fresh40 + early stop

    NN chỉ học vector field. Solver **torchdiffeq RK3(2), bosh3** chọn bước bằng sai số số học,
    luôn tới **T=1**. Không learned h, không GT/task-stop, không projected Euler.
    582.912 parameters; adaptive NFE thường lớn hơn 4, phải đo sau train.
    Upload nguyên folder `AnchorFlow_v11_NODE` vào MyDrive. Dùng lại **hai teacher TAR đã có**.
    1.600 train / 400 validation / 1.000 anonymous test; không thay split/depth protocol.
    Fresh student epoch0; chỉ RGB encoder ImageNet pretrained. Không warm-start V8/V9/V10.
    Max40 + early stop min20/patience8. Mục tiêu RMSE<0.9, iRMSE<3.2 **chưa được chứng minh**.
    '''),code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import hashlib,json,os,shutil,subprocess,sys,gc
    BUNDLE=Path('/content/drive/MyDrive/AnchorFlow_v11_NODE')
    DRIVE_DATA=Path('/content/drive/MyDrive/GeoLift_Data')
    DRIVE_RUNS=Path('/content/drive/MyDrive/GeoLift_RT_Runs')
    CODE=Path('/content/anchorflow_v11_code')
    WORK=Path('/content/anchorflow_v11_work')
    VARIANT='dual_teacher' # dual_teacher | metric_kd | gt_only
    AMP_REQUEST='auto' # native BF16 GPU -> BF16 CNN; T4/V100 -> FP32
    BATCH_SIZE=4
    WORKERS=2
    RUN_TAG='_bosh3_T1' # Keep unchanged to resume; change for a genuinely new run
    RUN_SOLVER_AUDIT=True # Same best checkpoint: default/tighter RK3(2) and fixed midpoint8
    RUN_EXPORT=True # STATIC MIDPOINT8 export, NOT adaptive RK3(2) export
    assert BUNDLE.is_dir(),f'Upload extracted folder: {BUNDLE}'
    assert VARIANT in ('dual_teacher','metric_kd','gt_only')
    assert shutil.disk_usage('/content').free/2**30>25,'Need >25GiB local Colab SSD, not PC SSD'
    '''),md('''
    ## Verify → install → contract tests

    Không reinstall Torch/CUDA. Checksum bỏ qua `.ipynb` UI vì Colab sửa output/metadata;
    toàn bộ code/config và notebook `_contract.json` vẫn được verify nghiêm ngặt.
    Không bỏ qua test failure hoặc tự đổi precision để chữa một lỗi run.
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
    import torch,torchdiffeq
    assert torch.cuda.is_available(),'Select a GPU runtime in Colab'
    assert torchdiffeq.__version__=='0.2.5'
    native_bf16=torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported(including_emulation=False)
    assert AMP_REQUEST in ('auto','bf16','fp32')
    if AMP_REQUEST=='bf16':assert native_bf16,'No native BF16: select fp32, not FP16'
    ACTUAL_AMP=('bf16' if native_bf16 else 'fp32') if AMP_REQUEST=='auto' else AMP_REQUEST
    print(torch.__version__,torch.cuda.get_device_name(0),'CNN:',ACTUAL_AMP,'ODE field/state: FP32')
    tested=subprocess.run([sys.executable,'-u','-m','unittest','discover','-s',str(CODE),'-v'],cwd=CODE,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    print(tested.stdout+tested.stderr)
    assert tested.returncode==0,'Contract test failed; read the actual traceback above'
    '''),md('''
    ## Freeze fresh40 configuration and source on Drive

    Solver tolerances are numerical hyperparameters, not learned parameters. Keep fixed for this run.
    No silent forced step when a numerical budget is exceeded. A changed solver/batch/loss/source
    requires a new RUN_TAG; resume restores the same full training recipe.
    '''),code('''
    cfg=json.loads((CODE/'config.json').read_text(encoding='utf-8'))
    cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(DRIVE_RUNS),work=str(WORK),batch_size=BATCH_SIZE,
        workers=WORKERS,amp=ACTUAL_AMP,amp_request=AMP_REQUEST,
        teacher_enabled=VARIANT!='gt_only',relative_enabled=VARIANT=='dual_teacher',init_checkpoint=None)
    cfg['run_name']=f'AnchorFlow_v11_NODE_Fresh40_ES_{ACTUAL_AMP}'+RUN_TAG
    assert cfg['epochs']==40 and cfg['ode_terminal_time']==1. and cfg['ode_method']=='bosh3'
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
    for name in names:
        dest=snapshot/name
        if dest.exists():assert dest.read_bytes()==(CODE/name).read_bytes(),f'Frozen recipe changed: {name}; use a new RUN_TAG'
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in names:
        dest=snapshot/name
        if not dest.exists():shutil.copy2(CODE/name,dest)
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
        if status:raise RuntimeError(f'{action} failed; traceback above; Drive console: {target/logfile.name}')
    '''),md('''
    ## Data gate → paired preview → real GPU smoke/backward

    Dùng metric `D_cm/C_cm` và relative `R_T/C_T` DA3 đã generate; không chạy teacher online.
    Chỉ cache training1600; validation không load teacher. Không cần geometry_fused/DSINE.
    Test lấy `GeoLift_Data/test_1000/kitti_test_1000.tar` nếu có; nếu thiếu dùng KITTI official như flow cũ.
    Extract/cache ở SSD `/content`; kết quả/checkpoint được đồng bộ Drive.
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
    gc.collect();torch.cuda.empty_cache()
    command('smoke')
    print(json.loads((WORK/'smoke_report.json').read_text()))
    '''),md('''
    ## Train / resume

    Fresh epoch0..39 tối đa; early stop min20/patience8/min_delta0.001m.
    Học field end-to-end ngay từ đầu. Inverse-RMSE loss ramp4, relative ramp3, tail ramp2.
    Mỗi epoch backup CSV/JSONL/checkpoints. Ngắt Colab: chạy lại cùng config/VARIANT/RUN_TAG,
    tự resume `last.pth` + optimizer + RNG. Không train lại từ đầu nếu đã có last.pth cùng recipe.
    '''),code('''
    command('train')
    print(json.loads((RUN_DIR/'training_status.json').read_text()))
    print('Log:',RUN_DIR/'train_log.csv','Best:',RUN_DIR/'best.pth')
    '''),md('''
    ## Evaluate same-checkpoint metrics and numerical diagnostics

    `best.pth`: minRMSE; `best_inverse.pth`: miniRMSE; `best_joint.pth`: minmax(RMSE/.9, iRMSE/3.2).
    Không ghép hai metric từ hai epoch để tuyên bố cùng đạt target. D4_step1/2 = t=.5/1,
    KHÔNG phải solver step1/2. Adaptive NFE gồm các trial bị reject.
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
    print('ODE:',best['integration'])
    rows=[]
    for label,file in [('RMSE','best_val_metrics.json'),('inverse','best_inverse_val_metrics.json'),('joint','best_joint_val_metrics.json')]:
        r=json.loads((RUN_DIR/file).read_text());s=r['final']['all']
        rows.append({'selection':label,'epoch':r['epoch'],'RMSE m':s['rmse_m'],'iRMSE km^-1':s['irmse_km_inv'],
                     'both_targets':s['rmse_m']<.9 and s['irmse_km_inv']<3.2})
    table=pd.DataFrame(rows);display(table);table.to_csv(RUN_DIR/'checkpoint_selection_metrics.csv',index=False)
    '''),code('''
    log=pd.read_csv(RUN_DIR/'train_log.csv')
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    axes[0].plot(log.epoch,log.val_rmse_m);axes[0].axhline(.9,color='red',ls='--');axes[0].set_title('RMSE m')
    axes[1].plot(log.epoch,log.val_irmse_km_inv);axes[1].axhline(3.2,color='red',ls='--');axes[1].set_title('iRMSE km^-1')
    axes[2].plot(log.epoch,log.val_executed_nfe,label='mean NFE')
    axes[2].plot(log.epoch,log.val_nfe_p95,label='P95 NFE');axes[2].set_title('Actual solver calls');axes[2].legend()
    for ax in axes:ax.grid(alpha=.3);ax.set_xlabel('Epoch')
    plt.tight_layout();plt.show();fig.savefig(RUN_DIR/'learning_curves.png',dpi=150)
    log[['epoch']+[k for k in log if k.startswith('loss_weighted_')]].to_csv(RUN_DIR/'loss_budget.csv',index=False)
    display(log[['epoch','val_executed_nfe','val_nfe_p95','val_accepted_h_mean','val_rejected_steps','val_terminal_time']].tail(10))
    def record(label,r):
        s=r['final'];a=s['all']
        return {'model':label,'RMSE m':a['rmse_m'],'MAE m':a['mae_m'],'iRMSE km^-1':a['irmse_km_inv'],
            **{f'RMSE {k}':s[k]['rmse_m'] for k in ('0-20','20-40','40-60','60-80','80-120','edge')},
            'GT boundary3':r['gt_boundary']['bands']['3']['rmse_m']}
    rows=[record(label,json.loads((CODE/file).read_text())) for label,file in
        [('V8','v8_val_metrics.json'),('V9','v9_val_metrics.json'),('V9.1','v9_1_val_metrics.json'),('V10','v10_val_metrics.json')]]
    rows.append(record('V11 fresh40',best));comparison=pd.DataFrame(rows);display(comparison)
    comparison.to_csv(RUN_DIR/'historical_comparison.csv',index=False)
    print('Historical budgets/init differ. V11 changes solver AND state chart; not a pure solver ablation.')
    '''),md('''
    ## Same checkpoint: solver audit → test1000 → GPU profile → static export

    Audit đầy đủ400 val: adaptive default, tighter numerical tolerance, fixed midpoint8.
    Giảm solver error không đồng nghĩa giảm GT error; không retune tolerance trên val rồi gọi independent test.
    Test1000 dùng adaptive default + bestRMSE, không có public GT.
    Profile cả adaptive/fixed với cùng checkpoint: actual NFE, wall median/P95 gồm host decisions, VRAM.
    ONNX **chỉ fixed midpoint8**; parity với chính graph PyTorch fixed. Không claim adaptive export.
    Chưa có TensorRT benchmark; chỉ giữ fixed solver deploy nếu accuracy audit chấp nhận được.
    '''),code('''
    if RUN_SOLVER_AUDIT:
        command('solver_audit')
        display(pd.DataFrame(json.loads((RUN_DIR/'solver_comparison.json').read_text()))[
            ['solver','epoch','rmse_m','irmse_km_inv','nfe_mean','nfe_p95','rejected_steps_mean','terminal_time_mean']])
    command('test')
    command('profile')
    for file in ('profile.json','profile_fixed_midpoint8.json'):
        p=json.loads((RUN_DIR/file).read_text())
        print(file,'params:',p['total_parameters'],'MAC:',p['total_conv_linear_macs'],
              'wall median/P95:',p['wall_median_ms'],p['wall_p95_ms'],'real100:',p['real_scene_profile'])
    if RUN_EXPORT:command('export')
    print('DONE:',RUN_DIR,'Predictions:',RUN_DIR/'kitti_test_predictions.zip')
    ''')]
    name='Train_AnchorFlow_V11_NODE_TAR2000_Fresh40.ipynb'
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':name}})
    for cell in cells:
        if cell.cell_type=='code':ast.parse(cell.source)
    nb.validate(book);nb.write(book,OUT/name)
    (OUT/(Path(name).stem+'_contract.json')).write_text(nb.writes(book),encoding='utf-8')
    print('Notebook:',name,len(cells),'cells')

if __name__=='__main__':main()
