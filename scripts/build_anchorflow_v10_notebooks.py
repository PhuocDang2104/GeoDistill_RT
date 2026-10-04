"""Two upload-ready workflows, immutable notebook snapshots, no student weights."""
import ast
import json
import textwrap
from pathlib import Path
import nbformat as nb
from build_anchorflow_v9_notebooks import COMMAND,STAGING

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'drive_upload/AnchorFlow_v10_AdaptiveJet'
def md(s): return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s): return nb.v4.new_code_cell(textwrap.dedent(s).strip())

def write(name,cells):
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':name}})
    nb.validate(book)
    for cell in cells:
        if cell.cell_type=='code': ast.parse(cell.source)
    nb.write(book,OUT/name)
    (OUT/(Path(name).stem+'_contract.json')).write_text(nb.writes(book),encoding='utf-8')

def main():
    previous=nb.read(ROOT/'drive_upload/AnchorFlow_v9_Consensus/01_Generate_Relative_Teacher_DA3MONO_TAR2000.ipynb',as_version=4)
    cells=[md('''
    # 01 — Optional relative teacher cache (same recipe as completed V9)

    **Đã có relative_teacher_2000_DA3MONO_LARGE.tar thì bỏ qua notebook01.**
    V10 dùng lại đúng metric/relative teacher đang có, không cần generate mới.
    Notebook này chỉ dùng khi chưa có TAR: per-image Drive resume, pinned DA3MONO-LARGE,
    RGB-only, không GT/sparse. Chỉ GPU native BF16; T4 train student FP32 ở notebook02.
    ''')]
    for cell in previous.cells[1:]:
        source=cell.source.replace('AnchorFlow_v9_Consensus','AnchorFlow_v10_AdaptiveJet').replace('anchorflow_v9_teacher','anchorflow_v10_teacher')
        if cell.cell_type=='code' and 'torch.cuda.is_available()' in source:
            source+='\nassert torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported(including_emulation=False), "Teacher generation requires native BF16; reuse existing TAR on T4 and run notebook02."'
        cells.append(code(source) if cell.cell_type=='code' else md(source))
    write('01_Optional_Relative_Teacher_Resume.ipynb',cells)
    cells=[md('''
    # AnchorFlow V10 — Fresh student, max40 epochs + early stop

    Upload extracted folder **AnchorFlow_v10_AdaptiveJet** vào MyDrive, chọn GPU, chạy từ trên xuống.
    Default: **fresh student từ epoch0; chỉ RGB encoder ImageNet pretrained**, không load V8/V9/V9.1.
    Data/teacher/test dùng thẳng GeoLift_Data; không upload lại dữ liệu. 1.600train/400val/1.000test.
    Max40 epochs (index0…39), early-stop min20/patience8/min_delta0.001m; chọn best.pth.
    V8/V9/V9.1 historical khác budget/init, không phải causal ablation. Mục tiêu <0.9m chưa được bảo đảm.
    ''') ,code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import hashlib,json,os,shutil,subprocess,sys
    BUNDLE=Path('/content/drive/MyDrive/AnchorFlow_v10_AdaptiveJet')
    DRIVE_DATA=Path('/content/drive/MyDrive/GeoLift_Data')
    DRIVE_RUNS=Path('/content/drive/MyDrive/GeoLift_RT_Runs')
    assert BUNDLE.is_dir(),f'Upload extracted folder to {BUNDLE}'
    '''),code('''
    CODE=Path('/content/anchorflow_v10_code')
    WORK=Path('/content/anchorflow_v10_work')
    POLICY='adaptive' # fixed3=B0; fixed4=A1; learned4=A2; adaptive=A3
    RUN_TAG=''        # Change tag for new seed/recipe; same tag resumes exact run only
    AMP_REQUEST='auto' # Native BF16 -> bf16; T4/V100 -> explicit fp32 before config freeze
    EPOCHS=40
    BATCH_SIZE=4
    WORKERS=2
    VARIANT='dual_teacher'
    RUN_POLICY_AUDIT=True
    RUN_EXPORT=True
    assert POLICY in ('fixed3','fixed4','learned4','adaptive')
    assert 1<=EPOCHS<=40
    assert shutil.disk_usage('/content').free/2**30>25,'Need >25GiB local SSD free; data extracted on /content, not your PC SSD'
    '''),md('''
    ## Verify sources → stage local code → dependencies → contract tests

    Mutable .ipynb UI không bị kiểm checksum; immutable _contract.json + code/config vẫn strict.
    Không cài lại Torch/CUDA runtime. Native BF16 checked with including_emulation=False.
    ''') ,code(textwrap.dedent(STAGING)+textwrap.dedent('''
    subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(CODE/'requirements.txt')],check=True)
    import torch
    assert torch.cuda.is_available(),'Select GPU runtime'
    sys.path.insert(0,str(CODE))
    from run import select_training_precision,native_bf16_supported
    ACTUAL_AMP=select_training_precision(AMP_REQUEST)
    print(torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))
    print('Resolved precision:',ACTUAL_AMP,'native BF16:',native_bf16_supported())
    tested=subprocess.run([sys.executable,'-u','-m','unittest','discover','-s',str(CODE),'-v'],cwd=CODE,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    print(tested.stdout+tested.stderr)
    if tested.returncode: raise RuntimeError('Contract tests failed; read full traceback above')
    ''')),md('''
    ## Freeze fresh40 configuration — isolated per policy/precision/tag

    Train tất cả4 trajectory states; readout chỉ chạy1 lần ở state được chọn.
    Hard stop supervision dùng future-benefit GT trong **loss**, không đưa GT vào model/controller.
    All baseline losses giữ V9.1; chỉ thay dynamics_aux và thêm0.02stop_BCE.
    Không dùng warm-start/StageA; tail ramps2epoch và relative ramps3epoch như baseline.
    ''') ,code(textwrap.dedent('''
    cfg=json.loads((CODE/'config.json').read_text())
    RUN_NAME=f'AnchorFlow_v10_{POLICY}_Fresh{EPOCHS}_ES_{ACTUAL_AMP}'+RUN_TAG
    cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(DRIVE_RUNS),work=str(WORK),run_name=RUN_NAME,
        epochs=EPOCHS,batch_size=BATCH_SIZE,workers=WORKERS,integration_policy=POLICY,
        amp=ACTUAL_AMP,amp_request=AMP_REQUEST,precision_policy='native_bf16_else_explicit_fp32',
        init_checkpoint=None,initialization='fresh_student_imagenet_rgb_only')
    assert cfg['init_checkpoint'] is None
    CONFIG=CODE/'resolved_config.json'
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    RUN_ROOT=DRIVE_RUNS/RUN_NAME
    RUN_DIR=RUN_ROOT/VARIANT
    SUBSET=DRIVE_DATA/'teacher_subset_2000'
    for name in ('selected_2000_ids.json','kitti_trainval_2000.tar',cfg['metric_tar'],cfg['relative_tar']):
        assert (SUBSET/name).is_file(),f'Missing {SUBSET/name}; reuse existing relative TAR from V9'
    # Check the entire snapshot FIRST, then write any missing files.
    snapshot=RUN_ROOT/'source_bundle'
    names=[n for n in manifest['files'] if not n.endswith('.ipynb')]+['resolved_config.json','bundle_manifest.json']
    for name in names:
        dest=snapshot/name
        if dest.exists(): assert dest.read_bytes()==(CODE/name).read_bytes(),f'Frozen recipe differs: {name}; change RUN_TAG'
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in names:
        src,dest=CODE/name,snapshot/name
        if not dest.exists() and src.resolve()!=dest.resolve(): shutil.copy2(src,dest)
    print('Fresh student, ImageNet RGB only. Output:',RUN_DIR)
    print(json.dumps(cfg,indent=2))
    ''')+COMMAND.replace('    if exit_code:',
        '    diagnostic=Path(current["work"])/"gpu_failure_report.json"\n'
        '    if diagnostic.is_file(): shutil.copy2(diagnostic,target/(action+"_gpu_failure_report.json"))\n'
        '    if exit_code:')),md('''
    ## Real data gate + preview + real teacher smoke/backward

    Metric D_cm/C_cm và relative R_T/C_T audit đủ2.000, chỉ cache1.600train; validation không teacher.
    Test đọc GeoLift_Data/test_1000/kitti_test_1000.tar, nếu thiếu sẽ tải official KITTI.
    Giữ notebook parent không có model GPU; smoke/train chạy subprocess để tránh chiếm VRAM kép.
    ''') ,code('''
    command('prepare')
    contract=json.loads((WORK/'data_contract.json').read_text())
    assert (contract['train'],contract['val'],contract['test'])==(1600,400,1000)
    assert not contract['validation_teacher_used']
    assert contract['relative_teacher']['cached_train']==1600 and contract['relative_teacher']['cached_val']==0
    shutil.copy2(WORK/'data_contract.json',RUN_ROOT/'data_contract.json')
    import matplotlib.pyplot as plt
    from data import KITTIDataset
    train=KITTIDataset(cfg,'train',teacher=True)[0]
    val=KITTIDataset(cfg,'val',teacher=False)[0]
    assert not {'teacher','relative','confidence','relative_confidence'}.intersection(val)
    fig,axes=plt.subplots(1,6,figsize=(24,4))
    axes[0].imshow(train['rgb'].permute(1,2,0));axes[0].set_title('RGB')
    for ax,key in zip(axes[1:],('sparse','gt','teacher','relative','relative_confidence')):
        lo,hi=(-2,2) if key=='relative' else ((0,1) if key=='relative_confidence' else (0,80))
        ax.imshow(train[key][0],cmap='turbo_r',vmin=lo,vmax=hi);ax.set_title(key)
    for ax in axes: ax.axis('off')
    plt.tight_layout();plt.show();fig.savefig(RUN_ROOT/'data_preview.png',dpi=120)
    del train,val
    import gc
    gc.collect();torch.cuda.empty_cache()
    command('smoke')
    smoke=json.loads((WORK/'smoke_report.json').read_text())
    assert not smoke['student_checkpoint_loaded']
    print(smoke)
    '''),md('''
    ## Train / resume

    Chạy lại từ đầu notebook sau mất Colab: source/config/data giống nhau thì tự resume last.pth,
    khôi phục optimizer/scheduler(global_step)/RNG/early-stop. Có thay config phải đổi RUN_TAG.
    Log/checkpoints đồng bộ Drive sau mỗi epoch; mất runtime giữa epoch sẽ chạy lại epoch đó.
    Nonfinite sẽ dừng trước optimizer/checkpoint, không NaN masking hoặc skip batch.
    ''') ,code('''
    command('train')
    print(json.loads((RUN_DIR/'training_status.json').read_text()))
    print('Logs:',RUN_DIR/'train_log.csv','Checkpoints:',RUN_DIR/'best.pth')
    '''),md('''
    ## Evaluate best: global/ranges/boundary/tail + dynamics trajectory/controller

    Validation luôn rollout4 để audit; selected_nfe là policy statistic, executed_nfe=4,
    **không phải latency saving**. Chỉ profile adaptive_batch1 mới đo thực tế dừng sớm.
    ''') ,code('''
    command('evaluate')
    import pandas as pd
    best=json.loads((RUN_DIR/'val_metrics.json').read_text())
    assert best['final']['all']['pixels']==25424992,'GT support/protocol changed'
    display(pd.DataFrame(best['final']).T)
    display(pd.DataFrame(best['gt_boundary']['bands']).T)
    display(pd.DataFrame(best['final']['all']['error_tail']).T)
    display(pd.DataFrame(best['integration']['quarter_trajectory']).T)
    print('Controller:',{k:v for k,v in best['integration'].items() if k!='quarter_trajectory'})
    print('Best epoch:',best['checkpoint_epoch'],'RMSE<0.9:',best['final']['all']['rmse_m']<.9)
    '''),code('''
    log=pd.read_csv(RUN_DIR/'train_log.csv')
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    axes[0].plot(log.epoch,log.val_rmse_m);axes[0].axhline(.9,ls='--',color='red');axes[0].set_title('Validation RMSE m')
    for k in (1,2,3,4): axes[1].plot(log.epoch,log[f'val_native_D4_step{k}_rmse_m'],label=f'j{k}')
    axes[2].plot(log.epoch,log.val_selected_nfe,label='Selected NFE')
    axes[2].plot(log.epoch,log.val_terminal_time,label='Terminal tau')
    for ax in axes:ax.grid(alpha=.3);ax.set_xlabel('Epoch')
    for ax in axes[1:]:ax.legend()
    plt.tight_layout();plt.show();fig.savefig(RUN_DIR/'v10_learning_curves.png',dpi=150)
    log[['epoch']+[k for k in log if k.startswith('loss_weighted_')]].to_csv(RUN_DIR/'loss_budget.csv',index=False)
    display(log.tail())
    '''),md('''
    ## Optional readout-exit audit, test1000, real-scene runtime, two ONNX graphs

    policy_audit compare adaptive/fixed exit2/3/4 **cùng checkpoint**, không phải4run training ablation.
    ONNX masked_adaptive và static4 đều compute4, không export conditional saving.
    Python adaptive_batch1 runtime tính cả host sync; nếu sync đắt hơn saved call, không claim nhanh hơn.
    Anonymous KITTI test không có GT công khai; ZIP predictions chỉ để nhìn/submit.
    ''') ,code('''
    if RUN_POLICY_AUDIT: command('policy_audit')
    command('test')
    command('profile')
    profile=json.loads((RUN_DIR/'profile.json').read_text())
    display(pd.DataFrame(profile['policy_profiles']).T)
    print('Parameters:',profile['total_parameters'],'ConvLinear MAC:',profile['total_conv_linear_macs'])
    if RUN_EXPORT: command('export')
    '''),md('''
    ## Historical V8/V9/V9.1 comparison — not equal-budget causality

    V9.1 warm-start V9 best23, thêm8epoch; V10 fresh tối đa40. Precision/backend cũng phải matched
    khi so runtime. Để isolated fresh40 baseline, đổi POLICY='fixed3' và chạy vào run folder khác.
    Không tự thay stop threshold để tối ưu val; default0.5 frozen. Calibrate sau phải báo riêng.
    '''),code('''
    def record(label,report):
        score=report['final'];all_=score['all']
        return {'model':label,'RMSE m':all_['rmse_m'],'MAE m':all_['mae_m'],'iRMSE km^-1':all_['irmse_km_inv'],
            **{f'RMSE {k}':score[k]['rmse_m'] for k in ('0-20','20-40','40-60','60-80','80-120','edge')},
            'GT-boundary3':report['gt_boundary']['bands']['3']['rmse_m'],
            'tail>5 SSE':all_['error_tail']['5']['sse_m2']}
    rows=[record(label,json.loads((CODE/file).read_text())) for label,file in (
        ('V8 historical','v8_val_metrics.json'),('V9 historical','v9_val_metrics.json'),('V9.1 historical fine-tune','v9_1_val_metrics.json'))]
    rows.append(record('V10 fresh '+POLICY,best))
    comparison=pd.DataFrame(rows);display(comparison);comparison.to_csv(RUN_DIR/'historical_comparison.csv',index=False)
    print('Historical comparison is NOT an equal-budget architecture ablation.')
    print('DONE:',RUN_DIR,'test predictions:',RUN_DIR/'kitti_test_predictions.zip')
    ''')]
    write('02_Train_V10_Fresh40_EarlyStop.ipynb',cells)
    print('Built two notebooks and immutable contracts')

if __name__=='__main__':main()
