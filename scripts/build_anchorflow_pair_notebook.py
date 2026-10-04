"""Build one self-contained Colab notebook; only upload this .ipynb + two bundles."""
import ast
import hashlib
from pathlib import Path
import textwrap
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb'
def md(s):return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s):return nb.v4.new_code_cell(textwrap.dedent(s).strip())

def main():
    source=(ROOT/'scripts/anchorflow_pair_benchmark.py').read_text(encoding='utf-8')
    embedded=f"RUNTIME_SOURCE = {source!r}\nRUNTIME_SHA256 = {hashlib.sha256(source.encode()).hexdigest()!r}\n"
    cells=[md('''
    # V10.1 vs V11 — parallel fresh40 train → isolated benchmark

    **Một notebook điều phối cả hai**, không sửa model bundle. Upload notebook này và hai folder:
    `MyDrive/AnchorFlow_v10_1_LiteMetric`, `MyDrive/AnchorFlow_v11_NODE`.
    Data giữ ở `MyDrive/GeoLift_Data/teacher_subset_2000/`; hai cached teachers như hiện tại.
    1.600 train / 400 val / 1.000 anonymous test, fresh student, max 40 + early stop.
    Early stop: ít nhất 20 epoch, patience 8, minimum improvement 0.001 m.

    Mặc định khởi chạy **hai tiến trình train cùng lúc**. Một GPU thì chia VRAM/compute,
    không bảo đảm nhanh hơn chạy lần lượt. Hai GPU thì mỗi model một GPU.
    **Final evaluate và profile luôn chạy tuần tự trên cùng GPU**, chỉ sau khi cả hai train kết thúc.
    Training epoch time dưới contention không được dùng để claim isolated model speed.
    '''),code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import gc,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,time
    DRIVE_DATA=Path('/content/drive/MyDrive/GeoLift_Data')
    DRIVE_RUNS=Path('/content/drive/MyDrive/GeoLift_RT_Runs')
    BUNDLES={
        'V10_1':Path('/content/drive/MyDrive/AnchorFlow_v10_1_LiteMetric'),
        'V11':Path('/content/drive/MyDrive/AnchorFlow_v11_NODE'),
    }
    LOCAL_ROOT=Path('/content/anchorflow_pair_benchmark')
    WORK=LOCAL_ROOT/'shared_data' # Final per-experiment subfolder resolved with AMP/BENCH_TAG below.
    BENCH_TAG='pair01' # New tag = new experiment; unchanged tag resumes both independently.
    PARALLEL_TRAIN=True
    TRAIN_GPU_INDICES=None # None: [0,1] if >=2 GPUs, otherwise [0,0]. Can set [0,0].
    PROFILE_GPU_INDEX=0 # BOTH final evaluate/profile on this SAME GPU.
    AMP_REQUEST='auto' # Common native BF16 or explicit FP32; never FP16.
    BATCH_SIZE=4
    ACCUMULATION=1 # On smaller GPUs, choose BATCH_SIZE=2, ACCUMULATION=2 BEFORE starting a new tag.
    WORKERS=2 # Per model: parallel creates up to 2*WORKERS loader workers.
    CPU_THREADS_PER_MODEL=4
    SHARED_GPU_MIN_FREE_GIB=20. # Conservative preflight heuristic, NOT an OOM guarantee.
    RUN_EXTRA_SELECTIONS=True # Isolated evaluate best_inverse and best_joint too.
    RUN_SOLVER_AUDIT=True # Same V11 checkpoint: adaptive/default, tighter, static midpoint8.
    RUN_TEST=True # Both anonymous1000 prediction ZIPs, no public GT metric.
    RUN_EXPORT=False # Optional slow CPU ONNX checks; V11 export is STATIC midpoint8 only.
    assert re.fullmatch(r'[A-Za-z0-9_-]+',BENCH_TAG),'BENCH_TAG: only letters, digits, underscore, hyphen'
    assert AMP_REQUEST in ('auto','bf16','fp32')
    assert BATCH_SIZE>=1 and ACCUMULATION>=1 and WORKERS>=0
    for folder in BUNDLES.values():assert folder.is_dir(),f'Upload folder: {folder}'
    for name in ('selected_2000_ids.json','kitti_trainval_2000.tar','metric_coarse_train_2000.tar',
                 'relative_teacher_2000_DA3MONO_LARGE.tar'):
        assert (DRIVE_DATA/'teacher_subset_2000'/name).is_file(),f'Missing Drive data: {name}'
    test_archive=DRIVE_DATA/'test_1000'/'kitti_test_1000.tar'
    if not test_archive.is_file():print('Official test TAR absent: prepare will download KITTI anonymous1000, not regenerate teachers.')
    assert shutil.disk_usage('/content').free/2**30>25,'Need >25GiB Colab local SSD, not PC SSD'
    LOCAL_ROOT.mkdir(parents=True,exist_ok=True)
    '''),md('''
    ## Helper tự chứa trong notebook

    Cell dưới chứa nguyên script điều phối/report; **không cần upload một folder helper thứ ba**.
    Hai model chạy trong hai Python interpreter riêng để tránh trùng module `model` / `run`.
    Không tự bật NVIDIA MPS, không đổi CUDA/Torch hoặc sửa source model.
    '''),nb.v4.new_code_cell(embedded+'''\nrunner_file=LOCAL_ROOT/'anchorflow_pair_benchmark.py'
assert hashlib.sha256(RUNTIME_SOURCE.encode('utf-8')).hexdigest()==RUNTIME_SHA256
runner_file.write_text(RUNTIME_SOURCE,encoding='utf-8')
spec=importlib.util.spec_from_file_location('anchorflow_pair_benchmark',runner_file)
bench=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bench;spec.loader.exec_module(bench)
print('Coordinator ready:',runner_file)
''',metadata={'collapsed':True}),md('''
    ## Verify both bundles → install common requirements → CPU contracts

    Checksum bỏ qua notebook UI `.ipynb` (Colab thay metadata/output), nhưng vẫn strict với mọi
    code/config/notebook snapshot. Không bypass test failure. Cài cùng environment, không reinstall Torch.
    '''),code('''
    manifests={};CODES={}
    for label,folder in BUNDLES.items():
        CODES[label]=LOCAL_ROOT/('code_'+label)
        manifests[label]=bench.stage_bundle(folder,CODES[label])
    for name in bench.SHARED_SOURCES:
        assert (CODES['V10_1']/name).read_bytes()==(CODES['V11']/name).read_bytes(),f'Shared primitive differs: {name}'
    args=[sys.executable,'-m','pip','install','-q']
    for folder in CODES.values():args.extend(['-r',str(folder/'requirements.txt')])
    subprocess.run(args,check=True)
    import torch,torchdiffeq
    assert torch.cuda.is_available(),'Select GPU runtime in Colab'
    assert torchdiffeq.__version__=='0.2.5'
    for label,folder in CODES.items():
        tested=subprocess.run([sys.executable,'-u','-m','unittest','discover','-s',str(folder),'-v'],cwd=folder,
            capture_output=True,text=True,encoding='utf-8',errors='replace',
            env=bench.process_environment(bench.visible_gpu_token(PROFILE_GPU_INDEX),folder,CPU_THREADS_PER_MODEL))
        print(label,tested.stdout+tested.stderr)
        assert tested.returncode==0,f'{label} tests failed; traceback above'
    '''),md('''
    ## Match recipe and freeze source/config

    Same seed42, max40, pretrained RGB policy, batch/accumulation, optimizer/loss/teachers/early stop.
    V10.1 giữ learned h + fixed2 Euler; V11 giữ adaptive RK3(2) + smooth chart + fixedT1.
    Đây là architecture bundle comparison, **không phải pure solver-only ablation** hoặc identical-all-weights init.
    Không đổi batch/loss/precision/source giữa chừng để resume. PARALLEL_TRAIN có thể đổi sang False
    để tránh contention khi resume; scheduling history được log riêng.
    '''),code('''
    n_gpu=torch.cuda.device_count()
    train_indices=TRAIN_GPU_INDICES if TRAIN_GPU_INDICES is not None else ([0,1] if n_gpu>=2 else [0,0])
    assert len(train_indices)==2 and all(isinstance(i,int) and 0<=i<n_gpu for i in train_indices)
    assert 0<=PROFILE_GPU_INDEX<n_gpu
    involved=sorted(set(train_indices+[PROFILE_GPU_INDEX]))
    gpu_info=[]
    for i in involved:
        with torch.cuda.device(i):
            free,total=torch.cuda.mem_get_info()
            native=torch.cuda.get_device_capability(i)[0]>=8 and torch.cuda.is_bf16_supported(including_emulation=False)
        gpu_info.append({'logical_index':i,'token':bench.visible_gpu_token(i),'name':torch.cuda.get_device_name(i),
                         'native_bf16':native,'free_GiB':free/2**30,'total_GiB':total/2**30})
    common_native=all(r['native_bf16'] for r in gpu_info)
    if AMP_REQUEST=='bf16':assert common_native,'Select fp32; every training/profile GPU must support native BF16'
    ACTUAL_AMP=('bf16' if common_native else 'fp32') if AMP_REQUEST=='auto' else AMP_REQUEST
    shared_gpu=PARALLEL_TRAIN and train_indices[0]==train_indices[1]
    if shared_gpu:
        info=next(r for r in gpu_info if r['logical_index']==train_indices[0])
        assert info['free_GiB']>=SHARED_GPU_MIN_FREE_GIB,(
            f"Shared GPU free={info['free_GiB']:.1f}GiB < conservative guard={SHARED_GPU_MIN_FREE_GIB}. "
            'Use PARALLEL_TRAIN=False; or before starting a NEW tag choose batch2/accum2 and a justified lower guard. '
            'No automatic batch/precision fallback.')
        print('WARNING: both train jobs share one GPU. More VRAM and possible time-slicing; NOT guaranteed faster.')
    PROFILE_GPU=bench.visible_gpu_token(PROFILE_GPU_INDEX)
    COMPARE_ROOT=DRIVE_RUNS/f'Compare_V10_1_V11_Fresh40_{ACTUAL_AMP}_{BENCH_TAG}'
    # Isolate local checkpoints/logs too: runner stores them under WORK/runs/<model>.
    # A NEW tag must never append epoch0 to a prior experiment's local train CSV.
    WORK=LOCAL_ROOT/'shared_data'/f'{ACTUAL_AMP}_{BENCH_TAG}'
    configs={};JOBS=[]
    for label,index in zip(bench.LABELS,train_indices):
        cfg=bench.read_json(CODES[label]/'config.json')
        cfg.update(drive_data=str(DRIVE_DATA),drive_runs=str(COMPARE_ROOT),work=str(WORK),run_name=label,
            epochs=40,seed=42,batch_size=BATCH_SIZE,accumulation=ACCUMULATION,workers=WORKERS,
            amp=ACTUAL_AMP,amp_request=AMP_REQUEST,init_checkpoint=None,encoder_pretrained=True,
            teacher_enabled=True,relative_enabled=True,checkpoint_selection='rmse',compile=False,
            profile_warmup=30,profile_runs=100)
        configs[label]=cfg
    recipe=bench.matched_recipe(configs)
    for label,index in zip(bench.LABELS,train_indices):
        cfg=configs[label]
        bench.freeze_snapshot(COMPARE_ROOT/'source_bundle'/label,CODES[label],cfg)
        JOBS.append({'label':label,'code':str(CODES[label]),'variant':'dual_teacher','config':cfg,
                     'run_dir':str(COMPARE_ROOT/label/'dual_teacher'),'gpu':bench.visible_gpu_token(index),
                     'cpu_threads':CPU_THREADS_PER_MODEL})
    frozen_runner=COMPARE_ROOT/'source_bundle'/'anchorflow_pair_benchmark.py'
    if frozen_runner.exists():assert frozen_runner.read_bytes()==runner_file.read_bytes(),'Coordinator changed; choose a new BENCH_TAG'
    else:shutil.copy2(runner_file,frozen_runner)
    execution={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'parallel_train':PARALLEL_TRAIN,
               'train_gpu_indices':train_indices,'profile_gpu_index':PROFILE_GPU_INDEX,'gpu_info':gpu_info,
               'cpu_threads_per_model':CPU_THREADS_PER_MODEL,'shared_gpu':shared_gpu,
               'torch':str(torch.__version__),'coordinator_sha256':RUNTIME_SHA256,'recipe':recipe}
    bench.atomic_json(COMPARE_ROOT/f'execution_{time.time_ns()}.json',execution)
    bench.atomic_json(COMPARE_ROOT/'benchmark_recipe.json',recipe)
    print(json.dumps(execution,indent=2));print('COMPARISON OUTPUT:',COMPARE_ROOT)
    '''),md('''
    ## Prepare shared data ONCE, smoke both SEQUENTIALLY

    Loader/cache sources của hai bundle được kiểm tra byte-identical. Prepare/audit dùng V10.1 một lần;
    cả hai sau đó chỉ đọc chung RGB/sparse/GT/K và train-only teacher caches.
    Không run prepare/smoke đồng thời. Không generate lại relative teacher.
    '''),code('''
    bench.run_stage([JOBS[0]],'prepare',gpu=PROFILE_GPU)
    contract=bench.read_json(WORK/'data_contract.json')
    for cfg in configs.values():bench.verify_data_contract(contract,cfg)
    bench.atomic_json(COMPARE_ROOT/'data_contract.json',contract)
    for job in JOBS:
        since=time.time()
        try:
            bench.run_stage([job],'smoke',gpu=PROFILE_GPU)
        finally:
            diagnostic=WORK/'gpu_failure_report.json'
            if diagnostic.exists() and diagnostic.stat().st_mtime>=since:
                shutil.copy2(diagnostic,Path(job['run_dir'])/'smoke_gpu_failure_report.json')
        report=bench.read_json(WORK/'smoke_report.json')
        bench.atomic_json(Path(job['run_dir'])/'smoke_report.json',report)
        print(job['label'],'SMOKE PASS:',report)
    gc.collect();torch.cuda.empty_cache()
    '''),md('''
    ## RUN BOTH TRAIN JOBS — một cell

    Console có prefix [V10_1/train] / [V11/train], tự cập nhật Drive console mỗi30giây.
    Checkpoint/CSV/JSONL sync mỗi epoch vào folder model riêng. Một model lỗi: model kia vẫn chạy,
    cell báo thất bại sau khi peer kết thúc; không bỏ NaN, không tự đổi precision/batch.
    Stop cell sẽ dừng hai nhóm tiến trình do cell tạo; checkpoint hoàn chỉnh gần nhất được giữ.
    Colab ngắt: chạy lại notebook cùng BENCH_TAG/config, hai model tự resume độc lập.
    Không mở hai notebook cùng train vào cùng COMPARE_ROOT. Nếu OOM: chờ peer kết thúc,
    chọn PARALLEL_TRAIN=False và chạy lại; giữ nguyên batch/precision/config của experiment.
    '''),code('''
    bench.run_stage(JOBS,'train',parallel=PARALLEL_TRAIN,status_path=COMPARE_ROOT/'train_pair_status.json')
    for job in JOBS:
        status=bench.read_json(Path(job['run_dir'])/'training_status.json')
        print(job['label'],status)
    print('BOTH TRAIN PROCESSES EXITED. Final evaluation/profile can now run without peer contention.')
    bench.atomic_json(COMPARE_ROOT/'gpu_after_training.json',bench.gpu_snapshot())
    '''),md('''
    ## Final evaluate / test / profile — KHÔNG SONG SONG

    Cả hai evaluate/profile trên cùng PROFILE_GPU, cùng precision/B1/image-size/real100 scenes.
    BestRMSE, bestInverse và bestJoint luôn giữ metric của đúng checkpoint đó.
    V11 static-midpoint8 được đo như một solver approximation khác, không trộn với adaptive accuracy.
    Anonymous1000 test không có public GT; không gọi validation metric là leaderboard result.
    '''),code('''
    bench.run_stage(JOBS,'evaluate',parallel=False,gpu=PROFILE_GPU,
                    status_path=COMPARE_ROOT/'evaluate_pair_status.json')
    if RUN_EXTRA_SELECTIONS:
        for selection in ('inverse','joint'):
            bench.run_stage(JOBS,'evaluate',parallel=False,gpu=PROFILE_GPU,
                            extra=('--checkpoint-selection',selection),suffix='_'+selection)
    if RUN_SOLVER_AUDIT:bench.run_stage([JOBS[1]],'solver_audit',gpu=PROFILE_GPU)
    if RUN_TEST:bench.run_stage(JOBS,'test',parallel=False,gpu=PROFILE_GPU,
                              status_path=COMPARE_ROOT/'test_pair_status.json')
    bench.atomic_json(COMPARE_ROOT/'gpu_before_isolated_profiles.json',bench.gpu_snapshot())
    bench.run_stage(JOBS,'profile',parallel=False,gpu=PROFILE_GPU,
                    status_path=COMPARE_ROOT/'profile_pair_status.json')
    if RUN_EXPORT:bench.run_stage(JOBS,'export',parallel=False,gpu=PROFILE_GPU)
    '''),md('''
    ## Compare accuracy, real latency, budget, same-checkpoint solver trade-off

    Inference runtime dùng real100 **wall median/P95**, không dùng training epoch_seconds đang tranh GPU.
    Early stop có thể khác số epoch: báo actual budget và best trong common logged epoch prefix riêng.
    Common-prefix là epoch log, không giả vờ checkpoint của epoch đó còn tồn tại.
    '''),code('''
    summary=bench.compare_reports(JOBS,COMPARE_ROOT)
    import pandas as pd
    from IPython.display import display,Image,Markdown
    for file in ('comparison_accuracy.csv','comparison_checkpoint_selections.csv','comparison_efficiency.csv',
                 'comparison_training_budget.csv','comparison_common_epoch_budget.csv','comparison_stages.csv',
                 'comparison_tails.csv','comparison_GT_boundaries.csv','comparison_sensor_policies.csv'):
        print(file);display(pd.read_csv(COMPARE_ROOT/file))
    solver=COMPARE_ROOT/'comparison_v11_solvers.csv'
    if solver.exists():display(pd.read_csv(solver))
    print('DELTA:',json.dumps(summary['deltas'],indent=2))
    display(Image(filename=str(COMPARE_ROOT/'comparison_dashboard.png')))
    display(Markdown((COMPARE_ROOT/'COMPARISON.md').read_text(encoding='utf-8')))
    print('REPORTS:',COMPARE_ROOT)
    for job in JOBS:print(job['label'],'logs/checkpoints/test:',job['run_dir'])
    ''')]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':OUT.name}})
    for cell in cells:
        if cell.cell_type=='code':ast.parse(cell.source)
    nb.validate(book);nb.write(book,OUT)
    print('Notebook:',OUT,len(cells),'cells',OUT.stat().st_size,'bytes')

if __name__=='__main__':main()
