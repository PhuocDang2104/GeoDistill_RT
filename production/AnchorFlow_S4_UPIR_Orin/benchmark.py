"""Jetson GPU FP32 TensorRT10, native CUDA graph, real changing inputs.

Build on target. No training, no dependency on the repository or PyTorch.
"""
import argparse
import ctypes as C
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from cuda_runtime import CUDA

INPUTS = {'rgb': (1, 3, 352, 1216), 'sparse': (1, 1, 352, 1216),
          'mask': (1, 1, 352, 1216), 'K': (1, 3, 3)}
OUTPUT = (1, 1, 352, 1216)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parity(actual, expected):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {'finite': False, 'mean_abs_m': None, 'max_abs_m': None, 'passed': False}
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    mean, maximum = float(delta.mean()), float(delta.max())
    return {'finite': True, 'mean_abs_m': mean, 'max_abs_m': maximum,
            'passed': mean <= .001 and maximum <= .06}


def stats(values):
    x = np.asarray(values, np.float64)
    return {'median_ms': float(np.median(x)), 'p95_ms': float(np.percentile(x, 95)),
            'mean_ms': float(x.mean()), 'fps_from_median': float(1000/np.median(x))}


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)


def validate_sample(sample):
    for key, shape in {**INPUTS, 'reference': OUTPUT}.items():
        value = sample[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'Invalid shape/dtype/nonfinite: {key}')
    if sample['rgb'].min() < 0 or sample['rgb'].max() > 1:
        raise ValueError('RGB must be [0,1], not ImageNet normalized or uint8')
    if not np.isin(sample['mask'], [0, 1]).all() or (sample['sparse'] < 0).any():
        raise ValueError('Sparse depth must be nonnegative meters; mask binary')


def build(trt, logger, onnx, engine_path, workspace):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace*2**30))
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError('Native TensorRT build failed; no fallback or precision change')
    engine_path.write_bytes(bytes(plan))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='NEW directory per run; engine built here')
    p.add_argument('--workspace-gib', type=float, default=1.)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--timed-images', type=int, default=100)
    p.add_argument('--power-note', required=True, help='Record board/RAM/power mode/cooling/clocks')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new output directory; existing evidence is never overwritten')
    if a.iterations < 1 or a.warmup < 1 or a.timed_images < 2 or a.workspace_gib <= 0:
        raise ValueError('Invalid iteration/image/workspace count')
    manifest = json.loads((a.assets/'io_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (a.assets/name).resolve()
        if a.assets.resolve() not in path.parents or digest(path) != expected:
            raise RuntimeError(f'Asset path/hash changed: {name}')
    paths = sorted((a.assets/'samples').glob('*.npz'))
    if len(paths) != manifest['frames'] or len(paths) < a.timed_images:
        raise ValueError('Missing samples; reduce --timed-images explicitly for a separate protocol')
    if any(path.relative_to(a.assets).as_posix() not in manifest['files'] for path in paths):
        raise ValueError('Unmanifested sample')
    import tensorrt as trt
    if int(trt.__version__.split('.')[0]) != 10:
        raise RuntimeError('Requires TensorRT10 (JetPack6.x); no automatic pip replacement')
    a.output.mkdir(parents=True)
    report = {'qualified': False, 'variant': manifest['variant'],
        'checkpoint_sha256': manifest['checkpoint_sha256'], 'onnx_sha256': manifest['files']['model_fp32_op17.onnx'],
        'asset_manifest_sha256': digest(a.assets/'io_manifest.json'),
        'precision': 'strongly typed FP32, TF32 OFF; GPU, not DLA',
        'tensorrt': trt.__version__, 'platform': platform.platform(),
        'board': command_output(['sh', '-c', 'tr -d "\\000" </proc/device-tree/model']),
        'l4t': command_output(['head', '-1', '/etc/nv_tegra_release']),
        'nvpmodel': command_output(['nvpmodel', '-q']), 'power_note': a.power_note,
        'workspace_gib': a.workspace_gib,
        'scope': 'GPU-resident changing inputs; D2D staging included, excludes file IO/decode/preprocess/H2D/D2H/metrics/build/capture. Not end-to-end.',
        'reference': 'PyTorch FP32 from SAME checkpoint; this new opset17 graph needs target qualification',
        'parity_limits_m': {'mean': .001, 'max': .06}, 'parity': []}
    cuda, context, engine, runtime = None, None, None, None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        engine_path = a.output/'orin_fp32.plan'
        build(trt, logger, a.assets/'model_fp32_op17.onnx', engine_path, a.workspace_gib)
        report['engine_sha256'] = digest(engine_path)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError('Deserialize failed')
        names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        if names != set(INPUTS) | {'D_full'}:
            raise ValueError(f'Engine IO differs: {names}')
        for name, shape in {**INPUTS, 'D_full': OUTPUT}.items():
            mode = trt.TensorIOMode.INPUT if name in INPUTS else trt.TensorIOMode.OUTPUT
            if (tuple(engine.get_tensor_shape(name)) != shape or engine.get_tensor_dtype(name) != trt.float32
                    or engine.get_tensor_mode(name) != mode):
                raise ValueError(f'Unexpected engine IO: {name}')
        cuda = CUDA()
        version = C.c_int()
        cuda.call('cudaRuntimeGetVersion', C.byref(version))
        report['cuda_runtime_version'] = version.value
        stream = cuda.handle('cudaStreamCreateWithFlags', 'cudaStreamDestroy', 1)
        buffers = {k: cuda.malloc(int(np.prod(shape))*4) for k, shape in {**INPUTS, 'D_full': OUTPUT}.items()}
        context = engine.create_execution_context()
        for key, pointer in buffers.items():
            if not context.set_tensor_address(key, pointer.value):
                raise RuntimeError(f'Tensor address rejected: {key}')
        resident = []
        for path in paths[:a.timed_images]:
            with np.load(path, allow_pickle=False) as sample:
                validate_sample(sample)
                resident.append({k: cuda.upload(np.ascontiguousarray(sample[k])) for k in INPUTS})
        def stage(frame):
            for key, shape in INPUTS.items():
                cuda.call('cudaMemcpyAsync', buffers[key], frame[key], int(np.prod(shape))*4, 3, stream)
        def enqueue():
            if not context.execute_async_v3(stream_handle=stream.value):
                raise RuntimeError('TensorRT enqueue failed')
        def download():
            result = np.empty(OUTPUT, np.float32)
            cuda.call('cudaStreamSynchronize', stream)
            cuda.call('cudaMemcpy', result.ctypes.data, buffers['D_full'], result.nbytes, 2)
            return result
        stage(resident[0])
        for _ in range(3):
            enqueue()
        cuda.call('cudaStreamSynchronize', stream)
        # Capture only enqueue: no CPU allocation/copy; addresses remain stable.
        cuda.call('cudaStreamBeginCapture', stream, 0)
        enqueue()
        graph = C.c_void_p()
        cuda.call('cudaStreamEndCapture', stream, C.byref(graph))
        cuda.resources.append(('cudaGraphDestroy', graph))
        executable = cuda.handle('cudaGraphInstantiateWithFlags', 'cudaGraphExecDestroy', graph, 0)
        def replay():
            cuda.call('cudaGraphLaunch', executable, stream)
        aba = []
        for idx in (0, 1, 0):
            stage(resident[idx]);enqueue();eager = download()
            stage(resident[idx]);replay();actual = download()
            with np.load(paths[idx], allow_pickle=False) as sample:
                reference = parity(actual, sample['reference'])
            graph_gate = parity(actual, eager)
            report['parity'].append({'frame': idx, 'vs_reference': reference, 'vs_trt_eager': graph_gate})
            if not reference['passed'] or not graph_gate['passed']:
                raise RuntimeError('A/B/A parity failed; no timing qualification or tolerance relaxation')
            aba.append(actual)
        if not np.array_equal(aba[0], aba[2]) or np.array_equal(aba[0], aba[1]):
            raise RuntimeError('Replay freshness/repeatability failed')
        sse, invse, ae, count, reference_sse, reference_invse = 0., 0., 0., 0, 0., 0.
        for i, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as sample:
                validate_sample(sample)
                # Non-resident validation frames: H2D OUTSIDE graph AND timing.
                if i < len(resident):
                    stage(resident[i])
                else:
                    for key in INPUTS:
                        arr = np.ascontiguousarray(sample[key])
                        cuda.call('cudaMemcpy', buffers[key], arr.ctypes.data, arr.nbytes, 1)
                replay();pred = download()
                gate = parity(pred, sample['reference'])
                report['parity'].append({'frame': i, 'vs_reference': gate})
                if not gate['passed']:
                    raise RuntimeError(f'Pixel parity failed on frame {i}')
                valid = sample['gt_mask'] > 0
                gt, prediction, ref = (x[valid].astype(np.float64) for x in (sample['gt'], pred, sample['reference']))
                if not np.isfinite(gt).all() or (gt <= 0).any():
                    raise ValueError('Invalid GT support')
                sse += float(np.square(prediction-gt).sum());ae += float(np.abs(prediction-gt).sum())
                invse += float(np.square(1000/np.maximum(prediction, 1e-6)-1000/np.maximum(gt, 1e-6)).sum())
                reference_sse += float(np.square(ref-gt).sum())
                reference_invse += float(np.square(1000/np.maximum(ref, 1e-6)-1000/np.maximum(gt, 1e-6)).sum())
                count += gt.size
        if count == 0 or (len(paths) == 400 and count != 25424992):
            raise RuntimeError('GT support changed/empty')
        report['accuracy'] = {'frames': len(paths), 'pixels': count, 'rmse_m': (sse/count)**.5,
            'irmse_km_inv': (invse/count)**.5, 'mae_m': ae/count,
            'full400': len(paths) == 400,
            'note': '100-frame metrics are NOT historical full400 metrics' if len(paths) != 400 else 'Original full400 support'}
        report['metric_drift_vs_fp32'] = {'rmse_m': abs((sse/count)**.5-(reference_sse/count)**.5),
            'irmse_km_inv': abs((invse/count)**.5-(reference_invse/count)**.5)}
        if report['metric_drift_vs_fp32']['rmse_m'] > .002 or report['metric_drift_vs_fp32']['irmse_km_inv'] > .02:
            raise RuntimeError('Global accuracy drift failed')
        start = cuda.handle('cudaEventCreate', 'cudaEventDestroy')
        end = cuda.handle('cudaEventCreate', 'cudaEventDestroy')
        report['timing'] = {}
        for label, runner in (('trt_eager', enqueue), ('trt_cuda_graph', replay)):
            for i in range(a.warmup):
                stage(resident[i % len(resident)]);runner()
            cuda.call('cudaStreamSynchronize', stream)
            wall, gpu, host = [], [], []
            for i in range(a.iterations):
                t0 = time.perf_counter()
                cuda.call('cudaEventRecord', start, stream)
                stage(resident[i % len(resident)]);runner()
                cuda.call('cudaEventRecord', end, stream)
                t1 = time.perf_counter()
                cuda.call('cudaStreamSynchronize', stream)
                t2 = time.perf_counter()
                elapsed = C.c_float()
                cuda.call('cudaEventElapsedTime', C.byref(elapsed), start, end)
                wall.append((t2-t0)*1000);gpu.append(elapsed.value);host.append((t1-t0)*1000)
            report['timing'][label] = {'synchronized_wall': stats(wall), 'gpu_stream': stats(gpu),
                'host_enqueue': stats(host), 'warmup': a.warmup, 'iterations': a.iterations,
                'distinct_images': len(resident)}
        report['cuda_memory_after'] = cuda.memory()
        report['qualified'] = True
        print(json.dumps(report['accuracy'], indent=2))
        print(json.dumps(report['timing'], indent=2))
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        if cuda is not None:
            # CUDA context/graph/resource lifetime is process-scoped. Synchronize
            # before release; process exit also frees memory on failure.
            try:
                if 'stream' in locals():
                    cuda.call('cudaStreamSynchronize', stream)
                context = engine = runtime = None
                gc.collect()
                cuda.close()
            except Exception as cleanup_error:
                # A failed capture can leave a CUDA stream invalid. Preserve the
                # original failure report; OS process teardown releases resources.
                report['cleanup_error'] = str(cleanup_error)
                report['qualified'] = False
        (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        if report.get('cleanup_error') and not report.get('error'):
            raise RuntimeError('CUDA cleanup failed; report unqualified')


if __name__ == '__main__':
    main()
