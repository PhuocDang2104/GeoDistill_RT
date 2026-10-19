"""Same-session matched capacity comparison; historical G is context only."""
import csv
import json
from pathlib import Path
import numpy as np
import experiment as E
from data import write_json


def build_rows(reports, cfg):
    if len(reports) != len(E.VARIANTS): raise ValueError('Need all three completed arms')
    base_contract = reports[0]['training_contract']
    base_images = [r['sample_id'] for r in reports[0]['per_image']]
    if len(base_images) != 400 or len(set(base_images)) != 400:
        raise RuntimeError('Expected 400 distinct matched validation IDs')
    shared = reports[0]['initial_shared_sha256']
    rows = []
    for variant, report in zip(E.VARIANTS, reports):
        if report['variant'] != variant or not report['qualification_passed']:
            raise RuntimeError(f'Unqualified/mismatched result: {variant}')
        for key in ('gpu', 'torch', 'cuda', 'build', 'parity_gates'):
            if report[key] != reports[0][key]: raise RuntimeError(f'Runtime/precision protocol differs: {key}')
        contract = report['training_contract']
        if any(contract[k] != base_contract[k] for k in ('config', 'data', 'source_sha256')):
            raise RuntimeError('Source/config/data changed across arms')
        if report['initial_shared_sha256'] != shared:
            raise RuntimeError('Retained original initial tensor slices differed')
        if [r['sample_id'] for r in report['per_image']] != base_images:
            raise RuntimeError('Validation image order/IDs changed')
        if report['training_epochs'] != 30 or report['inference_precision'] != 'explicit_FP32_TF32_OFF':
            raise RuntimeError('Epoch or precision budget changed')
        t = report['timing']['trt_graph']
        if (t['warmup'], t['iterations'], t['distinct_images']) != (100, 300, 100):
            raise RuntimeError('Timing protocol changed')
        s = report['trt_graph']['all']
        if s['pixels'] != 25424992: raise RuntimeError('GT support differs')
        rows.append({'variant': variant, 'checkpoint_epoch': report['checkpoint_epoch'],
            'rmse_m': s['rmse_m'], 'irmse_km_inv': s['irmse_km_inv'], 'mae_m': s['mae_m'],
            'boundary3_rmse_m': report['trt_graph_boundary']['bands']['3']['rmse_m'],
            'far80_120_rmse_m': report['trt_graph']['80-120']['rmse_m'],
            'tail10_sse_fraction': s['error_tail']['10']['sse_fraction'],
            'parameters': report['parameters'],
            'median_ms': t['synchronized_wall']['median_ms'],
            'p95_ms': t['synchronized_wall']['p95_ms'], 'matched_training_verified': True})
    for row in rows:
        row.update(fps=1000/row['median_ms'],
            rmse_improvement_vs_G0_pct=100*(rows[0]['rmse_m']-row['rmse_m'])/rows[0]['rmse_m'],
            latency_increase_vs_G0_pct=100*(row['median_ms']/rows[0]['median_ms']-1),
            accuracy_not_worse_than_G0=row['rmse_m'] <= rows[0]['rmse_m'] and row['irmse_km_inv'] <= rows[0]['irmse_km_inv'],
            latency_within_5pct_G0=row['median_ms'] <= cfg['capacity_latency_ratio_limit']*rows[0]['median_ms'])
    return rows


def paired_bootstrap(reference, candidate, seed=42):
    """Image-cluster resampling; development uncertainty, NOT independent proof."""
    n = np.array([r['pixels'] for r in reference], np.float64)
    if not np.array_equal(n, [r['pixels'] for r in candidate]): raise ValueError('Support mismatch')
    x = np.array([r['sse_m2'] for r in reference], np.float64)
    y = np.array([r['sse_m2'] for r in candidate], np.float64)
    picks = np.random.default_rng(seed).integers(0, len(n), (2000, len(n)))
    delta = np.sqrt(y[picks].sum(1)/n[picks].sum(1))-np.sqrt(x[picks].sum(1)/n[picks].sum(1))
    return {'delta_rmse_candidate_minus_G0_m': float(np.sqrt(y.sum()/n.sum())-np.sqrt(x.sum()/n.sum())),
        'image_bootstrap_95pct_interval_m': np.percentile(delta, [2.5, 97.5]).tolist(),
        'resamples': 2000, 'seed': seed,
        'caveat': 'Image resampling ignores sequence dependence and repeated validation selection; not independent confirmation.'}


def compare_history(cfg):
    root = Path(cfg['drive_runs'])/cfg['run_name']
    reports = [json.loads((root/v/'trt_benchmark.json').read_text()) for v in E.VARIANTS]
    rows = build_rows(reports, cfg)
    with (root/'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/'comparison.json', rows)
    write_json(root/'paired_rmse_bootstrap.json', {v: paired_bootstrap(reports[0]['per_image'], report['per_image'])
        for v, report in zip(E.VARIANTS[1:], reports[1:])})
    historical = json.loads((E.ROOT/'evidence/historical_comparison.json').read_text())
    historical_g = next(r for r in historical if r['variant'] == 'G_s4_upir_norel')
    write_json(root/'historical_G_context.json', {'historical': historical_g,
        'interpretation': 'Old session/engine: context only. Primary control is new G0, not the historical 3.826771ms.'})
    import cv2
    preview = root/'comparison_previews';preview.mkdir(exist_ok=True)
    for index in range(0, 1000, 100):
        pair = cv2.imread(str(root/E.VARIANTS[0]/'test_preview'/f'{index:04d}_rgb_depth.png'))
        if pair is None: raise RuntimeError('Missing qualified test previews')
        panels = [('RGB', pair[:, :1216])]
        panels += [(v, cv2.imread(str(root/v/'test_preview'/f'{index:04d}_depth.png'))) for v in E.VARIANTS]
        named = []
        for name, panel in panels:
            if panel is None: raise RuntimeError('Missing test preview')
            title = np.full((32, panel.shape[1], 3), 245, np.uint8)
            cv2.putText(title, name, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (25,25,25), 1, cv2.LINE_AA)
            named.append(np.concatenate((title, panel)))
        cv2.imwrite(str(preview/f'{index:04d}_RGB_G0_H_I.png'), np.concatenate(named))
    print(json.dumps(rows, indent=2))
