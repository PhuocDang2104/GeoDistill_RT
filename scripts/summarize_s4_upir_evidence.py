"""Rebuild curated numerical evidence from user-supplied logs (not GPU execution)."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'benchmarks/s4_upir_norel_2026_10_19'


def main():
    decoder = json.JSONDecoder()
    text = (ROOT/'comparison_user_log.txt').read_text(encoding='utf-8')
    comparison, _ = decoder.raw_decode(text[text.index('['):])
    text = (ROOT/'benchmark_user_log.txt').read_text(encoding='utf-8')
    report, _ = decoder.raw_decode(text[text.index('{'):])
    rows = list(csv.DictReader((ROOT/'train_log.csv').open(encoding='utf-8', newline='')))
    assert len(rows) == 30 and [int(r['epoch']) for r in rows] == list(range(30))
    assert all(float(r['loss_weighted_relative']) == 0 for r in rows)
    assert all(float(r['loss_relative_gradient']) == 0 for r in rows)
    best = min(rows, key=lambda r: float(r['val_rmse_m']))
    assert int(best['epoch']) == 27
    g = next(r for r in comparison if r['variant'] == 'G_s4_upir_norel')
    assert g['checkpoint_epoch'] == 27 and g['parameters'] == 583880
    assert report['all']['pixels'] == 25424992
    assert abs(g['rmse_m']-report['all']['rmse_m']) < 1e-12
    deltas = {}
    for r in comparison:
        if r['variant'] == g['variant']: continue
        deltas[r['variant']] = {k: 100*(r[k]-g[k])/r[k]
            for k in ('rmse_m', 'irmse_km_inv', 'mae_m', 'median_ms')}
    summary = {'source': 'User-pasted completed A100 logs + original training CSV; not a local rerun',
        'variant': g['variant'], 'epoch_zero_based': 27, 'epochs': 30,
        'best_native_row': best, 'actual_trt_result': g, 'improvement_pct_vs_history': deltas,
        'first_inverse_feasible_epoch': next(int(r['epoch']) for r in rows if float(r['val_irmse_km_inv']) <= 3.2),
        'train_seconds_sum': sum(float(r['train_seconds']) for r in rows),
        'epoch_seconds_sum': sum(float(r['epoch_seconds']) for r in rows),
        'relative_zero_all_epochs': True,
        'acceptance': {'rmse_not_worse_than_D': True, 'irmse_not_worse_than_D': True,
            'wall_median_within_target': False, 'all_targets_met': False,
            'wall_median_limit_ms': 3.8153860000420536},
        'attribution': 'D/G confounded by uncertainty architecture AND relative KD; F dual-teacher UPIR result not supplied. No isolated architectural gain established.',
        'qualification_limits': 'Only pasted report supplied: actual checkpoint hash/engine/parity detail must be supplied/exported before independent artifact verification.',
        'raw_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('comparison_user_log.txt', 'benchmark_user_log.txt', 'train_log.csv')}}
    for name, value in (('comparison.json', comparison), ('benchmark_snapshot.json', report), ('summary.json', summary)):
        (ROOT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'best_epoch': 27, 'deltas': deltas, 'acceptance': summary['acceptance']}, indent=2))


if __name__ == '__main__': main()
