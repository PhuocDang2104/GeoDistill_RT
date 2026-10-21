"""Audited, bounded float32 confidence repair for the original Colab export.

This is a supplemental numerical policy, not a rewrite of the base recipe or
provider files. Already valid confidence tensors are returned byte-for-byte.
Metric depth is never modified. Material errors and nonfinite data still fail.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np

TOLERANCE = 1e-6
BASE_SOURCES = {
    'teachers.py': '9a80712193cb546abc7733bc0aea7429a728f6eda93dccb28664561aba1f2890',
    'colab_dmd3cpp_full_train.py': '39007266a5c3d33394ddcc1b399adc02057f00adcef6770002ea1892107b7172',
}


def bound_metric_confidence(arrays, sid):
    """Clip only finite float32 rounding excursions, preserving valid arrays."""
    if arrays.shape != (2, 352, 1216) or arrays.dtype != np.float32 or not np.isfinite(arrays).all():
        raise RuntimeError(f'Invalid metric teacher tensor before confidence repair: {sid}')
    c = arrays[1]
    low, high = float(c.min()), float(c.max())
    if low < -TOLERANCE or high > 1 + TOLERANCE:
        raise RuntimeError(f'Material confidence error for {sid}: min={low:.12g}, max={high:.12g}; '
                           f'only float32 boundary errors <= {TOLERANCE} may be repaired')
    count = int(((c < 0) | (c > 1)).sum())
    if not count:
        return arrays, None
    repaired = arrays.copy()
    np.clip(c, 0, 1, out=repaired[1])
    return repaired, {'sid': sid, 'confidence_min_before': low, 'confidence_max_before': high,
                      'corrected_pixels': count, 'tolerance': TOLERANCE}


def install(output_root):
    """Install the wrapper and publish immutable, independently hashed provenance."""
    import teachers
    import colab_dmd3cpp_full_train as exporter
    from full_data import SIZE
    from utils import atomic_json, sha256, identity

    if tuple(SIZE) != (352, 1216):
        raise RuntimeError('Unexpected metric output contract')
    source_root = Path(teachers.__file__).parent
    actual = {name: sha256(source_root/name) for name in BASE_SOURCES}
    if actual != BASE_SOURCES:
        raise RuntimeError('This repair applies only to the original pinned export/provider; source mismatch')
    output_root = Path(output_root)
    if not output_root.is_dir():
        raise RuntimeError('Mount Drive and run the export settings cell before installing the repair')
    patch_file = Path(__file__)
    policy = {'schema': 1, 'name': 'metric-confidence-float32-boundary-v1',
              'tolerance': TOLERANCE, 'base_source_sha256': actual,
              'repair_source_sha256': sha256(patch_file),
              'depth_unchanged': True, 'in_range_arrays_unchanged': True,
              'material_or_nonfinite_errors': 'raise',
              'base_recipe': 'recipe.json retained; this manifest is a supplemental numerical policy',
              'existing_shards': 'strictly validated original arrays remain byte-for-byte unchanged'}
    repair_root = output_root/'repairs'/'confidence_roundoff_v1'
    repair_root.mkdir(parents=True, exist_ok=True)
    manifest = repair_root/'policy.json'
    if manifest.exists() and json.loads(manifest.read_text()) != policy:
        raise RuntimeError('Existing confidence repair policy differs; refusing silent replacement')
    exporter.publish(patch_file, repair_root/'dmd3cpp_confidence_repair.py')
    atomic_json(manifest, policy)
    installed = getattr(teachers.MetricTeacher.predict, '_confidence_policy_sha256', None)
    policy_sha = identity(policy)
    if installed is not None:
        if installed != policy_sha:
            raise RuntimeError('Another confidence patch is already installed')
        if getattr(exporter.audit, '_confidence_policy_sha256', None) != policy_sha:
            raise RuntimeError('Confidence patch installation was interrupted before audit integration')
        print('Confidence boundary guard already installed:', policy_sha, flush=True)
        return policy
    original_predict = teachers.MetricTeacher.predict

    def guarded_predict(self, sample):
        arrays = original_predict(self, sample)
        arrays, event = bound_metric_confidence(arrays, sample['sid'])
        if event is not None:
            event['policy_sha256'] = policy_sha
            recipe_file = output_root/'recipe.json'
            event['base_recipe_sha256'] = identity(json.loads(recipe_file.read_text()))
            event['recorded_unix_seconds'] = time.time()
            with (repair_root/'events.jsonl').open('a') as log:
                log.write(json.dumps(event, allow_nan=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
            print(f'Confidence roundoff repaired: {sample["sid"]}; '
                  f'range=[{event["confidence_min_before"]:.9g}, {event["confidence_max_before"]:.9g}]; '
                  f'pixels={event["corrected_pixels"]}', flush=True)
        return arrays

    guarded_predict._confidence_policy_sha256 = policy_sha
    guarded_predict.__wrapped__ = original_predict
    original_audit = exporter.audit

    def audited_with_policy(cfg):
        commits, rows, records, report = original_audit(cfg)
        if Path(cfg['drive_output']).resolve() != output_root.resolve():
            raise RuntimeError('Confidence policy is installed for a different Drive output folder')
        events_path = repair_root/'events.jsonl'
        report['supplemental_confidence_policy'] = {
            'path': manifest.relative_to(output_root).as_posix(), 'sha256': sha256(manifest),
            'policy_sha256': policy_sha,
            'correction_event_log_sha256': sha256(events_path) if events_path.exists() else None,
            'prior_strictly_validated_shards_retained': True}
        exporter.save_json(output_root/'audit.json', report)
        return commits, rows, records, report

    audited_with_policy._confidence_policy_sha256 = policy_sha
    audited_with_policy.__wrapped__ = original_audit
    teachers.MetricTeacher.predict = guarded_predict
    exporter.audit = audited_with_policy
    print('Installed audited confidence boundary guard; tolerance=1e-6. '
          'Base recipe and previously committed tensors retained.', flush=True)
    return policy
