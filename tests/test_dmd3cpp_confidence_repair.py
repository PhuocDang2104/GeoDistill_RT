"""Boundary policy tests; no CUDA/model substitution in generated datasets."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('confidence_repair', ROOT/'scripts/dmd3cpp_confidence_repair.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class ConfidenceBoundaryContracts(unittest.TestCase):
    def arrays(self):
        return np.stack((np.full((352, 1216), 10, np.float32), np.full((352, 1216), .5, np.float32)))

    def test_valid_arrays_return_original_object_unchanged(self):
        a = self.arrays()
        a[1, 0, :2] = [0, 1]
        before = a.tobytes()
        result, event = repair.bound_metric_confidence(a, 'valid')
        self.assertIs(result, a)
        self.assertIsNone(event)
        self.assertEqual(result.tobytes(), before)

    def test_float32_boundary_errors_clipped_and_depth_preserved(self):
        a = self.arrays()
        a[1, 0, 0] = np.nextafter(np.float32(1), np.float32(2))
        a[1, 0, 1] = -np.finfo(np.float32).eps
        before = a.copy()
        result, event = repair.bound_metric_confidence(a, 'roundoff')
        np.testing.assert_array_equal(result[0], before[0])
        np.testing.assert_array_equal(a, before)
        np.testing.assert_array_equal(result[1, 1:], before[1, 1:])
        self.assertEqual(event['corrected_pixels'], 2)
        self.assertEqual(result[1, 0, 0], 1)
        self.assertEqual(result[1, 0, 1], 0)

    def test_material_errors_still_fail_with_sample_id_and_range(self):
        for bad in (-.001, 1.001):
            a = self.arrays()
            a[1, 0, 0] = bad
            with self.assertRaisesRegex(RuntimeError, 'bad-sample: min='):
                repair.bound_metric_confidence(a, 'bad-sample')

    def test_nonfinite_depth_or_confidence_still_fails(self):
        for channel in (0, 1):
            for bad in (np.nan, np.inf, -np.inf):
                a = self.arrays()
                a[channel, 0, 0] = bad
                with self.assertRaisesRegex(RuntimeError, 'Invalid metric teacher tensor'):
                    repair.bound_metric_confidence(a, 'nonfinite')

    def test_install_records_provenance_and_keeps_base_sources_and_recipe(self):
        sys.path.insert(0, str(ROOT/'production/AnchorFlow_V11_3_Fixed6_FullKITTI'))
        sys.path.insert(0, str(ROOT/'scripts'))
        import teachers
        import colab_dmd3cpp_full_train as exporter
        from utils import sha256
        before = {name: sha256((Path(teachers.__file__).parent if name == 'teachers.py' else ROOT/'scripts')/name)
                  for name in repair.BASE_SOURCES}
        a = self.arrays()
        a[1, 0, 0] = -np.finfo(np.float32).eps
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            recipe = {'fixture_base_recipe': True}
            exporter.save_json(out/'recipe.json', recipe)
            # Real installation requires both pinned files in the embedded source directory.
            # Route the exporter's byte-identical local file for this CPU fixture.
            real_sha = sha256
            def fixture_sha(path):
                path = Path(path)
                if path.name == 'colab_dmd3cpp_full_train.py' and not path.exists():
                    path = ROOT/'scripts'/path.name
                return real_sha(path)
            def fake_predict(self, sample):
                return a
            def fake_audit(cfg):
                return [], [], [], {'passed': True}
            with patch('utils.sha256', fixture_sha), patch.object(teachers.MetricTeacher, 'predict', fake_predict), patch.object(exporter, 'audit', fake_audit):
                policy = repair.install(out)
                repair.install(out)  # Idempotent on a repeated notebook cell.
                result = teachers.MetricTeacher.predict(None, {'sid': 'cpu-fixture'})
                self.assertEqual(result[1, 0, 0], 0)
                np.testing.assert_array_equal(result[0], a[0])
                events = [json.loads(line) for line in (out/'repairs/confidence_roundoff_v1/events.jsonl').read_text().splitlines()]
                self.assertEqual(events[0]['sid'], 'cpu-fixture')
                self.assertEqual(policy['base_source_sha256'], before)
                self.assertEqual(exporter.read_json(out/'recipe.json'), recipe)
                report = exporter.audit({'drive_output': str(out)})[-1]
                self.assertIn('supplemental_confidence_policy', report)
        after = {name: sha256((Path(teachers.__file__).parent if name == 'teachers.py' else ROOT/'scripts')/name)
                 for name in repair.BASE_SOURCES}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
