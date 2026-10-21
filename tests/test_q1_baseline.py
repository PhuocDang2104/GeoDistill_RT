"""Promotion/evidence/cleanup invariants outside the immutable training snapshot."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100'
EVIDENCE=ROOT/'benchmarks/q1_n1fast_2026_10_21'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Baseline(unittest.TestCase):
    def test_adopted_manifest_and_frozen_bytes(self):
        package=load('q1_package',ROOT/'scripts/package_current_baseline.py')
        manifest=package.verify()
        self.assertEqual(len(manifest['files']),40)
        cfg=json.loads((BUNDLE/'config.json').read_text())
        self.assertEqual(cfg['epochs'],30);self.assertFalse(cfg['relative_enabled'])
        self.assertEqual((cfg['ode_method'],cfg['ode_fixed_steps'],cfg['ode_max_nfe']),('midpoint',2,4))
        self.assertFalse(cfg['tf32']);self.assertEqual(cfg['relative_weight'],0)

    def test_console_sha_and_actual_thresholds(self):
        evidence=json.loads((EVIDENCE/'evidence.json').read_text())
        self.assertEqual(evidence['user_console_sha256'],hashlib.sha256((EVIDENCE/'user_console.txt').read_bytes()).hexdigest())
        self.assertEqual(evidence['evidence_level'],'user_supplied_console_snapshot')
        rows=evidence['comparison']['rows'];self.assertEqual(len(rows),4)
        self.assertEqual(len({r['benchmark_session_id'] for r in rows}),1)
        new=next(r for r in rows if (r['variant'],r['backend'])==('Q1_s4_n1_fast','trt_fast'))
        self.assertLess(new['rmse_m'],.98);self.assertLess(new['irmse_km_inv'],3.2);self.assertLessEqual(new['mean_ms'],3.9)
        self.assertAlmostEqual(new['fps_from_mean'],1000/new['mean_ms'],places=10)
        self.assertFalse(evidence['independent_A100_rerun_by_local_agent'])

    def test_same_checkpoint_speed_not_accuracy_claim(self):
        evidence=json.loads((EVIDENCE/'evidence.json').read_text());a=evidence['attribution']
        self.assertAlmostEqual(a['same_Q1_checkpoint_wall_reduction_pct'],9.049117142104834)
        self.assertLess(abs(a['same_Q1_checkpoint_rmse_drift_m']),1e-7)
        self.assertIn('Different trained weights',a['quality_attribution'])
        self.assertIn('both use CUDA Graph',a['speed_attribution'])

    def test_unknown_private_artifacts_not_invented(self):
        adopted=json.loads((ROOT/'production/AnchorFlow_S4_Q1_Fast/baseline.json').read_text())
        self.assertEqual(adopted['baseline_id'],'Q1_s4_n1_fast')
        self.assertEqual(adopted['status'],'current_subset_development_baseline')
        self.assertEqual(adopted['new_neural_parameters_vs_N1'],0)
        self.assertIsNone(adopted['checkpoint_sha256']);self.assertIsNone(adopted['checkpoint_epoch_0based'])
        self.assertFalse(adopted['full_KITTI_evaluated']);self.assertFalse(adopted['Orin_measured'])

    def test_no_generator_or_reverification_mutates_frozen_snapshot(self):
        wrapper=(ROOT/'scripts/package_s4_n1fast.py').read_text()
        self.assertIn('from package_current_baseline import main',wrapper)
        self.assertNotIn('PARENT=',wrapper)
        verifier=(ROOT/'scripts/verify_s4_n1fast.py').read_text()
        self.assertIn('local_reverification.json',verifier)
        self.assertNotIn("(CODE/'local_verification.json').write_text",verifier)
        self.assertNotIn('nbf.write',wrapper)

    def test_archive_receipt_and_hashes_when_available(self):
        receipt=json.loads((EVIDENCE/'cleanup_manifest.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(receipt['state'],'complete');self.assertEqual(len(receipt['items']),25)
        self.assertEqual(receipt['file_count'],407)
        self.assertTrue(receipt['frozen_Q1_untouched']);self.assertTrue(receipt['preexisting_disconnected_junctions_untouched'])
        for item in receipt['items']:
            self.assertTrue(item['verified'])
            destination=Path(item['destination'])
            if not destination.exists():continue  # local cold storage is not cloned by Git
            for record in item['files']:
                path=destination/record['relative'] if item['directory'] else destination
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])

    def test_current_docs_links_and_legacy_exporter_warning(self):
        paths=['README.md','docs/AnchorFlow_V11_Baseline.md','docs/AnchorFlow_Q1_Theory.md',
               'docs/README.md','drive_upload/README.md','production/AnchorFlow_S4_Q1_Fast/README.md',
               'benchmarks/q1_n1fast_2026_10_21/README.md']
        for name in paths:
            path=ROOT/name;source=path.read_text(encoding='utf-8')
            for link in re.findall(r'\]\(([^)]+)\)',source):
                if '://' in link or link.startswith('#'):continue
                self.assertTrue((path.parent/link.split('#')[0]).exists(),(name,link))
        orin=(ROOT/'production/AnchorFlow_S4_UPIR_Orin/README.md').read_text(encoding='utf-8')
        self.assertIn('not Q1',orin)


if __name__=='__main__':unittest.main()
