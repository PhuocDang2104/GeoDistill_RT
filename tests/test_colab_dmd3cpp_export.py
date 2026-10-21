"""CPU export/recovery tests; these do not qualify the real DMD3C++ CUDA model."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'production/AnchorFlow_V11_3_Fixed6_FullKITTI'))
spec = importlib.util.spec_from_file_location('colab_export', ROOT/'scripts/colab_dmd3cpp_full_train.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class LocalZip:
    extract = export.RangeZip.extract
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
    def close(self):
        self.z.close()


class ExportContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out, self.local = self.root/'drive', self.root/'session'
        self.out.mkdir()
        self.local.mkdir()
        self.cfg = {'drive_output': str(self.out), 'local_root': str(self.local)}
        self.drive = '2011_09_26_drive_0001_sync'

    def fixture_archives(self):
        payloads = {'gt': {}, 'sparse': {}, 'raw': {}, 'calib': {}}
        for frame in range(2):
            name = f'{frame:010d}.png'
            gt = np.zeros((16, 32), np.uint16)
            gt[8:, 8:24] = 2560
            sparse = np.zeros_like(gt)
            sparse[12, 16] = 2560
            rgb = np.full((16, 32, 3), 100, np.uint8)
            for role, arr, path in (
                ('gt', gt, f'train/{self.drive}/proj_depth/groundtruth/image_02/{name}'),
                ('sparse', sparse, f'train/{self.drive}/proj_depth/velodyne_raw/image_02/{name}'),
                ('raw', rgb, f'2011_09_26/{self.drive}/image_02/data/{name}')):
                ok, png = cv2.imencode('.png', arr)
                self.assertTrue(ok)
                payloads[role][path] = png.tobytes()
        payloads['calib']['2011_09_26/calib_cam_to_cam.txt'] = b'P_rect_02: 700 0 16 0 0 700 8 0 0 0 1 0\n'
        archives = {}
        for role, items in payloads.items():
            path = self.root/(role+'.zip')
            with zipfile.ZipFile(path, 'w') as z:
                for name, data in items.items():
                    z.writestr(name, data)
            archives[role] = LocalZip(path)
            self.addCleanup(archives[role].close)
        return archives

    def test_full_count_pairing_and_no_silent_subset(self):
        z = self.fixture_archives()
        with self.assertRaisesRegex(RuntimeError, '85898'):
            export.enumerate_train(z['gt'], z['sparse'])
        with patch.object(export, 'EXPECTED_TRAIN', 2):
            groups = export.enumerate_train(z['gt'], z['sparse'])
        plan = export.plan_shards(groups, 1)
        self.assertEqual([len(p['samples']) for p in plan], [1, 1])
        self.assertTrue(all(p['drive'] == self.drive for p in plan))

    def test_copy_commit_rejects_corrupt_or_conflicting_artifact(self):
        source = self.local/'artifact.bin'
        source.write_bytes(b'verified content')
        target = self.out/'artifact.bin'
        record = export.publish(source, target)
        self.assertEqual(export.publish(source, target), record)
        self.assertEqual(record['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        target.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            export.publish(source, target)

    def test_zip_traversal_is_rejected(self):
        path = self.local/'bad.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('../outside.txt', 'unsafe')
        z = LocalZip(path)
        self.addCleanup(z.close)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            z.extract('../outside.txt', self.out)
        self.assertFalse((self.root/'outside.txt').exists())

    def test_sharded_export_finalize_and_corruption_gate(self):
        from full_data import SIZE
        archives = self.fixture_archives()
        with patch.object(export, 'EXPECTED_TRAIN', 2):
            groups = export.enumerate_train(archives['gt'], archives['sparse'])
            plan = export.plan_shards(groups, 1)
            recipe = {'teacher': {'schema': 1, 'test_fixture': True}}
            recipe_sha = export.identity(recipe)
            export.save_json(self.out/'recipe.json', recipe)
            export.save_json(self.out/'plan.json', {'parts': plan})
            class FakeTeacher:
                def predict(inner, sample):
                    return np.stack((np.full(SIZE, 10, np.float32), np.ones(SIZE, np.float32)))
            for part in plan:
                stage = self.local/part['key']
                stage.mkdir()
                rows = export.make_rows(part, stage, archives['gt'], archives['sparse'], archives['raw'], archives['calib'])
                input_zip, teacher_zip = stage/'inputs.zip', stage/'teachers.zip'
                export.make_input_shard(rows, stage, input_zip)
                first_hash = export.digest(input_zip)
                export.make_input_shard(rows, stage, input_zip)
                self.assertEqual(first_hash, export.digest(input_zip))
                with patch('torch.cuda.synchronize'):
                    records = export.make_teacher_shard(rows, stage, teacher_zip, FakeTeacher(), recipe_sha, 1024)
                inputs = export.publish(input_zip, self.out/'inputs'/(part['key']+'.zip'))
                teachers = export.publish(teacher_zip, self.out/'teachers'/(part['key']+'.zip'))
                commit = {'sample_ids': [r['sid'] for r in rows], 'recipe_sha256': recipe_sha,
                          'inputs': inputs, 'teachers': teachers, 'train_rows': rows, 'teacher_records': records}
                export.save_json(self.out/'commits'/(part['key']+'.json'), commit)
                self.assertIsNotNone(export.check_committed(self.out, part, recipe_sha))
            with patch.object(export, 'disk_gate'):
                export.finalize(self.cfg)
            complete = export.read_json(self.out/'COMPLETE.json')
            self.assertTrue(complete['passed'])
            self.assertEqual(complete['audit']['train_samples'], 2)
            target = self.out/'kitti_train_full_85898.zip'
            self.assertEqual(export.digest(target), complete['full_train_zip']['sha256'])
            with zipfile.ZipFile(target) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(len(json.loads(z.read('train_index.json'))), 2)
                self.assertEqual(len([n for n in z.namelist() if 'calib_cam_to_cam' in n]), 1)
            with patch.object(export, 'disk_gate'):
                export.finalize(self.cfg)
            with (self.out/'teachers'/'part-00000.zip').open('ab') as f:
                f.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError, 'corrupt'):
                export.audit(self.cfg)
            self.assertFalse(export.read_json(self.out/'audit.json')['passed'])
            self.assertFalse(export.read_json(self.out/'COMPLETE.json')['passed'])


if __name__ == '__main__':
    unittest.main()
