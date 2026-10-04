"""Execute the actual Colab parent/snapshot cell without Colab dependencies."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import torch


class NotebookContracts(unittest.TestCase):
    def test_parent_resolution_snapshot_and_safe_rerun(self):
        folder=Path(__file__).parent
        n=json.loads((folder/"AnchorFlow_v7_Jet_TAR2000_FT15.ipynb").read_text(encoding="utf-8"))
        cell=compile("".join(n["cells"][6]["source"]),"parent_cell","exec")
        for source in ("bundled_v6","drive_v6"):
            with self.subTest(source=source),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); code=root/"code"; code.mkdir()
                for name in ("config.json","init_v6_best.pth"):
                    shutil.copy2(folder/name,code/name)
                subset=root/"data/teacher_subset_2000"; subset.mkdir(parents=True)
                for name in ("selected_2000_ids.json","kitti_trainval_2000.tar","metric_coarse_train_2000.tar"):
                    (subset/name).write_bytes(b"fixture-presence-only")
                manifest={"files":{"config.json":"fixture"},"parent_v6_sha256":hashlib.sha256((code/"init_v6_best.pth").read_bytes()).hexdigest()}
                (root/"bundle_manifest.json").write_text(json.dumps(manifest))
                env=dict(Path=Path,json=json,hashlib=hashlib,torch=torch,shutil=shutil,subprocess=subprocess,sys=sys,
                    CODE=code,BUNDLE=root,DRIVE_DATA=root/"data",DRIVE_RUNS=root/"runs",WORK=root/"work",
                    RUN_NAME="fixture",RUN_ROOT=root/"runs/fixture",MODEL_NAME="v7_jet",INIT_SOURCE=source,
                    PARENT_CHECKPOINT=code/"init_v6_best.pth",EPOCHS=15,BATCH_SIZE=4,WORKERS=2,
                    BOUNDARY_WEIGHT=.05,BARRIER_WEIGHT=.01,TAIL_WEIGHT=.1,TAIL_THRESHOLD_M=2.,manifest=manifest)
                exec(cell,env)
                self.assertEqual(env["cfg"]["parent_cumulative_epoch"],59)
                self.assertEqual(env["cfg"]["parent_epoch"],14)
                self.assertTrue((env["snapshot"]/"init_parent.pth").is_file())
                exec(cell,env)
                env["BATCH_SIZE"]=2
                with self.assertRaisesRegex(AssertionError,"RUN_TAG"):
                    exec(cell,env)


if __name__=="__main__":
    unittest.main()
