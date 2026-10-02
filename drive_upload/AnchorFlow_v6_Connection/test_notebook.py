"""Execute the Colab parent/config/snapshot cell in an isolated local fixture."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from model import AnchorFlowEdge


class NotebookContracts(unittest.TestCase):
    def test_parent_resolution_snapshot_and_safe_rerun(self):
        folder = Path(__file__).parent
        notebook = json.loads((folder/"AnchorFlow_v6_TAR2000_15ep.ipynb").read_text(encoding="utf-8"))
        cell = compile("".join(notebook["cells"][6]["source"]),"colab_parent_cell","exec")
        for source in ("drive_v5","bundled_v3"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                code = root/"code"
                code.mkdir()
                shutil.copy2(folder/"config.json",code/"config.json")
                shutil.copy2(folder/"init_v3_best.pth",code/"init_v3_best.pth")
                subset = root/"data/teacher_subset_2000"
                subset.mkdir(parents=True)
                for name in ("selected_2000_ids.json","kitti_trainval_2000.tar","metric_coarse_train_2000.tar"):
                    (subset/name).write_bytes(b"fixture-presence-only")
                parent = root/"v5_best.pth"
                torch.save({"epoch":7,"best_rmse":1.03,"model":AnchorFlowEdge(model_name="v5_piecewise").state_dict(),
                    "protocol":{"config":{"architecture":"AnchorFlow-v5-PiecewiseSurface","parent_cumulative_epoch":29}}},parent)
                manifest = {"files":{"config.json":"fixture"}}
                (root/"bundle_manifest.json").write_text(json.dumps(manifest))
                env = dict(Path=Path,json=json,hashlib=hashlib,torch=torch,shutil=shutil,
                    subprocess=subprocess,sys=sys,CODE=code,BUNDLE=root,DRIVE_DATA=root/"data",
                    DRIVE_RUNS=root/"runs",WORK=root/"work",RUN_NAME="fixture",RUN_ROOT=root/"runs/fixture",
                    MODEL_NAME="v6_connection",INIT_SOURCE=source,PARENT_CHECKPOINT=parent,EPOCHS=15,
                    BATCH_SIZE=4,WORKERS=2,BOUNDARY_WEIGHT=.05,BARRIER_WEIGHT=.01,ROBUST_MSE_WEIGHT=.1,
                    manifest=manifest)
                exec(cell,env)
                cfg = env["cfg"]
                self.assertEqual(cfg["parent_cumulative_epoch"],37 if source=="drive_v5" else 29)
                self.assertEqual(cfg["parent_epoch"],7 if source=="drive_v5" else 14)
                self.assertTrue((env["snapshot"]/"init_parent.pth").is_file())
                self.assertEqual(cfg["init_sha256"],hashlib.sha256((code/"init_parent.pth").read_bytes()).hexdigest())
                exec(cell,env)  # same source/config/parent: safe rerun, no SameFileError
                env["BATCH_SIZE"] = 2
                with self.assertRaisesRegex(AssertionError,"RUN_TAG"):
                    exec(cell,env)  # reject silently changing the config in a frozen run


if __name__ == "__main__":
    unittest.main()
