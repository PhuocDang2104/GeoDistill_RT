import ast
import json
from pathlib import Path
import unittest


class NotebookContracts(unittest.TestCase):
    def test_clean_cells_and_fresh_config(self):
        root=Path(__file__).parent
        notebook=json.loads((root/"notebook_contract.json").read_text(encoding="utf-8"))
        codes=[]
        literals={}
        for cell in notebook["cells"]:
            if cell["cell_type"]!="code": continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"],[])
            source="".join(cell["source"])
            codes.append(source)
            tree=ast.parse(source)
            for assignment in tree.body:
                if not isinstance(assignment,ast.Assign): continue
                try: value=ast.literal_eval(assignment.value)
                except (ValueError,TypeError): continue
                for target in assignment.targets:
                    if isinstance(target,ast.Name): literals[target.id]=value
        ast.parse("\n".join(codes))
        cfg=json.loads((root/"config.json").read_text())
        self.assertEqual(literals["EPOCHS"],cfg["epochs"])
        self.assertEqual(literals["BATCH_SIZE"],cfg["batch_size"])
        self.assertEqual(literals["MODEL_NAME"],cfg["model_name"])
        self.assertEqual(literals["EARLY_STOP_PATIENCE"],cfg["early_stop_patience"])
        self.assertEqual(literals["EARLY_STOP_MIN_EPOCHS"],cfg["early_stop_min_epochs"])
        self.assertEqual(literals["EARLY_STOP_MIN_DELTA_M"],cfg["early_stop_min_delta_m"])
        self.assertEqual(literals["RUN_NAMES"][cfg["model_name"]],cfg["run_name"])
        self.assertTrue(cfg["encoder_pretrained"])
        self.assertNotIn("init_checkpoint",cfg)
        self.assertNotIn('command("parent_evaluate")',"\n".join(codes))
        for action in ("prepare","smoke","train","evaluate","test","profile","export"):
            self.assertIn(f'command("{action}")',"\n".join(codes))

    def test_pretrained_flag_only_on_fresh_model(self):
        # Guard against the old warm-start workflow's hardcoded pretrained=False.
        source=Path(__file__).with_name("run.py").read_text(encoding="utf-8")
        self.assertIn('pretrained=config["encoder_pretrained"] and not resume_file.is_file()',source)
        self.assertNotIn("load_parent_state",source)
        self.assertNotIn("warm_start(",source)

    def test_colab_copy_and_test_reporting_contract(self):
        snapshot=json.loads(Path(__file__).with_name("notebook_contract.json").read_text(encoding="utf-8"))
        source="\n".join("".join(c["source"]) for c in snapshot["cells"] if c["cell_type"]=="code")
        # Mutable .ipynb must be skipped BEFORE digest, while snapshot JSON is
        # included by the normal verified copy. Failure tracebacks stay visible.
        self.assertLess(source.index('if name.endswith(".ipynb"):'),source.index('hashlib.sha256(source.read_bytes())'))
        self.assertIn('print(test_log,flush=True)',source)
        self.assertIn('capture_output=True',source)
        self.assertIn('unit_test_output.txt',source)


if __name__=="__main__": unittest.main()
