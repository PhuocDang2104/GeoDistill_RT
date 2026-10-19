import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
import experiment as E
from deployment import typed_deployment, export_onnx
from objective import loss_fn

torch.set_num_threads(2)
CFG = json.loads((E.ROOT/'config.json').read_text())


class Contracts(unittest.TestCase):
    def inputs(self):
        rgb = torch.rand(1, 3, 32, 64)
        m = (torch.rand(1, 1, 32, 64) > .9).float()
        return rgb, 20*m, m, torch.tensor([[[50.,0.,31.],[0.,50.,15.],[0.,0.,1.]]])

    def test_same_shared_slices_and_baseline_hash(self):
        nets = [E.make_model(CFG, v).eval() for v in E.VARIANTS]
        hashes = [E.common_weight_hash(n) for n in nets]
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(hashes[0], E.B.common_weight_hash(nets[0]))
        self.assertEqual(sum(p.numel() for p in nets[0].parameters()), 583880)

    def test_widening_counts_layers_and_initial_function(self):
        args = self.inputs()
        nets = [E.make_model(CFG, v).eval() for v in E.VARIANTS]
        with torch.no_grad():
            expected = nets[0](*args)
            for net in nets[1:]:
                actual = net(*args)
                for key in ('D16', 'D8', 'D0', 'D4', 'D2', 'D_full'):
                    torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-5)
                self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in net.modules()),
                                 sum(isinstance(m, nn.Conv2d) for m in nets[0].modules()))
                self.assertGreater(E.architecture(net)['parameters'], 583880)
                self.assertEqual(E.architecture(net)['new_layers'], 0)

    def test_four_real_calls_no_fine_and_finite_gradients(self):
        for variant in E.VARIANTS:
            n = E.make_model(CFG, variant).eval();calls = []
            hook = n.dynamics.reaction.register_forward_hook(lambda *args: calls.append(1))
            out = n(*self.inputs());hook.remove()
            self.assertEqual(len(calls), 4);self.assertIsNone(n.fine_node)
            torch.testing.assert_close(out['D2'], out['D2_pre_fine'], atol=0, rtol=0)
            out['D_full'].mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in n.parameters()))

    def test_new_hidden_lanes_can_learn(self):
        n = E.make_model(CFG, E.VARIANTS[1]).eval()
        block = n.decoder.refine8
        optimizer = torch.optim.SGD(n.parameters(), lr=1e-3)
        args = self.inputs()
        for _ in range(2):
            optimizer.zero_grad();n(*args)['D8'].mean().backward();optimizer.step()
        self.assertGreater(float(block.body[2][0].weight.grad[:, 128:].abs().sum()), 0)
        self.assertGreater(float(block.body[0][0].weight.grad[128:].abs().sum()), 0)

    def test_relative_forbidden_and_modules_absent(self):
        self.assertFalse((E.ROOT/'vendor/relative_loss.py').exists())
        self.assertFalse((E.ROOT/'vendor/relative_data.py').exists())
        for key in ('relative_enabled', 'relative_weight'):
            cfg = dict(CFG);cfg[key] = True
            with self.assertRaises(ValueError): E.make_model(cfg, E.VARIANTS[0])
        import data
        with self.assertRaises(ValueError): data.prepare(dict(CFG, relative_enabled=True))
        self.assertNotIn('relative_train', (E.ROOT/'vendor/data.py').read_text())

    def test_objective_needs_no_relative(self):
        n = E.make_model(CFG, E.VARIANTS[1]).eval();args = self.inputs()
        pred = n(*args)
        batch = dict(zip(E.INPUTS, args), gt=torch.full_like(args[1], 20.), gt_mask=torch.ones_like(args[1]))
        # Keep architecture comparison's real GT losses, teacher disabled only in this CPU fixture.
        cfg = dict(CFG, teacher_enabled=False)
        loss, stats = loss_fn(pred, batch, torch.zeros_like(args[1]), 5., cfg)
        loss.backward();self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(stats['weighted_relative']), 0)

    def test_export_preserves_all_eval_modes_and_bn_state(self):
        n = E.make_model(CFG, E.VARIANTS[2]).eval()
        module = typed_deployment(n)
        before = {k: v.clone() for k, v in module.state_dict().items()}
        args = self.inputs()
        with torch.no_grad(): expected = module(*args).clone()
        with tempfile.TemporaryDirectory() as tmp:
            export_onnx(module, dict(zip(E.INPUTS, args)), Path(tmp)/'test.onnx')
        self.assertTrue(all(not m.training for m in module.modules()))
        for k, value in module.state_dict().items(): torch.testing.assert_close(value, before[k], atol=0, rtol=0)
        with torch.no_grad(): torch.testing.assert_close(module(*args), expected, atol=0, rtol=0)

    def test_changed_recipe_rejected(self):
        for change in ({'epochs': 26}, {'ode_max_nfe': 6}, {'decoder_expansion': 8}, {'tf32': True}):
            with self.assertRaises(ValueError): E.assert_config(dict(CFG, **change))

    def test_global_metrics(self):
        from metrics import Metrics
        m = Metrics('cpu');p = torch.tensor([[[[2.,4.]]]])
        m.update(p, torch.tensor([[[[1.,2.]]]]), torch.ones_like(p), torch.zeros(1,3,1,2))
        self.assertAlmostEqual(m.report()['all']['rmse_m'], (2.5)**.5)

    def test_validation_cannot_load_teachers(self):
        from data import KITTIDataset
        with self.assertRaises(ValueError): KITTIDataset(CFG, 'val', teacher=True)
        with self.assertRaises(ValueError): KITTIDataset(CFG, 'test', teacher=True)

    def test_bootstrap_identity(self):
        from compare_history import paired_bootstrap
        values = [{'pixels': 10, 'sse_m2': 20.}, {'pixels': 20, 'sse_m2': 40.}]
        result = paired_bootstrap(values, values)
        self.assertEqual(result['delta_rmse_candidate_minus_G0_m'], 0)
        self.assertEqual(result['image_bootstrap_95pct_interval_m'], [0, 0])

    def test_comparison_contracts_and_failed_targets(self):
        from compare_history import build_rows
        images = [{'sample_id': str(i), 'pixels': 10, 'sse_m2': 10.} for i in range(400)]
        reports = []
        for i, variant in enumerate(E.VARIANTS):
            reports.append({'variant':variant,'qualification_passed':True,
                'training_contract':{'variant':variant,'config':{},'data':{},'source_sha256':'source'},
                'initial_shared_sha256':'shared','per_image':images,'gpu':'A100','torch':'same','cuda':'same',
                'build':{},'parity_gates':{},'training_epochs':30,'inference_precision':'explicit_FP32_TF32_OFF',
                'checkpoint_epoch':27,'parameters':583880+i,
                'trt_graph':{'all':{'pixels':25424992,'rmse_m':1+i*.01,'irmse_km_inv':3.17,
                    'mae_m':.25,'error_tail':{'10':{'sse_fraction':.57}}},'80-120':{'rmse_m':3.}},
                'trt_graph_boundary':{'bands':{'3':{'rmse_m':1.6}}},
                'timing':{'trt_graph':{'warmup':100,'iterations':300,'distinct_images':100,
                    'synchronized_wall':{'median_ms':4+i,'p95_ms':4.1+i}}}})
        rows = build_rows(reports, CFG)
        self.assertFalse(rows[1]['accuracy_not_worse_than_G0'])
        self.assertFalse(rows[1]['latency_within_5pct_G0'])
        reports[1]['initial_shared_sha256'] = 'different'
        with self.assertRaises(RuntimeError): build_rows(reports, CFG)

    def test_notebook_syntax(self):
        path = E.ROOT/'Train_S4_UPIR_DecoderCapacity_Fresh30_A100.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))

    def test_prepare_never_imports_relative(self):
        import data
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(CFG, work=tmp, expected_subset_sha256=None)
            fake = {'teacher_report': {'selected_teacher_content_sha256':
                '06bd17a84faa5020692873ee598498486d2030f588b7429321fa0a70d9ebb7a6'}}
            def prepare_metric(_): (Path(tmp)/'data_contract.json').write_text(json.dumps(fake))
            with patch.object(data, 'prepare_metric', side_effect=prepare_metric): data.prepare(cfg)
            self.assertEqual(json.loads((Path(tmp)/'data_contract.json').read_text()), fake)

    def test_syntax(self):
        for path in E.ROOT.rglob('*.py'):
            if '__pycache__' not in path.parts: ast.parse(path.read_text(encoding='utf-8'))


if __name__ == '__main__': unittest.main()
