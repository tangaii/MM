"""Small PPU correctness tests, followed by actual paired-anchor checks."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

if "--paired" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

from drpa import (DRPA, EmbeddingHooks, cross_site_reliability, disease_reliability,
                  normalized_gate, shuffled_gate)
from run import (BASE, anchor_dir, array_hash, atomic_json, binary_metrics, capture_rng,
                 check_splits, completed, config, context, core_hash, export_core, import_core,
                 load_torch, make_options, private_seed, require_ppu, restore_rng,
                 rng_hash, state_hash)
from model.mm_gtunets import MM_GTUNets, CustomConv2d
from model.rp_graph import create_reward_penalty_graph
from utils.mydataloader import MyDataloader


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = require_ppu()
        cls.cfg = config()
        cls.temp = tempfile.TemporaryDirectory(dir=BASE / "artifacts", prefix="correctness_")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def data(self, n=24):
        device = self.device
        hi = torch.randn(n, 16, device=device, requires_grad=True)
        hp = torch.randn(n, 16, device=device, requires_grad=True)
        y = torch.arange(n, device=device) % 2
        sites = torch.arange(n, device=device) % 3
        indices = torch.arange(18, device=device)
        return hi, hp, y, sites, indices

    def test_official_forward_gradient_and_step_disabled(self):
        opt = make_options(911, self.temp.name, initialize=True)
        opt.num_subjects = 24
        core = MM_GTUNets(opt, 0).to(self.device)
        self.assertIsInstance(core.rp_attention.conv, CustomConv2d)
        self.assertNotIn("rp_attention.weights", core.state_dict())
        torch.save(core.vae.state_dict(), core.vae_save_path)
        x = torch.randn(24, 500)
        ph = torch.rand(24, 3)
        y = np.arange(24) % 2
        splits = (np.arange(16), np.arange(16, 20), np.arange(20, 24))
        phenotypes = {"SITE_ID": np.arange(24) % 3, "SEX": y, "AGE_AT_SCAN": np.arange(24)}
        affinity = torch.from_numpy(create_reward_penalty_graph(phenotypes, y, *splits, opt)).float()
        hi = torch.as_tensor(splits[0], device=self.device)
        labels = torch.as_tensor(y, device=self.device, dtype=torch.long)
        initial = export_core(core)
        rng = capture_rng()
        core.train()
        original_optimizer = torch.optim.Adam(core.parameters(), lr=opt.lr, weight_decay=opt.wd)
        original_optimizer.zero_grad()
        original_logits, original_graph = core(x, ph, affinity)
        original_loss = F.cross_entropy(original_logits[hi], labels[hi]) + original_graph
        original_loss.backward()
        original_grads = {n: p.grad.detach().clone() for n, p in core.named_parameters() if p.grad is not None}
        original_optimizer.step()
        original_after = export_core(core)
        rng_after = rng_hash(capture_rng())
        original_logits = original_logits.detach().clone()
        original_graph = original_graph.detach().clone()
        import_core(core, initial)
        restore_rng(rng)
        repeat_optimizer = torch.optim.Adam(core.parameters(), lr=opt.lr, weight_decay=opt.wd)
        repeat_optimizer.zero_grad()
        repeated_logits, repeated_graph = core(x, ph, affinity)
        (F.cross_entropy(repeated_logits[hi], labels[hi]) + repeated_graph).backward()
        repeat_grad_error = {n: float((p.grad - original_grads[n]).abs().max())
                             for n, p in core.named_parameters() if n in original_grads}
        repeat_optimizer.step()
        repeated_after = export_core(core)
        repeat_step_error = {n: float((v - original_after["state"][n]).abs().max())
                             for n, v in repeated_after["state"].items()}
        numeric_control = {"official_repeat_forward_max_abs": float((repeated_logits - original_logits).abs().max()),
                           "official_repeat_gradient_max_abs_by_parameter": repeat_grad_error,
                           "official_repeat_step_max_abs_by_parameter": repeat_step_error}
        print("OFFICIAL REPEAT NUMERIC CONTROL", json.dumps(numeric_control), flush=True)
        import_core(core, initial)
        restore_rng(rng)
        before = rng_hash(capture_rng())
        module = DRPA.isolated(self.cfg, 91204, self.device)
        self.assertEqual(before, rng_hash(capture_rng()))
        observer = EmbeddingHooks(core)
        heads_before = state_hash(module.state_dict())
        wrapped_optimizer = torch.optim.Adam(list(core.parameters()) + list(module.parameters()), lr=opt.lr, weight_decay=opt.wd)
        wrapped_optimizer.zero_grad()
        logits, graph = core(x, ph, affinity)
        self.assertEqual(tuple(observer.img.shape), (24, 16))
        self.assertEqual(tuple(observer.ph.shape), (24, 16))
        torch.testing.assert_close(logits, original_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(graph, original_graph, rtol=1e-5, atol=1e-6)
        observer.img.retain_grad()
        observer.ph.retain_grad()
        loss = F.cross_entropy(logits[splits[0]], labels[splits[0]]) + graph
        loss.backward()
        hook_grad_errors = {}
        for name, param in core.named_parameters():
            if name in original_grads:
                hook_grad_errors[name] = float((param.grad - original_grads[name]).abs().max())
                torch.testing.assert_close(param.grad, original_grads[name], rtol=1e-4, atol=1e-5, msg=name)
        self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertIsNotNone(observer.img.grad)
        self.assertIsNotNone(observer.ph.grad)
        wrapped_optimizer.step()
        self.assertEqual(heads_before, state_hash(module.state_dict()))
        for name, value in original_after["state"].items():
            torch.testing.assert_close(core.state_dict()[name].cpu(), value, rtol=1e-5, atol=1e-5, msg=name)
        numeric_control["hook_gradient_max_abs_by_parameter"] = hook_grad_errors
        atomic_json(numeric_control, BASE / "artifacts/numeric_repeat_control.json")
        self.assertEqual(rng_after, rng_hash(capture_rng()))
        observer.close()
        snapshot = {"core": export_core(core), "aux": module.state_dict()}
        path = Path(self.temp.name) / "restore.pt"
        torch.save(snapshot, path)
        with torch.no_grad():
            next(core.parameters()).add_(1)
            next(module.parameters()).add_(1)
        restored = load_torch(path)
        import_core(core, restored["core"])
        module.load_state_dict(restored["aux"])
        self.assertEqual(core_hash(restored["core"]), core_hash(export_core(core)))
        self.assertEqual(state_hash(restored["aux"]), state_hash(module.state_dict()))

    def test_only_training_rows_and_labels_enter_objective(self):
        module = DRPA.isolated(self.cfg, 101, self.device)
        hi, hp, y, sites, idx = self.data()
        expected = module.objective(hi, hp, y, sites, idx, "E6", apply_shuffle=False)
        altered_i, altered_p = hi.detach().clone(), hp.detach().clone()
        altered_y, altered_sites = y.clone(), sites.clone()
        altered_i[18:] = 123; altered_p[18:] = -321
        altered_y[18:] = 1 - altered_y[18:]; altered_sites[18:] = 999
        actual = module.objective(altered_i, altered_p, altered_y, altered_sites, idx, "E6", apply_shuffle=False)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[2]["weights"], expected[2]["weights"], rtol=0, atol=0)
        (expected[0] + expected[1]).backward()
        self.assertEqual(float(hi.grad[18:].abs().sum()), 0)
        self.assertEqual(float(hp.grad[18:].abs().sum()), 0)

    def test_stop_gradient_and_auxiliary_gradients(self):
        for method in ("E2", "E3", "E4", "E5", "E6"):
            with self.subTest(method=method):
                module = DRPA.isolated(self.cfg, 202, self.device)
                hi, hp, y, sites, idx = self.data()
                aux, align, details, value = module.objective(hi, hp, y, sites, idx, method)
                self.assertFalse(details["weights"].requires_grad)
                self.assertFalse(details["q_pred"].requires_grad)
                self.assertFalse(details["q_site"].requires_grad)
                align.backward(retain_graph=True)
                if method in ("E4", "E5", "E6"):
                    self.assertIsNone(hi.grad)
                    self.assertTrue(all(p.grad is None for p in module.project_img.parameters()))
                    self.assertIsNotNone(hp.grad)
                else:
                    self.assertIsNotNone(hi.grad); self.assertIsNotNone(hp.grad)
                module.zero_grad(); hi.grad = None; hp.grad = None
                (0.05 * aux + 0.1 * align).backward()
                self.assertTrue(all(p.grad is not None for p in module.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        module = DRPA.isolated(self.cfg, 303, self.device)
        values = self.data()
        aux, align, _, _ = module.objective(*values, "E1")
        self.assertIsNone(align)
        aux.backward()
        self.assertTrue(all(p.grad is not None for p in module.parameters()))

    def test_site_exclusion_and_missing_class_fallback(self):
        si = F.normalize(torch.randn(16, 16, device=self.device), dim=1)
        sp = F.normalize(torch.randn(16, 16, device=self.device), dim=1)
        y = torch.arange(16, device=self.device) % 2
        sites = torch.arange(16, device=self.device) // 4
        first, _ = cross_site_reliability(si, sp, y, sites)
        changed_i, changed_p = si.clone(), sp.clone()
        changed_i[1:4] = F.normalize(torch.randn(3, 16, device=self.device), dim=1)
        changed_p[1:4] = F.normalize(torch.randn(3, 16, device=self.device), dim=1)
        changed, _ = cross_site_reliability(changed_i, changed_p, y, sites)
        torch.testing.assert_close(first[0], changed[0], rtol=1e-5, atol=1e-6)
        single_site, information = cross_site_reliability(si, sp, y, torch.zeros_like(sites))
        self.assertEqual(float(single_site.sum()), 0)
        self.assertTrue(information["fallback"].all())
        no_other_class, information = cross_site_reliability(si, sp, y, y)
        self.assertEqual(float(no_other_class.sum()), 0)
        self.assertTrue(information["fallback"].all())

    def test_gate_zero_nonfinite_and_shuffle_distribution(self):
        zeros, empty = normalized_gate(torch.zeros(12, device=self.device), 0.15)
        self.assertTrue(empty); self.assertEqual(float(zeros.sum()), 0)
        q = torch.linspace(0, 1, 24, device=self.device, requires_grad=True)
        weights, empty = normalized_gate(q, 0.15)
        self.assertFalse(empty); self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights.mean(), weights.new_tensor(1.0))
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(FloatingPointError):
                normalized_gate(torch.tensor([invalid], device=self.device), 0.15)
        y = torch.arange(24, device=self.device) % 2
        sites = torch.arange(24, device=self.device) % 3
        before = rng_hash(capture_rng())
        generator = torch.Generator().manual_seed(444)
        shuffled, detail = shuffled_gate(weights, y, sites, generator)
        self.assertEqual(before, rng_hash(capture_rng()))
        torch.testing.assert_close(shuffled.sort().values, weights.sort().values, rtol=0, atol=0)
        self.assertGreater(detail["shuffle_moved_fraction"], 0)
        generator = torch.Generator().manual_seed(444)
        singleton, info = shuffled_gate(weights[:2], y[:2], sites[:2], generator)
        torch.testing.assert_close(singleton, weights[:2], rtol=0, atol=0)
        self.assertEqual(info["shuffle_isolated_singletons"], 2)

    def test_actual_indices_and_metric_recalculation(self):
        ctx = context()
        for seed in self.cfg["seeds"]:
            opt = make_options(seed, self.temp.name)
            loader = MyDataloader(opt, features=np.zeros((871, 1)), labels=ctx["labels"])
            splits = loader.data_split(10, val_ratio=0.1)
            for split in splits:
                check_splits(split)
            self.assertTrue(np.array_equal(np.sort(np.concatenate([s[2] for s in splits])), np.arange(871)))
        scores = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.9, 0.4, 0.6, 0.1])
        self.assertEqual(scores["AUC"], 0.75)
        self.assertEqual(scores["ACC"], 0.5)
        self.assertEqual(scores["BA"], 0.5)
        self.assertEqual(scores["F1"], 0.5)
        from analyze import metrics_from_rows
        rows = [{"label_original": str(y), "pred_original": str(p), "prob_asd": str(q)}
                for y, p, q in zip([0, 0, 1, 1], [0, 1, 0, 1], [0.9, 0.4, 0.6, 0.1])]
        self.assertEqual(scores, metrics_from_rows(rows))


def paired_checks(available_only=False):
    # Compare identical mathematical paths with deterministic reductions. The
    # production source itself can change TopK selections when repeated.
    # This is a test-only backend setting; formal training stays unchanged.
    torch.use_deterministic_algorithms(True)
    device, cfg, ctx = require_ppu(), config(), context()
    records = []
    for seed in cfg["seeds"]:
        for fold in range(10):
            if available_only and not completed("E0", seed, fold, verify=False):
                continue
            adir = anchor_dir(seed, fold)
            start = load_torch(adir / "train_start.pt")
            split = load_torch(adir / "splits.pt")
            splits = [split[k] for k in ("train", "val", "test")]
            x = load_torch(adir.parent / "selected_features.pt")
            opt = make_options(seed, adir.parent / "vae")
            core = MM_GTUNets(opt, fold).to(device)
            import_core(core, start["core"])
            labels = torch.as_tensor(ctx["labels"], device=device, dtype=torch.long)
            idx = torch.as_tensor(split["train"], device=device)
            ph = torch.from_numpy(ctx["ph_data"]).float()
            affinity = torch.from_numpy(create_reward_penalty_graph(ctx["ph_dict"], ctx["labels"], *splits, opt)).float()
            restore_rng(start["rng"])
            if capture_rng()["backend"] != start["rng"]["backend"]:
                raise AssertionError("Actual anchor backend flags were not restored")
            optimizer = torch.optim.Adam(core.parameters(), lr=opt.lr, weight_decay=opt.wd)
            optimizer.zero_grad(); core.train()
            baseline, graph = core(x, ph, affinity)
            (F.cross_entropy(baseline[splits[0]], labels[splits[0]]) + graph).backward()
            gradients = {n: p.grad.detach().clone() for n, p in core.named_parameters() if p.grad is not None}
            optimizer.step()
            after = export_core(core)
            rng_after = rng_hash(capture_rng())
            baseline, graph = baseline.detach().clone(), graph.detach().clone()
            import_core(core, start["core"])
            restore_rng(start["rng"])
            auxiliary = DRPA.isolated(cfg, private_seed(seed, fold, "heads"), device)
            observer = EmbeddingHooks(core)
            optimizer = torch.optim.Adam(list(core.parameters()) + list(auxiliary.parameters()), lr=opt.lr, weight_decay=opt.wd)
            optimizer.zero_grad()
            replay, replay_graph = core(x, ph, affinity)
            torch.testing.assert_close(replay, baseline, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(replay_graph, graph, rtol=1e-5, atol=1e-6)
            observer.img.retain_grad()
            observer.ph.retain_grad()
            (F.cross_entropy(replay[splits[0]], labels[splits[0]]) + replay_graph).backward()
            max_gradient_error = 0.0
            for name, param in core.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(param.grad, gradients[name], rtol=1e-4, atol=1e-5)
                    max_gradient_error = max(max_gradient_error, float((param.grad - gradients[name]).abs().max()))
            if any(p.grad is not None for p in auxiliary.parameters()):
                raise AssertionError("Disabled heads received gradients")
            optimizer.step()
            max_step_error = 0.0
            for name, tensor in core.state_dict().items():
                torch.testing.assert_close(tensor.cpu(), after["state"][name], rtol=1e-5, atol=1e-5)
                max_step_error = max(max_step_error, float((tensor.cpu() - after["state"][name]).abs().max()))
            if rng_after != rng_hash(capture_rng()):
                raise AssertionError("Disabled module perturbed official RNG")
            if capture_rng()["backend"] != start["rng"]["backend"]:
                raise AssertionError("Disabled module perturbed official backend flags")
            records.append({"seed": seed, "fold": fold, "passed": True,
                "max_forward_abs_error": float((replay - baseline).abs().max()),
                "graph_abs_error": float((graph - replay_graph).abs()),
                "max_gradient_abs_error": max_gradient_error, "max_parameter_step_abs_error": max_step_error,
                "rng_equal": True, "saved_backend_flags_equal": True,
                "source_and_candidate_test_backend_equal": True,
                "test_only_deterministic_algorithms": True, "extra_gradients": False})
            print("PAIRED DISABLED CHECK", json.dumps(records[-1]), flush=True)
            observer.close()
            del core, auxiliary, replay, baseline, gradients, affinity
    destination = "paired_disabled_partial.json" if available_only else "paired_disabled_checks.json"
    atomic_json({"passed": True, "n_actual_seed_folds": len(records),
                 "is_final_30_anchor_check": not available_only,
                 "equivalence_profile": {"deterministic_algorithms": True,
                     "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                     "scope": "Both source and candidate, numerical verification only; no formal training setting is changed"},
                 "production_repeatability_limit": "Actual default-backend source repeats can change TopK selections and differ by about 0.005 in logits. This mathematical-equivalence test is not a claim of bitwise production replay.",
                 "checks": records}, BASE / "artifacts" / destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--available-only", action="store_true",
                        help="Diagnostic check of completed anchors only; does not satisfy the final 30-anchor gate")
    args = parser.parse_args()
    if args.paired:
        paired_checks(args.available_only)
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Correctness)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        atomic_json({"passed": result.wasSuccessful(), "tests": result.testsRun,
                     "failures": len(result.failures), "errors": len(result.errors)}, BASE / "artifacts/correctness.json")
        raise SystemExit(0 if result.wasSuccessful() else 1)
