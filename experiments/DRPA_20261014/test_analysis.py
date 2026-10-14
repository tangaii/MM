"""CPU-only statistics regression tests; no neural model is trained."""

import json
import unittest

from analyze import KEYS, paired_statistics


SEEDS = [911, 912, 913]


class AnalysisTests(unittest.TestCase):
    def test_numpy_seed_comparisons_are_json_serializable(self):
        differences = []
        for seed, acc, ba, auc in zip(SEEDS, [0.01, -0.02, 0.0],
                                     [0.02, 0.0, 0.01], [-0.01, -0.02, -0.03]):
            for fold in range(10):
                row = {"seed": seed, "fold": fold, **{k: acc for k in KEYS}}
                row.update(BA=ba, AUC=auc)
                differences.append(row)
        report = paired_statistics(differences, SEEDS)
        restored = json.loads(json.dumps(report, allow_nan=False))
        self.assertEqual(restored, report)
        self.assertEqual(report["improved_seeds"], {"ACC": 1, "BA": 2, "AUC": 0})
        self.assertEqual(report["improved_folds"], {"ACC": 10, "BA": 20, "AUC": 0})
        self.assertEqual(report["tied_folds"], {"ACC": 10, "BA": 10, "AUC": 0})
        for value in report["improved_seeds"].values():
            self.assertIs(type(value), int)
        self.assertAlmostEqual(report["mean_delta"]["ACC"], -0.01 / 3)
        self.assertAlmostEqual(report["mean_delta"]["BA"], 0.01)
        self.assertAlmostEqual(report["mean_delta"]["AUC"], -0.02)
        self.assertEqual(len(report["fold0"]), 3)
        self.assertEqual(report["paired_folds"], differences)

    def test_all_tied_results_remain_zero_not_missing(self):
        rows = [{"seed": seed, "fold": fold, **{k: 0.0 for k in KEYS}}
                for seed in SEEDS for fold in range(10)]
        report = paired_statistics(rows, SEEDS)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["improved_seeds"], {k: 0 for k in ("ACC", "BA", "AUC")})
        self.assertEqual(report["tied_folds"], {k: 30 for k in ("ACC", "BA", "AUC")})
        self.assertEqual(report["mean_delta"], {k: 0.0 for k in KEYS})

    def test_fold_zero_is_not_in_other_fold_mean(self):
        rows = [{"seed": seed, "fold": fold, **{k: 0.3 if fold == 0 else -0.1 for k in KEYS}}
                for seed in SEEDS for fold in range(10)]
        report = paired_statistics(rows, SEEDS)
        for key in KEYS:
            self.assertAlmostEqual(report["other_folds_mean_delta"][key], -0.1)
            self.assertAlmostEqual(report["mean_delta"][key], -0.06)


if __name__ == "__main__":
    unittest.main(verbosity=2)
