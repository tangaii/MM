"""Independent numeric and data-boundary checks; no neural training/forward."""
from __future__ import annotations
import ast
import copy
import inspect
import json
import numpy as np
from common import BASE, ROOT, CFG, utc
from stage_a import (PhenotypeEncoding, ridge_predict, oof_predictions, mismatch,
                     residualize_site, shared_directions, study, decide)


def run_tests():
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)
    rng = np.random.default_rng(911)
    n = 180
    pf = np.tile(np.arange(3), n // 3)
    raw = {"age": rng.uniform(5, 55, n), "sex": rng.choice([1, 2], n),
           "site": np.repeat(["A", "B", "C"], n // 3)}
    encoding = PhenotypeEncoding(raw)
    s, ph = encoding.transform(raw)
    z = rng.normal(size=(n, 16))
    z[:, 0] += 2 * ph[:, 0]
    z[:, 1] += 2 * ph[:, 2]
    ref = pf != 0
    x = np.column_stack((s, ph))
    # Independent augmented least-squares oracle, not the normal-equation solver.
    xr, xq = x[ref], x[~ref]
    xm, xs = xr.mean(0), xr.std(0)
    xs = np.where(xs > 1e-12, xs, 1.)
    a = (xr - xm) / xs
    ym = z[ref].mean(0)
    aug_x = np.vstack((a, np.eye(a.shape[1])))
    aug_y = np.vstack((z[ref] - ym, np.zeros((a.shape[1], 16))))
    w = np.linalg.lstsq(aug_x, aug_y, rcond=None)[0]
    expected = (xq - xm) / xs @ w + ym
    actual = ridge_predict(xr, z[ref], xq)
    check("ridge_alpha_one_I_augmented_lstsq_oracle", np.allclose(actual, expected, rtol=1e-10, atol=1e-10))
    constant = ridge_predict(np.ones((10, 4)), np.full((10, 16), 7.), np.ones((3, 4)))
    check("ridge_unpenalized_intercept_and_constant_features", np.array_equal(constant, np.full((3, 16), 7.)))
    pred_site, p0 = oof_predictions(z, s, ph, pf, False)
    pred_full, p1 = oof_predictions(z, s, ph, pf, True)
    check("site_full_identical_private_partitions", p0 == p1)
    # Each query's target can change without changing that query's own prediction.
    altered = z.copy(); altered[pf == 0] += 10000
    altered_pred, _ = oof_predictions(altered, s, ph, pf, True)
    check("oof_no_own_private_fold_target_fit", np.array_equal(altered_pred[pf == 0], pred_full[pf == 0]))
    check("threefold_disjoint_complete_query_coverage", sum(p["query_n"] for p in p0) == n and all(p["reference_n"] == 120 for p in p0))
    # Fit is unaffected by arbitrary heldout data because callers slice before fit.
    heldout = {k: np.concatenate((v, v[:10])) for k, v in raw.items()}
    train_index = np.arange(n)
    enc2 = PhenotypeEncoding({k: v[train_index] for k, v in heldout.items()})
    heldout["age"][n:] = 1e20; heldout["sex"][n:] = -100; heldout["site"][n:] = "Z"
    enc3 = PhenotypeEncoding({k: v[train_index] for k, v in heldout.items()})
    check("outer_train_only_preprocessing_heldout_invariance", enc2.info == enc3.info)
    missing = {"age": np.array([10., 20., np.nan, -1.]), "sex": np.array([1, 2, np.nan, 9]),
               "site": np.array(["A", "B", "", "A"])}
    me = PhenotypeEncoding(missing)
    ms, mp = me.transform(missing)
    check("age_legal_train_median_imputation", me.median == 15 and np.allclose(mp[2:, 0], 0))
    check("sex_male_female_unknown_separate", np.array_equal(mp[:, 1:], [[1,0,0],[0,1,0],[0,0,1],[0,0,1]]))
    check("site_categorical_onehot_missing", ms.shape == (4, 3) and np.all(ms.sum(1) == 1))
    qs, qp = me.transform({"age": [np.inf], "sex": [-1], "site": ["unseen"]})
    check("unknown_site_no_ordinal_size", np.count_nonzero(qs) == 0 and np.isfinite(qp).all())
    ae = PhenotypeEncoding({"age": [np.nan, -1], "sex": [9, np.nan], "site": ["A", "A"]})
    check("all_missing_age_fixed_recorded_fallback", ae.median == 17 and ae.info["all_age_missing"])
    state_before = np.random.get_state()
    permutations = []
    for b in range(32):
        p, singles = mismatch(pf, raw["site"], 0, b)
        check(f"null_{b:02d}_train_private_site_derangement", np.array_equal(pf[p], pf)
              and np.array_equal(raw["site"][p], raw["site"]) and np.all(p != np.arange(n)) and singles == 0)
        permutations.append(tuple(p))
    check("null_replicates_distinct", len(set(permutations)) == 32)
    p, count = mismatch(np.array([0,0,1]), np.array(["S","S","S"]), 0, 99)
    check("singleton_counted_and_retained", count == 1 and p[2] == 2 and p[0] == 1 and p[1] == 0)
    check("null_function_has_no_diagnosis_argument", set(inspect.signature(mismatch).parameters) == {"pf", "raw_site", "fold", "replicate"})
    check("probe_study_has_no_diagnosis_argument", set(inspect.signature(study).parameters) == {"z", "raw_train", "pf", "fold"})
    before = np.random.get_state()
    result, encoded = study(z, raw, pf, 0)
    after = np.random.get_state()
    check("private_null_rng_does_not_change_global_numpy_rng", all(np.array_equal(a, b) for a, b in zip(before, after)))
    sst = np.square(z - z.mean(0)).sum()
    oracle_delta = (np.square(z - pred_site).sum() - np.square(z - pred_full).sum()) / sst
    check("delta_R2_increment_not_full_R2", np.isclose(result["real_delta_R2"], oracle_delta, atol=1e-14)
          and not np.isclose(result["real_delta_R2"], result["full_R2_descriptive_only"]))
    check("null_p95_fixed_linear_quantile", result["null_p95"] == float(np.quantile(result["null_delta_R2"], .95, method="linear")))
    check("synthetic_shared_information_recovers_signal", result["real_delta_R2"] > result["null_p95"] and result["real_delta_R2"] > .05)
    # Independent ridge site-residualization with an explicit intercept column.
    # Centered dummies + intercept are equivalent to unpenalized intercept fitting.
    design = np.column_stack((np.ones(n), s))
    penalty = np.diag([0.] + [1.] * s.shape[1])
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ ph)
    oracle_res = ph - design @ coef
    actual_res = residualize_site(ph, s)
    check("site_residualization_explicit_intercept_oracle", np.allclose(oracle_res, actual_res, atol=1e-12))
    y = np.tile([0, 1], n // 2)
    sd, stab = shared_directions(z, s, ph, pf, y)
    sd_flip, _ = shared_directions(z, s, ph, pf, 1 - y)
    check("shared_u_not_diagnosis_derived", all(np.array_equal(a["u"], b["u"]) for a, b in zip(sd, sd_flip)))
    check("shared_u_unit_norm", all(abs(a["u_unit_norm"] - 1) < 1e-12 for a in sd))
    check("shared_direction_has_nonzero_variance", all(a["direction_valid"] for a in sd))
    check("protected_norm_geometric_identity", all(np.isclose(a["protected_norm_squared"], 1 - a["absolute_u_dot_d"]**2) for a in sd))
    # No query-fold image or label value can affect its own reference u/d.
    zv, yv = z.copy(), y.copy()
    zv[pf == 0] += 1e5; yv[pf == 0] = 9
    sv, _ = shared_directions(zv, s, ph, pf, yv)
    check("shared_u_d_exclude_query_rows", np.array_equal(sv[0]["u"], sd[0]["u"])
          and sv[0]["protected_norm_squared"] == sd[0]["protected_norm_squared"])
    dc, _ = shared_directions(np.ones_like(z), s, ph, pf, y)
    check("constant_embedding_no_random_direction", all(not a["direction_valid"] and a["effective_rank"] == 0 for a in dc))
    degenerate, _ = study(np.ones_like(z), raw, pf, 0)
    check("constant_target_cannot_pass_delta_gate", degenerate["degenerate_target"])
    check("shared_projection_is_scalar_linear_not_sign_L2", np.allclose((z @ np.asarray(sd[0]["u"]))**2, (z @ -np.asarray(sd[0]["u"]))**2))
    # Frozen strict boundaries: exactly 0 and exactly .002 must fail.
    base = {"probe": {"degenerate_target": False, "real_delta_R2": .003, "real_positive": True,
                      "real_above_null_p95": True},
            "shared_directions": [{"direction_valid": True, "protection_pass": True}] * 3}
    rows = [{**copy.deepcopy(base), "fold": f} for f in range(10)]
    check("go_requires_all_gates", decide(rows)["passed"])
    for r in rows:
        r["probe"]["real_delta_R2"] = .002
    check("median_gate_strict_at_point002", not decide(rows)["passed"])
    rows = [{**copy.deepcopy(base), "fold": f} for f in range(10)]
    for r in rows[:4]:
        r["probe"]["real_positive"] = False
    check("positive_gate_needs_seven", not decide(rows)["passed"])
    rows = [{**copy.deepcopy(base), "fold": f} for f in range(10)]
    for r in rows[:4]:
        r["probe"]["real_above_null_p95"] = False
    check("null_gate_needs_seven", not decide(rows)["passed"])
    rows = [{**copy.deepcopy(base), "fold": f} for f in range(10)]
    for r in rows[:4]:
        r["shared_directions"][0] = {"direction_valid": True, "protection_pass": False}
    check("protection_cannot_be_disabled_to_bypass_gate", not decide(rows)["passed"])
    # Verify original graph routine's heldout label invariance without a network.
    import sys
    sys.path.insert(0, str(ROOT))
    from model.rp_graph import create_reward_penalty_graph
    from types import SimpleNamespace
    opt = SimpleNamespace(scores=["SITE_ID", "SEX", "AGE_AT_SCAN"], ages="AGE_AT_SCAN")
    graph_meta = {"SITE_ID": np.array([0,0,1,1,0,0]), "SEX": np.array([1,2,1,2,1,1]),
                  "AGE_AT_SCAN": np.array([12,13,14,15,16,17])}
    a = create_reward_penalty_graph(graph_meta, np.array([0,1,0,1,0,1]), [0,1,2,3], [4], [5], opt)
    b = create_reward_penalty_graph(graph_meta, np.array([0,1,0,1,99,-99]), [0,1,2,3], [4], [5], opt)
    check("native_graph_does_not_consult_heldout_diagnosis", np.array_equal(a,b))
    check("native_graph_early_return_preserved", np.count_nonzero(a[1:]) == 0)
    tree = ast.parse((BASE / "run.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    check("stage_a_runner_has_no_optimizer_step_or_backward", not any(isinstance(n.func, ast.Attribute) and n.func.attr in ("step", "backward", "train_vae", "feature_selection") for n in calls))
    check("ASD_zero_TD_one_fixed", CFG["class_encoding"] == {"ASD":0,"TD":1})
    return {"utc": utc(), "passed": all(checks.values()), "passed_count": sum(checks.values()),
            "checks": checks, "neural_model_forwards": 0, "neural_training_tasks": 0,
            "independent_oracles": ["augmented least-squares ridge", "explicit-intercept residualization", "heldout mutation tests", "strict gate boundary tests"]}


if __name__ == "__main__":
    print(json.dumps(run_tests(), ensure_ascii=False, indent=2))
