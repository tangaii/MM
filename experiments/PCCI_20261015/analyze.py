"""Independent paired statistics, OOF validation and counterfactual analysis."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import run as r


def packed_metrics(rows):
    seeds = {}
    for seed in r.config()["seeds"]:
        selected = [row for row in rows if row["seed"] == seed]
        if len(selected) != 10:
            raise ValueError("Each seed must have ten folds")
        seeds[str(seed)] = {k: float(np.mean([row["metrics"][k] for row in selected])) for k in r.KEYS}
    return {"seeds": seeds,
            "mean": {k: float(np.mean([v[k] for v in seeds.values()])) for k in r.KEYS},
            "seed_std_ddof1": {k: float(np.std([v[k] for v in seeds.values()], ddof=1)) for k in r.KEYS},
            "fold_std_ddof0_descriptive": {k: float(np.std([row["metrics"][k] for row in rows])) for k in r.KEYS}}


def fold_data(method, seed, fold, state, aux=False):
    if method == "C0":
        old = state["audit"]["baseline"][f"{seed}/{fold}"]
        saved = {**old, "oof": {k: old[k] for k in ("indices", "labels", "pred", "prob_asd", "ids", "sites", "metrics")}}
        saved["frozen"] = state.get("baseline_frozen", {}).get(f"{seed}/{fold}")
        return saved
    if not r.verify_result(method, seed, fold, state, require_aux=aux):
        raise ValueError(f"Missing validated {method}/{seed}/{fold}")
    return r.load_torch(r.result_path(method, seed, fold), mmap=True)


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    result = float(spearmanr(a, b).statistic)
    return result if np.isfinite(result) else None


def pooled_oof(rows):
    result = {}
    for seed in r.config()["seeds"]:
        selected = [row for row in rows if row["seed"] == seed]
        idx = np.concatenate([np.asarray(row["indices"]) for row in selected])
        if not np.array_equal(np.sort(idx), np.arange(871)):
            raise ValueError("OOF coverage is not exactly once per subject")
        y = np.concatenate([np.asarray(row["labels"]) for row in selected])
        pred = np.concatenate([np.asarray(row["pred"]) for row in selected])
        p = np.concatenate([np.asarray(row["prob_asd"]) for row in selected])
        result[str(seed)] = r.binary_metrics(y, pred, p)
    return result


def stratified_oof(rows, metadata):
    partitions = {f"site_{name}": np.where(metadata["site"] == name)[0] for name in sorted(set(metadata["site"]))}
    partitions.update({f"sex_{name}": np.where(metadata["sex"] == name)[0] for name in r.config()["sex_categories"]})
    for low, high in zip(r.config()["age_bins"][:-1], r.config()["age_bins"][1:]):
        partitions[f"age_{low}_{high}"] = np.where((metadata["age"] >= low) & (metadata["age"] < high))[0]
    answer = {}
    for name, indices in partitions.items():
        seed_metrics = {}
        for seed in r.config()["seeds"]:
            selected = [row for row in rows if row["seed"] == seed]
            idx = np.concatenate([np.asarray(row["indices"]) for row in selected])
            ordered = np.argsort(idx)
            y = np.concatenate([np.asarray(row["labels"]) for row in selected])[ordered][indices]
            p = np.concatenate([np.asarray(row["prob_asd"]) for row in selected])[ordered][indices]
            pred = np.concatenate([np.asarray(row["pred"]) for row in selected])[ordered][indices]
            if not len(y):
                continue
            seed_metrics[str(seed)] = r.binary_metrics(y, pred, p)
        answer[name] = {"subjects_per_seed": len(indices), "seeds": seed_metrics,
                        "mean": {k: float(np.mean([m[k] for m in seed_metrics.values() if m[k] is not None]))
                                 if any(m[k] is not None for m in seed_metrics.values()) else None for k in r.KEYS}}
    sites = [v["mean"] for k, v in answer.items() if k.startswith("site_")]
    return {"strata": answer, "site_macro_std_ddof0_descriptive": {
        k: float(np.std([m[k] for m in sites if m[k] is not None])) if any(m[k] is not None for m in sites) else None for k in r.KEYS},
        "warning": "Small or single-class strata have unstable or undefined metrics; this is within-site CV, not held-out-site generalization"}


def summarize_group(method, state, auxiliary=False):
    rows, repeat, oof, counts, seconds, gate, residual, instability, draw_rows = [], [], [], [], [], [], [], [], [[] for _ in range(5)]
    ctx = r.load_torch(r.OLD / "artifacts/context.pt")
    metadata = r.raw_metadata(ctx)
    repeated_oof, frozen_gates, frozen_residuals, shuffle = [], [], [], []
    fingerprints, gpu = {}, {"0": 0, "1": 0}
    for seed in r.config()["seeds"]:
        for fold in range(10):
            saved = fold_data(method, seed, fold, state, auxiliary)
            row = {"seed": seed, "fold": fold, "metrics": saved["oof"]["metrics"],
                   "best_epoch": saved["best_epoch"], "epochs_run": saved["epochs_run"],
                   "seconds": saved["classifier_seconds"]}
            rows.append(row)
            oof.append({"seed": seed, **saved["oof"]})
            counts.append(saved["parameters"]); seconds.append(saved["classifier_seconds"])
            if method != "C0":
                gpu[str(saved["physical_gpu"])] += 1
                fingerprints[f"{seed}/{fold}"] = saved["initial_pcci_sha256"]
                gate.append(saved["gate"]["test"]["gate"]["mean"])
                residual.append(saved["gate"]["test"]["residual_l2"]["mean"])
                if method == "C6":
                    shuffle.append({"seed": seed, "fold": fold,
                                    "moved_fraction": saved["encoding"]["shuffle_moved_fraction"],
                                    "changed_age_sex_fraction": saved["encoding"]["shuffle_changed_age_sex_fraction"],
                                    "by_site": saved["encoding"]["shuffle_by_site"]})
            frozen = saved.get("frozen")
            if auxiliary:
                if frozen is None:
                    raise ValueError("Frozen inference incomplete")
                repeat.append({"seed": seed, "fold": fold, "metrics": frozen["mean_probability_metrics"]})
                avg = np.asarray(frozen["test_prob_asd"], dtype=np.float32).mean(0)
                repeated_oof.append({"seed": seed, "indices": saved["oof"]["indices"], "labels": saved["oof"]["labels"],
                                     "pred": np.where(avg >= .5, 0, 1), "prob_asd": avg})
                if method != "C0":
                    idx = np.asarray(saved["oof"]["indices"])
                    frozen_gates.append(np.asarray(frozen["gate"])[:, idx].ravel())
                    frozen_residuals.append(np.asarray(frozen["residual_l2"])[:, idx].ravel())
                instability.append(frozen["test_prediction_instability_fraction"])
                for d in range(5):
                    draw_rows[d].append({"seed": seed, "fold": fold, "metrics": frozen["draw_metrics"][d]})
    result = {"method": method, "name": r.config()["methods"][method]["name"], "completed": 30,
              "formal": packed_metrics(rows), "folds": rows, "pooled_oof_metrics_not_main_table": pooled_oof(oof),
              "fold0": {str(seed): next(v for v in rows if v["seed"] == seed and v["fold"] == 0) for seed in r.config()["seeds"]},
              "classifier_seconds": {"sum": sum(seconds), "per_fit": r.distribution(seconds)},
              "parameters": {k: r.distribution([x[k] for x in counts if k in x and isinstance(x[k], (int, float))])
                             for k in sorted(set().union(*(x.keys() for x in counts)))
                             if any(k in x and isinstance(x[k], (int, float)) for x in counts)},
              "parameters_per_fold": counts,
              "physical_gpu_fits": gpu, "initial_pcci_fingerprints": fingerprints,
              "formal_test_gate_mean_across_folds": r.distribution(gate),
              "formal_test_residual_l2_mean_across_folds": r.distribution(residual)}
    result["formal_stratified_oof"] = stratified_oof(oof, metadata)
    if shuffle:
        result["C6_shuffle"] = {"folds": shuffle,
                               "moved_fraction": r.distribution([v["moved_fraction"] for v in shuffle]),
                               "actual_changed_age_sex_fraction": r.distribution([v["changed_age_sex_fraction"] for v in shuffle]),
                               "joint_age_sex_and_sites_preserved": True}
    if auxiliary:
        result["frozen_probability_mean"] = packed_metrics(repeat)
        result["frozen_folds"] = repeat
        result["frozen_draws"] = [packed_metrics(v) for v in draw_rows]
        result["frozen_test_instability"] = r.distribution(instability)
        result["frozen_metric_draw_distribution"] = {k: r.distribution([packed_metrics(v)["mean"][k] for v in draw_rows]) for k in r.KEYS}
        result["frozen_stratified_oof"] = stratified_oof(repeated_oof, metadata)
        result["frozen_test_gate_distribution"] = r.distribution(np.concatenate(frozen_gates) if frozen_gates else [])
        result["frozen_test_residual_l2_distribution"] = r.distribution(np.concatenate(frozen_residuals) if frozen_residuals else [])
    return result


def paired(a, b, scope):
    field = "folds" if scope == "formal" else "frozen_folds"
    base = {(v["seed"], v["fold"]): v for v in b[field]}
    differences = [{"seed": v["seed"], "fold": v["fold"],
                    "metrics": {k: v["metrics"][k] - base[v["seed"], v["fold"]]["metrics"][k] for k in r.KEYS}}
                   for v in a[field]]
    packed = packed_metrics(differences)
    return {"a": a["method"], "b": b["method"], "scope": scope, **packed,
            "improved_folds": {k: sum(v["metrics"][k] > 1e-12 for v in differences) for k in r.KEYS},
            "tied_folds": {k: sum(abs(v["metrics"][k]) <= 1e-12 for v in differences) for k in r.KEYS},
            "positive_seeds": {k: sum(v[k] > 0 for v in packed["seeds"].values()) for k in r.KEYS},
            "paired_folds": differences, "independence_warning": "Three repeated CV seeds share the same 871 subjects; 30 folds are not independent clinical cohorts"}


def comparisons(groups, auxiliary=False):
    pairs = list(r.config()["comparisons"])
    # The task also explicitly names C1 as C3's direct capacity control.
    if ["C3", "C1"] not in pairs:
        pairs.append(["C3", "C1"])
    for method in ("C1", "C2", "C4", "C5", "C6"):
        if [method, "C0"] not in pairs:
            pairs.append([method, "C0"])
    answer = {}
    for a, b in pairs:
        if a in groups and b in groups:
            answer[f"{a}-{b}"] = {"formal": paired(groups[a], groups[b], "formal")}
            if auxiliary and "frozen_folds" in groups[a] and "frozen_folds" in groups[b]:
                answer[f"{a}-{b}"]["frozen_probability_mean"] = paired(groups[a], groups[b], "frozen_probability_mean")
    return answer


def mechanism(method, state):
    ctx = r.load_torch(r.OLD / "artifacts/context.pt")
    metadata = r.raw_metadata(ctx)
    per_seed, all_records = {}, []
    for seed in r.config()["seeds"]:
        fields = {k: np.zeros((5, 871), dtype=np.float64) for k in ("true", "shuffled", "reference", "off", "gate", "residual")}
        coverage = np.zeros(871, dtype=int)
        for fold in range(10):
            saved = fold_data(method, seed, fold, state, True)
            test, frozen = np.asarray(saved["oof"]["indices"]), saved["frozen"]
            coverage[test] += 1
            for key in ("true", "shuffled", "reference", "off"):
                fields[key][:, test] = frozen["counterfactual_prob_asd"][key][:, test]
            fields["gate"][:, test] = frozen["gate"][:, test]
            fields["residual"][:, test] = frozen["residual_l2"][:, test]
        if not np.all(coverage == 1):
            raise ValueError("Counterfactual OOF coverage mismatch")
        y = np.asarray(ctx["labels"]) == 0
        def ce(p):
            correct = np.where(y[None, :], p, 1 - p)
            return -np.log(np.clip(correct, 1e-7, 1 - 1e-7))
        benefit = ce(fields["off"]) - ce(fields["true"])
        pair_advantage = ce(fields["shuffled"]) - ce(fields["true"])
        ref_advantage = ce(fields["reference"]) - ce(fields["true"])
        mean_probability = {name: fields[name].astype(np.float32).mean(0) for name in ("true", "shuffled", "reference", "off")}
        historical_prediction = np.zeros(871, dtype=int)
        for fold in range(10):
            baseline = state["audit"]["baseline"][f"{seed}/{fold}"]
            historical_prediction[np.asarray(baseline["indices"])] = baseline["pred"]
        for i in range(871):
            all_records.append({"seed": seed, "index": i, "benefit": float(benefit[:, i].mean()),
                "benefit_positive_draws": int(np.sum(benefit[:, i] > 0)),
                "pair_advantage": float(pair_advantage[:, i].mean()), "reference_advantage": float(ref_advantage[:, i].mean()),
                "delta_probability": float((fields["true"][:, i] - fields["off"][:, i]).mean()),
                "absolute_delta_probability": float(np.abs(fields["true"][:, i] - fields["off"][:, i]).mean()),
                "pair_perturbation": float(np.abs(fields["true"][:, i] - fields["shuffled"][:, i]).mean()),
                "gate": float(fields["gate"][:, i].mean()), "residual": float(fields["residual"][:, i].mean()),
                "off_probability": float(mean_probability["off"][i]),
                "base_correct": bool((mean_probability["off"][i] >= .5) == y[i]),
                "historical_C0_correct": bool(historical_prediction[i] == ctx["labels"][i]),
                "flip_fraction": float(np.mean((fields["true"][:, i] >= .5) != (fields["off"][:, i] >= .5))),
                "true_probability": float(mean_probability["true"][i]),
                "shuffled_probability": float(mean_probability["shuffled"][i]),
                "reference_probability": float(mean_probability["reference"][i])})
        per_seed[str(seed)] = {"mean_delta_ce_true_vs_off": float(benefit.mean()),
                              "mean_delta_ce_true_vs_shuffled": float(pair_advantage.mean()),
                              "mean_delta_ce_true_vs_reference": float(ref_advantage.mean()),
                              "counterfactual_probability_mean_metrics": {k: r.binary_metrics(ctx["labels"], np.where(mean_probability[k] >= .5, 0, 1), mean_probability[k]) for k in ("true", "shuffled", "reference", "off")}}
    def describe(records):
        if not records:
            return {"n": 0}
        val = lambda key: np.array([row[key] for row in records])
        return {"n": len(records), "subjects_unique": len(set(row["index"] for row in records)),
                **{key: r.distribution(val(key)) for key in ("benefit", "pair_advantage", "reference_advantage", "gate", "residual", "delta_probability", "absolute_delta_probability", "pair_perturbation")},
                "benefit_positive_fraction": float(np.mean(val("benefit") > 0)),
                "stable_benefit_4_of_5_draws_fraction": float(np.mean(val("benefit_positive_draws") >= 4)),
                "stable_harm_4_of_5_draws_fraction": float(np.mean(val("benefit_positive_draws") <= 1)),
                "mean_label_flip_fraction": float(val("flip_fraction").mean()),
                "gate_vs_benefit_spearman": correlation(val("gate"), val("benefit")),
                "gate_vs_abs_delta_probability_spearman": correlation(val("gate"), val("absolute_delta_probability")),
                "gate_vs_pair_perturbation_spearman": correlation(val("gate"), val("pair_perturbation")),
                "gate_vs_benefit_spearman_per_seed": {str(seed): correlation(
                    [v["gate"] for v in records if v["seed"] == seed],
                    [v["benefit"] for v in records if v["seed"] == seed]) for seed in r.config()["seeds"]},
                "flip_weight_near_off_boundary_005": float(np.sum(val("flip_fraction") * (np.abs(val("off_probability") - .5) < .05)) / max(np.sum(val("flip_fraction")), 1e-12)),
                "probability_mean_metrics_by_seed": {str(seed): {condition: r.binary_metrics(
                    np.array([ctx["labels"][v["index"]] for v in records if v["seed"] == seed]),
                    np.where(np.array([v[f"{condition}_probability"] for v in records if v["seed"] == seed]) >= .5, 0, 1),
                    np.array([v[f"{condition}_probability"] for v in records if v["seed"] == seed]))
                    for condition in ("true", "shuffled", "reference", "off")}
                    for seed in r.config()["seeds"] if any(v["seed"] == seed for v in records)}}
    partitions = {"ASD": [v for v in all_records if ctx["labels"][v["index"]] == 0],
                  "HC": [v for v in all_records if ctx["labels"][v["index"]] == 1],
                  "base_correct": [v for v in all_records if v["base_correct"]],
                  "base_wrong": [v for v in all_records if not v["base_correct"]],
                  "historical_C0_correct": [v for v in all_records if v["historical_C0_correct"]],
                  "historical_C0_wrong": [v for v in all_records if not v["historical_C0_correct"]],
                  "benefit_positive": [v for v in all_records if v["benefit"] > 0],
                  "benefit_nonpositive": [v for v in all_records if v["benefit"] <= 0]}
    for sex in r.config()["sex_categories"]:
        partitions[f"sex_{sex}"] = [v for v in all_records if metadata["sex"][v["index"]] == sex]
    bins = r.config()["age_bins"]
    for low, high in zip(bins[:-1], bins[1:]):
        partitions[f"age_{low}_{high}"] = [v for v in all_records if low <= metadata["age"][v["index"]] < high]
    for site in sorted(set(metadata["site"])):
        partitions[f"site_{site}"] = [v for v in all_records if metadata["site"][v["index"]] == site]
    stable_across_seeds = []
    for i in range(871):
        selected = [v for v in all_records if v["index"] == i]
        stable_across_seeds.append(sum(v["benefit"] > 0 for v in selected))
    return {"method": method, "scope": "test-only cross-fitted patients; five common RNG draws; no labels used for intervention construction",
            "definition_delta_ce": "CE(current trained core with residual off) - CE(true condition); positive is beneficial",
            "off_is_not_C0": True, "per_seed": per_seed, "overall": describe(all_records),
            "strata": {k: describe(v) for k, v in partitions.items()},
            "benefit_positive_in_all_three_seeds_fraction": float(np.mean(np.array(stable_across_seeds) == 3)),
            "warning": "Counterfactual associations within one trained model do not establish causal benefit; repeated seeds are not independent patients"}


def selectivity_comparison(state):
    records = []
    ctx = r.load_torch(r.OLD / "artifacts/context.pt")
    for seed in r.config()["seeds"]:
        for fold in range(10):
            c3, c5 = [fold_data(method, seed, fold, state, True) for method in ("C3", "C5")]
            idx = np.asarray(c3["oof"]["indices"])
            y = (ctx["labels"][idx] == 0)[None, :]
            values = {}
            for method, saved in (("C3", c3), ("C5", c5)):
                frozen = saved["frozen"]
                p = frozen["counterfactual_prob_asd"]
                true, off = p["true"][:, idx], p["off"][:, idx]
                ce = lambda v: -np.log(np.clip(np.where(y, v, 1 - v), 1e-7, 1 - 1e-7))
                values[method] = {"benefit": (ce(off) - ce(true)).mean(0),
                                  "impact": np.abs(true - off).mean(0),
                                  "gate": frozen["gate"][:, idx].mean(0),
                                  "residual": frozen["residual_l2"][:, idx].mean(0)}
            for i in range(len(idx)):
                records.append({"seed": seed, "index": int(idx[i]), "C3_harmful": bool(values["C3"]["benefit"][i] <= 0),
                                **{f"{method}_{key}": float(value[i]) for method, v in values.items() for key, value in v.items()}})
    strata = {}
    for name, selected in (("C3_harmful_same_patients", [v for v in records if v["C3_harmful"]]),
                           ("C3_beneficial_same_patients", [v for v in records if not v["C3_harmful"]]), ("all", records)):
        strata[name] = {"n_seed_subjects": len(selected), **{method: {key: r.distribution([v[f"{method}_{key}"] for v in selected])
                                                                   for key in ("benefit", "impact", "gate", "residual")} for method in ("C3", "C5")}}
    return {"definition": "Compare C3 and C5 on identical patients; harm grouping defined post hoc by frozen C3 CE benefit, not used for optimization or model selection",
            "strata": strata, "warning": "Descriptive matched sensitivity, not a learned reliability guarantee or causal effect"}


def final_checks(state):
    checked = 0
    for seed in r.config()["seeds"]:
        for fold in range(10):
            expected = [r.private_seed(seed, fold, f"PCCI-repeat-{base}") % (2**32) for base in r.config()["repeat_seed_bases"]]
            for method in r.config()["methods"]:
                saved = fold_data(method, seed, fold, state, True)
                if method != "C0" and saved["physical_gpu"] != fold % 2:
                    exception = state.get("scheduling", {}).get("affinity_exceptions", {}).get(f"train/{method}/{seed}/{fold}", {})
                    if exception.get("actual_gpu") != saved["physical_gpu"]:
                        raise RuntimeError("Physical PPU assignment changed without a recorded scheduling exception")
                frozen = saved["frozen"]
                if frozen["draw_seeds"] != expected:
                    raise RuntimeError("Common frozen RNG pairing failed")
                p = np.asarray(frozen["test_prob_asd"], dtype=np.float32)
                y = saved["oof"]["labels"]
                if p.shape != (5, len(y)) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise RuntimeError("Frozen OOF probability validation failed")
                avg = p.mean(0)
                metrics = r.binary_metrics(y, np.where(avg >= .5, 0, 1), avg)
                if any(abs(metrics[k] - frozen["mean_probability_metrics"][k]) > 1e-12 for k in r.KEYS):
                    raise RuntimeError("Frozen probability-mean metrics failed independent recomputation")
                predictions = np.asarray(frozen["test_pred"])
                instability = float(np.mean(np.any(predictions != predictions[0], axis=0)))
                if abs(instability - frozen["test_prediction_instability_fraction"]) > 1e-12:
                    raise RuntimeError("Frozen instability statistic mismatch")
                if method in ("C2", "C3", "C4", "C5"):
                    if not np.array_equal(frozen["counterfactual_prob_asd"]["true"][:, np.asarray(saved["oof"]["indices"])], p):
                        raise RuntimeError("Counterfactual true condition did not reuse the original frozen realization")
                checked += 1
    state["frozen_pairing_verification"] = {"passed": True, "checkpoints": checked, "draws_per_checkpoint": 5,
        "mean_probability_metrics_independently_recomputed": True, "source_probability_dtype": "float32",
        "actual_seed_formula": "int(SHA256(seed/fold/PCCI-repeat-base).hexdigest()[:15],16) % 2**32; same for all seven methods"}
    state["selectivity_comparison"] = selectivity_comparison(state)
    state["analysis_provenance"] = {"time_utc": r.utc(),
        "sha256": {name: r.sha_file(r.BASE / name) for name in ("run.py", "pcci.py", "launch.py", "analyze.py", "config.json")},
        "scope": "Predefined fixed post-hoc strata, inherited-C0 correctness strata, matched C3/C5 sensitivity, independent frozen OOF recomputation; training and frozen outputs unchanged"}


def number(value, scale=100):
    return "NA" if value is None else f"{scale * value:.2f}"


def main_table(groups, scope):
    lines = ["| Method | ACC | BA | AUC | SEN-ASD | SPE-HC | F1 | delta ACC vs C0 | delta AUC vs C0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    if "C0" not in groups:
        return lines
    base = groups["C0"][scope]["mean"]
    for method, group in sorted(groups.items()):
        metrics = group[scope]
        values = [f"{number(metrics['mean'][k])} +/- {number(metrics['seed_std_ddof1'][k])}" for k in r.KEYS]
        lines.append("| " + " | ".join([method + " " + group["name"], *values,
                     number(metrics["mean"]["ACC"] - base["ACC"]), number(metrics["mean"]["AUC"] - base["AUC"])]) + " |")
    return lines


def research_answers(state):
    c, m = state["comparisons"], state.get("mechanisms", {})
    def evidence(pair, scope="formal"):
        v = c[pair][scope]
        return f"ACC {number(v['mean']['ACC'])} pp / AUC {number(v['mean']['AUC'])} pp; ACC正向{v['positive_seeds']['ACC']}/3 seed、{v['improved_folds']['ACC']}/30 fold"
    strong = lambda pair: all(c[pair]["frozen_probability_mean"]["positive_seeds"][k] == 3 and c[pair]["frozen_probability_mean"]["mean"][k] > 0 for k in ("ACC", "AUC"))
    main_support = strong("C3-C0")
    interaction_support = strong("C3-C2")
    pair_support = strong("C3-C6") and all(v["mean_delta_ce_true_vs_shuffled"] > 0 for v in m["C3"]["per_seed"].values())
    answers = [
        "Q1 原融合之外的增量：C3-C0正式单次 " + evidence("C3-C0") + "；冻结5次概率平均 " + evidence("C3-C0", "frozen_probability_mean") + ("。在当前协议内有三seed同向支持。" if main_support else "。未获ACC/AUC跨三seed一致的增量支持。") + "C3对C1容量对照：" + evidence("C3-C1", "frozen_probability_mean") + "。这是沿用原始泄漏协议下的配对结果，不是无泄漏临床泛化分数。",
        "Q2 加性条件是否已足够：C2-C1 " + evidence("C2-C1") + ("。当前协议支持加性相对固定条件的收益，但是否足够还取决于C3-C2。" if strong("C2-C1") else "。加性条件相对固定条件的增益本身尚未跨seed稳定成立，不能将乘性不显著直接解释为加性已足够。") + "C2仍保留GELU及原分类器非线性，不能称为全线性模型。",
        "Q3 乘性交互：C3-C2 " + evidence("C3-C2") + "；冻结5次概率平均 " + evidence("C3-C2", "frozen_probability_mean") + ("。当前协议支持乘性相对加性增量，但仍需外部验证。" if interaction_support else "。当前结果未稳定证明动态乘性优于加性；不能凭最高单次分数主张优越。"),
        "Q4 站点条件：C4-C3 " + evidence("C4-C3") + "。站点内随机CV的提升不能证明跨站点可靠，需独立LOSO验证。",
        "Q5 正确患者配对：C3-C6 " + evidence("C3-C6") + "；冻结5次概率平均 " + evidence("C3-C6", "frozen_probability_mean") + f"；C3固定当前模型内真实vs错配的平均CE收益 {m['C3']['overall']['pair_advantage']['mean']:.6f}。" + ("两种证据共同支持配对相关性，但不是因果结论。" if pair_support else "两种证据尚未共同达到跨seed稳定支持，不能断言真实患者配对是当前性能的关键。"),
        "Q6 门控选择性：" + "; ".join(f"{method} gate均值{m[method]['overall']['gate']['mean']:.4f}, SD{m[method]['overall']['gate']['std']:.4f}, gate-实际收益Spearman={m[method]['overall']['gate_vs_benefit_spearman']}" for method in ("C3", "C5")) + "。门控与收益的事后相关及同患者干预变化不足以证明可靠性识别；gate整体变小或接近常数本身尤其不是选择性证据。",
        "Q7 RNG稳健性：正式单次及固定5次概率平均见两张主表，逐draw跨度与标签不稳定性另列；不选取最好draw。" + ("两项主比较的ACC/AUC均在3 seed同向为正。" if strong("C3-C0") and strong("C3-C2") else "至少一项主比较未达到ACC/AUC在3 seed均同向为正的严格一致性，应判为证据有限。"),
        "Q8 研究决策：" + ("满足预定的保守延续条件，但仍需无泄漏外部验证、有效容量匹配和站点外验证；FiLM本身不构成创新。" if strong("C3-C0") and strong("C3-C2") and strong("C3-C6") and m["C3"]["overall"]["pair_advantage"]["mean"] > 0 else "目前不足以将PCCI作为已成立的MICCAI创新模块；仅当真实配对收益、优于加性/容量对照和跨seed稳健性共同成立才值得进一步扩展，不建议据单项最高ACC包装成功。")]
    flips = [f"{pair}/{metric}" for pair in ("C3-C0", "C3-C2") for metric in ("ACC", "AUC")
             if np.sign(c[pair]["formal"]["mean"][metric]) != np.sign(c[pair]["frozen_probability_mean"]["mean"][metric])]
    answers[6] = "Q7 RNG稳健性：" + ("正式单次与5次概率平均的主比较方向不一致项为 " + ", ".join(flips) + "，应判为推理敏感。" if flips else "两项主比较的ACC/AUC均值方向在正式单次和5次概率平均间一致。") + "方向一致也不等于跨seed稳定获益，逐draw跨度与预测不稳定性见表；不选择最好draw。"
    return answers


def write_report(state):
    groups = state.get("groups", {})
    final = state.get("phase") == "complete"
    lines = ["# Experiment 3: PCCI 正式研究报告", "", f"更新时间：{r.utc()}；状态：{state.get('phase')}。", "",
             "## 审计与协议", "",
             "历史C0为已核验的3 seed x 10 fold；C1-C6各新增30折，共180折。原始模型、标签ASD=0/HC=1、跨折RFE覆盖行为、VAE、图构造、dropout和早停不改。保留这些已知问题仅为公平源代码复现，绝不将分数解释为无泄漏临床性能。", "",
             "每个新组均使用同一24维条件编码和相同模块参数量，主干、VAE、RFE、折分、训练起点RNG严格配对。训练CE+原图损失；仅C5增加0.005乘训练人群平均gate；验证一律只用原CE+图损失。零初始化输出投影；原分类器每次主干前向仅调用一次，BatchNorm不额外更新。", "",
             "30折无残差及零初始化等价检查在独立PPU诊断进程中使用确定性算术；正式训练不启用该额外确定性模式，保留历史随机后端及原始随机VAE/dropout。历史C0与本轮调度负载不同，无法保证长程训练逐bit重放；数值等价不等于新增可训练分支的优化轨迹相同。", "",
             "主表：每seed先取10折均值，再取3个seed均值及样本标准差(ddof=1)，单位%。30个重复CV fold不是30个独立临床队列。不作伪独立显著性检验。", "",
             "## 正式单次测试", "", *main_table(groups, "formal"), "",
             "## 逐 Seed 与耗时", "", "| Method | Seed | ACC | BA | AUC | SEN | SPE | F1 |", "|---|---|---|---|---|---|---|---|"]
    for method, group in sorted(groups.items()):
        for seed, metrics in group["formal"]["seeds"].items():
            lines.append("| " + " | ".join([method, seed, *[number(metrics[k]) for k in r.KEYS]]) + " |")
    lines += ["", "| Method | 分类器累计秒 | 每折平均秒 | Core参数 | 新参数 | 新参数有grad | 新参数实际非零grad(抽样并集) | Gate均值 | 残差L2均值 |", "|---|---|---|---|---|---|---|---|---|"]
    for method, group in sorted(groups.items()):
        p = group["parameters"]
        value = lambda k: f"{p[k]['mean']:.0f}" if k in p else "NA"
        gate, residual = group["formal_test_gate_mean_across_folds"], group["formal_test_residual_l2_mean_across_folds"]
        lines.append(f"| {method} | {group['classifier_seconds']['sum']:.2f} | {group['classifier_seconds']['per_fit']['mean']:.2f} | {value('core_total')} | {value('new_total')} | {value('new_receiving_gradients')} | {value('new_nonzero_scalar_gradients_sampled_union')} | {gate.get('mean', 0):.4f} | {residual.get('mean', 0):.5f} |")
    lines += ["", "新增参数总数相同不意味着有效容量相同：C1输入固定、C2的gamma不含个体差异、无site组的site输入列恒零。梯度非零计数为每25 epoch抽样并集下界，不把零初始化首步上游梯度为零误判为断路。C0引用历史耗时并未在本轮重训，新增组VAE预训练耗时为0；并发负载和写盘方式不同，不把时间差解释为算法加速。", "",
              "## Fold 0", "", "| Method | Seed | ACC | BA | AUC | SEN | SPE | F1 |", "|---|---|---|---|---|---|---|---|"]
    for method, group in sorted(groups.items()):
        for seed, v in group["fold0"].items():
            lines.append("| " + " | ".join([method, seed, *[number(v["metrics"][k]) for k in r.KEYS]]) + " |")
    lines += ["",
              "## 配对比较", "", "| Contrast | Scope | delta ACC pp | delta AUC pp | ACC正向seed | ACC改善fold |", "|---|---|---|---|---|---|"]
    for pair, scopes in state.get("comparisons", {}).items():
        for scope, v in scopes.items():
            lines.append(f"| {pair} | {scope} | {number(v['mean']['ACC'])} | {number(v['mean']['AUC'])} | {v['positive_seeds']['ACC']}/3 | {v['improved_folds']['ACC']}/30 |")
    if final:
        lines += ["", "## 固定五次冻结推理", "", *main_table(groups, "frozen_probability_mean"), "",
                  "所有组对相同seed/fold使用相同5个预定RNG，不选最优draw，不重训、不重新选checkpoint。表中按5次概率平均再重算指标；不是取5次指标最大值。原始单次正式测试与此辅助稳健性分析分开。", "",
                  "| Method | ACC跨draw最小/最大% | AUC跨draw最小/最大% | 发生标签变化的测试被试比例% |", "|---|---|---|---|"]
        for method, group in sorted(groups.items()):
            d = group["frozen_metric_draw_distribution"]
            lines.append(f"| {method} | {number(d['ACC']['min'])}/{number(d['ACC']['max'])} | {number(d['AUC']['min'])}/{number(d['AUC']['max'])} | {number(group['frozen_test_instability']['mean'])} |")
        lines += ["", "## 冻结反事实机制", "",
                  "每个checkpoint、每个RNG只运行一次原始编码/图/融合，复用相同imaging和joint表示，仅重算新增残差及eval分类器。off指当前新模型训练后的主干，不是C0。真实、站点内联合错配、训练参考、关闭残差全部在同一编码实现上比较，不受额外VAE/dropout抽样干扰。", "",
                  "正的delta CE表示真实条件相对当前主干off降低诊断CE；配对收益表示CE(shuffled)-CE(true)。概率在计算CE时截到[1e-7,1-1e-7]避免log(0)。分层均为测试OOF，完整逐seed/年龄/性别/站点/当前off主干正确与错误/历史C0正确与错误统计见summary.json。", "",
                  "| Method | delta CE true-off | delta CE true-shuffle | delta CE true-reference | 受益比例% | 4/5 draw稳定受益% | Gate-benefit Spearman | Gate SD |", "|---|---|---|---|---|---|---|---|"]
        for method, mechanism_value in state["mechanisms"].items():
            v = mechanism_value["overall"]
            lines.append(f"| {method} | {v['benefit']['mean']:.6f} | {v['pair_advantage']['mean']:.6f} | {v['reference_advantage']['mean']:.6f} | {number(v['benefit_positive_fraction'])} | {number(v['stable_benefit_4_of_5_draws_fraction'])} | {v['gate_vs_benefit_spearman']} | {v['gate']['std']:.4f} |")
        v = state["mechanisms"]["C3"]["overall"]
        stable = state["mechanisms"]["C3"]["benefit_positive_in_all_three_seeds_fraction"]
        site_benefits = [v["benefit"]["mean"] for name, v in state["mechanisms"]["C3"]["strata"].items() if name.startswith("site_") and v["n"]]
        lines += ["", f"C3每draw发生标签翻转的测试被试比例均值 {number(v['mean_label_flip_fraction'])}%；这些翻转中off概率位于0.45-0.55的比例 {number(v['flip_weight_near_off_boundary_005'])}%。同一患者在三个seed中平均CE均受益的比例 {number(stable)}%；站点平均CE收益为正 {sum(x>0 for x in site_benefits)}/{len(site_benefits)}，为负 {sum(x<0 for x in site_benefits)}/{len(site_benefits)}。", ""]
        if "frozen_test_gate_distribution" in groups["C3"]:
            lines += ["| Method | Gate mean | SD | P10 | P50 | P90 | >0.5比例% | 残差L2 mean |", "|---|---|---|---|---|---|---|---|"]
            for method, group in sorted(groups.items()):
                gate = group.get("frozen_test_gate_distribution", {})
                if not gate.get("n"):
                    continue
                residual = group["frozen_test_residual_l2_distribution"]
                lines.append(f"| {method} | {gate['mean']:.4f} | {gate['std']:.4f} | {gate['p10']:.4f} | {gate['p50']:.4f} | {gate['p90']:.4f} | {number(gate['above_half_fraction'])} | {residual['mean']:.6f} |")
        if "C6_shuffle" in groups["C6"]:
            shuffle = groups["C6"]["C6_shuffle"]
            lines += ["", f"C6置换索引变化比例均值 {number(shuffle['moved_fraction']['mean'])}%；实际年龄/性别值变化比例均值 {number(shuffle['actual_changed_age_sex_fraction']['mean'])}%。联合字段与站点分布保持；重复年龄/性别组合可导致置换索引已变而字段数值不变。逐站点比例见summary。"]
        if "selectivity_comparison" in state:
            lines += ["", "### C5 对同一患者的选择性", "", "| C3定义的患者分组 | N(seed-subject) | C3 gate | C5 gate | C3绝对概率扰动 | C5绝对概率扰动 | C3 CE收益 | C5 CE收益 |", "|---|---|---|---|---|---|---|---|"]
            for name, v in state["selectivity_comparison"]["strata"].items():
                if not v["n_seed_subjects"]:
                    lines.append(f"| {name} | 0 | NA | NA | NA | NA | NA | NA |")
                    continue
                lines.append(f"| {name} | {v['n_seed_subjects']} | {v['C3']['gate']['mean']:.4f} | {v['C5']['gate']['mean']:.4f} | {v['C3']['impact']['mean']:.6f} | {v['C5']['impact']['mean']:.6f} | {v['C3']['benefit']['mean']:.6f} | {v['C5']['benefit']['mean']:.6f} |")
            lines += ["", "同患者匹配比较为事后描述，不用于训练、门控输入或调参；gate整体缩小不等于患者级选择性成立。"]
        if "frozen_stratified_oof" in groups["C3"]:
            lines += ["", "### 站点条件对照", "", "| Site | N/seed | C3 ACC | C4 ACC | C3 AUC | C4 AUC | C3真-off CE收益 | C4真-off CE收益 |", "|---|---|---|---|---|---|---|---|"]
            for name, value in groups["C3"]["frozen_stratified_oof"]["strata"].items():
                if not name.startswith("site_"):
                    continue
                other = groups["C4"]["frozen_stratified_oof"]["strata"][name]
                m3, m4 = [state["mechanisms"][method]["strata"][name] for method in ("C3", "C4")]
                lines.append(f"| {name[5:]} | {value['subjects_per_seed']} | {number(value['mean']['ACC'])} | {number(other['mean']['ACC'])} | {number(value['mean']['AUC'])} | {number(other['mean']['AUC'])} | {m3['benefit']['mean']:.6f} | {m4['benefit']['mean']:.6f} |")
            lines += ["", "### 年龄与性别分层", "", "| Method | Stratum | N/seed | 真-off CE收益 | 真-错配CE收益 | 稳定受益比例% |", "|---|---|---|---|---|---|"]
            for method, value in state["mechanisms"].items():
                for name, v in value["strata"].items():
                    if name.startswith(("age_", "sex_")) and v["n"]:
                        lines.append(f"| {method} | {name} | {v['n']//3} | {v['benefit']['mean']:.6f} | {v['pair_advantage']['mean']:.6f} | {number(v['stable_benefit_4_of_5_draws_fraction'])} |")
        lines += ["", "## 八个研究问题", "", *[answer + "\n" for answer in research_answers(state)], "",
                  "## 限制与解释边界", "",
                  "只有3个重复CV seed，同871人，不能当独立临床重复。历史RFE存在跨折标签泄漏，保留作源码公平控制不代表临床无泄漏。原始图在训练构造时可使用训练标签，测试标签仅用于最终评分与事后机制分层。C6使用全体无标签表型站点内置换，是已声明的transductive对照。未知站点固定one-hot槽；本实验不是跨站点测试。Frozen概率平均改变推理估计器，不是替代正式结果。门控-收益相关不是因果证据。"]
    lines += ["", "## 论文依据与创新边界", "",
              "[MM-GTUNets原文](https://arxiv.org/html/2406.14455v3) Fig.1及3.2-3.5显示表型已参与图构造及融合，Table5中的表型单支路接近机会水平。因此这里只检验现有融合之上的增量条件价值。", "",
              "[PhenoBrain](https://proceedings.mlr.press/v306/meng26f.html)研究表型条件化个体通信核；[SMGD](https://proceedings.mlr.press/v306/shi26ag.html)区分多种跨模态诊断因素；[MGDR](https://papers.miccai.org/miccai-2024/514-Paper2494.html)研究公共/私有图表征；[To Align or Not to Align](https://ojs.aaai.org/index.php/AAAI/article/view/39248)讨论对齐与模态特异信息的权衡。PCCI不把FiLM、门控或条件残差本身主张为新发明，潜在研究价值必须来自可信的增量诊断和患者匹配机制证据。", "",
              f"训练源码指纹：`{state['training_code_sha256']}`。本地/远端基准提交：`{state['audit']['head']}`。只读输入总数：{len(state['audit']['manifest'])}。正确性检查：{len(state.get('checks', {}))}/30。最终只读哈希复核：{state.get('final_readonly_verification', 'pending')}。", ""]
    if "scheduling" in state:
        scheduling = state["scheduling"]
        legacy = scheduling.get("legacy_reservation_reconciliation", {})
        interrupted = sum(value["actual_started_executions"] > 0 for value in legacy.values())
        lines += ["## 任务调度", "",
                  f"C1-C6统一队列，每卡最多{scheduling['maximum_workers_per_gpu']}个独立Worker；显存/CPU/RAM不足时暂停新任务、等待在途任务自然完成，不中断健康Worker。最新指令覆盖prompt旧的实验组串行及每卡1进程要求，训练配置文件保持原样。",
                  f"接管时跳过已校验正式折 {scheduling.get('validated_fits_skipped', 0)}；并发峰值 {scheduling.get('peak_workers', {})}；正式训练每卡完成数 {state.get('physical_gpu_training_totals', 'pending')}。",
                  f"同seed/fold优先原物理卡；为避免空闲卡等待而记录的正式任务分配例外 {len(scheduling.get('affinity_exceptions', {}))} 个，逐项见summary.json。",
                  f"旧长驻Worker接管后退出，影响 {len(legacy)} 个预留任务，其中 {interrupted} 个已有启动记录、{len(legacy)-interrupted} 个尚未启动；退出码未能取得，不能排除旧执行会话清理的影响。只恢复缺失产物，实际执行次数与固定重试预算核对保留在summary/run.log。",
                  f"统一队列至全部分析完成 {scheduling.get('elapsed_seconds', 'pending')} 秒；从首次实验初始化到完成 {scheduling.get('experiment_elapsed_seconds', 'pending')} 秒。", ""]
    if "formal_field_preservation" in state:
        lines += ["正式字段独立核验：`" + json.dumps(state["formal_field_preservation"], ensure_ascii=False) + "`。", ""]
    if "final_acceptance" in state:
        lines += ["最终验收（含只读续跑校验及最终handoff更新）：`" + json.dumps(state["final_acceptance"], ensure_ascii=False) + "`。", ""]
    (r.BASE / "report.md").write_text("\n".join(lines))


def write_handoff(state):
    path = r.ROOT / "handoff.md"
    original = path.read_text() if path.exists() else "# Research Handoff\n"
    start, end = "<!-- PCCI_20261015_START -->", "<!-- PCCI_20261015_END -->"
    lines = [start, "## Experiment 3: PCCI_20261015", "", f"更新时间 {r.utc()}；状态 {state['phase']}。",
             "假设：正确患者年龄/性别经条件乘性交互，可在既有多模态图融合之上增加诊断证据；主要比较C3-C0和C3-C2。", "",
             "所有组保留原始协议；24维训练折编码；同主干/RFE/VAE/折分/起始RNG；C1固定条件、C2固定gamma、C3年龄性别、C4加站点、C5加训练gate正则、C6仅旁路站点内联合错配。", "",
             *main_table(state.get("groups", {}), "formal"), ""]
    for method, group in sorted(state.get("groups", {}).items()):
        p = group["parameters"]
        params = {k: round(v["mean"], 2) for k, v in p.items()}
        lines += [f"### {method}: {group['name']}", f"完成 {group['completed']}/30；参数 {json.dumps(params, ensure_ascii=False)}；分类器累计 {group['classifier_seconds']['sum']:.1f} 秒。",
                  "逐seed六指标：`" + json.dumps(group["formal"]["seeds"], ensure_ascii=False) + "`。",
                  "Gate/残差均值分布：`" + json.dumps({"gate": group["formal_test_gate_mean_across_folds"], "residual": group["formal_test_residual_l2_mean_across_folds"]}) + "`。"]
        for key, value in state.get("comparisons", {}).items():
            if key.startswith(method + "-"):
                lines.append(f"{key} 正式差值：`{json.dumps(value['formal']['mean'])}`；ACC改善fold {value['formal']['improved_folds']['ACC']}/30。")
        if method in state.get("mechanisms", {}):
            v = state["mechanisms"][method]["overall"]
            lines.append(f"真实-错配CE收益 {v['pair_advantage']['mean']:.6f}；真实-off CE收益 {v['benefit']['mean']:.6f}；gate-benefit Spearman {v['gate_vs_benefit_spearman']}。")
        else:
            lines.append("真实/错配及冻结推理机制：待全部组训练完后统一执行，不提前据测试指标改配置。")
        lines.append("")
    if state["phase"] == "complete":
        scheduling = state.get("scheduling", {})
        legacy = scheduling.get("legacy_reservation_reconciliation", {})
        interrupted = sum(value["actual_started_executions"] > 0 for value in legacy.values())
        lines += ["### 最终执行验收",
                  f"C1-C6正式训练180/180；历史C0为30/30复用；冻结推理210个checkpoint各5次；只读输入复核 {state.get('final_readonly_verification', {})}。",
                  f"每卡正式任务完成数 {state.get('physical_gpu_training_totals', {})}；Worker峰值 {scheduling.get('peak_workers', {})}；调度接管后至全部任务完成 {scheduling.get('elapsed_seconds', 'NA')} 秒；实验总墙钟 {scheduling.get('experiment_elapsed_seconds', 'NA')} 秒。",
                  f"模型/训练配置指纹未变；接管前已完成结果未覆盖 {scheduling.get('preexisting_artifacts_unchanged_after_training', 'NA')}；资源调度不强杀在途Worker。接管时旧Worker退出，影响 {interrupted} 个已启动和 {len(legacy)-interrupted} 个未启动任务，已按原锚点恢复；不能将本轮描述为完全零中断。明细保留在summary/run.log。",
                  "### 最终研究判断", *[answer + "\n" for answer in research_answers(state)],
                  "详细报告与全部配对/机制统计：experiments/PCCI_20261015/report.md、summary.json。未push GitHub。"]
        if "formal_field_preservation" in state:
            lines.append("180个产物正式字段指纹与冻结推理期间快照逐项一致（排除追加的frozen和完成状态字段）；模型权重、OOF、RNG和训练记录保持一致。")
        if "resume_verification" in state:
            lines.append("完成后断点续跑验证：`" + json.dumps(state["resume_verification"], ensure_ascii=False) + "`。")
        if "final_acceptance" in state:
            lines.append("全部prompt任务及验证结束后的最终交接更新：`" + json.dumps(state["final_acceptance"], ensure_ascii=False) + "`。")
    else:
        lines += ["当前不作成功判断；不支持假设的证据将按预定对照如实报告。下一步：完成剩余固定配置训练、统一5次冻结推理、同编码反事实分析与八项研究结论。"]
    lines += [end]
    block = "\n".join(lines)
    if start in original:
        if end not in original:
            raise RuntimeError("Incomplete PCCI handoff marker")
        head, tail = original.split(start, 1)
        _, tail = tail.split(end, 1)
        output = head + block + tail
    else:
        output = original.rstrip() + "\n\n" + block + "\n"
    path.write_text(output)


def update(state, methods=None, auxiliary=False):
    groups = state.setdefault("groups", {})
    for method in methods or list(r.config()["methods"]):
        groups[method] = summarize_group(method, state, auxiliary)
    state["comparisons"] = comparisons(groups, auxiliary)
    r.atomic_json(state)
    write_report(state); write_handoff(state)


if __name__ == "__main__":
    state = json.loads((r.BASE / "summary.json").read_text())
    if r.code_hash() != state["training_code_sha256"]:
        raise RuntimeError("Training implementation changed; refuse to analyze mixed provenance")
    if state["phase"] == "complete":
        state["mechanisms"] = {method: mechanism(method, state) for method in ("C2", "C3", "C4", "C5")}
        final_checks(state)
    update(state, auxiliary=state["phase"] == "complete")
