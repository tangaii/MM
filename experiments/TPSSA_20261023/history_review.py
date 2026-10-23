"""Read-only review of prior failure mechanisms, including row-order oracles."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from common import (BASE, ROOT, D4, HISTORY, sha_file, load_torch,
                    historical_module, atomic_json, utc)


def review():
    old = Path("/tmp/PCRPA_20261022")
    manifest = json.loads((old / "manifest.json").read_text())
    frozen = {name: {"actual_sha256": sha_file(old / name),
                     "registered_sha256": manifest["training_code_files_sha256"][name],
                     "equal": sha_file(old / name) == manifest["training_code_files_sha256"][name]}
              for name in ("residual.py", "run.py", "prototype.py")}
    assert all(v["equal"] for v in frozen.values())
    ctx = load_torch(HISTORY / "artifacts/context.pt")
    row_order = []
    for fold in range(10):
        checkpoint = load_torch(D4 / f"runs/D4/s911_f{fold}.pt")
        prep = checkpoint["prototype_preparation"]
        labels = np.asarray(ctx["labels"])[prep["train_indices"]]
        pf = np.asarray(prep["prototype_fold"])
        cat_order = np.concatenate([np.flatnonzero(pf == g) for g in range(3)])
        row_order.append({"fold": fold, "n_train": len(labels),
                          "logit_row_index_changes": int(np.sum(cat_order != np.arange(len(labels)))),
                          "diagnosis_targets_misaligned": int(np.sum(labels != labels[cat_order])),
                          "diagnosis_targets_misaligned_fraction": float(np.mean(labels != labels[cat_order]))})
    # Only synthetic embedding arithmetic. No model instance, network forward,
    # optimizer, data-fitting neural parameters, or historical model inference.
    import torch
    import torch.nn.functional as F
    module = historical_module("tpssa_review_residual", old / "residual.py")
    proto = historical_module("tpssa_review_prototype", old / "prototype.py")
    rg = np.random.default_rng(911)
    n = 120
    y = np.tile([0, 1], n // 2).astype(np.int64)
    meta = {"age": rg.uniform(8, 30, n), "sex": rg.choice([1, 2], n),
            "site": np.repeat(["S0", "S1", "S2", "S3"], n // 4)}
    prep = proto.prepare(y, meta, np.arange(n), 911, 0, module.HISTORICAL_CFG)
    zi = rg.normal(0, .05, (n, 16)); zi[:, 0] += np.where(y == 0, 1.0, -1.0)
    zp = rg.normal(size=(n, 16))
    obj = module.ResidualPrototypeObjective(prep, torch.device("cpu"))
    with torch.no_grad():
        value = obj(torch.tensor(zi, dtype=torch.float32), torch.tensor(zp, dtype=torch.float32), "R3")
        detail = value["residual_detail"]
        order = detail["query_indices"].numpy()
        valid = detail["valid"]
        logits = detail["logits"]
        old_ce = F.cross_entropy(logits, torch.tensor(prep["y"]), reduction="none")
        correct_ce = F.cross_entropy(logits, torch.tensor(prep["y"][order]), reduction="none")
        old_loss = float((old_ce * valid).sum() / valid.sum())
        correct_loss = float((correct_ce * valid).sum() / valid.sum())
        assert abs(old_loss - float(value["residual_ce"])) < 1e-7
        assert not np.array_equal(order, np.arange(n)) and abs(old_loss-correct_loss) > 1e-4
        p1 = obj.raw_p1_pair_detail(torch.tensor(zi, dtype=torch.float32), value["d4_detail"])
        p1_old = F.cross_entropy(p1["logits"], torch.tensor(prep["y"]), reduction="none")
        p1_correct = F.cross_entropy(p1["logits"], torch.tensor(prep["y"][order]), reduction="none")
        p1_correct_loss = float((p1_correct*p1["valid"]).sum()/p1["valid"].sum())
        assert abs(float((p1_old*p1["valid"]).sum()/p1["valid"].sum())-float(p1["loss"])) < 1e-7
    # An independent asymmetric toy reveals experiment22's candidate/reference reversal.
    ns = {}
    a_source = ROOT / "experiments/PCDPACM_20261021/analyze.py"
    import ast
    tree = ast.parse(a_source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "patient_changes")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(a_source), "exec"), ns)
    toy = {"candidate": {
        "a": {"label": 0, "fold": 0, "prediction": 0},
        "b": {"label": 1, "fold": 0, "prediction": 1},
        "c": {"label": 0, "fold": 0, "prediction": 1}},
        "reference": {
        "a": {"label": 0, "fold": 0, "prediction": 1},
        "b": {"label": 1, "fold": 0, "prediction": 0},
        "c": {"label": 0, "fold": 0, "prediction": 0}}}
    ns["collect_oof"] = lambda records, key: records[key]
    old_transition = ns["patient_changes"](toy, "candidate", "reference")
    assert old_transition["net_correct_change"] == -1  # Correct candidate-reference net is +1.
    result = {"utc": utc(), "review_only": True, "new_network_forwards": 0, "new_neural_training_tasks": 0,
              "paper": "https://arxiv.org/html/2406.14455v3", "repository": "https://github.com/tangaii/MM",
              "code_priority": "actual server source", "experiment23_frozen_sources_match": frozen,
              "experiment23_additional_implementation_failure": {
                  "file": str(old / "residual.py"), "logit_concat_line": 191, "loss_label_line": 193,
                  "raw_P1_diagnostic_loss_label_line": 228,
                  "problem": "Logits/valid are concatenated by private g, but CE uses original train-order self.y instead of self.y[cat(query_indices)]. The raw P1 comparison repeats the same ordering mistake.",
                  "historical_train_fold_target_misalignment": row_order,
                  "mean_misaligned_fraction": float(np.mean([r["diagnosis_targets_misaligned_fraction"] for r in row_order])),
                  "synthetic_oracle": {"n": n, "reported_R3_loss": float(value["residual_ce"]),
                                       "independent_same_logits_original_order_loss": old_loss,
                                       "same_logits_correct_query_target_loss": correct_loss,
                                       "reported_raw_P1_diagnostic_loss": float(p1["loss"]),
                                       "same_raw_P1_logits_correct_target_loss": p1_correct_loss},
                  "implication": "The reported 0.99918 R3/raw-P1 gradient cosine compares two objectives sharing an incorrect label order; it does not establish near-collinearity with the correctly ordered historical P1 objective. This confounds mechanism attribution; it does not uniquely explain every classification change.",
                  "historical_repairs_or_retrains": 0},
              "experiment22_analysis_direction_failure": {
                  "file": str(a_source), "line": 174,
                  "asymmetric_toy_expected_candidate_net": 1,
                  "historical_function_net": old_transition["net_correct_change"],
                  "historical_selftest_limitation": "The two toy groups were identical, so a sign reversal remained invisible.",
                  "historical_files_modified": False},
              "failures_review": {
                  "20_M3": "Non-image student training accuracy ~0.5018, entropy ~0.6926, ph CE ~0.694; KL rises ~0.0012 to ~0.2601. A nonzero KL gradient did not establish learned disease semantics; AUC below D4 and shuffled M4.",
                  "21_N3": "Age-sex weighted group teacher ~0.4885 ASD probability, cross-patient SD ~0.0169, accuracy ~0.5088. Averaging diluted individual disease information; student stays near chance.",
                  "22_P1_P3": "Paired matching improves age-sex balance, but embedding-derived coupling does not stabilize improvements; P3-P4 AUC change is fold-sensitive. Patient change direction is reversed in the historical analyzer.",
                  "23_R3": "Reported age residualization and near-parallel CE gradients are additionally confounded by the verified auxiliary label-order error above. Site fallback is frequent; small AUC differences coexist with worse ACC/SPE.",
                  "DRPA": "Direct/shared feature alignment and gates did not show consistent independent diagnostic gains; learned projections need not identify useful shared factors.",
                  "PCCI": "Fusion residuals reduce classification performance; correct correspondence evidence is weak and high perturbation need not help diagnosis.",
                  "CMCV": "Raw age/sex did not establish stable independent probability-AUC/CE increment; this is not proof of universal absence of useful nonlinear information.",
                  "LCRA": "Local relation loss is learnable but effective graph/forward probability perturbations are very small; optimization of an auxiliary quantity does not prove an effective prediction mechanism."}}
    atomic_json(result, BASE / "history_review.json")
    return result


if __name__ == "__main__":
    result = review()
    print(json.dumps({"experiment23_mean_target_misalignment": result["experiment23_additional_implementation_failure"]["mean_misaligned_fraction"],
                      "synthetic_oracle": result["experiment23_additional_implementation_failure"]["synthetic_oracle"],
                      "experiment22_direction_check": result["experiment22_analysis_direction_failure"]}, indent=2, ensure_ascii=False))
