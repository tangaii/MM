"""Independent PC-RPA artifact checks, four-metric summaries, and paired OOF analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/MM-GTUNets")
sys.path.insert(0, str(BASE))
import run

GROUPS = ("R1", "R2", "R3", "R4")
METRICS = ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC")
HIST = {"historical_D4": (ROOT / "experiments/PCDCA_20261019", "D4"),
        "historical_M3": (ROOT / "experiments/PCDPAX_20261021", "M3"),
        "historical_P1": (ROOT / "experiments/PCDPACM_20261021", "P1")}


def _metric(labels, probability, prediction):
    return run.formal_metrics(labels, probability, prediction)


def _record(value, source):
    oof = value["oof"]
    metrics = _metric(oof["labels"], oof["probability"], oof["prediction"])
    return {"artifact": value, "oof": oof, "metrics": metrics, "source": source}


def _load_history(name):
    folder, group = HIST[name]
    result = {}
    for fold in range(10):
        path = folder / "runs" / group / f"s911_f{fold}.pt"
        value = run.load_torch(path)
        if "payload" in value and "content_sha256" in value:
            if run.payload_hash(value["payload"]) != value["content_sha256"]:
                raise AssertionError(f"Historical wrapper checksum mismatch: {path}")
            value = value["payload"]
        if value.get("fold") != fold or value.get("seed") != 911 or value.get("group") != group:
            raise AssertionError(f"Historical artifact identity mismatch: {path}")
        item = _record(value, name)
        if len(item["oof"]["labels"]) != len(item["oof"]["indices"]):
            raise AssertionError(f"Historical OOF shape mismatch: {path}")
        result[fold] = item
    return result


def _load_new(audit):
    result, missing = {}, []
    for group in GROUPS:
        for fold in range(10):
            try:
                value = run.valid_result(group, 911, fold, audit, required=False)
            except Exception as error:
                missing.append({"group": group, "fold": fold, "error": repr(error)})
                continue
            if value is not None:
                result[(group, fold)] = _record(value, group)
    return result, missing


def _folds(records, group):
    if group in records and isinstance(records[group], dict):
        return records[group]
    return {fold: records[(group, fold)] for fold in range(10) if (group, fold) in records}


def _mean_sd(values):
    x = np.asarray(values, dtype=float) * 100.0
    return {"mean_percent": float(x.mean()) if len(x) else None,
            "sample_sd_pp": float(x.std(ddof=1)) if len(x) > 1 else None, "n": int(len(x))}


def _summarize(records, group):
    rows = _folds(records, group)
    return {"folds_complete": len(rows),
            "metrics": {key: _mean_sd([rows[f]["metrics"][key] for f in sorted(rows)]) for key in METRICS},
            "best_epoch_by_fold": {str(f): int(rows[f]["artifact"].get("best_epoch", -1)) for f in sorted(rows)},
            "best_val_loss_by_fold": {str(f): float(rows[f]["artifact"].get("best_val_loss", np.nan)) for f in sorted(rows)},
            "training_seconds_by_fold": {str(f): float(rows[f]["artifact"].get("training_seconds", 0)) for f in sorted(rows)}}


def _paired(records, left, right):
    a, b = _folds(records, left), _folds(records, right)
    result = {}
    for metric in METRICS:
        deltas = {f: 100 * (a[f]["metrics"][metric] - b[f]["metrics"][metric])
                  for f in sorted(set(a) & set(b))}
        loo = {str(drop): float(np.mean([v for f, v in deltas.items() if f != drop]))
               for drop in deltas if len(deltas) > 1}
        result[metric] = {"fold_delta_pp": {str(f): float(v) for f, v in deltas.items()},
                          "mean_delta_pp": _mean_sd([v / 100 for v in deltas.values()]),
                          "improved_folds": int(sum(v > 0 for v in deltas.values())),
                          "worsened_folds": int(sum(v < 0 for v in deltas.values())),
                          "leave_one_fold_out_min_pp": min(loo.values()) if loo else None,
                          "leave_one_fold_out_max_pp": max(loo.values()) if loo else None,
                          "loo_direction_reversal": bool(loo and min(loo.values()) <= 0 <= max(loo.values()))}
    return result


def _oof(rows):
    result = {}
    for fold, item in rows.items():
        o = item["oof"]
        for i, subject in enumerate(o["ids"]):
            if subject in result:
                raise AssertionError(f"Duplicate OOF subject {subject}")
            result[subject] = {"fold": fold, "label": int(o["labels"][i]),
                               "prediction": int(o["prediction"][i]),
                               "probability": float(o["probability"][i]),
                               "logits": np.asarray(o["logits"][i]).tolist()}
    return result


def _changes(reference, candidate, expected_count=871):
    common = sorted(set(reference) & set(candidate))
    if len(common) != expected_count:
        raise AssertionError(f"Expected {expected_count} matched subjects, found {len(common)}")
    for subject in common:
        if reference[subject]["fold"] != candidate[subject]["fold"] or reference[subject]["label"] != candidate[subject]["label"]:
            raise AssertionError(f"OOF identity/label mismatch for {subject}")
    corrected = [s for s in common if reference[s]["prediction"] != reference[s]["label"] and candidate[s]["prediction"] == candidate[s]["label"]]
    newly_wrong = [s for s in common if reference[s]["prediction"] == reference[s]["label"] and candidate[s]["prediction"] != candidate[s]["label"]]
    return {"reference_to_candidate_corrected": len(corrected),
            "reference_to_candidate_new_errors": len(newly_wrong),
            "net_correct_subjects": len(corrected) - len(newly_wrong),
            "ASD_net_correct": int(sum(reference[s]["label"] == 0 for s in corrected) - sum(reference[s]["label"] == 0 for s in newly_wrong)),
            "TD_net_correct": int(sum(reference[s]["label"] == 1 for s in corrected) - sum(reference[s]["label"] == 1 for s in newly_wrong)),
            "corrected_subject_ids": corrected, "new_error_subject_ids": newly_wrong}


def _direction_contract_test():
    old = {"a": {"fold": 0, "label": 0, "prediction": 1}, "b": {"fold": 0, "label": 1, "prediction": 1}}
    new = {"a": {"fold": 0, "label": 0, "prediction": 0}, "b": {"fold": 0, "label": 1, "prediction": 0}}
    result = _changes(old, new, expected_count=2)
    assert result["reference_to_candidate_corrected"] == 1 and result["reference_to_candidate_new_errors"] == 1
    return True


def _write_tables(summary, records, oof_by_group):
    metrics_path = BASE / "fold_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "fold", *METRICS, "best_epoch", "best_val_loss", "training_seconds"])
        writer.writeheader()
        for group in ["historical_D4", "historical_M3", "historical_P1", *GROUPS]:
            for fold, row in sorted(_folds(records, group).items()):
                writer.writerow({"group": group, "fold": fold,
                                 **{m: row["metrics"][m] for m in METRICS},
                                 "best_epoch": row["artifact"].get("best_epoch", ""),
                                 "best_val_loss": row["artifact"].get("best_val_loss", ""),
                                 "training_seconds": row["artifact"].get("training_seconds", "")})
    with (BASE / "oof.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "subject_id", "fold", "label", "prediction", "ASD_probability"])
        writer.writeheader()
        for group in GROUPS:
            for subject, value in sorted(oof_by_group.get(group, {}).items()):
                writer.writerow({"group": group, "subject_id": subject, "fold": value["fold"],
                                 "label": value["label"], "prediction": value["prediction"],
                                 "ASD_probability": value["probability"]})


def analyze():
    audit = run.guard(require_manifest=True)
    records = {name: _load_history(name) for name in HIST}
    new, missing = _load_new(audit)
    records.update(new)
    summaries = {name: _summarize(records, name) for name in [*HIST, *GROUPS]}
    pairs = {pair: _paired(records, a, b) for pair, (a, b) in {
        "R2-R1": ("R2", "R1"), "R3-R2": ("R3", "R2"), "R3-R4": ("R3", "R4"),
        "R3-historical_M3": ("R3", "historical_M3"), "R3-historical_P1": ("R3", "historical_P1"),
        "R3-historical_D4": ("R3", "historical_D4")}.items()}
    oof_by_group = {name: _oof(_folds(records, name)) for name in [*HIST, *GROUPS] if len(_folds(records, name)) == 10}
    transitions = {}
    if "R3" in oof_by_group and "R4" in oof_by_group:
        transitions["R3_vs_R4"] = _changes(oof_by_group["R4"], oof_by_group["R3"])
    loo_auc = {}
    for name in [*HIST, *GROUPS]:
        rows = _folds(records, name)
        vals = {str(f): float(np.mean([rows[x]["metrics"]["probability_AUC"] for x in rows if x != f]) * 100)
                for f in rows if len(rows) > 1}
        loo_auc[name] = {"min_percent": min(vals.values()) if vals else None,
                         "max_percent": max(vals.values()) if vals else None, "by_omitted_fold": vals}
    task_mechanisms = {}
    for group in GROUPS:
        rows = _folds(records, group)
        task_mechanisms[group] = {str(f): {"best_epoch": rows[f]["artifact"].get("best_epoch"),
            "best_val_loss": rows[f]["artifact"].get("best_val_loss"),
            "residual_mechanism": [e.get("residual_mechanism") for e in rows[f]["artifact"].get("epochs", [])
                                   if e.get("residual_mechanism") is not None],
            "gradient_diagnostics": [e.get("embedding_gradient_diagnostics") for e in rows[f]["artifact"].get("epochs", [])
                                     if e.get("embedding_gradient_diagnostics") is not None]}
            for f in rows}
    result = {"experiment": "PCRPA_20261022", "method": "PC-RPA", "status": "COMPLETE" if len(new) == 40 and not missing else "PARTIAL",
              "formal_metrics": summaries, "paired_deltas": pairs, "loo_auc": loo_auc,
              "patient_changes": transitions, "mechanisms": task_mechanisms,
              "historical_groups_read_only": list(HIST), "new_task_count": len(new), "expected_task_count": 40,
              "missing_or_invalid_tasks": missing,
              "new_oof_rows": sum(sum(len(v["oof"]["ids"]) for v in _folds(records, g).values()) for g in GROUPS),
              "analyzer_direction_contract_passed": _direction_contract_test(),
              "analysis_code_sha256": run.sha_file(BASE / "analyze.py"),
              "training_code_sha256": audit["training_code_sha256"]}
    _write_tables(result, records, oof_by_group)
    run.atomic_json(result, BASE / "summary.json")
    lines = ["# Experiment 23: PC-RPA", "", f"Status: {result['status']}",
             f"Tasks: {len(new)}/40; OOF rows: {result['new_oof_rows']}/3484", "",
             "| Group | ACC | SEN-ASD | SPE-TD | probability AUC |", "|---|---:|---:|---:|---:|"]
    for group, row in summaries.items():
        vals = [row["metrics"][m] for m in METRICS]
        fmt = ["n/a" if x["mean_percent"] is None else f"{x['mean_percent']:.2f} +/- {x['sample_sd_pp']:.2f}" for x in vals]
        lines.append(f"| {group} | " + " | ".join(fmt) + " |")
    lines += ["", "Paired deltas in percentage points are recorded per fold in `summary.json`; OOF patient transition direction is reference to candidate.",
              "Historical D4, M3, and P1 artifacts are read-only. No historical model was retrained.", ""]
    (BASE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    analyze()
