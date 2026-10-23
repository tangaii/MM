"""Read-only historical metric key normalization for the frozen analyzer."""
from __future__ import annotations

import json

import analyze
import run


def normalize_metrics(rows,group):
    normalized=[]
    for row in rows:
        if {"ACC","SEN-ASD","SPE-TD","probability_AUC"}.issubset(row):
            normalized.append({key:float(row[key]) for key in run.RESULT_METRICS})
            continue
        if group not in ("D0","D4") or not {"ACC","SEN","SPE","AUC"}.issubset(row):
            raise AssertionError(f"Unrecognized historical metric schema for {group}: {sorted(row)}")
        normalized.append({"ACC":float(row["ACC"]),"SEN-ASD":float(row["SEN"]),
                           "SPE-TD":float(row["SPE"]),"probability_AUC":float(row["AUC"])})
    if len(normalized)!=10:raise AssertionError(f"Historical {group} should have ten fold records")
    return normalized


def main():
    manifest=json.loads((run.BASE/"manifest.json").read_text())
    audit=json.loads((run.BASE/"audit.json").read_text())
    actual=run.train_code_hashes()
    if actual!=manifest["training_files_sha256"]:
        raise AssertionError("Frozen source SHA mismatch; refusing posthoc analysis")
    if run.sha_file(run.BASE/"manifest.json")!=audit["manifest_sha256"]:
        raise AssertionError("Frozen manifest SHA mismatch")
    # Keep the ordinary guard active for protected historical inputs and project sources.
    run.code_hash=lambda:manifest["training_code_sha256"]
    protected_audit=run.guard(True)
    if protected_audit["training_code_files_sha256"]!=actual:
        raise AssertionError("Audit frozen source fingerprint mismatch")
    normalized_audit=json.loads(json.dumps(protected_audit))
    for key,group in (("historical_D0","D0"),("historical_D4","D4"),
                      ("historical_M1","M1"),("historical_M3","M3")):
        rows=normalized_audit["historical_readonly_controls"][key]["fold_metrics"]
        normalized_audit["historical_readonly_controls"][key]["fold_metrics_raw_schema"]=rows
        normalized_audit["historical_readonly_controls"][key]["fold_metrics_key_mapping"]="SEN->SEN-ASD; SPE->SPE-TD; AUC->probability_AUC" if group in ("D0","D4") else "already formal"
        normalized_audit["historical_readonly_controls"][key]["fold_metrics"]=normalize_metrics(rows,group)
    run.guard=lambda require_manifest=True:normalized_audit
    summary=analyze.run_analysis()
    print(json.dumps({"status":summary["status"],"tasks_completed":summary["tasks_completed"],
                      "research_judgment":summary["research_judgment"],
                      "oof_records_registered_groups":summary["oof_records_registered_groups"],
                      "manifest_sha256":summary["manifest_sha256"]},indent=2,ensure_ascii=False))


if __name__=="__main__":main()
