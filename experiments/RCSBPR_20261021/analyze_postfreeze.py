"""Re-run post-freeze analysis with a type-correct RNG fingerprint comparison."""
from __future__ import annotations

import hashlib
import csv
import json
import sys
import statistics
from pathlib import Path


ANALYZER = Path(__file__).with_name("analyze.py").resolve()
WRAPPER = Path(__file__).resolve()
sys.path.insert(0, str(ANALYZER.parent))

source = ANALYZER.read_text(encoding="utf-8")
old = '            assert len({v["formal_start_rng_sha256"] for v in vals.values()}) == 1\n'
new = '            assert all(v["formal_start_rng_sha256"] == vals["K1"]["formal_start_rng_sha256"] for v in vals.values())\n'
if source.count(old) != 1:
    raise RuntimeError("Expected exactly one frozen RNG comparison to repair")
old_pairs = ('PAIRS = (("K3", "K1"), ("K3", "K2"), ("K3", "K4"),\n'
             '         ("K3", "historical_J1"), ("K3", "historical_D4"))')
new_pairs = ('PAIRS = (("K3", "K1"), ("K3", "K2"), ("K3", "K4"),\n'
             '         ("K3", "historical_J1"), ("K3", "historical_D4"),\n'
             '         ("K1", "historical_J1"), ("K2", "historical_J1"))')
if source.count(old_pairs) != 1:
    raise RuntimeError("Expected exactly one frozen comparison list to extend")
old_history = '    histories, hist_metrics = history_rows(audit, ctx)\n'
new_history = (
    '    histories, hist_metrics = history_rows(audit, ctx)\n'
    '    for history_group, history_values in histories.items():\n'
    '        for history_value in history_values:\n'
    '            fold_rows.append({"group": history_group, "seed": 911, "fold": history_value["fold"],\n'
    '                             **percent(history_value["metrics"])})\n'
)
if source.count(old_history) != 1:
    raise RuntimeError("Expected exactly one historical fold analysis point to extend")
source = source.replace(old_pairs, new_pairs).replace(old_history, new_history)
source = source.replace(old, new)

sys.argv = [str(ANALYZER), *sys.argv[1:]]
exec(compile(source, str(ANALYZER), "exec"),
     {"__name__": "__main__", "__file__": str(ANALYZER)})

if "--final" in sys.argv[1:]:
    import run

    summary_path = ANALYZER.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (ANALYZER.parent / "fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], {})[int(row["fold"])] = {
            key: float(row[key]) for key in ("ACC", "SEN", "SPE", "AUC")
        }

    def paired_metrics(left, right):
        out = {}
        for metric in ("ACC", "SEN", "SPE", "AUC"):
            diffs = [by_group[left][fold][metric] - by_group[right][fold][metric]
                     for fold in range(10)]
            out[metric] = {
                "mean_pp": statistics.mean(diffs),
                "sample_sd_pp": statistics.stdev(diffs),
                "improved_folds": sum(value > 1e-12 for value in diffs),
                "ties_folds": sum(abs(value) <= 1e-12 for value in diffs),
                "fold_differences_pp": diffs,
                "leave_one_fold_out_mean_pp": {
                    str(fold): statistics.mean(diffs[:fold] + diffs[fold + 1:])
                    for fold in range(10)
                },
            }
        return {"metrics": out, "paired_by_fold": True}

    summary["paired_comparisons"]["K1-historical_J1"] = paired_metrics("K1", "historical_J1")
    summary["paired_comparisons"]["K2-historical_J1"] = paired_metrics("K2", "historical_J1")
    summary["analysis_repair"] = {
        "method": "in-memory equality comparison of four RNG fingerprint fields",
        "frozen_analyzer_sha256": run.sha_file(ANALYZER),
        "repair_wrapper_sha256": hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
        "training_code_sha256_preserved": summary["code_sha256"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                            encoding="utf-8")
    report_path = ANALYZER.parent / "report.md"
    report = report_path.read_text(encoding="utf-8")
    supplement = [
        "",
        "## 全组逐折四指标",
        "",
        "数值为百分比；历史组来自只读 artifact。",
        "",
        "| Fold | Group | ACC | SEN-ASD | SPE-TD | Probability AUC |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    group_order = ("historical_D4", "historical_G2", "historical_G3", "historical_J1", "K1", "K2", "K3", "K4")
    for fold in range(10):
        for group in group_order:
            metrics = by_group[group][fold]
            supplement.append(f"| {fold} | {group} | " + " | ".join(
                f"{metrics[key]:.4f}" for key in ("ACC", "SEN", "SPE", "AUC")) + " |")
    supplement += [
        "",
        "## K3 逐折 AUC 差",
        "",
        "单位为百分点，逐折顺序为 Fold 0–9；历史 J1 对照来自只读 artifact。",
        "",
        "| Fold | K3−K1 | K3−K2 | K3−K4 | K3−J1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold in range(10):
        vals = []
        for pair in ("K3-K1", "K3-K2", "K3-K4", "K3-historical_J1"):
            vals.append(summary["paired_comparisons"][pair]["metrics"]["AUC"]["fold_differences_pp"][fold])
        supplement.append(f"| {fold} | " + " | ".join(f"{value:+.4f}" for value in vals) + " |")
    report += "\n".join(supplement) + "\n"
    report += ("\n## 分析器修复说明\n\n"
               "冻结分析器将四段 RNG 指纹字典当作集合元素，导致最终比较抛出 TypeError。"
               "本次由 analyze_postfreeze.py 在内存中改为与 K1 指纹逐项比较；冻结分析器、"
               "训练代码和 manifest 均未改写，未重训模型。修复包装器 SHA256："
               f"{summary['analysis_repair']['repair_wrapper_sha256']}。\n")
    report_path.write_text(report, encoding="utf-8")
