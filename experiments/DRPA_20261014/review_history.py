"""Explicitly document a diagnosed, non-exact replay; never alter raw tolerances."""

import json
from pathlib import Path
import re

from run import (BASE, anchor_dir, atomic_json, completed, config, core_hash,
                 fold_dir, load_torch, rng_hash, sha_file, source_guard, state_hash,
                 training_code_hash, utc)


def review():
    audit = source_guard()
    if not all(completed("E0", 911, f) for f in range(10)):
        raise RuntimeError("Review requires all ten canonical seed911 results")
    history_path = BASE / "artifacts/history_check.json"
    history = json.loads(history_path.read_text())
    if any(abs(history[key]) > config()["history_mean_tolerance"]
           for key in ("mean_delta_ACC", "mean_delta_AUC")):
        raise RuntimeError("Mean historical discrepancy still exceeds 0.5 points; further diagnosis is required")
    evidence_path = BASE / "artifacts/baseline_diagnostics/verification.json"
    evidence = json.loads(evidence_path.read_text())
    if (not evidence["verified"] or not evidence["same_initialization_vae_and_test_rng"]
            or evidence["same_physical_card_acc_span"] <= 0
            or evidence["data_digest"] != audit["data_digest"]
            or evidence["protected_source_sha256"] != audit["source_sha256"]):
        raise RuntimeError("Native-repeat evidence is missing, invalid, or from another input/source")
    folder = Path(config()["history_folder"])
    text = (folder / "train.log").read_text()
    historical_epochs = {int(f): int(e) for f, e in re.findall(
        r"Fold (\d+) best val_loss [0-9.]+, val_acc [0-9.]+, epoch (\d+)", text)}
    historical_vae = {int(re.search(r"fold(\d+)_", p.name)[1]): p
                      for p in (folder / "save_model").rglob("fold*_pretrain.pth")}
    ledger, digests = [], {}
    for f in range(10):
        path = fold_dir("E0", 911, f) / "metrics.json"
        result = json.loads(path.read_text())
        if result["training_code_sha256"] != training_code_hash():
            raise AssertionError("Baseline mixes different training implementations")
        digests[str(f)] = sha_file(path)
        new_start = load_torch(anchor_dir(911, f) / "train_start.pt")
        new_initial = load_torch(anchor_dir(911, f) / "initial.pt")
        row = {**history["per_fold"][f], "historical_best_epoch": historical_epochs[f],
               "canonical_best_epoch": result["best_epoch"],
               "historical_vae_state_equal": state_hash(load_torch(historical_vae[f]))
                   == state_hash(load_torch(anchor_dir(911, f).parent / "vae" / f"fold{f}_pretrain.pth"))}
        old_folder = BASE / "artifacts/diagnostic_replay_v1" / "seed_911" / f"fold_{f}"
        if (old_folder / "train_start.pt").exists():
            old_start = load_torch(old_folder / "train_start.pt")
            old_initial = load_torch(old_folder / "initial.pt")
            row["archived_initial_core_equal"] = core_hash(new_initial["core"]) == core_hash(old_initial["core"])
            row["archived_pre_init_rng_equal"] = {
                key: rng_hash(new_initial["pre_init_rng"])[key] == rng_hash(old_initial["pre_init_rng"])[key]
                for key in ("python", "numpy", "cpu", "cuda")}
            row["archived_train_start_core_equal"] = core_hash(new_start["core"]) == core_hash(old_start["core"])
            row["archived_train_start_rng_equal"] = rng_hash(new_start["rng"]) == rng_hash(old_start["rng"])
        ledger.append(row)
    if not all(r["historical_vae_state_equal"] for r in ledger[:2]):
        raise AssertionError("Initial historical VAE anchors differ; further input/RNG diagnosis is required")
    if not all(r.get("archived_train_start_core_equal") and r.get("archived_train_start_rng_equal")
               for r in ledger[:2]):
        raise AssertionError("First two replay anchors differ; no execution-drift diagnosis can be accepted")
    value = {
        "time_utc": utc(), "accepted_for_paired_research": True,
        "decision": "均值偏差满足0.5个百分点要求，逐折漂移经诊断后继续本轮官方行为配对研究；不声称历史逐折精确复现",
        "training_code_sha256": training_code_hash(), "data_digest": audit["data_digest"],
        "history_check_sha256": sha_file(history_path),
        "diagnostic_verification_sha256": sha_file(evidence_path),
        "seed911_result_sha256": digests, "raw_strict_check_passed": history["passed"],
        "mean_delta_ACC": history["mean_delta_ACC"], "mean_delta_AUC": history["mean_delta_AUC"],
        "registered_mean_tolerance": config()["history_mean_tolerance"],
        "registered_fold_tolerance": config()["history_fold_tolerance"],
        "mean_tolerance_passed": True,
        "per_fold_diagnosis": ledger,
        "first_changed_best_epoch_fold": next((r["fold"] for r in ledger
            if r["canonical_best_epoch"] != r["historical_best_epoch"]), None),
        "established": "Three verbatim official-function replays demonstrate non-bitwise training with matched initialization, VAE, backend flags, and test RNG. First two historical VAE states reproduce exactly; the canonical first two pretraining anchors equal the archived replay. Data and original source hashes are unchanged.",
        "explanation_with_limits": "Small accelerator arithmetic differences can be amplified during classifier optimization and top-k. The first changed best epoch changes total training RNG consumption; later VAE and classifier trajectories need not match. CPU initial cores can remain identical while PPU RNG differs. Historical per-epoch RNG was not saved and the precise nondeterministic kernel is not identified; therefore this is a diagnosed non-exact new replay, not an exact restoration of historical training.",
        "continuation_guard": "All 30 actual-anchor disabled-loss forward/gradient/Adam-step/RNG/backend checks must pass before E1-E7. Any mismatch, NaN, missing gradients, input change, or pairing error still stops affected experiments.",
        "selection_policy": "Keep the first canonical v2 run regardless of score. Diagnostic replays and archived v1 results are excluded from formal 240 tasks. No registered training hyperparameter, test-selected checkpoint, root algorithm, or numeric tolerance was changed."
    }
    atomic_json(value, BASE / "artifacts/history_acceptance.json")
    print(json.dumps(value, indent=2), flush=True)
    return value


if __name__ == "__main__":
    review()
