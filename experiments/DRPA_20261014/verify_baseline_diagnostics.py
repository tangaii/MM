"""Verify the original-function repeat controls without fitting another model."""

import hashlib
import inspect
import json
import os
import re

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from run import (BASE, ROOT, atomic_json, config, core_hash, load_torch, rng_hash,
                 sha_file, source_guard, state_hash, utc)
import train_mm_gtunets as official


def verify():
    audit = source_guard()
    controls = []
    physical_cards = {"original_function_1": "0", "original_function_2": "1",
                      "original_function_3": "0"}
    for tag, card in physical_cards.items():
        path = BASE / "artifacts/baseline_diagnostics" / tag / "metrics.json"
        record = json.loads(path.read_text())
        if not record["official_functions"] or record["model_hooks"] or record["extra_backward"]:
            raise AssertionError("Repeat control did not use the unmodified official functions")
        if not record["cudnn_deterministic"] or record["cudnn_benchmark"]:
            raise AssertionError("Official backend settings changed in a repeat control")
        log = (BASE / "artifacts" / f"{tag}.log").read_text()
        matches = re.findall(r"Fold 0 best val_loss ([0-9.]+), val_acc ([0-9.]+), epoch (\d+)", log)
        if len(matches) != 1:
            raise AssertionError("Missing original training checkpoint-selection record")
        loss, acc, epoch = matches[0]
        controls.append({**record, "physical_ppu_from_execution_ledger": card,
                         "best_validation_original_loss": float(loss),
                         "best_validation_acc": float(acc), "best_epoch": int(epoch),
                         "metrics_file": str(path.relative_to(BASE)),
                         "metrics_sha256": sha_file(path)})
    first = controls[0]
    shared_fields = ("initial_core_fingerprint", "vae_state_fingerprint",
                     "before_test_rng", "end_rng", "best_epoch")
    for record in controls[1:]:
        for key in shared_fields:
            if record[key] != first[key]:
                raise AssertionError(f"Non-identical repeat-control anchor: {key}")
    archive = BASE / "artifacts/diagnostic_replay_v1"
    original_anchor = load_torch(archive / "seed_911/fold_0/train_start.pt")
    current_anchor = load_torch(BASE / "artifacts/seed_911/fold_0/train_start.pt")
    if core_hash(original_anchor["core"]) != first["initial_core_fingerprint"]:
        raise AssertionError("Original replay does not use the archived initial core")
    if core_hash(current_anchor["core"]) != first["initial_core_fingerprint"]:
        raise AssertionError("Canonical E0 fold0 initialization changed")
    if rng_hash(current_anchor["rng"]) != rng_hash(original_anchor["rng"]):
        raise AssertionError("Canonical E0 fold0 RNG changed")
    historical_vae = next((ROOT / "experiments/official_source_ppu_20261014/save_model").rglob("fold0_pretrain.pth"))
    if state_hash(load_torch(historical_vae)) != first["vae_state_fingerprint"]:
        raise AssertionError("Historical VAE pretraining did not reproduce exactly")
    current = json.loads((BASE / "outputs/E0/seed_911/fold_0/metrics.json").read_text())
    for key, destination in (("before_test_rng", "test_forward_rng"),
                             ("end_rng", "end_rng_fingerprint")):
        if current[destination] != first[key]:
            raise AssertionError("Canonical fold0 random consumption differs from official replay")
    checkpoints = {tag: load_torch(BASE / "artifacts/baseline_diagnostics" / tag / "fold0.pth")
                   for tag in physical_cards}
    original_states = list(checkpoints.values())
    native_errors = {name: float((tensor - original_states[2][name]).abs().max())
                     for name, tensor in original_states[0].items() if tensor.is_floating_point()}
    native_state_hashes = {tag: state_hash(state) for tag, state in checkpoints.items()}
    if len(set(native_state_hashes.values())) == 1:
        raise AssertionError("Diagnostic training checkpoints are identical; inspect evaluation only")
    value = {
        "time_utc": utc(), "verified": True, "n_original_function_replays": 3,
        "controls": controls, "same_initialization_vae_and_test_rng": True,
        "same_physical_card_acc_span": abs(controls[0]["ACC"] - controls[2]["ACC"]),
        "all_cards_acc_span": max(c["ACC"] for c in controls) - min(c["ACC"] for c in controls),
        "all_cards_auc_span": max(c["AUC"] for c in controls) - min(c["AUC"] for c in controls),
        "canonical_fold0_official_metrics": current["official_metrics"],
        "canonical_fold0_matches_first_original_replay_at_5_decimals": all(
            round(current["official_metrics"][k], 5) == round(first[k], 5) for k in ("ACC", "AUC")),
        "original_function_best_checkpoint_state_hashes": native_state_hashes,
        "same_card_best_checkpoint_max_abs_difference_by_tensor": native_errors,
        "official_function_source_sha256": {name: hashlib.sha256(inspect.getsource(
            getattr(official, name)).encode()).hexdigest() for name in ("train", "evaluate")},
        "protected_source_sha256": audit["source_sha256"],
        "data_digest": audit["data_digest"],
        "backend_from_canonical_anchor": current_anchor["rng"]["backend"],
        "established": "Non-bitwise PPU training repeatability exists with original functions, identical initialization, VAE, backend flags, and test RNG. This is not specific to DRPA hooks or additional backward diagnostics.",
        "inference_not_kernel_localization": "Accelerator floating-point reductions are the leading explanation. Adam and top-k can amplify small errors; a changed early-stop epoch then changes later folds' RNG consumption. The precise responsible kernel has not been isolated.",
        "use": "Evidence for diagnosing non-exact historical replay, not evidence of exact historical replication, not a change of training hyperparameters or a selection of the best replay. E0-to-E7 must still pass anchor-level disabled-loss equivalence checks."
    }
    atomic_json(value, BASE / "artifacts/baseline_diagnostics/verification.json")
    print(json.dumps(value, indent=2), flush=True)
    return value


if __name__ == "__main__":
    verify()
