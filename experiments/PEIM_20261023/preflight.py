"""Single native fold-0 train-start forward and frozen-mechanism checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit import make_anchor_records
from evidence import fit_reference
from runtime import (BASE, CFG, PCDCA, Loaded, InteractionHooks, capture_rng, encode_conditions,
                     export_core, fixed_outer_permutation, historical_prototype_module,
                     import_core, payload_hash, private_group_map, restore_rng, state_hash, utc)
from interaction import LowRankInteraction


def _parameters_hash(core):
    return state_hash({name: value.detach().cpu() for name, value in core.named_parameters()})


def _parts(z, labels, d, sites, train, private_group, split, method):
    result = []
    for group in range(3):
        query = np.flatnonzero(private_group == group)
        if not len(query):
            continue
        reference = train[private_group[train] != group]
        fit = fit_reference(z, reference, labels, d, sites, CFG, method)
        result.append({"indices": query, "mean": fit["mean"], "std": fit["std"], "P": fit["P"],
                       "prototypes": fit["prototypes"], "tau": fit["tau"],
                       "rank": fit["rank"], "valid": fit["valid"]})
    heldout = np.concatenate((split["val"], split["test"])).astype(np.int64)
    fit = fit_reference(z, train, labels, d, sites, CFG, method)
    result.append({"indices": heldout, "mean": fit["mean"], "std": fit["std"], "P": fit["P"],
                   "prototypes": fit["prototypes"], "tau": fit["tau"],
                   "rank": fit["rank"], "valid": fit["valid"]})
    return result


def _module_with_weight(device):
    module = LowRankInteraction().to(device)
    with torch.no_grad():
        values = torch.linspace(-0.35, 0.4, 32, device=device).reshape(16, 2)
        module.W_d.copy_(values)
    return module


def main():
    audit = json.loads((BASE / "audit.json").read_text())
    if audit.get("formal_authorized"):
        raise RuntimeError("Preflight must run before formal authorization")
    loaded = Loaded(0, 0, audit, train_only_labels=True)
    historical = historical_prototype_module()
    old_cfg = json.loads((PCDCA / "config.json").read_text())
    train = np.asarray(loaded.split["train"], dtype=np.int64)
    prepared = historical.prepare(loaded.diagnostic_labels, loaded.meta, train, 911, 0, old_cfg)
    private_rows, private_group = private_group_map(prepared)
    if not np.array_equal(private_rows, train):
        raise AssertionError("Fold-0 D4 private-fold train order changed")
    d_true, condition_stats = encode_conditions(loaded.meta, train)
    permutation, permutation_stats = fixed_outer_permutation(
        loaded.split, loaded.meta["site"], loaded.meta["age"], loaded.meta["sex"])
    d_shuffled = d_true[permutation]
    module = LowRankInteraction().to(loaded.device)
    if sum(p.numel() for p in module.parameters()) != 32:
        raise AssertionError("PEIM must add exactly 32 trainable parameters")
    objective = historical.PrototypeObjective(prepared, "D4", old_cfg, loaded.device)

    import_core(loaded.core, loaded.start["core"])
    restore_rng(loaded.start["rng"])
    rng_start_sha = payload_hash(capture_rng())
    start_core_sha = state_hash(export_core(loaded.core))
    start_params_sha = _parameters_hash(loaded.core)
    loaded.core.train()
    module.train()
    hooks = InteractionHooks(loaded, module, "E2", d_true, d_true, private_group, CFG)
    hooks.set_epoch(CFG["rho_warmup_last_epoch"])
    attention_input = {}
    observer = loaded.core.mm_attention.register_forward_pre_hook(
        lambda _module, args: attention_input.__setitem__("image", args[0].detach().clone()))
    try:
        loaded.forward_count = 0
        logits, graph = loaded.forward()
        if loaded.forward_count != 1:
            raise AssertionError("Fold-0 preflight must execute exactly one native core forward")
        raw = hooks.raw_img
        if raw is None or hooks.raw_ph is None or "image" not in attention_input:
            raise AssertionError("Fold-0 forward did not capture the required native tensors")
        if not torch.equal(raw, hooks.modified) or not torch.equal(raw, attention_input["image"]):
            raise AssertionError("Zero-initialized interaction must be exact identity at the native attention input")
        if _parameters_hash(loaded.core) != start_params_sha:
            raise AssertionError("A real train forward changed model parameters before optimizer.step")
        bn_start = int(loaded.start["core"]["state"]["clf.2.num_batches_tracked"])
        bn_after = int(loaded.core.clf[2].num_batches_tracked)
        if bn_after != bn_start + 1:
            raise AssertionError("Expected exactly one classifier BN update in the train-mode preflight")

        labels = loaded.diagnostic_labels
        z_np = raw.detach().cpu().numpy().astype(np.float64)
        sites = loaded.meta["site"]
        p_main = _parts(z_np, labels, d_true, sites, train, private_group, loaded.split, "main")
        p_int = _parts(z_np, labels, d_true, sites, train, private_group, loaded.split, "interaction")
        parts_by_group = {"E1": p_main, "E2": p_int, "E3": p_int, "E4": p_int}
        zero_outputs = {}
        active_outputs = {}
        group_stats = {}
        for group in CFG["groups"]:
            zero = LowRankInteraction().to(loaded.device)
            out, stats = zero.apply(raw.detach(), torch.as_tensor(d_true, device=loaded.device, dtype=raw.dtype),
                                    parts_by_group[group], 9, CFG, selective=group in ("E3", "E4"))
            zero_outputs[group] = out
            nonzero = _module_with_weight(loaded.device)
            d_used = d_shuffled if group == "E4" else d_true
            active, active_stats = nonzero.apply(
                raw.detach(), torch.as_tensor(d_used, device=loaded.device, dtype=raw.dtype),
                parts_by_group[group], 9, CFG, selective=group in ("E3", "E4"))
            active_outputs[group] = active
            group_stats[group] = {"mean_relative_l2": float(active_stats["relative_l2"].mean()),
                                  "mean_gate": float(active_stats["gate"].mean()),
                                  "cap_fraction": float(active_stats["cap_hit"].float().mean()),
                                  "rank_mean": float(np.mean([part["rank"] for part in parts_by_group[group]]))}
        if not all(torch.equal(zero_outputs["E1"], value) for value in zero_outputs.values()):
            raise AssertionError("E1-E4 zero interaction outputs differ")
        if np.array_equal(p_main[0]["P"], p_int[0]["P"]):
            raise AssertionError("Fold-0 main-effect and interaction projectors are identical")
        e21 = float(torch.norm(active_outputs["E2"] - active_outputs["E1"]))
        e32 = float(torch.norm(active_outputs["E3"] - active_outputs["E2"]))
        e34 = float(torch.norm(active_outputs["E3"] - active_outputs["E4"]))
        if e21 <= 0 or e32 <= 0 or e34 <= 0:
            raise AssertionError("Preflight did not produce distinct registered mechanism contrasts")
        native = F.cross_entropy(logits[train], loaded.labels[train]) + graph
        w_grad = torch.autograd.grad(native, module.W_d, allow_unused=True)[0]
        if w_grad is None or not torch.isfinite(w_grad).all() or float(w_grad.norm()) <= 0:
            raise AssertionError("PEIM parameters did not receive a finite nonzero native gradient")
        d4_loss, d4_detail = objective(raw)
        if not torch.isfinite(d4_loss):
            raise FloatingPointError("Historical D4 objective is nonfinite")
        if not torch.isfinite(logits).all() or not torch.isfinite(graph):
            raise FloatingPointError("Native fold-0 output is nonfinite")

        record = {"status": "passed", "fold": 0, "start": "historical train_start.pt",
                  "native_forward_count": loaded.forward_count, "additional_core_forwards": 0,
                  "classifier_bn_start": bn_start, "classifier_bn_after_forward": bn_after,
                  "core_parameters_unchanged_before_optimizer": True,
                  "rng_start_sha256": rng_start_sha, "train_start_core_sha256": start_core_sha,
                  "native_logits_shape": list(logits.shape), "raw_z_img_shape": list(raw.shape),
                  "raw_z_ph_shape": list(hooks.raw_ph.shape), "native_graph_loss": float(graph.detach()),
                  "historical_D4_loss": float(d4_loss.detach()),
                  "zero_init_attention_input_exact_identity": True,
                  "zero_init_E1_E4_outputs_equal": True,
                  "E1_E2_nonzero_weight_output_l2_difference": e21,
                  "E2_E3_gate_output_l2_difference": e32,
                  "E3_E4_correct_vs_shuffled_output_l2_difference": e34,
                  "train_condition_stats": condition_stats, "shuffle_stats": permutation_stats,
                  "group_diagnostics": group_stats,
                  "W_d_parameter_count": 32, "W_d_initial_sha256": state_hash(module.state_dict()),
                  "W_d_native_gradient_norm": float(w_grad.norm()),
                  "D4_uses_unmodified_raw_z_img": True,
                  "graph_uses_original_core_embeddings": True,
                  "graph_value_may_change_via_original_attention_weights": True,
                  "heldout_labels_used_for_fit_or_direction": False,
                  "heldout_labels_masked_in_native_graph": True,
                  "native_logits_same_as_zero_interaction_forward": True,
                  "raw_classifier_logits_saved_for_mechanism_only": False,
                  "completed_utc": utc()}
    finally:
        observer.remove()
        hooks.close()
        import_core(loaded.core, loaded.start["core"])
        restore_rng(loaded.start["rng"])
        if state_hash(export_core(loaded.core)) != start_core_sha:
            raise AssertionError("Fold-0 train_start core was not restored after preflight")
        if payload_hash(capture_rng()) != rng_start_sha:
            raise AssertionError("Fold-0 train_start RNG was not restored after preflight")
        loaded.close()
    out_path = BASE / "preflight.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
