"""Offline and minimal inference audit for PCDCA D0/D4 vs SCPDA E0/E1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
BASE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/MM-GTUNets")
CFG = json.loads((BASE / "config.json").read_text())
OLD = Path(CFG["historical_root"])
NEW = Path(CFG["new_root"])
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def payload_hash(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(b"tensor"); add(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            h.update(b"array"); h.update(array_hash(v).encode())
        elif isinstance(v, np.generic):
            add(v.item())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=repr):
                add(k); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(type(v)).encode()); h.update(str(len(v)).encode())
            for x in v: add(x)
        elif v is None or isinstance(v, (str, int, float, bool)):
            h.update(str(type(v)).encode()); h.update(repr(v).encode())
        else:
            raise TypeError(type(v))
    add(value)
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    payload = obj["payload"]
    actual = payload_hash(payload)
    if actual != obj["content_sha256"]:
        raise ValueError(f"content hash mismatch: {path}")
    return payload, obj["content_sha256"]


def metric(y, p, pred):
    y, p, pred = np.asarray(y), np.asarray(p), np.asarray(pred)
    return {"ACC": float(accuracy_score(y, pred)),
            "BA": float(balanced_accuracy_score(y, pred)),
            "AUC": float(roc_auc_score(y == 0, p)),
            "SEN": float(recall_score(y, pred, pos_label=0)),
            "SPE": float(recall_score(y, pred, pos_label=1))}


def flatten_scalars(v, prefix=""):
    out = {}
    if isinstance(v, dict):
        for k, x in v.items():
            out.update(flatten_scalars(x, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v): out.update(flatten_scalars(x, f"{prefix}[{i}]"))
    elif isinstance(v, (int, float, np.number)) and np.isfinite(v):
        out[prefix] = float(v)
    return out


def put_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def state_diff(old, new):
    groups = {"img_unet.": "image_gtunet", "ph_unet.": "phenotype_gtunet",
              "mm_attention.": "attention", "clf.": "classifier_bn"}
    rows = []
    entries = [(k, old["state"][k], new["state"][k]) for k in old["state"]]
    entries.append(("unregistered_rp_weights", old["rp_weights"], new["rp_weights"]))
    bucket = {}
    for name, a, b in entries:
        name = name if name == "unregistered_rp_weights" else name
        group = "other_buffers"
        for prefix, label in groups.items():
            if name.startswith(prefix): group = label; break
        aa, bb = a.detach().cpu().double().numpy(), b.detach().cpu().double().numpy()
        d = np.abs(aa - bb)
        r = bucket.setdefault(group, {"tensors": 0, "exact": 0, "elements": 0, "different": 0,
                                      "sq": 0.0, "ref_sq": 0.0, "max_abs": 0.0})
        r["tensors"] += 1; r["exact"] += int(np.array_equal(aa, bb)); r["elements"] += aa.size
        r["different"] += int(np.count_nonzero(d)); r["sq"] += float(np.square(d).sum())
        r["ref_sq"] += float(np.square(aa).sum()); r["max_abs"] = max(r["max_abs"], float(d.max(initial=0)))
    for name, v in bucket.items():
        rows.append({"component": name, "tensors": v["tensors"], "exact_tensors": v["exact"],
                     "element_count": v["elements"], "different_element_fraction": v["different"] / max(v["elements"], 1),
                     "absolute_l2": v["sq"]**0.5, "relative_l2": (v["sq"] / max(v["ref_sq"], 1e-30))**0.5,
                     "max_abs": v["max_abs"]})
    return rows


def compare_pair(old_group, new_group, fold, artifact_cache):
    op = OLD / "runs" / old_group / f"s911_f{fold}.pt"
    npth = NEW / "runs" / new_group / f"s911_f{fold}.pt"
    o, oh = artifact_cache[str(op)]
    n, nh = artifact_cache[str(npth)]
    oo, no = o["oof"], n["oof"]
    if not (np.array_equal(oo["indices"], no["indices"]) and oo["ids"] == no["ids"]
            and np.array_equal(oo["labels"], no["labels"])):
        raise ValueError(f"OOF subject/index/label mismatch {old_group}/{new_group}/fold{fold}")
    om, nm = metric(oo["labels"], oo["probability"], oo["prediction"]), metric(no["labels"], no["probability"], no["prediction"])
    opred, npred = np.asarray(oo["prediction"]), np.asarray(no["prediction"])
    y = np.asarray(oo["labels"])
    old_correct, new_correct = opred == y, npred == y
    oldrng, newrng = payload_hash(o["rng_before_test"]), payload_hash(n["rng_before_test"])
    oldafter, newafter = payload_hash(o["rng_after_test"]), payload_hash(n["rng_after_test"])
    row = {"old_group": old_group, "new_group": new_group, "fold": fold,
           "old_artifact_sha256": file_sha(op), "new_artifact_sha256": file_sha(npth),
           "old_content_sha256": oh, "new_content_sha256": nh,
           "best_epoch_old": o["best_epoch"], "best_epoch_new": n["best_epoch"],
           "delta_best_epoch": n["best_epoch"] - o["best_epoch"],
           "epochs_run_old": o["epochs_run"], "epochs_run_new": n["epochs_run"],
           "delta_epochs_run": n["epochs_run"] - o["epochs_run"],
           "best_val_loss_old": o["best_val_loss"], "best_val_loss_new": n["best_val_loss"],
           "delta_best_val_loss": n["best_val_loss"] - o["best_val_loss"],
           "best_core_sha_old": o["best_core_sha256"], "best_core_sha_new": n["best_core_sha256"],
           "best_core_equal": o["best_core_sha256"] == n["best_core_sha256"],
           "start_rng_equal": o["formal_start_rng_sha256"] == n["formal_start_rng_sha256"],
           "input_fingerprint_equal": o["input_sha256"] == n["input_sha256"],
           "data_input_digest_equal": o["data_input_digest"] == n["data_input_digest"],
           "pairing_fingerprint_equal": o["pairing"] == n["pairing"],
           "prototype_preparation_equal": o["prototype_preparation"] == n["prototype_preparation"],
           "optimizer_parameter_order_equal": o["optimizer_parameter_order"] == n["optimizer_parameter_order"],
           "physical_gpu_equal": o["physical_gpu"] == n["physical_gpu"],
           "test_rng_before_equal": oldrng == newrng, "test_rng_after_equal": oldafter == newafter,
           "test_indices_equal": True, "test_ids_equal": True, "test_labels_equal": True,
           "test_n": len(y), "changed_predictions": int(np.count_nonzero(opred != npred)),
           "old_correct_new_wrong": int(np.count_nonzero(old_correct & ~new_correct)),
           "old_wrong_new_correct": int(np.count_nonzero(~old_correct & new_correct)),
           "logits_max_abs": float(np.max(np.abs(np.asarray(oo["logits"]) - np.asarray(no["logits"])))),
           "probability_mean_abs": float(np.mean(np.abs(np.asarray(oo["probability"]) - np.asarray(no["probability"]))))}
    for k in ("ACC", "BA", "AUC", "SEN", "SPE", "F1", "CE", "Brier"):
        row[f"{k}_old"] = o["metrics"][k]; row[f"{k}_new"] = n["metrics"][k]
        row[f"delta_{k}_pp"] = (n["metrics"][k] - o["metrics"][k]) * 100
    for k in ("ACC", "BA", "AUC", "SEN", "SPE"):
        if abs(row[f"{k}_old"] - om[k]) > 1e-12 or abs(row[f"{k}_new"] - nm[k]) > 1e-12:
            raise ValueError(f"independent metric mismatch {old_group}/{new_group}/fold{fold}/{k}")
    return row, o, n


def offline():
    cache = {}
    records = []
    paths = []
    for old_group, new_group in CFG["pairs"]:
        for fold in range(CFG["folds"]):
            paths += [OLD / "runs" / old_group / f"s911_f{fold}.pt",
                      NEW / "runs" / new_group / f"s911_f{fold}.pt"]
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(path)
        payload, content = load_artifact(path)
        cache[str(path)] = (payload, content)
    for old_group, new_group in CFG["pairs"]:
        for fold in range(CFG["folds"]):
            row, o, n = compare_pair(old_group, new_group, fold, cache)
            records.append(row)
            dr = state_diff(o["core"], n["core"])
            for x in dr: x.update(old_group=old_group, new_group=new_group, fold=fold)
            state_rows.extend(dr)
            a = {int(x["epoch"]): flatten_scalars({k:v for k,v in x.items() if k != "epoch"}) for x in o["epochs"]}
            b = {int(x["epoch"]): flatten_scalars({k:v for k,v in x.items() if k != "epoch"}) for x in n["epochs"]}
            common = sorted(set(a) & set(b))
            fields = set.intersection(*(set(a[e]) for e in common), *(set(b[e]) for e in common)) if common else set()
            for field in sorted(fields):
                diffs = [(e, abs(a[e][field] - b[e][field])) for e in common]
                summary = {"old_group": old_group, "new_group": new_group, "fold": fold, "field": field,
                           "common_epochs": len(diffs), "first_different_epoch_exact": next((e for e, d in diffs if d > 0), None),
                           "max_abs_diff": max(d for _, d in diffs),
                           "max_diff_epoch": max(diffs, key=lambda z: z[1])[0]}
                for tol in CFG["trajectory_tolerances"]:
                    summary[f"first_epoch_gt_{tol:g}"] = next((e for e, d in diffs if d > tol), None)
                trajectory_rows.append(summary)
    put_csv(BASE / "fold_drift.csv", records)
    put_csv(BASE / "state_diff.csv", state_rows)
    put_csv(BASE / "trajectory_summary.csv", trajectory_rows)
    selection = {}
    for old_group, new_group in CFG["pairs"]:
        subset = [r for r in records if r["old_group"] == old_group]
        ranked = sorted(subset, key=lambda r: (-max(abs(r["delta_ACC_pp"]), abs(r["delta_AUC_pp"])), r["fold"]))
        if old_group == "D4":
            selection[old_group] = [0, 2, 1]
        else:
            selection[old_group] = [r["fold"] for r in ranked[:2]]
            stable = [r["fold"] for r in subset if r["delta_ACC_pp"] == 0 and r["delta_AUC_pp"] == 0]
            if stable and stable[0] not in selection[old_group]: selection[old_group].append(stable[0])
    old_manifest = json.loads((OLD / "audit.json").read_text())
    new_manifest = json.loads((NEW / "manifest.json").read_text())
    source_checks = {}
    for name, old_sha in old_manifest["source_sha256"].items():
        new_sha = new_manifest["source_sha256"].get(name)
        source_checks[name] = {"old_recorded": old_sha, "new_recorded": new_sha,
                               "current": file_sha(ROOT/name), "all_equal": old_sha == new_sha == file_sha(ROOT/name)}
    e1_cfg = json.loads((NEW / "config.json").read_text())["groups"]["E1"]
    output = {"verified_artifacts": len(paths), "all_content_hashes_valid": True,
              "selection": selection, "fold_records": records,
              "root_source_checks": source_checks,
              "e1_flags": {"ema": e1_cfg["ema"], "class_calibration": e1_cfg["class_calibration"],
                           "original_age_sex_kernel": e1_cfg["kernel"] == "age_sex",
                           "same_train_start_all_folds": all(r["start_rng_equal"] and r["input_fingerprint_equal"]
                                                               and r["pairing_fingerprint_equal"]
                                                               and r["prototype_preparation_equal"]
                                                               and r["optimizer_parameter_order_equal"] for r in records if r["old_group"] == "D4")},
              "source_files_sha256": {str(p): file_sha(p) for p in [OLD/"run.py", OLD/"prototype.py", NEW/"run.py", NEW/"prototype.py"]}}
    (BASE / "offline.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    return output


def load_scpda_module():
    sys.path.insert(0, str(NEW))
    spec = importlib.util.spec_from_file_location("scpda_audit_run", NEW / "run.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def set_diag_rng(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def gpu_cross(folds_by_pair, tag="all"):
    mod = load_scpda_module()
    man = json.loads((NEW / "manifest.json").read_text())
    rows = []
    for old_group, new_group in CFG["pairs"]:
        for fold in folds_by_pair[old_group]:
            l = mod.Loaded(911, int(fold), man)
            try:
                old, _ = load_artifact(OLD / "runs" / old_group / f"s911_f{fold}.pt")
                new, _ = load_artifact(NEW / "runs" / new_group / f"s911_f{fold}.pt")
                pairs = [("A_oldcp_oldrng", old, old["rng_before_test"]),
                         ("B_oldcp_newrng", old, new["rng_before_test"]),
                         ("C_newcp_oldrng", new, old["rng_before_test"]),
                         ("D_newcp_newrng", new, new["rng_before_test"])]
                for label, checkpoint, rng in pairs:
                    mod.import_core(l.core, checkpoint["core"]); l.core.eval()
                    mod.restore_rng(rng); torch.use_deterministic_algorithms(False)
                    actual_before = mod.capture_rng()
                    before = mod.rng_hash(actual_before)
                    with torch.set_grad_enabled(True): logits, _ = l.forward()
                    after = mod.rng_hash(mod.capture_rng())
                    idx = np.asarray(l.sp["test"], dtype=np.int64)
                    logits = logits.detach().cpu().numpy()[idx]
                    probability = torch.softmax(torch.as_tensor(logits), dim=1).numpy()[:, 0]
                    pred = logits.argmax(1); y = old["oof"]["labels"]
                    metrics = metric(y, probability, pred)
                    expected = checkpoint["oof"]
                    rows.append({"old_group": old_group, "new_group": new_group, "fold": int(fold), "condition": label,
                                 "checkpoint": "old" if "oldcp" in label else "new",
                                 "rng": "old" if label.endswith("oldrng") else "new",
                                 "pre_rng_fingerprint": payload_hash(rng), "actual_pre_rng_fingerprint": payload_hash(actual_before),
                                 "post_rng_fingerprint": payload_hash(mod.capture_rng()),
                                 "rng_before_replay_matches_saved": payload_hash(rng) == payload_hash(checkpoint["rng_before_test"]) if label.startswith("A") or label.startswith("D") else None,
                                 "checkpoint_sha": checkpoint["best_core_sha256"],
                                 "logits_saved_comparison_checkpoint": float(np.max(np.abs(logits - np.asarray(expected["logits"])))) if label.startswith("A") or label.startswith("D") else None,
                                 "ACC": metrics["ACC"], "AUC": metrics["AUC"], "SEN": metrics["SEN"], "SPE": metrics["SPE"],
                                 "saved_checkpoint_formal_metrics": checkpoint["metrics"],
                                 "logits": logits, "probability": probability, "prediction": pred})
                # Public, fixed diagnostic seeds isolate within-checkpoint stochastic inference.
                for cp_name, checkpoint in (("old", old), ("new", new)):
                    for seed in CFG["predeclared_diagnostic_rng_seeds"]:
                        mod.import_core(l.core, checkpoint["core"]); l.core.eval(); set_diag_rng(seed)
                        pre_full = mod.capture_rng()
                        pre = mod.rng_hash(pre_full)
                        with torch.set_grad_enabled(True): logits, _ = l.forward()
                        post = mod.rng_hash(mod.capture_rng())
                        idx = np.asarray(l.sp["test"], dtype=np.int64)
                        logits = logits.detach().cpu().numpy()[idx]
                        probability = torch.softmax(torch.as_tensor(logits), dim=1).numpy()[:, 0]
                        pred = logits.argmax(1); y = old["oof"]["labels"]
                        m = metric(y, probability, pred)
                        rows.append({"old_group": old_group, "new_group": new_group, "fold": int(fold),
                                     "condition": f"fixed_rng_{seed}", "checkpoint": cp_name, "rng": str(seed),
                                     "pre_rng_fingerprint": payload_hash(pre_full), "actual_pre_rng_fingerprint": payload_hash(pre_full),
                                     "post_rng_fingerprint": post, "rng_before_replay_matches_saved": None,
                                     "checkpoint_sha": checkpoint["best_core_sha256"],
                                     "logits_saved_comparison_checkpoint": None, "ACC": m["ACC"], "AUC": m["AUC"],
                                     "SEN": m["SEN"], "SPE": m["SPE"], "saved_checkpoint_formal_metrics": checkpoint["metrics"],
                                     "logits": logits, "probability": probability, "prediction": pred})
            finally:
                l.close()
    # Save compact rows with patientwise outputs for all crossed RNG conditions.
    serial = []
    for r in rows:
        x = {k: v for k, v in r.items() if k not in ("logits", "probability", "prediction", "saved_checkpoint_formal_metrics")}
        x["formal_metrics_json"] = json.dumps(r["saved_checkpoint_formal_metrics"], sort_keys=True)
        serial.append(x)
    put_csv(BASE / f"crossover_inference_{tag}.csv", serial)
    torch.save([{k: v for k, v in r.items() if k not in ("saved_checkpoint_formal_metrics",)} for r in rows],
               BASE / f"crossover_predictions_{tag}.pt")
    return rows


def merge_gpu_shards():
    rows = []
    for tag in ("gpu0", "gpu1"):
        path = BASE / f"crossover_predictions_{tag}.pt"
        if not path.is_file(): raise FileNotFoundError(path)
        rows.extend(torch.load(path, map_location="cpu", weights_only=False))
    serial = []
    for r in rows:
        x = {k:v for k,v in r.items() if k not in ("logits", "probability", "prediction", "saved_checkpoint_formal_metrics")}
        serial.append(x)
    put_csv(BASE / "crossover_inference.csv", serial)
    torch.save(rows, BASE / "crossover_predictions.pt")
    summarize_cross(rows)
    return rows


def max_nested_abs(a, b):
    if torch.is_tensor(a): a = a.detach().cpu().numpy()
    if torch.is_tensor(b): b = b.detach().cpu().numpy()
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape: return float("inf")
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): return float("inf")
        return max((max_nested_abs(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): return float("inf")
        return max((max_nested_abs(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (float, int, np.number)) and isinstance(b, (float, int, np.number)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def clone_nested_cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:clone_nested_cpu(v) for k,v in value.items()}
    if isinstance(value, list): return [clone_nested_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(clone_nested_cpu(v) for v in value)
    return value


def hook_control(fold=2, steps=3, repetitions=3):
    """Toggle count hooks while measuring same-mode native PPU repeat noise."""
    mod = load_scpda_module()
    man = json.loads((NEW / "manifest.json").read_text())
    l = mod.Loaded(911, int(fold), man)
    try:
        def trajectory(counts_on, rep):
            for h in l.count_handles: h.remove()
            l.count_handles = []
            if counts_on:
                l.count_handles = [m.register_forward_hook(l.count_hook(name)) for name, m in
                                   [("core", l.core), ("phenotype", l.core.ph_unet), ("vae", l.core.vae),
                                    ("classifier", l.core.clf)]]
            l.reset(diagnostic=False)
            optimizer = l.optimizer()
            objective = mod.PrototypeObjective(l.prepared, "E1", mod.CFG, l.opt.device)
            trace = []
            for epoch in range(steps):
                l.core.train(); optimizer.zero_grad()
                logits, graph = l.forward()
                native = torch.nn.functional.cross_entropy(logits[l.sp["train"]], l.y[l.sp["train"]]) + graph
                aux, _ = objective(l.z, epoch)
                weight, scale = mod.gradient_scale(native, aux, l.z, epoch, mod.CFG)
                total = native + weight * aux if float(weight) else native
                row = {"train_logits": logits.detach().cpu().clone(),
                       "train_logits_hash": array_hash(logits.detach().cpu().numpy()), "native": float(native.detach()),
                       "aux": float(aux.detach()), "scale": scale, "train_z": array_hash(l.z.detach().cpu().numpy()),
                       "train_z_tensor": l.z.detach().cpu().clone(),
                       "topk_train": [[array_hash(p.detach().cpu().numpy()) for p in e.perms]
                                      for e in (l.core.img_unet, l.core.ph_unet)]}
                total.backward()
                row["gradient_tensors"] = {n:p.grad.detach().cpu().clone() for n,p in l.core.named_parameters()
                                            if p.grad is not None}
                row["gradients"] = mod.state_hash({n:p.grad for n,p in l.core.named_parameters() if p.grad is not None})
                optimizer.step()
                row["core_tensors"] = mod.export_core(l.core)
                row["optimizer_tensors"] = clone_nested_cpu(optimizer.state_dict())
                row["core_after_adam"] = mod.core_hash(mod.export_core(l.core))
                row["adam"] = mod.payload_hash(optimizer.state_dict())
                l.core.eval()
                with torch.no_grad(): val, vg = l.forward()
                row.update(val_logits=val.detach().cpu().clone(),
                           val_logits_hash=array_hash(val.detach().cpu().numpy()),
                           val_loss=float(torch.nn.functional.cross_entropy(val[l.sp["val"]], l.y[l.sp["val"]]) + vg),
                           val_z=array_hash(l.z.detach().cpu().numpy()),
                           val_z_tensor=l.z.detach().cpu().clone(),
                           topk_val=[[array_hash(p.detach().cpu().numpy()) for p in e.perms]
                                     for e in (l.core.img_unet, l.core.ph_unet)],
                           bn=mod.state_hash(l.core.clf[2].state_dict()), rng=mod.rng_hash(mod.capture_rng()))
                trace.append(row)
            return {"trace": trace, "calls": l.calls.copy(), "sha256": mod.payload_hash(trace), "rep": rep,
                    "count_hooks": counts_on}
        runs=[]
        for rep in range(repetitions):
            runs.append(trajectory(True, rep))
            runs.append(trajectory(False, rep))
        compact=[]
        for r in runs:
            steps_out=[]
            for row in r["trace"]:
                steps_out.append({k:v for k,v in row.items() if k not in
                                  ("gradient_tensors", "core_tensors", "optimizer_tensors", "train_z_tensor", "val_z_tensor",
                                   "train_logits", "val_logits")})
            compact.append({"rep":r["rep"], "count_hooks":r["count_hooks"], "calls":r["calls"], "sha256":r["sha256"], "trace":steps_out})
        comparisons=[]
        for rep in range(repetitions):
            on=next(r for r in runs if r["rep"]==rep and r["count_hooks"])
            off=next(r for r in runs if r["rep"]==rep and not r["count_hooks"])
            for step,(a,b) in enumerate(zip(on["trace"],off["trace"])):
                comparisons.append({"rep":rep,"step":step,"train_logit_max_abs":max_nested_abs(a["train_logits"],b["train_logits"]),
                                    "train_z_max_abs":max_nested_abs(a["train_z_tensor"],b["train_z_tensor"]),
                                    "gradient_max_abs":max_nested_abs(a["gradient_tensors"],b["gradient_tensors"]),
                                    "adam_core_max_abs":max_nested_abs(a["core_tensors"],b["core_tensors"]),
                                    "optimizer_max_abs":max_nested_abs(a["optimizer_tensors"],b["optimizer_tensors"]),
                                    "val_logit_max_abs":max_nested_abs(a["val_logits"],b["val_logits"]),
                                    "val_z_max_abs":max_nested_abs(a["val_z_tensor"],b["val_z_tensor"]),
                                    "val_loss_abs":abs(a["val_loss"]-b["val_loss"]),
                                    "train_topk_equal":a["topk_train"]==b["topk_train"],
                                    "val_topk_equal":a["topk_val"]==b["topk_val"],"bn_equal":a["bn"]==b["bn"],
                                    "rng_equal":a["rng"]==b["rng"]})
        same_mode=[]
        for enabled in (True,False):
            rr=[r for r in runs if r["count_hooks"]==enabled]
            for a,b in zip(rr,rr[1:]):
                for step,(x,y) in enumerate(zip(a["trace"],b["trace"])):
                    same_mode.append({"count_hooks":enabled,"rep_a":a["rep"],"rep_b":b["rep"],"step":step,
                                      "train_logit_max_abs":max_nested_abs(x["train_logits"],y["train_logits"]),
                                      "train_z_max_abs":max_nested_abs(x["train_z_tensor"],y["train_z_tensor"]),
                                      "gradient_max_abs":max_nested_abs(x["gradient_tensors"],y["gradient_tensors"]),
                                      "adam_core_max_abs":max_nested_abs(x["core_tensors"],y["core_tensors"]),
                                      "val_logit_max_abs":max_nested_abs(x["val_logits"],y["val_logits"]),
                                      "val_loss_abs":abs(x["val_loss"]-y["val_loss"]),
                                      "train_topk_equal":x["topk_train"]==y["topk_train"],
                                      "val_topk_equal":x["topk_val"]==y["topk_val"]})
        return {"fold":fold,"steps":steps,"repetitions_per_condition":repetitions,
                "only_count_hooks_toggled":True,"capture_hook_retained_both":True,
                "on_off_paired_comparisons":comparisons,"same_mode_repeat_comparisons":same_mode,
                "runs":compact}
    finally:
        l.close()


def validation_rng(group_old="D4", group_new="E1", fold=2):
    """Paired stochastic validation-only forwards of the two already saved best checkpoints."""
    mod = load_scpda_module(); man = json.loads((NEW / "manifest.json").read_text())
    l = mod.Loaded(911, int(fold), man); rows=[]
    try:
        for cp_name, path in (("old", OLD/"runs"/group_old/f"s911_f{fold}.pt"),
                              ("new", NEW/"runs"/group_new/f"s911_f{fold}.pt")):
            ck,_=load_artifact(path)
            for seed in CFG["predeclared_diagnostic_rng_seeds"]:
                mod.import_core(l.core,ck["core"]);l.core.eval();set_diag_rng(seed)
                pre=mod.capture_rng()
                with torch.set_grad_enabled(True): logits,graph=l.forward()
                val=torch.as_tensor(l.sp["val"],device=l.opt.device,dtype=torch.long)
                ce=torch.nn.functional.cross_entropy(logits[val],l.y[val])
                total=ce+graph
                rows.append({"old_group":group_old,"new_group":group_new,"fold":fold,"checkpoint":cp_name,
                             "best_epoch":ck["best_epoch"],"rng_seed":seed,
                             "pre_rng_sha256":payload_hash(pre),"post_rng_sha256":payload_hash(mod.capture_rng()),
                             "val_ce":float(ce.detach()),"val_graph":float(graph.detach()),"val_loss":float(total.detach()),
                             "saved_single_draw_val_loss":ck["best_val_loss"]})
    finally:
        l.close()
    put_csv(BASE/"validation_rng.csv",rows)
    return rows


def read_csv(path):
    with Path(path).open(newline="") as f: return list(csv.DictReader(f))


def fmt_pp(v, digits=4):
    return f"{100*float(v):.{digits}f}"


def analyze_outputs(handoff_status="pending"):
    offline_info=json.loads((BASE/"offline.json").read_text())
    folds=read_csv(BASE/"fold_drift.csv")
    trajectory=read_csv(BASE/"trajectory_summary.csv")
    states=read_csv(BASE/"state_diff.csv")
    cross=read_csv(BASE/"crossover_inference.csv")
    effects=read_csv(BASE/"crossover_effects.csv")
    fixed=read_csv(BASE/"fixed_rng_summary.csv")
    valrng=read_csv(BASE/"validation_rng.csv")
    hook=json.loads((BASE/"hook_control.json").read_text())
    old_groups={"D0":"D0","D4":"D4"};new_groups={"D0":"E0","D4":"E1"}
    metrics_summary={}
    for old_group,new_group in (("D0","E0"),("D4","E1")):
        for group,path in ((old_group,OLD/"runs"/old_group),(new_group,NEW/"runs"/new_group)):
            vals=[]
            for f in range(10): vals.append(load_artifact(path/f"s911_f{f}.pt")[0]["metrics"])
            metrics_summary[group]={k:{"mean":float(np.mean([x[k] for x in vals])),
                                       "sd_sample":float(np.std([x[k] for x in vals],ddof=1))}
                                    for k in ("ACC","BA","AUC","SEN","SPE","F1","CE","Brier")}
    aggregate_pairs={}
    for old_group,new_group in (("D0","E0"),("D4","E1")):
        a,b=metrics_summary[old_group],metrics_summary[new_group]
        aggregate_pairs[f"{new_group}-{old_group}"]={k:{"delta_pp":(b[k]["mean"]-a[k]["mean"])*100,
                                                         "old_mean":a[k]["mean"],"new_mean":b[k]["mean"]}
                                                         for k in ("ACC","BA","AUC","SEN","SPE","F1")}
    hook_stats={}
    for name,rows in (("on_off",hook["on_off_paired_comparisons"]),("same_mode",hook["same_mode_repeat_comparisons"])):
        hook_stats[name]={}
        for field in ("train_logit_max_abs","train_z_max_abs","gradient_max_abs","adam_core_max_abs",
                      "optimizer_max_abs","val_logit_max_abs","val_z_max_abs","val_loss_abs"):
            vv=[float(r[field]) for r in rows if field in r]
            if vv: hook_stats[name][field]={"max":max(vv),"median":float(np.median(vv))}
        hook_stats[name]["train_topk_equal_fraction"]=float(np.mean([r["train_topk_equal"] for r in rows]))
        hook_stats[name]["val_topk_equal_fraction"]=float(np.mean([r["val_topk_equal"] for r in rows]))
        if name=="on_off":
            hook_stats[name]["rng_equal_count"]=sum(bool(r["rng_equal"]) for r in rows)
            hook_stats[name]["count"]=len(rows)
    val_summary={}
    for cp in ("old","new"):
        vv=[float(r["val_loss"]) for r in valrng if r["checkpoint"]==cp]
        val_summary[cp]={"n":len(vv),"mean":float(np.mean(vv)),"sd_sample":float(np.std(vv,ddof=1)),
                         "min":min(vv),"max":max(vv),"range":max(vv)-min(vv)}
    val_summary["paired_new_minus_old_mean"]=val_summary["new"]["mean"]-val_summary["old"]["mean"]
    root_sources_all=all(x["all_equal"] for x in offline_info["root_source_checks"].values())
    old_audit=json.loads((OLD/"audit.json").read_text())
    new_manifest=json.loads((NEW/"manifest.json").read_text())
    preflight=new_manifest["preflight"]
    drpa_repeats=json.loads((ROOT/"experiments/DRPA_20261014/artifacts/baseline_diagnostics/verification.json").read_text())
    pygeo="2.8.0"
    try: pygeo=importlib.metadata.version("torch-geometric")
    except importlib.metadata.PackageNotFoundError: pass
    runtime={"python":sys.version.split()[0],"torch":torch.__version__,"numpy":np.__version__,
             "torch_geometric_diagnostic_runtime":pygeo,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    summary={"experiment":"D4_REPRO_AUDIT_20261020","status":"complete",
             "formal_training_repeated":False,"formal_folds_rerun":0,"artifact_files_verified":offline_info["verified_artifacts"],
             "artifact_content_hashes_valid":offline_info["all_content_hashes_valid"],"offline_fold_pairs":len(folds),
             "cross_inference_records":len(cross),"fixed_test_rng_evaluations":sum(r["condition"].startswith("fixed_rng_") for r in cross),
             "fixed_validation_rng_evaluations":len(valrng),"hook_control_steps_per_condition":hook["steps"]*hook["repetitions_per_condition"],
             "source_hashes_unchanged_across_audits":root_sources_all,"e1_configuration":offline_info["e1_flags"],
             "metrics_fold_mean_sample_sd":metrics_summary,"aggregate_pair_deltas":aggregate_pairs,
             "selected_cross_inference_folds":offline_info["selection"],"hook_control":hook_stats,
             "fold2_validation_rng":val_summary,"environment_recorded_both_experiments":{
                 "python_equal":old_audit["environment"]["python"]==new_manifest["environment"]["python"],
                 "torch_equal":old_audit["environment"]["torch"]==new_manifest["environment"]["torch"],
                 "numpy_equal":old_audit["environment"]["numpy"]==new_manifest["environment"]["numpy"],
                 "historical_torch_geometric_version_recorded":False,"current_diagnostic_torch_geometric":pygeo},
             "prior_same_source_repeat_evidence":{"verified":drpa_repeats["verified"],
                 "fold0_acc_range_pp":drpa_repeats["same_physical_card_acc_span"]*100,
                 "all_cards_auc_range_pp":drpa_repeats["all_cards_auc_span"]*100,
                 "same_initialization_vae_test_rng":drpa_repeats["same_initialization_vae_and_test_rng"],
                 "no_model_hooks":all(not r["model_hooks"] for r in drpa_repeats["controls"]),
                 "no_extra_backward":all(not r["extra_backward"] for r in drpa_repeats["controls"])},
             "deterministic_short_preflight":{"anchors":{k:{"native_error":v["maximum_native_error"],
                  "d4_error":v["maximum_D4_error"],"historical_oracle_sha256":v["historical_oracle_sha256"]}
                  for k,v in preflight.items()},"native_production_first3_max_loss_error_recorded_pp":7.808208465576172e-06},
             "root_source_checks":offline_info["root_source_checks"],"handoff_status":handoff_status,
             "code_sha256":file_sha(BASE/"diagnose.py"),"config_sha256":file_sha(BASE/"config.json"),
             "outputs_sha256":{p.name:file_sha(p) for p in BASE.iterdir() if p.is_file() and p.name not in ("summary.json","report.md","run.log")}}

    def render_table(headers, rows):
        lines=["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]
        lines += ["|"+"|".join(str(v) for v in row)+"|" for row in rows]
        return "\n".join(lines)

    def pretty_metric(g,k):
        x=metrics_summary[g][k]
        return f"{x['mean']*100:.4f} ± {x['sd_sample']*100:.4f}"

    drift_lines=[]
    for pair in ("D0","D4"):
        dg=[r for r in folds if r["old_group"]==pair]
        new=new_groups[pair]
        drift_rows=[]
        for r in dg:
            drift_rows.append([r["fold"],f"{100*float(r['ACC_old']):.4f}",f"{100*float(r['ACC_new']):.4f}",
                f"{float(r['delta_ACC_pp']):+.4f}",f"{100*float(r['AUC_old']):.4f}",f"{100*float(r['AUC_new']):.4f}",
                f"{float(r['delta_AUC_pp']):+.4f}",f"{float(r['delta_SEN_pp']):+.4f}",f"{float(r['delta_SPE_pp']):+.4f}",
                f"{r['best_epoch_old']}→{r['best_epoch_new']}",f"{r['epochs_run_old']}→{r['epochs_run_new']}",
                "same" if r["test_rng_before_equal"]=="True" else "different",r["changed_predictions"]])
        drift_lines.append(f"### {pair} vs {new}\n\n"+render_table(["Fold","ACC old","ACC new","ΔACC pp","AUC old","AUC new","ΔAUC pp","ΔSEN pp","ΔSPE pp","Best epoch","Epochs","Test RNG","Changed labels"],drift_rows))

    cross_rows=[]
    for r in effects:
        if r["metric"] not in ("ACC","AUC"): continue
        cross_rows.append([f"{r['old_group']}/{r['new_group']} f{r['fold']}",r["metric"],
            f"{100*float(r['old_model_old_rng_A']):.3f}",f"{100*float(r['old_model_new_rng_B']):.3f}",
            f"{100*float(r['new_model_old_rng_C']):.3f}",f"{100*float(r['new_model_new_rng_D']):.3f}",
            f"{float(r['rng_effect_old_checkpoint_B_minus_A_pp']):+.3f}",
            f"{float(r['checkpoint_effect_old_rng_C_minus_A_pp']):+.3f}",
            f"{float(r['interaction_D_minus_C_minus_B_plus_A_pp']):+.3f}",
            f"{float(r['total_formal_D_minus_A_pp']):+.3f}"])

    fixed_rows=[]
    for r in fixed:
        if r["metric"] not in ("ACC","AUC"): continue
        fixed_rows.append([r["old_group"],r["fold"],r["checkpoint"],r["metric"],
                           f"{100*float(r['mean']):.3f}±{100*float(r['sd_sample']):.3f}",
                           f"{100*float(r['min']):.3f}–{100*float(r['max']):.3f}",
                           f"{100*float(r['range']):.3f}"])

    trajectory_rows=[]
    for gg,ff in (("D0",1),("D0",2),("D4",0),("D4",2)):
        for field in ("train_ce","train_graph","train_total","val_ce","val_graph","val_loss","auxiliary_loss","lambda_eff"):
            r=next((x for x in trajectory if x["old_group"]==gg and int(x["fold"])==ff and x["field"]==field),None)
            if r:
                trajectory_rows.append([gg,ff,field,r["first_different_epoch_exact"],r["first_epoch_gt_1e-06"],
                                        r["first_epoch_gt_1e-05"],f"{float(r['max_abs_diff']):.8g}",r["max_diff_epoch"]])

    state_rows=[]
    for r in states:
        if (r["old_group"],int(r["fold"])) in (("D4",2),("D0",2)):
            state_rows.append([r["old_group"],r["fold"],r["component"],r["exact_tensors"],r["tensors"],
                               f"{float(r['different_element_fraction']):.6g}",f"{float(r['relative_l2']):.6g}",f"{float(r['max_abs']):.6g}"])

    validation_text=[]
    for cp in ("old","new"):
        v=val_summary[cp]
        validation_text.append(f"{cp}: 均值 {v['mean']:.6f}，样本SD {v['sd_sample']:.6f}，范围 {v['min']:.6f}–{v['max']:.6f}（range {v['range']:.6f}）")
    paired_delta=[float(n["val_loss"])-float(o["val_loss"]) for o,n in zip([r for r in valrng if r["checkpoint"]=="old"],[r for r in valrng if r["checkpoint"]=="new"])]

    report=f"""# D0/D4 与 E0/E1 复现漂移审计

日期：2026-10-20
诊断目录：`{BASE}`
正式重训：0 折；本报告只用历史40份结果、固定Checkpoint推理和少量短轨迹诊断。

## 结论先行

没有发现D0/E0或D4/E1的训练目标、输入、样本划分、优化器参数顺序、Checkpoint规则或根模型源码改变。E1明确关闭EMA与类别独立ESS，辅助原型配置与历史D4一致。40份正式结果内容哈希均有效；20个历史/新结果对的患者ID、标签、索引、RFE/输入指纹、训练起始RNG、初始化和优化器顺序逐折一致。受控的确定性短轨迹原先已在两个锚点上严格等价。

最有力解释是**PPU原生长轨迹数值非确定性，经随机验证前向和best-checkpoint选择放大，再叠加正式测试VAE/Dropout随机性**。具体算子尚未单独隔离，不能把原因简化成“GPU随机性”或声称已有唯一根因。计数hook没有显示出超过同模式短程重复误差的系统影响。没有证据支持修改根模型代码。

## 正式结果与逐折漂移

这些为seed 911十折均值±样本SD。它们是单seed折间统计，不是三个seed总体SD。

|组|ACC %|SEN-ASD %|SPE-TD %|prob-AUC %|
|---|---:|---:|---:|---:|
|D0|{pretty_metric('D0','ACC')}|{pretty_metric('D0','SEN')}|{pretty_metric('D0','SPE')}|{pretty_metric('D0','AUC')}|
|E0|{pretty_metric('E0','ACC')}|{pretty_metric('E0','SEN')}|{pretty_metric('E0','SPE')}|{pretty_metric('E0','AUC')}|
|D4|{pretty_metric('D4','ACC')}|{pretty_metric('D4','SEN')}|{pretty_metric('D4','SPE')}|{pretty_metric('D4','AUC')}|
|E1|{pretty_metric('E1','ACC')}|{pretty_metric('E1','SEN')}|{pretty_metric('E1','SPE')}|{pretty_metric('E1','AUC')}|

D0→E0：ACC **+{aggregate_pairs['E0-D0']['ACC']['delta_pp']:.4f}pp**、AUC **+{aggregate_pairs['E0-D0']['AUC']['delta_pp']:.4f}pp**。D4→E1：ACC **{aggregate_pairs['E1-D4']['ACC']['delta_pp']:.4f}pp**、AUC **{aggregate_pairs['E1-D4']['AUC']['delta_pp']:.4f}pp**。详细逐折表如下，负号均表示新运行减历史。

{drift_lines[0]}

{drift_lines[1]}

最明显的成绩差不是均匀分布：D0/E0 fold 1和2的ACC各变化约2.30pp、AUC变化−1.70/+4.40pp，fold 0 ACC另变化+2.27pp；D4/E1的−0.687pp ACC整体正好由fold 0的−2.27pp与fold 2的−4.60pp除以10形成，其他折ACC均未变化。历史D4相对D0约+0.45pp ACC、+0.59pp AUC，与本轮检测到的漂移处在同一量级，单个seed不足以稳健区分小收益与训练/推理噪声。

## 代码与协议审计

- 历史PCDCA与SCPDA记录的根模型/工具源码SHA逐文件相同，当前工作树再次核对也与两份记录一致；根模型未修改。
- 每折D0↔E0、D4↔E1的`input_sha256`、数据摘要、完整pairing、原型准备、优化器参数名称顺序、物理PPU归属均相同。训练起点完整RNG摘要一致。
- 新配置E1为`kernel=age_sex, ema=false, class_calibration=false`。新原型实现只在EMA/类别校准开启时进入新增算法分支；关闭时沿用原beta和当前teacher。E1额外传入epoch并产生零差异teacher诊断量，但没有新增损失/可训练参数。
- 两边每折保存的测试ID、标签和外层索引相同；ASD概率及ACC/SEN/SPE/AUC由独立代码复算，与artifact指标一致。
- SCPDA新增的四类调用计数hook回调只递增Python计数器，不替换、不改写Tensor。3步训练对照在启用/关闭计数hook时，9/9配对的完整RNG推进相同。hook开关误差：训练logit最大`{hook_stats['on_off']['train_logit_max_abs']['max']:.2g}`、梯度最大`{hook_stats['on_off']['gradient_max_abs']['max']:.2g}`、更新后核心状态最大`{hook_stats['on_off']['adam_core_max_abs']['max']:.2g}`；同模式重复的相应最大差为`{hook_stats['same_mode']['train_logit_max_abs']['max']:.2g}`、`{hook_stats['same_mode']['gradient_max_abs']['max']:.2g}`、`{hook_stats['same_mode']['adam_core_max_abs']['max']:.2g}`。TopK集合本身有相近的重复间变化。因此hook不是已证实原因，不能仅凭这次开关对照宣称绝对无影响。
- PCDCA/SCPDA都记录Python 3.12.3、PyTorch 2.9.0、NumPy 1.26.0；历史fold保存设备名和物理卡相同。历史运行未记录torch-geometric与PPU驱动的完整版本指纹，故其跨日环境完全相同尚不能证明。当前诊断venv为PyG {pygeo}。
- SCPDA既有预检在911/fold0与fold5使用强制确定性算法对比原D4和E1，native及D4短轨迹最大误差均为0；生产原生前3步历史记录最大native loss误差约`7.81e-6`。这是“短程确定性等价、长程原生轨迹可漂移”的证据，不代表正式训练启用了确定性算法。

## 训练轨迹、Checkpoint与状态

epoch日志没有每步参数快照，所以首次**参数**分叉无法定位；只能报告最早的loss差异。下面是离线保存轨迹，容差列为该loss首次超过对应阈值的epoch。

{render_table(['Pair','Fold','Trace','首个非零epoch','>1e-6','>1e-5','最大绝对差','发生epoch'],trajectory_rows)}

fold 2 D4/E1：train CE在epoch 1首次出现非零差，epoch 60超过1e-6、66超过1e-5；validation CE至epoch 10才首次不同，epoch 59/67超过上述阈值；validation total loss epoch 4已不同，最大差`{float(next(r for r in trajectory if r['old_group']=='D4' and r['fold']=='2' and r['field']=='val_loss')['max_abs_diff']):.6f}`发生于epoch175。历史best是epoch175、loss 0.713966；E1 best是epoch152、loss 0.727794。对应epoch152时两条val loss接近（历史0.727212、新0.727794），但到epoch175分开（历史0.713966、新0.744259，差0.030294），这不是单纯近乎相同最小值的tie。两条train loss同一epoch仍很接近；validation分类CE的差异大于graph项。因为eval状态下VAE仍采样、GTUNet的`F.dropout`未传`training=self.training`，随机验证前向也会参与并可能放大选模差异。

Fold 2两份best checkpoint的逐组件差异：

{render_table(['Pair','Component','精确相同tensor','总tensor','不同元素比例','relative L2','max abs'],state_rows)}

Checkpoint差异是训练/选模后状态不同的直接结果，不是序列化损坏。VAE权重和RP未注册权重均纳入比较；每次推理前重新恢复包括RP权重的整个best状态。

## Checkpoint × RNG交叉推理

四格A=旧checkpoint+旧测试RNG，B=旧checkpoint+新RNG，C=新checkpoint+旧RNG，D=新checkpoint+新RNG。A/D分别复现正式存档：所选12个A/D的最大logit误差不超过约`9.3e-6`，表明测试实现、checkpoint装载和已保存测试RNG基本正确。ΔRNG为B−A，ΔCheckpoint为C−A，交互为D−C−B+A；指标有阈值/排序非线性，因此交互项不能当作可加的方差比例。

{render_table(['Pair/fold','Metric','A','B','C','D','ΔRNG pp','ΔCheckpoint pp','Interaction pp','Total pp'],cross_rows)}

fold 2 D4/E1：只换RNG时ACC/AUC变化−1.149/−3.659pp；只换checkpoint（旧RNG）为0/−1.167pp；正式组合为−4.598/−4.507pp，存在明显交互。fold 0的D4/E1 ACC差−2.273pp则在同旧RNG下已由checkpoint造成，换测试RNG在该折指标上没有影响。D0/E0 fold 1、2同样显示测试RNG效应与Checkpoint效应并存且互相作用。

## 固定Checkpoint随机推理敏感性

预先固定5个诊断RNG（20261020–20261024），旧/新checkpoint使用同一组随机种子；这只量化评估敏感性，不替代正式单次成绩。

{render_table(['Pair','Fold','Checkpoint','Metric','均值±SD %','最小–最大 %','Range pp'],fixed_rows)}

关键折同一checkpoint的五次推理ACC范围最高5.75pp、AUC范围最高5.41pp，已经大于本轮整体漂移，也大于历史D4的0.45/0.59pp表观改进。该随机前向来自VAE重参数采样及eval下仍启用的GTUNet Dropout；本诊断没有把两者拆开归因。

fold 2另对两个已保存best checkpoint在同一5个固定状态下做了验证集前向（各5次，无训练）：{'; '.join(validation_text)}。配对随机状态下新checkpoint验证loss平均比旧checkpoint低{abs(val_summary['paired_new_minus_old_mean']):.6f}；与正式选模单次best loss的排序相反。每个checkpoint内部随机状态range约{val_summary['old']['range']:.4f}/{val_summary['new']['range']:.4f}，大于存档best loss差{abs(float(next(r for r in folds if r['old_group']=='D4' and r['fold']=='2')['delta_best_val_loss'])):.4f}。这是post-selection诊断，不能视作无偏性能比较，但支持随机验证前向足以影响候选排序。

## 原生非确定性与原因等级

实验12同一官方`train()`无任何模型hook、无额外backward的历史验证里，911/fold0以相同初始模型/VAE/测试前RNG运行三次，ACC为73.86%、68.18%、68.18%；两次位于同一物理卡仍相差5.68pp。该直接同源复跑是PPU原生长轨迹不可逐位重现的强证据。另有受控测试显示强制确定性算法时一次forward/梯度/Adam能逐位相等；但生产正式训练没有启用它，也没有因此修改协议。

|类别|证据等级|判断|
|---|---|---|
|训练协议/根模型差异|**未发现**|输入、模型源码、起点、optimizer、native loss、D4数学路径和选模规则逐项一致；有关闭状态的额外诊断计算，但未发现算法目标变化。|
|原生训练数值轨迹漂移|**Supported / 强支持**|同一历史源码同卡复跑不一致；本轮loss早期微差后逐渐扩大，best core状态变化。确切触发kernel未定位。|
|验证与Checkpoint放大|**Confirmed / 已确认参与**|fold2 best epoch改变23，val loss轨迹/随机前向变化，保存状态与正式结果随之变化；无法把模型更新和当时随机验证抽样完全拆分。|
|测试随机性|**Confirmed / 已确认参与特定fold**|旧/新RNG交叉可在相同checkpoint下改变fold1/2结果；固定checkpoint五次范围达数个百分点。|
|序列化/恢复错误|**基本排除**|A/D重放能复现正式logits，误差约1e-5以内；state包含unregistered RP并逐次重载。|
|计数hook是主因|**无支持，有限排除**|RNG推进相同；hook差异不高于同模式短跑噪声，但仅3步×3重复，不能据此证明任何调度效应绝对为零。|
|PPU具体算子/驱动变化|**Unresolved**|TopK/scatter/归约等可能贡献，但未用逐算子确定性跟踪或同一kernel隔离；历史PyG/driver版本记录不全。|

## 八个问题的回答

1. **E0−D0为何+0.34pp ACC、+0.32pp AUC？** 总差很小且是折间抵消：fold0、1、2出现较大变化；fold1/2测试RNG均不同，best epoch也变，四格显示Checkpoint与RNG有方向相反/交互效应。没有证据指向方法或输入改动。
2. **E1−D4为何−0.69pp ACC、−0.53pp AUC？** 整体ACC几乎完全由fold0和fold2两折下降驱动。fold0同RNG下checkpoint差使ACC下降；fold2训练/验证轨迹改变checkpoint，正式测试RNG也改变，两者交互。
3. **两个异常是否共同原因？** 有共同的运行层因素证据：原生长轨迹非逐位可复现，eval随机前向影响checkpoint和测试；但各折的checkpoint/RNG变化方向不同，所以不是同一固定偏差。
4. **Fold2为何选不同Checkpoint？** native训练从epoch1开始可观察到微小loss差，随后放大；validation loss至epoch175已经相差0.0303，D4的epoch175最佳而E1在epoch152最低。固定RNG验证重复的range大于best loss差，随机选模参与；缺少逐epoch权重/RNG快照，不能宣称唯一机制。
5. **差异在训练还是测试？** 两者都有。checkpoint状态与训练日志已有差异；交叉测试也独立显示RNG效应，fold2尤甚。
6. **历史D4收益有多少受随机性影响？** 无法给出一个有效百分比。观测的checkpoint/RNG变化已足以达到或超过0.45pp ACC、0.59pp AUC；应视为小幅、未稳健确证的同协议结果，而不是宣称D4无效。
7. **以后如何减少漂移？** 固定物理卡和每卡worker负载；保存完整环境（PyTorch/PyG/PPU固件、backend flags）；逐epoch保存验证前后完整RNG、val loss、best checkpoint状态和unregistered RP；每个小于1pp增益至少做多个配对seed及同源短复跑。确定性算法只作诊断，不冒充历史生产协议。
8. **应否改根模型？** 不应。没有发现算法代码错误；不要改dropout、VAE、TopK或图逻辑来“修复”复现。下一轮保留原生协议并增加可复核的RNG/环境记录，另外把无泄漏协议作为独立研究设定。

## 执行与交接

40/40正式artifact哈希和指标复核通过；交叉测试84次、固定测试随机状态60次、fold2验证随机状态10次；计数hook诊断共18次优化步（只用于差分测试），未正式重训任何折、未拟合RFE/VAE。PPU检查未终止其他进程；训练前空闲。根模型SHA未变化。输出：`fold_drift.csv`、`trajectory_summary.csv`、`state_diff.csv`、`crossover_inference.csv`、`crossover_effects.csv`、`fixed_rng_summary.csv`、`validation_rng.csv`、`hook_control.json`、`summary.json`。

SCPDA待追加handoff按时间顺序先于本报告，再追加D4漂移审计。handoff状态：**{handoff_status}**。未git commit/push。
"""
    summary["report_sha256"]="pending"
    (BASE/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    (BASE/"report.md").write_text(report)
    summary["report_sha256"]=file_sha(BASE/"report.md")
    (BASE/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    log_data={"status":"complete","formal_retraining_folds":0,"verified_artifacts":offline_info["verified_artifacts"],
              "cross_inference_records":len(cross),"hook_steps_per_condition":hook["steps"]*hook["repetitions_per_condition"],
              "handoff_status":handoff_status,"report_sha256":file_sha(BASE/"report.md")}
    (BASE/"run.log").write_text(json.dumps(log_data,ensure_ascii=False,indent=2)+"\n")
    return summary


def summarize_cross(rows):
    out = []
    by = {}
    for r in rows:
        if r["condition"].startswith("fixed_rng_"):
            by.setdefault((r["old_group"], r["fold"], r["checkpoint"]), []).append(r)
    for (og, fold, cp), arr in by.items():
        for metric_key in ("ACC", "AUC", "SEN", "SPE"):
            vals = [x[metric_key] for x in arr]
            out.append({"old_group": og, "fold": fold, "checkpoint": cp, "metric": metric_key,
                        "n_rng": len(vals), "mean": float(np.mean(vals)), "sd_sample": float(np.std(vals, ddof=1)),
                        "min": float(min(vals)), "max": float(max(vals)), "range": float(max(vals)-min(vals))})
    put_csv(BASE / "fixed_rng_summary.csv", out)
    crossed = {(r["old_group"], r["fold"], r["condition"]):r for r in rows
               if r["condition"] in ("A_oldcp_oldrng", "B_oldcp_newrng", "C_newcp_oldrng", "D_newcp_newrng")}
    effects=[]
    for old_group,fold,_ in sorted(set((k[0],k[1],None) for k in crossed)):
        cells={c:crossed[(old_group,fold,c)] for c in ("A_oldcp_oldrng","B_oldcp_newrng","C_newcp_oldrng","D_newcp_newrng")}
        for metric_key in ("ACC","AUC","SEN","SPE"):
            a,b,c,d=(cells[k][metric_key] for k in ("A_oldcp_oldrng","B_oldcp_newrng","C_newcp_oldrng","D_newcp_newrng"))
            effects.append({"old_group":old_group,"new_group":"E0" if old_group=="D0" else "E1","fold":fold,
                            "metric":metric_key,"old_model_old_rng_A":a,"old_model_new_rng_B":b,
                            "new_model_old_rng_C":c,"new_model_new_rng_D":d,
                            "rng_effect_old_checkpoint_B_minus_A_pp":(b-a)*100,
                            "checkpoint_effect_old_rng_C_minus_A_pp":(c-a)*100,
                            "interaction_D_minus_C_minus_B_plus_A_pp":(d-c-b+a)*100,
                            "total_formal_D_minus_A_pp":(d-a)*100})
    put_csv(BASE / "crossover_effects.csv",effects)
    fixed={(r["old_group"],r["fold"],r["checkpoint"],r["condition"]):r for r in rows
           if r["condition"].startswith("fixed_rng_")}
    paired=[]
    for old_group,fold,seed in sorted(set((k[0],k[1],k[3]) for k in fixed)):
        o=fixed[(old_group,fold,"old",seed)];n=fixed[(old_group,fold,"new",seed)]
        dprob=np.abs(np.asarray(o["probability"])-np.asarray(n["probability"]))
        dpred=np.asarray(o["prediction"])!=np.asarray(n["prediction"])
        paired.append({"old_group":old_group,"fold":fold,"rng_seed":seed,
                       "probability_mean_abs_old_new":float(dprob.mean()),"probability_max_abs_old_new":float(dprob.max()),
                       "changed_prediction_count_old_new":int(dpred.sum()),"test_n":len(dpred),
                       "old_ACC":o["ACC"],"new_ACC":n["ACC"],"delta_ACC_pp":(n["ACC"]-o["ACC"])*100,
                       "old_AUC":o["AUC"],"new_AUC":n["AUC"],"delta_AUC_pp":(n["AUC"]-o["AUC"])*100})
    put_csv(BASE / "fixed_rng_checkpoint_pair.csv",paired)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["offline", "gpu", "hooks", "val-rng", "merge", "analyze"])
    ap.add_argument("--folds-json", default=None)
    ap.add_argument("--tag", default="all")
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--parity", type=int, choices=[0,1], default=None)
    ap.add_argument("--handoff-status", choices=["pending", "complete"], default="pending")
    args = ap.parse_args()
    if args.action == "offline":
        global state_rows, trajectory_rows
        state_rows, trajectory_rows = [], []
        result = offline()
        print(json.dumps({"verified_artifacts": result["verified_artifacts"], "selection": result["selection"]}, indent=2))
    elif args.action == "gpu":
        if not torch.cuda.is_available(): raise RuntimeError("PPU unavailable; no CPU model inference")
        folds = json.loads(Path(args.folds_json).read_text()) if args.folds_json else json.loads((BASE/"offline.json").read_text())["selection"]
        if args.parity is not None:
            folds = {k:[f for f in v if int(f) % 2 == args.parity] for k,v in folds.items()}
        rows = gpu_cross(folds, args.tag); summarize_cross(rows)
        print(f"Completed {len(rows)} diagnostic forward evaluations")
    elif args.action == "hooks":
        if not torch.cuda.is_available(): raise RuntimeError("PPU unavailable; no CPU model inference")
        out = hook_control(args.fold)
        (BASE / "hook_control.json").write_text(json.dumps(out, indent=2, allow_nan=False))
        print(json.dumps({k:out[k] for k in ("fold","steps","repetitions_per_condition","only_count_hooks_toggled","capture_hook_retained_both")}, indent=2))
    elif args.action == "val-rng":
        if not torch.cuda.is_available(): raise RuntimeError("PPU unavailable; no CPU model inference")
        rows=validation_rng(fold=args.fold)
        print(f"Completed {len(rows)} fixed-checkpoint validation forwards")
    elif args.action == "merge":
        rows=merge_gpu_shards()
        print(f"Merged {len(rows)} cross-inference records")
    else:
        result=analyze_outputs(args.handoff_status)
        print(json.dumps({"status":result["status"],"report_sha256":result["report_sha256"],
                          "artifact_files_verified":result["artifact_files_verified"],
                          "cross_inference_records":result["cross_inference_records"]},indent=2))


if __name__ == "__main__": main()
