"""Array-first, resumable diagnosis of immutable formal DRPA experiments."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.special import expit, softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "experiments/DRPA_20261014"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT))
import run as historical

EXPECTED = "9186fe8b76142e6ef114d850409cb7d8e9ac5f2691d8e132647c389ee60e5572"
REFERENCE = "a6471c54477d663b6321c706b21173e1db979b7e"
METHODS = [f"E{i}" for i in range(8)]
SUBSETS = ("train", "val", "test")
PRIMARY = {"E0": ["raw"], "E1": ["si_sp"], "E2": ["si_sp"],
           "E3": ["ti_sp", "si_tp"], **{f"E{i}": ["si_tp"] for i in range(4, 8)}}
PAIRS = {"raw": ("img_embed", "ph_embed"), "si_sp": ("si", "sp"),
         "si_tp": ("si", "tp"), "ti_sp": ("ti", "sp")}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return historical.sha_file(path)


def atomic_json(value):
    target = HERE / "summary.json"
    tmp = HERE / f"summary.json.tmp.{os.getpid()}"
    try:
        with tmp.open("w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "std": None, "median": None, "p10": None, "p90": None}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()), "median": float(np.median(x)),
            **{f"p{p}": float(np.quantile(x, p / 100)) for p in (10, 25, 50, 75, 90)}}


def mean_tree(records):
    records = [r for r in records if r is not None]
    if not records:
        return None
    if all(isinstance(r, dict) for r in records):
        keys = sorted(set().union(*(r.keys() for r in records)))
        return {k: mean_tree([r[k] for r in records if k in r]) for k in keys}
    if all(isinstance(r, (int, float, np.number, bool)) for r in records):
        return float(np.mean(records))
    if all(isinstance(r, list) for r in records) and len({len(r) for r in records}) == 1:
        return [mean_tree(list(values)) for values in zip(*records)]
    return records[0]


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def effective_rank(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(0)
    eigen = np.maximum(np.linalg.eigvalsh(centered.T @ centered / max(len(x), 1)), 0)
    total = float(eigen.sum())
    if total == 0:
        return {"effective_rank": 0.0, "top_eigenvalue_share": 0.0,
                "total_variance": 0.0, "zero_variance": True}
    p = eigen[eigen > 0] / total
    return {"effective_rank": float(np.exp(-np.sum(p * np.log(p)))),
            "top_eigenvalue_share": float(eigen[-1] / total),
            "total_variance": total, "zero_variance": False}


def mean_mask(values, mask):
    return float(values[mask].mean()) if np.any(mask) else None


def representation(x, labels, sites):
    x = np.asarray(x, dtype=np.float64)
    mu, n = x.mean(0), len(x)
    class_means = {int(c): x[labels == c].mean(0) for c in np.unique(labels)}
    between = sum(int(np.sum(labels == c)) * np.sum((v - mu) ** 2) for c, v in class_means.items()) / n
    within = sum(np.sum((x[labels == c] - v) ** 2) for c, v in class_means.items()) / n
    site_between = sum(int(np.sum(sites == s)) * np.sum((x[sites == s].mean(0) - mu) ** 2)
                       for s in np.unique(sites)) / n
    total = float(np.sum((x - mu) ** 2) / n)
    squared = np.sum(x * x, 1)
    distances = np.maximum(squared[:, None] + squared[None, :] - 2 * (x @ x.T), 0)
    off = ~np.eye(n, dtype=bool)
    same_site = sites[:, None] == sites[None, :]
    same_class = labels[:, None] == labels[None, :]
    return {"n": n, "mean_variance": float(x.var(0).mean()),
            "per_dimension_variance": x.var(0).tolist(), **effective_rank(x),
            "class_center_distance": float(np.linalg.norm(class_means[0] - class_means[1])) if len(class_means) == 2 else None,
            "class_between_variance": float(between), "class_within_variance": float(within),
            "class_between_within_ratio": float(between / within) if within > 0 else None,
            "class_variance_fraction": float(between / total) if total > 0 else None,
            "site_variance_fraction": float(site_between / total) if total > 0 else None,
            "same_site_squared_distance": mean_mask(distances, off & same_site),
            "cross_site_squared_distance": mean_mask(distances, off & ~same_site),
            "same_class_squared_distance": mean_mask(distances, off & same_class),
            "different_class_squared_distance": mean_mask(distances, off & ~same_class)}


def recall_at(similarity, targets, k):
    target_scores = similarity[np.arange(len(targets)), targets]
    greater = np.sum(similarity > target_scores[:, None], axis=1)
    tied = np.sum(similarity == target_scores[:, None], axis=1)
    # Exact ties receive their expected share under uniform random tie-breaking.
    return float(np.mean(np.clip((k - greater) / np.maximum(tied, 1), 0, 1)))


def permutation(groups, rng):
    result = np.arange(len(groups))
    for value in np.unique(groups):
        members = np.where(groups == value)[0]
        if len(members) > 1:
            shuffled = rng.permutation(members)
            result[shuffled] = np.roll(shuffled, 1)
    return result


def private_rng(*parts):
    return np.random.default_rng(int(hashlib.sha256(repr(parts).encode()).hexdigest()[:16], 16))


def alignment(x, y, labels, sites, identity):
    x, y = unit(x), unit(y)
    similarity, n = x @ y.T, len(x)
    off = ~np.eye(n, dtype=bool)
    targets = np.arange(n)
    same = float(np.trace(similarity) / n)
    different = mean_mask(similarity, off)
    result = {"n": n, "same_cos": same, "different_cos": different, "cos_gap": same - different,
              "r1_img_to_ph": recall_at(similarity, targets, 1),
              "r1_ph_to_img": recall_at(similarity.T, targets, 1),
              "r5_img_to_ph": recall_at(similarity, targets, 5),
              "r5_ph_to_img": recall_at(similarity.T, targets, 5),
              "chance_r1": 1 / n, "chance_r5": min(5 / n, 1),
              "variance_unit_img": float(x.var(0).mean()), "variance_unit_ph": float(y.var(0).mean()),
              "effective_rank_img": effective_rank(x)["effective_rank"],
              "effective_rank_ph": effective_rank(y)["effective_rank"], "controls": {}}
    result["r1_bidirectional"] = (result["r1_img_to_ph"] + result["r1_ph_to_img"]) / 2
    result["r5_bidirectional"] = (result["r5_img_to_ph"] + result["r5_ph_to_img"]) / 2
    for name, equal in (("class", labels[:, None] == labels[None, :]), ("site", sites[:, None] == sites[None, :])):
        result[f"same_{name}_cos"] = mean_mask(similarity, off & equal)
        result[f"different_{name}_cos"] = mean_mask(similarity, off & ~equal)
    controls = {"random": np.zeros(n), "within_class": labels, "within_site": sites,
                "within_class_site": labels * (int(np.max(sites)) + 1) + sites}
    for name, groups in controls.items():
        perm = permutation(groups, private_rng(*identity, name))
        moved = perm != targets
        inverse = np.argsort(perm)
        positive = similarity[targets, perm]
        result["controls"][name] = {
            "moved_fraction": float(moved.mean()), "same_cos": float(positive.mean()),
            "same_cos_moved_only": float(positive[moved].mean()) if moved.any() else None,
            "correct_minus_control_cos": same - float(positive.mean()),
            "r1_img_to_ph": recall_at(similarity, perm, 1),
            "r1_ph_to_img": recall_at(similarity.T, inverse, 1),
            "r5_img_to_ph": recall_at(similarity, perm, 5),
            "r5_ph_to_img": recall_at(similarity.T, inverse, 5),
            "negative_definition": "all true different identities; diagonal excluded before any permutation"}
    return result


def distribution(x):
    out = describe(x)
    x = np.asarray(x, dtype=np.float64)
    n, total = len(x), float(x.sum())
    out.update({"nonzero_fraction": float(np.mean(x > 0)),
                "cv": float(x.std() / abs(x.mean())) if x.mean() != 0 else None,
                "ess": float(total ** 2 / np.sum(x * x)) if np.sum(x * x) > 0 else 0.0,
                "ess_ratio": float(total ** 2 / (n * np.sum(x * x))) if np.sum(x * x) > 0 else 0.0,
                "top_decile_mass": float(np.sort(x)[-int(np.ceil(n / 10)):].sum() / total) if total > 0 else None})
    return out


def gate_moments(gate):
    w = gate["weights"]
    mean, var, n = w["mean"], w["std"] ** 2, w["n"]
    out = {**w, "cv": w["std"] / abs(mean) if mean else None,
           "ess_ratio": mean ** 2 / (mean ** 2 + var) if mean ** 2 + var > 0 else 0.0,
           "ess": n * mean ** 2 / (mean ** 2 + var) if mean ** 2 + var > 0 else 0.0,
           "q_pred": gate["q_pred"], "q_site": gate["q_site"],
           "classes": gate["classes"], "sites": gate["sites"],
           "p25": None, "p75": None, "top_decile_mass": None}
    for name, entries in (("class", gate["classes"]), ("site", {s: v["weights"] for s, v in gate["sites"].items()})):
        between = sum(v["n"] * (v["mean"] - mean) ** 2 for v in entries.values()) / n
        within = sum(v["n"] * v["std"] ** 2 for v in entries.values()) / n
        out[f"{name}_between_variance"] = between
        out[f"{name}_within_variance"] = within
        out[f"{name}_between_fraction"] = between / (between + within) if between + within > 0 else None
        for v in entries.values():
            if "ess_ratio" not in v:
                v["ess_ratio"] = v["mean"] ** 2 / (v["mean"] ** 2 + v["std"] ** 2) if v["mean"] ** 2 + v["std"] ** 2 > 0 else 0.0
    for key in ("shuffle_moved_fraction", "shuffle_changed_weight_fraction", "shuffle_cross_site_singletons"):
        if key in gate:
            out[key] = gate[key]
    return out


def posthoc_gate(z, indices, labels, sites, method):
    labels, sites = labels[indices], sites[indices]
    pi = softmax(z["logits_i"][indices].astype(np.float64), axis=1)
    pp = softmax(z["logits_p"][indices].astype(np.float64), axis=1)
    mean = (pi + pp) / 2
    jsd = ((pi * (np.log(np.maximum(pi, 1e-8)) - np.log(np.maximum(mean, 1e-8)))).sum(1)
           + (pp * (np.log(np.maximum(pp, 1e-8)) - np.log(np.maximum(mean, 1e-8)))).sum(1)) / 2
    q_pred = np.sqrt(pi[np.arange(len(indices)), labels] * pp[np.arange(len(indices)), labels]) * np.clip(1 - jsd / np.log(2), 0, 1)
    q_site, valid = np.ones(len(indices)), np.ones(len(indices), dtype=bool)
    if method in ("E6", "E7"):
        margins = []
        for key in ("si", "sp"):
            values = z[key][indices].astype(np.float64)
            margin = np.zeros(len(indices))
            for site in np.unique(sites):
                mask = sites == site
                pools = [values[(sites != site) & (labels == c)] for c in (0, 1)]
                if min(map(len, pools)) < 2:
                    valid[mask] = False
                prototypes = np.array([v.mean(0) if len(v) else np.zeros(values.shape[1]) for v in pools])
                if np.any(np.linalg.norm(prototypes, axis=1) <= 1e-8):
                    valid[mask] = False
                prototypes = unit(prototypes)
                sims = values[mask] @ prototypes.T
                yy = labels[mask]
                margin[mask] = sims[np.arange(len(yy)), yy] - sims[np.arange(len(yy)), 1 - yy]
            margins.append(margin)
        q_site = np.sqrt(expit(2 * margins[0]) * expit(2 * margins[1]))
        q_site[~valid] = 0
    raw = np.maximum(q_pred * q_site - 0.15, 0)
    weights = raw * (len(raw) / raw.sum()) if raw.sum() > 0 else np.zeros_like(raw)
    if method == "E4":
        weights = np.ones_like(weights)
        raw = np.ones_like(raw)
    out = {"q_pred": distribution(q_pred), "q_site": distribution(q_site),
           "raw_gate": distribution(raw), "weights": distribution(weights),
           "classes": {str(c): distribution(weights[labels == c]) for c in (0, 1)},
           "sites": {str(s): {"weights": distribution(weights[sites == s]),
                              "q_pred": distribution(q_pred[sites == s]), "q_site": distribution(q_site[sites == s])}
                     for s in np.unique(sites)}, "fallback_count": int(np.sum(~valid)),
           "weight_semantics": "unshuffled best-checkpoint post-hoc; not a historical training assignment"}
    overall = weights.mean()
    for name, groups in (("class", labels), ("site", sites)):
        between = sum(np.sum(groups == g) * (weights[groups == g].mean() - overall) ** 2 for g in np.unique(groups)) / len(weights)
        within = sum(np.sum((weights[groups == g] - weights[groups == g].mean()) ** 2) for g in np.unique(groups)) / len(weights)
        out[f"{name}_between_variance"] = float(between)
        out[f"{name}_within_variance"] = float(within)
        out[f"{name}_between_fraction"] = float(between / (between + within)) if between + within > 0 else None
    residual = weights.copy()
    for c in (0, 1):
        residual[labels == c] -= weights[labels == c].mean()
    residual_total = float(np.mean(residual ** 2))
    site_after_class = sum(np.sum(sites == s) * residual[sites == s].mean() ** 2 for s in np.unique(sites)) / len(weights)
    out["site_fraction_after_class_centering"] = float(site_after_class / residual_total) if residual_total > 0 else None
    return out


def prediction_description(logits, labels):
    probs = softmax(logits, axis=1)
    p, y, predicted = probs[:, 0], labels == 0, logits.argmax(1)
    conf, correct = probs.max(1), predicted == labels
    bins = []
    for lower in np.linspace(0, 0.9, 10):
        mask = (conf >= lower) & (conf < lower + 0.1 if lower < 0.9 else conf <= 1)
        bins.append({"lower": float(lower), "n": int(mask.sum()),
                     "confidence": float(conf[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    ece = sum(b["n"] / len(y) * abs(b["accuracy"] - b["confidence"]) for b in bins if b["n"])
    return {"AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "BA": float(balanced_accuracy_score(labels, predicted)),
            "brier_asd_binary": float(np.mean((p - y) ** 2)), "ece_confidence_10_bins": ece,
            "prob_asd": distribution(p), "true_asd_prob": distribution(p[y]),
            "true_hc_prob": distribution(p[~y]), "confidence": distribution(conf),
            "calibration_bins": bins}


def audit_inputs():
    historical.source_guard()
    if historical.training_code_hash() != EXPECTED:
        raise RuntimeError("Unexpected training-source hash")
    ctx = historical.context()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    try:
        remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True, timeout=30).split()[0]
    except (subprocess.SubprocessError, IndexError) as exc:
        remote = f"unavailable: {exc}"
    manifest, fits, splits, arrays = {}, {}, {}, {}
    def remember(path):
        relative = str(path.relative_to(ROOT))
        if relative not in manifest:
            manifest[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return manifest[relative]["sha256"]
    for name in historical.SOURCE_FILES:
        remember(ROOT / name)
    for name in ("run.py", "drpa.py", "analyze.py", "config.json", "README.md", "outputs/final_report.txt", "outputs/final_summary.json", "artifacts/context.pt", "artifacts/audit.json", "artifacts/paper/2406.14455v3.pdf"):
        remember(OLD / name)
    total_rows, epochs_count = 0, 0
    missing = []
    for seed in (911, 912, 913):
        remember(OLD / f"artifacts/seed_{seed}/selected_features.pt")
        remember(OLD / f"artifacts/seed_{seed}/rfe_support.npy")
        for fold in range(10):
            adir = historical.anchor_dir(seed, fold)
            for name in ("splits.pt", "train_start.pt", "fingerprints.json", "end_rng.pt"):
                remember(adir / name)
            remember(adir.parent / "vae" / f"fold{fold}_pretrain.pth")
            split = historical.load_torch(adir / "splits.pt")
            historical.check_splits([split[k] for k in SUBSETS])
            splits[(seed, fold)] = split
    for method in METHODS:
        for seed in (911, 912, 913):
            seen = []
            for fold in range(10):
                ident = f"{method}/{seed}/{fold}"
                out = historical.fold_dir(method, seed, fold)
                absent = [name for name in ("metrics.json", "status.json", "embeddings.npz", "epochs.jsonl", "oof.csv", "best.pt") if not (out / name).is_file()]
                if absent:
                    missing.append({"task": ident, "files": absent}); continue
                m = json.loads((out / "metrics.json").read_text())
                st = json.loads((out / "status.json").read_text())
                if m["training_code_sha256"] != EXPECTED or st["state"] != "complete":
                    raise ValueError(f"Nonformal/incomplete result {ident}")
                if remember(out / "metrics.json") != st["metrics_sha256"]:
                    raise ValueError(f"Metrics hash mismatch {ident}")
                remember(out / "status.json")
                for name, digest in m["output_sha256"].items():
                    if remember(out / name) != digest:
                        raise ValueError(f"Output hash mismatch {ident}/{name}")
                split = splits[(seed, fold)]
                schema = {}
                with np.load(out / "embeddings.npz", allow_pickle=False) as z:
                    for k in z.files:
                        value = z[k]
                        if not np.isfinite(value).all():
                            raise ValueError(f"Nonfinite array {ident}/{k}")
                        if k != "indices_test" and value.shape != (871, 2 if k.startswith("logits") else 16):
                            raise ValueError(f"Wrong array shape {ident}/{k}")
                        schema[k] = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
                    if not np.array_equal(z["indices_test"], split["test"]):
                        raise ValueError("Embedding index mismatch")
                    if method != "E0" and not set(("si", "sp", "ti", "tp", "logits_i", "logits_p")).issubset(z.files):
                        raise ValueError("Missing actual alignment space")
                checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False, mmap=True)
                if checkpoint["training_code_sha256"] != EXPECTED or checkpoint["epoch"] != m["best_epoch"]:
                    raise ValueError("Checkpoint identity mismatch")
                for state in [checkpoint["core"]["state"], checkpoint["auxiliary"] or {}]:
                    if any(not torch.isfinite(v).all() for v in state.values()):
                        raise ValueError("Nonfinite checkpoint")
                with (out / "oof.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                if [int(r["index"]) for r in rows] != list(split["test"]):
                    raise ValueError("OOF order/indices mismatch")
                for r in rows:
                    i = int(r["index"])
                    if r["subject_id"] != ctx["ids"][i] or int(r["label_original"]) != int(ctx["labels"][i]) or r["site"] != ctx["site_names"][i]:
                        raise ValueError("OOF subject/label/site mismatch")
                actual = historical.binary_metrics([int(r["label_original"]) for r in rows], [int(r["pred_original"]) for r in rows], [float(r["prob_asd"]) for r in rows])
                if any(abs(actual[k] - m["metrics"][k]) > 1e-10 for k in actual):
                    raise ValueError("Stored score does not reproduce OOF")
                ep = [json.loads(line) for line in (out / "epochs.jsonl").read_text().splitlines()]
                if len(ep) != m["epochs_run"]:
                    raise ValueError("Epoch log incomplete")
                seen.extend(int(r["index"]) for r in rows)
                total_rows += len(rows); epochs_count += len(ep)
                fits[ident], arrays[ident] = m, schema
            if len(seen) == 871 and sorted(seen) != list(range(871)):
                raise ValueError("Incorrect per-seed OOF coverage")
    previous = json.loads((OLD / "outputs/final_summary.json").read_text())
    formal = {m: {k: previous["groups"][m][k] for k in ("mean", "std_across_three_seed_means_ddof1", "comparisons", "auxiliary_test")} for m in METHODS}
    for method in METHODS:
        own = [v for v in fits.values() if v["method"] == method]
        if len(own) == 30:
            for key, value in formal[method]["mean"].items():
                if abs(np.mean([v["metrics"][key] for v in own]) - value) > 1e-12:
                    raise ValueError("Formal summary not consistent with fold metrics")
    return {"time_utc": utc(), "head": head, "remote_main": remote, "reference_commit": REFERENCE,
            "commits_match": head == remote == REFERENCE, "training_code_sha256": EXPECTED,
            "data_digest": ctx["data_digest"], "manifest": manifest, "array_schemas": arrays,
            "formal_fits": len(fits), "oof_rows": total_rows, "epochs_read": epochs_count,
            "missing": missing, "formal_results": formal}, fits, splits, ctx


def load_arrays(ident):
    method, seed, fold = ident.split("/")
    with np.load(historical.fold_dir(method, int(seed), int(fold)) / "embeddings.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def d1(fits, splits, ctx):
    records, aggregate = {}, {}
    for number, (ident, m) in enumerate(fits.items(), 1):
        z, method = load_arrays(ident), m["method"]
        keys = ["raw"] if method == "E0" else ["raw", "si_sp", "si_tp"] + (["ti_sp"] if method in ("E1", "E3") else [])
        spaces = {}
        for key in keys:
            a, b = PAIRS[key]
            spaces[key] = {}
            for subset, idx in splits[(m["seed"], m["fold"])].items():
                record = alignment(z[a][idx], z[b][idx], ctx["labels"][idx], ctx["site_codes"][idx], (ident, key, subset))
                record["saved_coordinate_variance_img"] = float(z[a][idx].astype(np.float64).var(0).mean())
                record["saved_coordinate_variance_ph"] = float(z[b][idx].astype(np.float64).var(0).mean())
                record["class_center_distance_img"] = float(np.linalg.norm(z[a][idx][ctx["labels"][idx] == 0].mean(0) - z[a][idx][ctx["labels"][idx] == 1].mean(0)))
                record["class_center_distance_ph"] = float(np.linalg.norm(z[b][idx][ctx["labels"][idx] == 0].mean(0) - z[b][idx][ctx["labels"][idx] == 1].mean(0)))
                spaces[key][subset] = record
        records[ident] = {"method": method, "seed": m["seed"], "fold": m["fold"], "spaces": spaces}
        if number % 30 == 0:
            print(f"D1 processed {number}/{len(fits)}", flush=True)
    for method in METHODS:
        rows = [r for r in records.values() if r["method"] == method]
        if rows:
            aggregate[method] = mean_tree([r["spaces"] for r in rows])
    correlations = {}
    for method in METHODS:
        for key in PRIMARY[method]:
            own = [ident for ident in records if records[ident]["method"] == method]
            values = {}
            for diagnostic in ("same_cos", "cos_gap", "r1_bidirectional", "effective_rank_ph"):
                x = np.array([records[i]["spaces"][key]["test"][diagnostic] for i in own])
                for metric in ("ACC", "BA", "AUC"):
                    y = np.array([fits[i]["metrics"][metric] for i in own])
                    xc, yc = x.copy(), y.copy()
                    for seed in (911, 912, 913):
                        mask = np.array([fits[i]["seed"] == seed for i in own])
                        xc[mask] -= xc[mask].mean(); yc[mask] -= yc[mask].mean()
                    values[f"{diagnostic}/{metric}"] = {
                        "spearman": float(spearmanr(x, y).statistic) if x.std() > 0 and y.std() > 0 else None,
                        "within_seed_pearson": float(np.corrcoef(xc, yc)[0, 1]) if xc.std() > 0 and yc.std() > 0 else None}
            correlations[f"{method}/{key}"] = values
    return {"state": "complete", "fits": records, "aggregate": aggregate, "correlations_descriptive_only": correlations,
            "pair_orientation": "image query first: si/tp evaluates the tp->stopgrad(si) objective in both retrieval directions",
            "retrieval": "Candidate pool equals subset. Exact ties get fractional expected recall; independent cyclic derangements within groups; singleton rows retained and counted.",
            "rank_definition": "float64 covariance eigvalsh; clamp negative eigenvalues to zero; entropy over positive eigenvalues; all-zero covariance has rank 0"}


def d2(fits, splits, ctx):
    records, aggregate = {}, {}
    for ident, m in fits.items():
        if m["method"] not in ("E4", "E5", "E6", "E7"):
            continue
        z = load_arrays(ident)
        post = posthoc_gate(z, splits[(m["seed"], m["fold"])]["train"], ctx["labels"], ctx["site_codes"], m["method"])
        reference = m["diagnostics"]["eval_train_gate_unshuffled"]
        errors = {name: abs(post[name]["mean"] - reference[name]["mean"]) for name in ("weights", "q_pred", "q_site")}
        if max(errors.values()) > 1e-4:
            raise ValueError(f"Gate reconstruction disagrees with saved post-hoc diagnostics: {ident}: {errors}")
        logs = [json.loads(line) for line in (historical.fold_dir(m["method"], m["seed"], m["fold"]) / "epochs.jsonl").read_text().splitlines()]
        sampled = [(ep["epoch"], gate_moments(ep["gate"])) for ep in logs if "gate" in ep]
        phases = {phase: mean_tree([g for e, g in sampled if min(int(3 * e / len(logs)), 2) == j])
                  for j, phase in enumerate(("early", "middle", "late"))}
        records[ident] = {"method": m["method"], "seed": m["seed"], "fold": m["fold"],
                          "posthoc_unshuffled": post, "reconstruction_mean_errors": errors,
                          "training_last_sampled": gate_moments(m["last_sampled_gate"]),
                          "training_phases": phases, "sampled_epoch_count": len(sampled),
                          "training_epoch_series": [{"epoch": e, "coverage": g["nonzero_fraction"], "cv": g["cv"],
                                                     "ess_ratio": g["ess_ratio"], "class_between_fraction": g["class_between_fraction"],
                                                     "q_pred_mean": g["q_pred"]["mean"], "q_site_mean": g["q_site"]["mean"]}
                                                    for e, g in sampled]}
    for method in ("E4", "E5", "E6", "E7"):
        own = [r for r in records.values() if r["method"] == method]
        aggregate[method] = {key: mean_tree([r[key] for r in own])
                             for key in ("training_last_sampled", "posthoc_unshuffled", "training_phases")}
    return {"state": "complete", "fits": records, "aggregate": aggregate,
            "limitations": ["Historical logs preserve moments and P10/P50/P90, not individual weights. Historical P25/P75/top-10%-mass unavailable, not reconstructed from best checkpoint.",
                            "E7 logged weights are actual shuffled training weights; reconstructed best-checkpoint gates are explicitly unshuffled.",
                            "Class and site one-way variance decompositions overlap and are not additive; class-centered residual site analysis is separate.",
                            "Per-class/site historical ESS/N is recovered exactly from recorded mean and population SD, up to float32 logging roundoff."]}


def d3_arrays(fits, splits, ctx):
    records, aggregate = {}, {}
    for ident, m in fits.items():
        z = load_arrays(ident)
        representations, auxiliary = {}, {}
        keys = [k for k in ("img_embed", "ph_embed", "si", "sp", "ti", "tp") if k in z]
        for subset, idx in splits[(m["seed"], m["fold"])].items():
            representations[subset] = {key: representation(z[key][idx], ctx["labels"][idx], ctx["site_codes"][idx]) for key in keys}
            if m["method"] != "E0":
                auxiliary[subset] = {name: prediction_description(z[key][idx], ctx["labels"][idx]) for name, key in (("img", "logits_i"), ("ph", "logits_p"))}
        logs = [json.loads(line) for line in (historical.fold_dir(m["method"], m["seed"], m["fold"]) / "epochs.jsonl").read_text().splitlines()]
        measurements = []
        for ep in logs:
            if "gradient_norms" in ep:
                g = ep["gradient_norms"]
                measurements.append({"epoch": ep["epoch"], "official": g["official"], "weighted_align": g["weighted_align"],
                                     "weighted_aux": g["weighted_aux"],
                                     "align_official_ratio": g["weighted_align"] / (g["official"] + 1e-12),
                                     "aux_official_ratio": g["weighted_aux"] / (g["official"] + 1e-12),
                                     **{k: g[k] for k in ("weighted_align_new_parameters", "weighted_aux_new_parameters") if k in g},
                                     **{k: ep[k] for k in ("train_official", "train_aux", "train_align", "train_graph", "train_ce", "train_total")}})
        def scales(values):
            keys = set().union(*(v.keys() for v in values)) if values else set()
            return {k: describe([v[k] for v in values if k in v]) for k in keys if k != "epoch"}
        phases = {phase: scales([v for v in measurements if min(int(3 * v["epoch"] / len(logs)), 2) == j])
                  for j, phase in enumerate(("early", "middle", "late"))}
        records[ident] = {"method": m["method"], "seed": m["seed"], "fold": m["fold"],
                          "representations": representations, "auxiliary": auxiliary,
                          "training_gradient_summary": scales(measurements), "training_gradient_phases": phases,
                          "training_measurements": measurements}
    for method in METHODS:
        own = [r for r in records.values() if r["method"] == method]
        all_measured = [v for r in own for v in r["training_measurements"]]
        aggregate[method] = {"representations": mean_tree([r["representations"] for r in own]),
                             "auxiliary": mean_tree([r["auxiliary"] for r in own]),
                             "training_gradient": {k: describe([v[k] for v in all_measured if k in v]) for k in set().union(*(v.keys() for v in all_measured)) if k != "epoch"},
                             "training_gradient_phases_fold_equal": mean_tree([r["training_gradient_phases"] for r in own]),
                             "training_gradient_fold_equal": mean_tree([r["training_gradient_summary"] for r in own])}
    return {"state": "arrays_complete", "fits": records, "aggregate": aggregate, "frozen_gradients": {},
            "limitations": ["Historical output gradient norms combine both original 871-row branch outputs; per-view/per-encoder direction was not logged.",
                            "Historical ratio statistics are ratios at each measurement, not ratio of mean norms; epoch-pooled and fold-equal aggregates are kept separately.",
                            "Frozen checkpoint gradients use training CE/aux/alignment and original all-node graph regularization; no held-out labels are consumed by losses/prototypes.",
                            "Frozen diagnostics use train mode on a single shared stochastic forward and a test-only deterministic backend. BN buffers and original in-place constraints are restored afterwards."]}


def gpu_tasks(fits, methods):
    tasks = []
    for seed in (911, 912, 913):
        for fold in (0, 5):
            for method in methods:
                ident = f"{method}/{seed}/{fold}"
                if ident in fits:
                    tasks.append({"method": method, "seed": seed, "fold": fold, "sha256": fits[ident]["output_sha256"]["best.pt"]})
    return tasks


def gpu_workers(state, stage, tasks):
    group = state["D3"]["frozen_gradients"] if stage == "gradients" else state["D4"]["checkpoints"]
    def key(t):
        return f"{t['method']}/{t['seed']}/{t['fold']}"
    pending = [t for t in tasks if not (key(t) in group and (group[key(t)].get("state") == "complete" if stage == "gradients" else group[key(t)].get(stage, {}).get("state") == "complete"))]
    if not pending:
        return
    telemetry = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=30)
    rows = list(csv.reader(telemetry.splitlines()))
    if len(rows) < 2 or any("PPU" not in r[1].upper() or float(r[3]) < 20000 for r in rows[:2]):
        raise RuntimeError(f"Two PPUs with >=20GiB free are required: {telemetry}")
    state.setdefault("gpu_telemetry", []).append({"time_utc": utc(), "stage": stage, "csv": telemetry.strip()})
    selector, processes = selectors.DefaultSelector(), []
    try:
        for gpu in (0, 1):
            own = [t for t in pending if (0 if t["fold"] == 0 else 1) == gpu]
            if not own:
                continue
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            if stage == "native":
                env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            process = subprocess.Popen([sys.executable, "-B", str(HERE / "replay.py"), "--stage", stage],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, env=env, cwd=ROOT)
            process.stdin.write(json.dumps(own)); process.stdin.close()
            selector.register(process.stdout, selectors.EVENT_READ, process)
            processes.append(process)
            print(f"{stage}: PPU {gpu} started {len(own)} checkpoint tasks (pid {process.pid})", flush=True)
        while selector.get_map():
            for selected, _ in selector.select(timeout=5):
                line = selected.fileobj.readline()
                if not line:
                    selector.unregister(selected.fileobj)
                    continue
                message = json.loads(line)
                ident = key(message["task"])
                if stage == "gradients":
                    group[ident] = message
                else:
                    group.setdefault(ident, {})[stage] = message
                if message["state"] != "complete":
                    state["errors"].append({"time_utc": utc(), "stage": stage, "task": ident, "error": message["error"]})
                atomic_json(state)
                print(f"{stage}: {ident}: {message['state']}", flush=True)
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError(f"Diagnostic worker exited with {process.returncode}")
        if any(key(t) not in group or (group[key(t)].get("state") != "complete" if stage == "gradients" else group[key(t)].get(stage, {}).get("state") != "complete") for t in tasks):
            raise RuntimeError(f"Incomplete {stage} checkpoint diagnostics; recorded successes can be resumed")
    finally:
        selector.close()
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            process.stdout.close()


def self_test():
    assert effective_rank(np.ones((8, 4)))["effective_rank"] == 0
    assert abs(effective_rank(np.vstack((np.eye(4), -np.eye(4))))["effective_rank"] - 4) < 1e-12
    s = np.eye(6)
    assert recall_at(s, np.arange(6), 1) == 1
    assert recall_at(np.ones((6, 6)), np.arange(6), 1) == 1 / 6
    p = permutation(np.zeros(6), private_rng("test"))
    assert np.all(p != np.arange(6)) and recall_at(s, p, 1) == 0
    d = distribution(np.ones(6))
    assert d["ess"] == 6 and d["cv"] == 0
    w = np.array([0.8, 0.8, 1.2, 1.2])
    assert abs(distribution(w)["ess_ratio"] - 1 / (1 + np.std(w) ** 2 / np.mean(w) ** 2)) < 1e-12
    model_files = list(HERE.glob("*.py"))
    for path in model_files:
        compile(path.read_text(), str(path), "exec")
    print("SELF TEST: covariance ranks, identity mapping, fractional tie recall, derangements, gate ESS and syntax passed.", flush=True)


def frozen_aggregate(state, method, group):
    records = [r["value"]["groups"][group] for r in state["D3"]["frozen_gradients"].values()
               if r.get("state") == "complete" and r["task"]["method"] == method]
    return mean_tree(records) if records else None


def stability_summary(state):
    result = {}
    for condition in ("original_free_rng", "same_rng", "fixed_vae_vary_dropout", "fixed_dropout_vary_vae", "deterministic_same_rng"):
        stage = "deterministic" if condition == "deterministic_same_rng" else "native"
        entries = [r[stage]["value"][condition] for r in state.get("D4", {}).get("checkpoints", {}).values()
                   if r.get(stage, {}).get("state") == "complete"]
        if not entries:
            continue
        pairs = [pair for entry in entries for pair in entry["pairwise"]]
        result[condition] = {"checkpoints": len(entries), "pairwise_comparisons": len(pairs),
                             **{k: max(p[k] for p in pairs) for k in ("max_abs_logits", "max_abs_probs", "max_abs_img", "max_abs_ph", "max_abs_vae", "label_flips_all", "label_flips_test")},
                             "any_label_flip_checkpoints": sum(any(p["label_flips_test"] > 0 for p in entry["pairwise"]) for entry in entries),
                             "max_acc_span": max(e["acc_span"] for e in entries),
                             "mean_acc_span": float(np.mean([e["acc_span"] for e in entries])),
                             "topk_order_changed_comparisons": sum(any(v["ordered_positions_changed"] > 0 for vv in p["topk"].values() for v in vv) for p in pairs),
                             "topk_set_changed_comparisons": sum(any(v["selected_set_symmetric_difference"] > 0 for vv in p["topk"].values() for v in vv) for p in pairs),
                             "all_initial_rng_equal": all(all(r["rng_before"] == e["repeats"][0]["rng_before"] for r in e["repeats"]) for e in entries)}
    return result


def analysis_fingerprint():
    names = {"describe", "mean_tree", "unit", "effective_rank", "mean_mask", "representation",
             "recall_at", "permutation", "private_rng", "alignment", "distribution", "gate_moments",
             "posthoc_gate", "prediction_description", "audit_inputs", "load_arrays",
             "d1", "d2", "d3_arrays", "gpu_tasks", "gpu_workers"}
    constants = {"EXPECTED", "REFERENCE", "METHODS", "SUBSETS", "PRIMARY", "PAIRS"}
    nodes = {}
    for node in ast.parse((HERE / "diagnose.py").read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            nodes[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    nodes[target.id] = ast.dump(node, include_attributes=False)
    if set(nodes) != names | constants:
        raise RuntimeError("Incomplete scientific-analysis fingerprint")
    return hashlib.sha256(json.dumps(nodes, sort_keys=True).encode()).hexdigest()


def record_code_revision(state):
    current = {name: sha(HERE / name) for name in ("diagnose.py", "replay.py")}
    previous = state["diagnostic_code_sha256"]
    if current == previous:
        return
    if (state.get("analysis_logic_sha256") != analysis_fingerprint()
            or current["replay.py"] != previous["replay.py"]):
        raise RuntimeError("Scientific computation changed; refuse to reuse diagnostic results")
    state.setdefault("executed_code_sha256", previous.copy())
    state.setdefault("code_revisions", []).append({
        "time_utc": utc(), "previous": previous, "current": current,
        "reason": "Report/context/resume presentation only; original numerical analysis AST and GPU replay unchanged"})
    state["diagnostic_code_sha256"] = current


def report_context(state):
    ctx = historical.context()
    splits, partition = {}, {}
    for ident, fit in state["D3"]["fits"].items():
        anchor = (fit["seed"], fit["fold"])
        if anchor not in splits:
            splits[anchor] = historical.load_torch(historical.anchor_dir(*anchor) / "splits.pt")
        subsets = {}
        for subset in SUBSETS:
            indices = splits[anchor][subset]
            n = len(indices)
            groups = {"class": len(np.unique(ctx["labels"][indices])),
                      "site": len(np.unique(ctx["site_codes"][indices]))}
            baseline = {key: (count - 1) / (n - 1) for key, count in groups.items()}
            subsets[subset] = {
                "n": n, "group_counts": groups, "random_partition_expected_fraction": baseline,
                "representations": {key: {
                    "observed_class_fraction": value["class_variance_fraction"],
                    "observed_site_fraction": value["site_variance_fraction"],
                    "class_fraction_minus_random_expectation": None if value["class_variance_fraction"] is None else value["class_variance_fraction"] - baseline["class"],
                    "site_fraction_minus_random_expectation": None if value["site_variance_fraction"] is None else value["site_variance_fraction"] - baseline["site"]}
                    for key, value in fit["representations"][subset].items()}}
        partition[ident] = {"method": fit["method"], "subsets": subsets}
    partition_aggregate = {method: mean_tree([r["subsets"] for r in partition.values() if r["method"] == method])
                           for method in METHODS}
    calibration = {}
    for method in ("E6", "E7"):
        rows = [r for r in state["D2"]["fits"].values() if r["method"] == method]
        calibration[method] = {}
        for source in ("training_last_sampled", "posthoc_unshuffled"):
            values = [r[source]["q_site"] for r in rows]
            calibration[method][source] = {
                "q_site_cv": describe([v["std"] / v["mean"] for v in values]),
                "constant_scale_effective_q_pred_threshold": describe([0.15 / v["mean"] for v in values])}
    q = np.array([0.2, 0.5, 0.8])
    scale, threshold = 0.576, 0.15
    left, right = np.maximum(scale * q - threshold, 0), np.maximum(q - threshold / scale, 0)
    error = float(np.max(np.abs(left / left.mean() - right / right.mean())))
    if error > 1e-12:
        raise RuntimeError("Constant-scale gate algebra check failed")
    directions = {}
    for method in ("E1", "E3", "E4", "E5", "E6"):
        directions[method] = {}
        for group in ("img_encoder", "ph_encoder", "raw_outputs_all"):
            values = [r["value"]["groups"][group]["official_align_cosine"]
                      for r in state["D3"]["frozen_gradients"].values()
                      if r["task"]["method"] == method and r.get("state") == "complete"]
            finite = [v for v in values if v is not None]
            directions[method][group] = {"cosine": describe(finite),
                                         "negative_count": sum(v < 0 for v in finite),
                                         "undefined_count": len(values) - len(finite)}
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch-geometric"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "derived_from": "Saved D1-D4 numerical results and previously hashed context/splits; no new network execution",
        "runtime": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda, "torch_git": torch.version.git_version, "packages": packages},
        "finite_partition": {"formula": "E[between/total] = (K-1)/(N-1) for unrestricted permutation of fixed group sizes and fixed nonzero-variance representations",
                             "limitation": "Descriptive group-count reference, not a stratified null, p-value or proof of site invariance",
                             "fits": partition, "aggregate": partition_aggregate},
        "gate_calibration": {"constant_scale_identity": "Normalize(ReLU(c*q_pred-tau)) = Normalize(ReLU(q_pred-tau/c)) for c>0 with nonzero mean",
                             "identity_unit_check_max_abs_error": error, "aggregate": calibration,
                             "limitation": "Algebraic counterfactual only: actual q_site varies; no new ablation or causal explanation of the E6-E5 score difference"},
        "frozen_direction_distribution": directions,
        "unavailable_historical_quantities": ["individual raw-gate values", "P25/P75/top-decile weight mass", "encoder gradient directions"]}


def update_handoff(state):
    lines = ["<!-- DRPA_MECHANISM_20261015_START -->", "# DRPA 机制诊断 20261015", f"更新时间 UTC：{utc()}",
             f"- 正式产物审计：{state['audit']['formal_fits']}/240 fits，{state['audit']['oof_rows']} 条 OOF，{state['audit']['epochs_read']} 个历史 epoch。",
             f"- 版本：本地/远程 {state['audit']['head']} / {state['audit']['remote_main']}；正式训练 SHA256 {EXPECTED}。",
             f"- 唯一产物目录：{HERE}。原实验只读，无训练、无 optimizer.step、无 push。"]
    if state.get("D1", {}).get("state") == "complete":
        a = state["D1"]["aggregate"]["E6"]["si_tp"]["test"]
        lines += ["## D1 实际对齐空间",
                  f"- 完成 {len(state['D1']['fits'])} fits 的 train/val/test：E6 si↔tp 测试 same/diff cos={a['same_cos']:.6f}/{a['different_cos']:.6f}；gap={a['cos_gap']:.3g}；双向 R@1={a['r1_bidirectional']*100:.3f}%，机会水平={a['chance_r1']*100:.3f}%。",
                  f"- 同类打乱的正确配对优势={a['controls']['within_class']['correct_minus_control_cos']:.3g}，同站点打乱优势={a['controls']['within_site']['correct_minus_control_cos']:.3g}。",
                  "- 决策：区分均值方向一致与患者身份匹配，不以高平均 cos 解释分类收益；后续不用 si↔sp 代替 E4–E7 的训练目标。",
                  "- 限制：仅保存的单次随机表示；折相关，不能从相关系数推出对齐的因果作用。"]
    if state.get("D2", {}).get("state") == "complete":
        g = state["D2"]["aggregate"]["E6"]["training_last_sampled"]
        lines += ["## D2 门控信息来源",
                  f"- 完成 {len(state['D2']['fits'])} fits 的历史门控和最佳 checkpoint 事后门控。E6 最后采样训练门控覆盖={g['nonzero_fraction']*100:.4f}%，CV={g['cv']:.5f}，ESS/N={g['ess_ratio']:.5f}，类别间方差占比={g['class_between_fraction']*100:.2f}%。",
                  f"- ASD/HC 权重均值={g['classes']['0']['mean']:.5f}/{g['classes']['1']['mean']:.5f}；类内 SD={g['classes']['0']['std']:.5f}/{g['classes']['1']['std']:.5f}。",
                  "- 决策：优先判断类内区分能力，而非只看覆盖率；不把 E7 事后未打乱权重写成训练分配。历史 P25/P75/top10 质量未保存，明确缺失。",
                  "- 限制：没有用额外训练分离门控公式与输入信息的因果效应，不据现有测试分数重新设计阈值。"]
    if state.get("D3", {}).get("state") == "complete":
        g = state["D3"]["aggregate"]["E6"]["training_gradient"]
        p = state["D3"]["aggregate"]["E6"]["auxiliary"]["test"]["ph"]
        frozen = frozen_aggregate(state, "E6", "ph_encoder")
        lines += ["## D3 梯度和表型语义",
                  f"- 完成 240 fits 的数组分析、210 variants 的历史梯度统计，以及 {len(state['D3']['frozen_gradients'])}/30 冻结 checkpoint 梯度诊断（每卡15）。",
                  f"- E6 历史逐次 align/official 比均值={g['align_official_ratio']['mean']*100:.4f}%，中位数={g['align_official_ratio']['median']*100:.4f}%；表型辅助 test AUC={p['AUC']*100:.2f}%，BA={p['BA']*100:.2f}%。",
                  f"- 关键区分：冻结表型编码器参数处 align/official={frozen['align_official_ratio']*100:.2f}%，方向cos={frozen['official_align_cosine']:.4f}；不能把输出处小梯度推成整个编码器几乎未受影响。H4仅部分支持。",
                  "- 决策：暂不支持直接增大对齐系数。冻结梯度方向单列，不冒充历史训练方向；非对称损失的影像目标 stop-gradient 是实际设计，不是实现错误。",
                  "- 限制：辅助头弱不等于原始三维数据没有任何互补价值；梯度范数不等于长期参数更新或因果贡献。"]
    if state.get("D4", {}).get("state") == "complete":
        a = stability_summary(state)
        lines += ["## D4 PPU 重复性",
                  "- 完成 E0/E6 × 3 seeds × folds0/5 共12 checkpoint，双卡各6；原始自由 RNG、同 RNG、随机源隔离、确定性后端分别三次。",
                  f"- 相同 RNG 原后端 max logits 差={a['same_rng']['max_abs_logits']:.6g}、最多 test 标签变化={a['same_rng']['label_flips_test']}；确定性诊断 max 差={a['deterministic_same_rng']['max_abs_logits']:.6g}。",
                  f"- 同RNG原后端TopK排序变化={a['same_rng']['topk_order_changed_comparisons']}/36，所选节点集合变化={a['same_rng']['topk_set_changed_comparisons']}/36；二者不混淆。明显推理波动主要来自VAE/Dropout抽样。",
                  f"- 自由 RNG 最大诊断 ACC 跨度={a['original_free_rng']['max_acc_span']*100:.3f}个百分点；不是新成绩，不选最好一次。",
                  "- 决策：新正式比较预先固定后端并重复运行验证；保持历史结果。小于1个百分点的差异不能凭当前三 seed 均值就归因于新增机制。",
                  "- 限制：本轮只检验固定已训练模型；未估计训练轨迹的数值误差放大。诊断采用锚点保存 RNG，非未保存的原正式测试前完整 RNG 状态。"]
    if state.get("state") == "complete":
        lines += ["## 研究决策总结", "- 已完成 D1→D2→D3→D4；完整证据和最终 H1–H4 判定见 report.md，所有必要逐折数值集中在 summary.json。",
                  "- q_site近似常数也会通过tau/c改变门控有效阈值；这是已核查的代数机制，不是已证实的E6性能下降原因。站点方差占比需参照有限样本组数基线，不以站点组数远多于类别导致的差异宣称站点语义更强。",
                  "- 下一轮优先检验条件互补性，其次更丰富且诊断前可获得的临床模态，再考虑局部类别条件关系。当前不继续调门控阈值或盲目提高 λalign。",
                  "- 候选方向不等于新颖性已证明，需与 MGDR/PhenoBrain/SMGD 原文区别核查，并使用无泄漏、预先固定的验证协议。"]
    lines.append("<!-- DRPA_MECHANISM_20261015_END -->")
    target = ROOT / "handoff.md"
    old = target.read_text() if target.exists() else ""
    start, end = lines[0], lines[-1]
    block = "\n\n".join(lines) + "\n"
    if start in old and end in old:
        text = old[:old.index(start)] + block + old[old.index(end) + len(end):].lstrip("\n")
    else:
        text = old.rstrip() + "\n\n" + block
    target.write_text(text)


def render_report(state):
    state["report_context"] = report_context(state)
    context = state["report_context"]
    a, b, c = state["D1"]["aggregate"], state["D2"]["aggregate"], state["D3"]["aggregate"]
    stability = stability_summary(state)
    state["D4"]["aggregate"] = stability
    e6 = a["E6"]["si_tp"]["test"]
    gate = b["E6"]["training_last_sampled"]
    frozen_ph = frozen_aggregate(state, "E6", "ph_encoder")
    ph_direction = context["frozen_direction_distribution"]["E6"]["ph_encoder"]
    lines = ["# DRPA 机制诊断研究报告", f"完成时间 UTC：{state.get('completed_utc', utc())}",
             "## 结论先行",
             f"实际对齐空间确有更高的平均方向一致性，但 E6 测试 same/different cos={e6['same_cos']:.6f}/{e6['different_cos']:.6f}，gap={e6['cos_gap']:.3g}；双向 R@1={e6['r1_bidirectional']*100:.3f}%，候选池机会水平={e6['chance_r1']*100:.3f}%。提高平均 cos 不等于学到了患者对应关系。",
             f"E6 最后采样训练门控类别间方差占比={gate['class_between_fraction']*100:.2f}%，ESS/N={gate['ess_ratio']:.4f}。门控的个体信息应看类内变化，不应由99.99%非零覆盖单独判定。",
             f"E6 历史逐次加权对齐/官方输出梯度比均值={c['E6']['training_gradient']['align_official_ratio']['mean']*100:.4f}%；表型辅助测试 AUC={c['E6']['auxiliary']['test']['ph']['AUC']*100:.2f}%。现有证据不支持仅增大对齐系数或继续调整可靠性阈值。",
             f"但冻结表型编码器参数处该梯度比为{frozen_ph['align_official_ratio']*100:.2f}%，不是近零；与官方梯度cos均值={frozen_ph['official_align_cosine']:.4f}，{ph_direction['negative_count']}/{ph_direction['cosine']['n']}个诊断点为负。H4只能部分支持，不能认定『所有GTUNet梯度都太弱』是唯一失败原因。",
             f"固定RNG原后端最大logits差仅{stability['same_rng']['max_abs_logits']:.3g}、测试标签和TopK所选集合不变；确定性诊断完全一致。原始自由RNG的诊断ACC平均跨度{stability['original_free_rng']['mean_acc_span']*100:.2f}个百分点，主要是原代码随机VAE/Dropout的影响，不是可直接归因PPU的硬件误差。",
             "## 审计与科学口径",
             f"本地 HEAD 与 GitHub main 均为 `{state['audit']['head']}`，参考版本一致={state['audit']['commits_match']}。正式训练代码 SHA256 `{EXPECTED}`。核验240 fits、20,904条OOF、64,411个历史epoch，所有保存数组的shape/dtype/有限性、划分、标签/ID/站点顺序和结果哈希均通过。完整输入清单在 summary.json.audit.manifest。未使用 diagnostic_replay_v1 归档结果。",
             "本轮无训练、无optimizer.step、无RFE预处理、无测试选模型/调参、无新checkpoint、无push。只在新目录保留5个文件，并更新根handoff.md。D1/D2/D3数组分析使用CPU NumPy；30个冻结梯度诊断、12个checkpoint的重复前向使用两张PPU。原静态reward/penalty/motivation输入在内存按保存划分及原函数恢复；学习到的AMRS参数读取checkpoint，不重新拟合。",
             "统计主单位为30个相关seed-fold的等权描述性平均，三个重复seed和871个受试者不是30个独立队列。D3另保留训练梯度测量点等权与fold等权两种聚合。ASD=0/HC=1训练编码不变；AUC、SEN/SPE/F1的研究口径以ASD为阳性。正式分类成绩仅读取旧产物，不以诊断重复推理替换。",
             f"诊断环境：Python {sys.version.split()[0]}，Torch {torch.__version__}，CUDA接口 {torch.version.cuda}；其余包版本见summary.json.report_context.runtime。执行数值计算时的代码哈希保存在executed_code_sha256；后续报告整理保留code_revisions，原D1–D3计算AST与replay.py未变。",
             f"既有正式分类背景（不重跑或替换）：E0 ACC/AUC={state['audit']['formal_results']['E0']['mean']['ACC']*100:.2f}%/{state['audit']['formal_results']['E0']['mean']['AUC']*100:.2f}%，E6={state['audit']['formal_results']['E6']['mean']['ACC']*100:.2f}%/{state['audit']['formal_results']['E6']['mean']['AUC']*100:.2f}%；E6−E5的ACC/BA/AUC分别为{state['audit']['formal_results']['E6']['comparisons']['E5']['mean_delta']['ACC']*100:.3f}/{state['audit']['formal_results']['E6']['comparisons']['E5']['mean_delta']['BA']*100:.3f}/{state['audit']['formal_results']['E6']['comparisons']['E5']['mean_delta']['AUC']*100:.3f}个百分点。小幅差异不单独作为机制成立的证据。",
             "### 原论文与实际代码",
             "论文的Modality Alignment用RFE/VAE适配维度；ARPG融合患者图；ACMGL将两分支输出及其均值做注意力融合；官方目标是CE与图/奖励正则。非影像单独判别较弱，但融合可能有互补作用。这些不等于显式疾病语义对齐。[MM-GTUNets原论文](https://arxiv.org/html/2406.14455v3)",
             "实际代码：VAE 3→256→1000，拆均值/对数方差后每次采样500维；GTUNet两分支输出各16维。全871人参加传导式图。eval仍执行随机VAE和未带training标记的F.dropout；边dropout仅训练启用。多模态shared为两原始嵌入的均值，而不是可识别的shared/private分解。奖励图循环首轮return、torch.tensor(value)断开相应奖励梯度、部分贡献权重detach均保留，未作修正。根源代码无改动。",
             "DRPA：si/sp为各自16→64→16投影后的L2归一化输出；ti/tp为Predictor输出，辅助头作用在ti/tp。E2对齐si/sp；E3对齐ti→stopgrad(sp)及tp→stopgrad(si)；E4–E7只有tp→stopgrad(si)。新增头不进入原推理分类器。",
             "## D1 实际对齐空间",
             "以下R@1/R@5是双向平均；单向值、原始坐标方差、归一化方差、类别/站点相似度、类别中心距离、完整validation结果在summary.json。每个子集只在该子集候选池检索；train约704人、val约79人、test87/88人，机会水平分别约0.142%、1.266%、1.148%。上一轮0.11%使用全871人，不能直接与本表test R@1比较。",
             "所有different-subject、同/异类及同/跨站点比较严格排除真实身份对角线。配对对照使用独立固定NumPy RNG和组内循环错配；候选池不变，无法打乱的singleton显式计入moved_fraction。检索以假配对为控制目标，不将标签传回模型。精确相同分数采用均匀随机tie-break的期望召回。有效秩采用float64协方差谱熵；负特征值截零，零项不参与log，全零协方差记0。",
             ]
    for subset, title in (("train", "训练子集"), ("test", "测试子集")):
        lines += [f"### {title}", "| Method | Evaluated Pair | Same Cos | Different Cos | Cos Gap | R@1 % | R@5 % | Effective Rank I/P |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            for key in PRIMARY[method]:
                v = a[method][key][subset]
                names = {"raw": "img_embed ↔ ph_embed", "si_sp": "si ↔ sp", "ti_sp": "ti ↔ sp", "si_tp": "tp ↔ si"}
                lines.append(f"| {method} | {names[key]} | {v['same_cos']:.6f} | {v['different_cos']:.6f} | {v['cos_gap']:.3g} | {v['r1_bidirectional']*100:.3f} | {v['r5_bidirectional']*100:.3f} | {v['effective_rank_img']:.2f}/{v['effective_rank_ph']:.2f} |")
    lines += ["### 公平对照与配对优势", "E1没有对齐损失，但同时计算E1的si/tp作为E4–E7的同空间容量控制；不能将E1的si/sp与E6的si/tp直接称为配对能力增益。",
              "| Method | Test si/tp Same Cos | Correct−Random Cos | Correct−Class-shuffle Cos | Correct−Site-shuffle Cos | R@1 % |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in ("E1", "E3", "E4", "E5", "E6", "E7"):
        v = a[method]["si_tp"]["test"]
        lines.append(f"| {method} | {v['same_cos']:.6f} | {v['controls']['random']['correct_minus_control_cos']:.3g} | {v['controls']['within_class']['correct_minus_control_cos']:.3g} | {v['controls']['within_site']['correct_minus_control_cos']:.3g} | {v['r1_bidirectional']*100:.3f} |")
    lines += ["E2实际si/sp测试gap仍接近零且为负、R@1接近机会，没有显示稳定患者匹配优势。E3–E7的高cos主要证明公共方向接近；正确配对不稳定优于同类/同站点错配。投影有效秩仍为约5–7，不能将低幅值或公共方向直接称为完整秩一坍塌。",
              "### 对齐诊断与分类的探索性关系",
              "| Method/Pair | Spearman: Same Cos vs AUC | Cos Gap vs BA | R@1 vs AUC |",
              "|---|---:|---:|---:|"]
    for method in METHODS:
        for key in PRIMARY[method]:
            v = state["D1"]["correlations_descriptive_only"][f"{method}/{key}"]
            values = [v[pair]["spearman"] for pair in ("same_cos/AUC", "cos_gap/BA", "r1_bidirectional/AUC")]
            shown = ["未定义" if value is None else f"{value:.4f}" for value in values]
            lines.append(f"| {method}/{key} | {' | '.join(shown)} |")
    lines += ["表内是各method内部30个seed-fold的Spearman；对seed中心化的Pearson另保留在summary。E6患者R@1与AUC关系接近零，没有稳定的『检索更好则分类更好』证据。不同方法及空间之间的相关性并不一致；E3某一空间较高也不能作为选择目标或因果证据。折相关、折难度、站点构成、多个探索性比较及单次随机表示均是限制。",
              "## D2 门控有效性", "### 真实训练门控（每折最后一次已记录采样）",
              "| Method | Nonzero Coverage % | Gate CV | ESS Ratio | ASD Mean | HC Mean | Within-Class SD ASD/HC | Class-between Variance % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("E5", "E6", "E7"):
        v = b[method]["training_last_sampled"]
        lines.append(f"| {method} | {v['nonzero_fraction']*100:.4f} | {v['cv']:.5f} | {v['ess_ratio']:.5f} | {v['classes']['0']['mean']:.5f} | {v['classes']['1']['mean']:.5f} | {v['classes']['0']['std']:.5f}/{v['classes']['1']['std']:.5f} | {v['class_between_fraction']*100:.2f} |")
    lines += ["### 最佳checkpoint事后未打乱门控（独立统计，非历史训练分配）",
              "| Method | Nonzero % | CV | ESS/N | Top10% Weight Mass % | Class-between Variance % | q_site Mean ± SD |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("E4", "E5", "E6", "E7"):
        v = b[method]["posthoc_unshuffled"]
        fraction = "不适用（等权）" if v['class_between_fraction'] is None else f"{v['class_between_fraction']*100:.2f}"
        lines.append(f"| {method} | {v['weights']['nonzero_fraction']*100:.4f} | {v['weights']['cv']:.5f} | {v['weights']['ess_ratio']:.5f} | {v['weights']['top_decile_mass']*100:.3f} | {fraction} | {v['q_site']['mean']:.6f}±{v['q_site']['std']:.6f} |")
    lines += ["历史日志只保留n/mean/SD/P10/P50/P90，无法推出P25/P75或top10%质量。本轮这些历史量明确为null；只对保存embedding重建的best-checkpoint事后门控计算全部分位数。ESS从n和总体方差恢复：ESS/N=mean²/(mean²+SD²)。每类/站点ESS及epoch演化均在summary。早/中/晚按照完成epoch数量的三等分，而非只对采样点分段。",
              "门控权重在每折归一化为均值1。99.99%覆盖仅证明几乎没有硬剔除，需结合CV、ESS、类内SD和方差分解解释软选择。类别之间与站点之间分别做一元方差分解，两者不能相加；best事后另分析去除类别均值后的站点残差。",
              "E7实际训练在同类/同站点内打乱，最大限度保留类/站点权重结构和总体分布；少数singleton按原规则回退。不要求E6/E7独立训练得到逐元素相同分布。E7没有稳定变差与弱个体选择相容，但不能证明任意打乱都无效，也不能把0.23个百分点差异单独归因于打乱。已证实的是权重结构，未证实的是公式缺陷或三维输入不足的因果分配。",
              "### q_site常数近似的阈值效应（代数核查，不是新训练消融）",
              "若q_site=c>0近似常数，则归一化ReLU(c*q_pred−tau)与归一化ReLU(q_pred−tau/c)完全相同（非零均值时）；因此乘一个近似常数并不会被最后的归一化完全抵消。固定tau=0.15时，E6还改变了有效q_pred阈值，而不只是增加个体可靠性信息。",
              "| Method | Source | q_site Mean ± Within-fit SD | Mean q_site CV | Constant-scale Effective q_pred Threshold |",
              "|---|---|---:|---:|---:|"]
    for method in ("E6", "E7"):
        for source, title in (("training_last_sampled", "历史最后采样"), ("posthoc_unshuffled", "best事后未打乱")):
            q = b[method][source]["q_site"]
            calibration = context["gate_calibration"]["aggregate"][method][source]
            lines.append(f"| {method} | {title} | {q['mean']:.6f}±{q['std']:.6f} | {calibration['q_site_cv']['mean']:.5f} | {calibration['constant_scale_effective_q_pred_threshold']['mean']:.5f} |")
    lines += [f"常数缩放恒等式数值单元检查误差={context['gate_calibration']['identity_unit_check_max_abs_error']:.3g}。E5对q_pred直接使用0.15阈值；E6训练常数近似对应约0.266，事后约0.294。这说明现有E6−E5也混合了门控校准/阈值效应，不能单独解释为站点信息有效或无效。真实q_site并非常数，此近似也没有证明−0.42个百分点的因果来源；未据测试结果更改阈值。",
              "## D3 梯度、表示与疾病信息", "### 真实历史输出梯度",
              "Official/Align为原GTUNet两个871行输出处的联合L2范数；Align已经乘0.1，Aux已经乘0.05。比例列是每次记录先求比，再聚合，不是平均范数相除。表中方向来自独立冻结checkpoint集合，严格单列说明。",
              "| Method | Official Grad | Align Grad | Align/Official % | Aux/Official % | Frozen Gradient Cosine (I/P Encoders) |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS[1:]:
        v = c[method]["training_gradient"]
        gs = [frozen_aggregate(state, method, key) for key in ("img_encoder", "ph_encoder")]
        cos = "/".join("未重算" if g is None else ("未定义（零梯度）" if g["official_align_cosine"] is None else f"{g['official_align_cosine']:.4f}") for g in gs)
        lines.append(f"| {method} | {v['official']['mean']:.6g} | {v['weighted_align']['mean']:.6g} | {v['align_official_ratio']['mean']*100:.4f} | {v['aux_official_ratio']['mean']*100:.4f} | {cos} |")
    lines += ["均值/中位数/P10/P90及早中晚训练比值、投影头联合梯度全部保存在summary。原日志没有各编码器方向，不能从范数推出方向冲突。冻结集合为E1/E3/E4/E5/E6×3seeds×folds0/5，共30个，每卡15个。单个checkpoint在相同forward中用autograd.grad分别求三个损失，分类CE/辅助/对齐仅用训练标签；原图正则仍跨871节点，不改变原目标。train模式的BN缓冲及原权重约束在内存执行后恢复，源文件和最终内存状态均校验不变。没有optimizer.step。这不是历史best epoch的实际梯度。",
              "### 冻结参数梯度（不同模块维度不同，不用范数直接比较谁吸收了多少比例）",
              "| Method | Group | Official L2 | Weighted Align L2 | Align RMS | Align/Official % | Cosine |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for method in ("E1", "E3", "E4", "E5", "E6"):
        for group in ("img_encoder", "ph_encoder", "project_ph", "predict_ph", "raw_outputs_all"):
            v = frozen_aggregate(state, method, group)
            cosine = "未定义" if v["official_align_cosine"] is None else f"{v['official_align_cosine']:.4f}"
            ratio = "不适用" if v['align_official_ratio'] is None else f"{v['align_official_ratio']*100:.4f}"
            lines.append(f"| {method} | {group} | {v['official']['norm']:.6g} | {v['weighted_align']['norm']:.6g} | {v['weighted_align']['rms']:.3g} | {ratio} | {cosine} |")
    lines += ["非对称E4–E7的si目标detach，所以影像编码器没有来自该目标的直接对齐梯度，是预注册损失本身的行为。新增projector/predictor处official梯度为0，Align/Official记不适用，参考其L2/RMS及连接维度。头部与编码器梯度强弱需结合雅可比和长期更新，不能把多个空间L2范数当成可加的损失份额。",
              f"关键反证：E6历史合并输出梯度比均值仅{c['E6']['training_gradient']['align_official_ratio']['mean']*100:.3f}%，但冻结表型编码器参数处为{frozen_ph['align_official_ratio']*100:.2f}%。后者有非忽略影响，不能把输出处比例外推为所有参数比例。表型参数cos均值{frozen_ph['official_align_cosine']:.4f}，{ph_direction['negative_count']}/{ph_direction['cosine']['n']}为负，范围{ph_direction['cosine']['min']:.4f}至{ph_direction['cosine']['max']:.4f}；说明局部有方向冲突，不是所有诊断点或整个训练轨迹都冲突。盲目提高lambda可能放大错误约束，当前没有受控训练证明其收益。",
              "### 表示诊断（test；train/val逐维结果保留在summary）",
              "Class Separation=类别间/类内方差比；Site Separation=站点间/总方差占比。同/跨站点平均平方距离、类中心L2和最大协方差特征值占比均保留。此类描述受站点/标签分布及小测试池偏差影响。",
              "| Method | Representation | Mean Variance | Effective Rank | Class Separation | Site Separation % | Top Eigenvalue % |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for method, key in (("E1", "img_embed"), ("E1", "ph_embed"), ("E1", "si"), ("E1", "sp"), ("E4", "sp"), ("E6", "img_embed"), ("E6", "ph_embed"), ("E6", "sp"), ("E6", "tp"), ("E6", "si")):
        v = c[method]["representations"]["test"][key]
        lines.append(f"| {method} | {key} | {v['mean_variance']:.6g} | {v['effective_rank']:.3f} | {v['class_between_within_ratio']:.6g} | {v['site_variance_fraction']*100:.3f} | {v['top_eigenvalue_share']*100:.2f} |")
    lines += ["### 有限样本分组参照（避免把组数效应当作站点语义）",
              "对固定非零方差表示、固定组大小且完全随机分组，组间/总方差占比的置换期望为(K−1)/(N−1)。类别只有2组，站点有十余组，所以原始21%站点方差与约1%类别方差不能直接比较。本表只计算这个代数期望与已保存统计之差，不重算表示或执行置换检验。",
              "| Method/Representation | Subset | Mean N / Site Groups | Class Observed / Random Expected % | Site Observed / Random Expected % | Site Excess (pp) |",
              "|---|---|---:|---:|---:|---:|"]
    for method, key in (("E1", "ph_embed"), ("E6", "ph_embed"), ("E6", "sp"), ("E6", "tp"), ("E6", "img_embed")):
        for subset in ("train", "test"):
            v = context["finite_partition"]["aggregate"][method][subset]
            r = v["representations"][key]
            expected = v["random_partition_expected_fraction"]
            lines.append(f"| {method}/{key} | {subset} | {v['n']:.1f}/{v['group_counts']['site']:.1f} | {r['observed_class_fraction']*100:.3f}/{expected['class']*100:.3f} | {r['observed_site_fraction']*100:.3f}/{expected['site']*100:.3f} | {r['site_fraction_minus_random_expectation']*100:.3f} |")
    lines += ["测试表型类别分离接近有限样本随机参照；站点是否有超出组数的结构还应看train、对照和站点保持的置换。该参照不处理图相关、站点/标签组成混杂或队列偏移，不是显著性检验，也不证明模型完全站点无关。",
              "### 辅助头预测（test）", "| Method | Modality | AUC % | BA % | Mean P(ASD) ± SD | Binary Brier | Confidence ECE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for method in METHODS[1:]:
        for modality in ("img", "ph"):
            v = c[method]["auxiliary"]["test"][modality]
            lines.append(f"| {method} | {modality} | {v['AUC']*100:.2f} | {v['BA']*100:.2f} | {v['prob_asd']['mean']:.4f}±{v['prob_asd']['std']:.4f} | {v['brier_asd_binary']:.4f} | {v['ece_confidence_10_bins']:.4f} |")
    lines += ["弱表型辅助AUC接近机会与均值概率/类内变化一起支持『当前表示/头缺少可靠疾病信号』，不证明输入完全无条件互补信息。图分支还混入共享population graph和随机VAE，不能把该辅助AUC当成原始3维表型的最优可达AUC。高effective rank可由噪声产生，低方差未必低秩；高img AUC但低BA也可能是排序尚可、预测阈值几乎只选多数类。暂不支持直接增大lambda_align，先确定目标语义和梯度方向。",
              "## D4 PPU数值与随机稳定性",
              "12个固定已训练checkpoint：E0/E6×911/912/913×fold0/5，每卡6个；每条件三次。每次恢复同一checkpoint、固定输入并保存前后Python/NumPy/Torch CPU/CUDA RNG指纹。TopK记录各层映射回871人全局索引的hash，分别比较排序及所选节点集合。",
              "native进程没有CUBLAS_WORKSPACE_CONFIG且deterministic_algorithms=False；deterministic为独立新进程，在CUDA初始化之前设置:4096:8并开启torch.use_deterministic_algorithms(True)。原cuDNN/TF32标志从保存锚点恢复。正式模型、训练后端和成绩不改。",
              "| Condition | Checkpoints | Max Logits Δ | Max Prob Δ | Max VAE Δ | Max Test Label Flips | TopK Set-change Pairs /36 | Max Diagnostic ACC Span (pp) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, v in stability.items():
        lines.append(f"| {name} | {v['checkpoints']} | {v['max_abs_logits']:.6g} | {v['max_abs_probs']:.6g} | {v['max_abs_vae']:.6g} | {v['label_flips_test']} | {v['topk_set_changed_comparisons']}/{v['pairwise_comparisons']} | {v['max_acc_span']*100:.3f} |")
    lines += ["original_free_rng反映正常VAE/Dropout抽样加内核误差；same_rng在每次forward前重置完整RNG，剩余差异才是当前条件下的数值不重复。不能将前者全部归因PPU。固定VAE/变化Dropout保留每次常规随机数消耗但复用同一epsilon；固定Dropout/变化VAE用私有CUDA generator提供不同epsilon，同时每次重置全局RNG并消耗原VAE抽样数量。干预仅作用诊断模型实例，恢复原方法，不修改源代码。两种影响包含图构造及非线性交互，不能相加得到方差百分比。",
              "相同初始RNG不意味着两个干预的随机源语义相同：fixed_dropout_vary_vae的私有generator另外记录。这里使用保存的训练起始锚点RNG，正式测试前完整RNG只保存hash，因此不能声称重现正式测试那一次随机抽样。每个case的原始差值、pool排序/集合变化及三次诊断ACC均集中保存，但没有选择最好一次或替换原成绩。",
              f"同RNG native最大logits差={stability['same_rng']['max_abs_logits']:.6g}，{stability['same_rng']['topk_order_changed_comparisons']}/36对TopK排序变化，但{stability['same_rng']['topk_set_changed_comparisons']}/36对节点集合变化、测试标签不变；确定性诊断输出及排序完全一致。不能把节点顺序变化说成选择了不同患者。原始自由RNG的平均/最大ACC跨度为{stability['original_free_rng']['mean_acc_span']*100:.3f}/{stability['original_free_rng']['max_acc_span']*100:.3f}个百分点；固定VAE后变化Dropout平均跨度{stability['fixed_vae_vary_dropout']['mean_acc_span']*100:.3f}，固定Dropout后变化VAE平均跨度{stability['fixed_dropout_vary_vae']['mean_acc_span']*100:.3f}，仅说明两种抽样均能产生明显波动。",
              "因此当前PPU在已验证固定checkpoint、固定RNG与确定性后端条件下可用于精细比较；原代码单次随机eval不足以可靠解释小于1个百分点的收益。本轮稳定不代表正式500epoch训练轨迹可精确复现；native舍入误差具体底层内核仍未被唯一定位，是否会放大为长期训练分歧未验证。不能以共享seed/初始化替代重复性检查。",
              "下一轮正式训练应事先固定并核验可用确定性后端、固定输入/RNG和划分、保留必要概率输出，针对同一模型做预先规定的独立重复；严格训练内特征选择、避免泄漏，另做留站点/外部验证。历史官方行为复现与新研究协议分开报告，不能把新协议成绩与旧81.88直接混成同一基线。",
              "## H1–H4判断",
              "| 假设 | 判定 | 证据与边界 |", "|---|---|---|",
              "| H1 测量空间错误是主要失败原因 | 部分支持 | 原报告空间确实不对应E3–E7损失，实际目标cos更高；是否存在患者匹配需看gap、正确−假配对及R@k。改诊断不能改变原分类成绩，不能据高均值cos认定主要失败原因已消除。 |",
              "| H2 门控近似类别重加权而非患者选择 | 支持 | 历史类间/类内方差、ESS和E7组内打乱支持权重结构判断；99.99%覆盖本身不构成充分证据。不证明任何可靠性公式都无效。 |",
              "| H3 弱表型疾病语义不足、个体对齐依据弱 | 部分支持 | 当前表型辅助判别、配对对照和类分离支持这一解释；未重新训练独立表型模型或验证更丰富模态，不能证明三维输入绝对信息上限或缺少所有条件互补性。 |",
              "| H4 对齐传回原GTUNet的梯度太弱 | 部分支持 | 历史合并输出处较小、影像侧直接梯度为零；但冻结表型参数处约7.27%并非近零，且4/6诊断点与官方目标方向为负。更像反向通路不对称和目标语义/方向问题，不能将所有编码器无效或仅增加lambda作为结论。 |",
              "PPU限制：普通随机抽样、同RNG内核差异和训练轨迹不稳定是不同层面。本轮结果只支持已运行的固定checkpoint条件，不将推理扰动幅度直接当成新模块效应置信区间。所有假设判断均为机制证据综合，不是独立临床验证。",
              "## 下一阶段方向（最多三个）",
              "### 1. D：可检验的表型条件互补性，而非整体患者对齐",
              "依据：高平均目标cos与接近机会的患者检索并存，表型辅助判别弱；非对称对齐并未给影像编码器直接新梯度。新科学问题是『表型是否改变某类影像证据的诊断价值』，不是『两个人的完整向量是否足够近』。可以保持当前数据，以图/患者分类证据的条件残差及事先规定的表型替换检验为对象；对真实缺失/错配给出可验证退化行为。",
              "最小必要消融：原融合控制；等参数无条件交互；表型条件交互；同类/站点保持的表型替换控制。评估与全部训练选择事先固定，所有组使用同一无泄漏协议及确定性条件。失败风险：只有站点/年龄/性别导致模型学到队列捷径；条件交互也可能没有可迁移信号。不能把一种conditional gate本身写成新颖性。",
              "### 2. C：引入诊断前可获得的真正临床配对模态",
              "依据：表型表示有个体差异不等于疾病语义，现有辅助头不足以支持患者级语义对齐。新科学问题是『什么信息应共享，什么临床信息只在条件下互补』及『缺失/错配临床模态如何影响选择性融合』。需要改变数据，可考虑有合规授权且预诊断可用的症状/发育/认知资料；不把直接定义ASD标签的诊断量表或诊断后文本作为无泄漏预测输入。",
              "最小必要消融：原三维表型；新增临床信息直接融合；等参数新增信息不对齐；有可辨识语义约束的选择性共享/条件互补；预先规定的临床缺失/错配控制。风险：样本交集缩小、缺失机制/标签泄漏、外部站点临床规范不一致，额外数据收益可能不是方法创新。",
              "### 3. B：只约束诊断相关局部关系/类别条件结构",
              "依据：患者正确配对优势很弱，个体整体余弦目标可能只学公共方向；先验证同类、同站点及跨站点关系的不同意义。新问题是『哪种局部诊断关系可以跨模态转移且不复制站点结构』，不再要求每个患者的全向量对应。可先不改变数据，但不因看到本轮test分数就选择邻居、阈值或lambda。",
              "最小必要消融：相同容量辅助控制；无条件关系匹配；训练标签/训练内局部结构条件匹配；保持站点/类别组成的关系置换控制；固定无泄漏CV和留站点验证。风险：弱表型下局部关系仍主要是类先验，且与已有关系几何方法明显重叠，需要充分检索后再决定是否投入正式训练。",
              "### 与已有方法的边界",
              "MGDR通过多图表示、公共/私有表示及私有对比约束做融合；简单加shared/private分支并非新的科学问题。[MGDR原论文](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)",
              "本报告指ICML 2026 PhenoBrain（不是同名罕见病EHR系统）：用表型条件的长程通信/attention机制调节脑网络。[PhenoBrain原文入口](https://proceedings.mlr.press/v306/meng26f.html)",
              "SMGD区分共享诊断、互补诊断、偶然一致、模态非稳健相关，以关系几何与正则抑制混杂。[SMGD原文入口](https://proceedings.mlr.press/v306/shi26ag.html)",
              "方向D不能只复制PhenoBrain的表型条件路由，应把可检验的条件诊断证据与干预退化作为问题；与MGDR区别不是另加私有头，与SMGD区别也不能只是给成分换名。方向C的数据语义、配对缺失/错配与诊断前可用性是问题来源，但仍需与上述融合/解耦方法直接比较。方向B与SMGD关系几何最接近，新颖性风险最高，只有在限定任务/监督来源/可验证局部传递机制上提出实质区别才值得推进。这些是研究候选和风险判断，不是已证明的创新点。",
              "不优先A：在当前三维信息上再设计复杂患者可靠性估计器，暂缺『个体可靠性可以被估计』的直接依据。不优先单独提高lambda：现有目标语义不足且反向通路不对称，增大系数可能只增强公共方向/噪声约束。",
              "## 局限与完成记录",
              "继承官方跨折RFE泄漏、同站点传导式图、仅3个seed，故本轮不能作为无泄漏临床基准结论。旧测试集标签只用于事后统计，不生成新表示、更新原型或选择模型。D1/D3仅分析正式保存的单次随机表示；D4额外重复仅用于稳定性，不替换原产物。门控历史缺失统计明确保留null。",
              f"完成 D1→D2→D3→D4；所有读取输入结束后SHA256复核通过，原数据/模型代码/历史产物保持只读。新目录5个文件。代码与输入指纹、每fold数字、错误/续跑记录、双卡分配均在summary.json，研究决策每组完成后更新handoff.md。"]
    state["hypotheses"] = {"H1": "部分支持", "H2": "支持", "H3": "部分支持", "H4": "部分支持"}
    state["recommended_directions"] = ["D: testable phenotype-conditioned complementarity", "C: richer pre-diagnostic paired clinical modality", "B: diagnostic local/conditional relational structure"]
    atomic_json(state)
    paragraphs = []
    for line in lines:
        if paragraphs and not (line.startswith("|") and paragraphs[-1].startswith("|")):
            paragraphs.append("")
        paragraphs.append(line)
    (HERE / "report.md").write_text("\n".join(paragraphs) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.self_test:
        self_test(); return
    with (HERE / "diagnose.py").open("r") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = HERE / "summary.json"
        if args.render_only:
            state = json.loads(path.read_text())
            historical.source_guard()
            record_code_revision(state)
            render_report(state); update_handoff(state); return
        audit, fits, splits, ctx = audit_inputs()
        code = {name: sha(HERE / name) for name in ("diagnose.py", "replay.py")}
        if path.exists():
            state = json.loads(path.read_text())
            if state["audit"]["manifest"] != audit["manifest"]:
                raise RuntimeError("Historical inputs changed; refuse to mix diagnostic runs")
            record_code_revision(state)
            state["resume_checks"].append({"time_utc": utc(), "unchanged_inputs": len(audit["manifest"])})
        else:
            state = {"schema": 1, "created_utc": utc(), "state": "running", "audit": audit,
                     "diagnostic_code_sha256": code, "executed_code_sha256": code.copy(),
                     "analysis_logic_sha256": analysis_fingerprint(), "code_revisions": [],
                     "resume_checks": [], "errors": [],
                     "config": {"source": str(OLD), "seeds": [911, 912, 913], "folds": list(range(10)),
                                "lambda_aux": 0.05, "lambda_align": 0.1, "gate_threshold": 0.15,
                                "gradient_methods": ["E1", "E3", "E4", "E5", "E6"], "replay_folds": [0, 5],
                                "stability_methods": ["E0", "E6"], "stability_repeats": 3,
                                "gpu_assignment": "fold 0 -> physical PPU 0; fold 5 -> physical PPU 1; one worker per PPU",
                                "retraining": False, "optimizer_steps": 0, "array_statistics_device": "CPU NumPy",
                                "retrieval_ties": "exact float64 score ties, uniform tie-breaking expected hit probability",
                                "phase_definition": "epoch / actual completed epoch count: thirds"}}
        atomic_json(state)
        try:
            if state.get("D1", {}).get("state") != "complete":
                state["D1"] = d1(fits, splits, ctx); state["D1"]["completed_utc"] = utc()
                atomic_json(state); update_handoff(state); print("D1 complete", flush=True)
            if state.get("D2", {}).get("state") != "complete":
                state["D2"] = d2(fits, splits, ctx); state["D2"]["completed_utc"] = utc()
                atomic_json(state); update_handoff(state); print("D2 complete", flush=True)
            if state.get("D3", {}).get("state") not in ("arrays_complete", "complete"):
                state["D3"] = d3_arrays(fits, splits, ctx); atomic_json(state)
            if state["D3"]["state"] != "complete":
                gpu_workers(state, "gradients", gpu_tasks(fits, ("E1", "E3", "E4", "E5", "E6")))
                state["D3"]["state"] = "complete"; state["D3"]["completed_utc"] = utc()
                atomic_json(state); update_handoff(state); print("D3 complete", flush=True)
            state.setdefault("D4", {"state": "running", "checkpoints": {}})
            if state["D4"]["state"] != "complete":
                tasks = gpu_tasks(fits, ("E0", "E6"))
                for stage in ("native", "deterministic"):
                    gpu_workers(state, stage, tasks)
                state["D4"]["state"] = "complete"; state["D4"]["completed_utc"] = utc()
                atomic_json(state); update_handoff(state); print("D4 complete", flush=True)
            for relative, entry in audit["manifest"].items():
                if sha(ROOT / relative) != entry["sha256"]:
                    raise RuntimeError(f"Read-only historical asset changed: {relative}")
            state["final_integrity"] = {"time_utc": utc(), "unchanged_input_files": len(audit["manifest"]), "passed": True}
            state["state"] = "complete"; state["completed_utc"] = utc()
            atomic_json(state); render_report(state); update_handoff(state)
            print(f"COMPLETE: {HERE / 'report.md'}", flush=True)
        except BaseException as exc:
            state["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            state["errors"].append({"time_utc": utc(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)})
            atomic_json(state)
            raise


if __name__ == "__main__":
    main()
