"""Official sequential anchors and paired, fold-independent PPU experiments."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
# Official caches contain trusted NumPy arrays; torch 2.0 loaded these by default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))

import fcntl
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.mydataloader import MyDataloader
from utils.tools import EarlyStopping, feature_selection, get_upper_triangle_networks
from utils.metrics import accuracy, auc, metrics
from drpa import DRPA, EmbeddingHooks, finite, gate_diagnostics


SOURCE_FILES = ["opt.py", "train_mm_gtunets.py", "model/mm_gtunets.py",
                "model/gtunet.py", "model/rp_graph.py", "utils/mydataloader.py",
                "utils/tools.py", "utils/metrics.py", "tests/test_ppu_custom_conv_equivalence.py"]


def config():
    return json.loads((BASE / "config.json").read_text())


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(array_hash(state[key]).encode())
    return h.hexdigest()


def training_code_hash():
    paths = [BASE / name for name in ("run.py", "drpa.py", "config.json")]
    return hashlib.sha256("".join(sha_file(p) for p in paths).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def atomic_torch(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def require_ppu():
    if not torch.cuda.is_available():
        raise RuntimeError("No accelerator: CPU training is explicitly forbidden")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "PPU" not in name.upper():
        raise RuntimeError(f"Expected this experiment's PPU backend, got {name}")
    return torch.device("cuda:0")


def make_options(seed, ckpt_path, initialize=False):
    saved_argv = sys.argv
    try:
        sys.argv = ["official_options"]
        settings = OptInit(model="MM_GTUNets", dataset="ABIDE", atlas="aal")
    finally:
        sys.argv = saved_argv
    cfg = config()
    args = settings.args
    args.seed = seed
    args.data_folder = cfg["data_folder"]
    args.ckpt_path = str(ckpt_path)
    args.epoch, args.early_stop = cfg["epochs"], cfg["early_stop"]
    args.lr, args.vae_lr, args.wd = cfg["lr"], cfg["vae_lr"], cfg["weight_decay"]
    args.device = require_ppu()
    if initialize:
        settings.initialize()
    return args


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone(),
            "backend": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], 0)
    backend = state.get("backend", {"cudnn_deterministic": True, "cudnn_benchmark": False})
    torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
    if "matmul_allow_tf32" in backend:
        torch.backends.cuda.matmul.allow_tf32 = backend["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = backend["cudnn_allow_tf32"]


def rng_hash(state):
    return {"python": hashlib.sha256(repr(state["python"]).encode()).hexdigest(),
            "numpy": hashlib.sha256(repr(state["numpy"][0]).encode()
                                    + state["numpy"][1].tobytes()
                                    + repr(state["numpy"][2:]).encode()).hexdigest(),
            "cpu": array_hash(state["cpu"]), "cuda": array_hash(state["cuda"])}


def export_core(core):
    return {"state": {k: v.detach().cpu().clone() for k, v in core.state_dict().items()},
            "rp_weights": core.rp_attention.weights.detach().cpu().clone()}


def import_core(core, state):
    core.load_state_dict(state["state"], strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(state["rp_weights"].to(core.device))


def core_hash(state):
    return state_hash({**state["state"], "unregistered_rp_weights": state["rp_weights"]})


def private_seed(seed, fold, purpose):
    return int(hashlib.sha256(f"{seed}/{fold}/{purpose}".encode()).hexdigest()[:15], 16)


def source_guard():
    audit_path = BASE / "artifacts/audit.json"
    if not audit_path.exists():
        raise RuntimeError("Audit must finish before any formal experiment")
    saved = json.loads(audit_path.read_text())
    for name, sha in saved["source_sha256"].items():
        if sha_file(ROOT / name) != sha:
            raise RuntimeError(f"Protected source changed: {name}")
    if saved["training_code_sha256"] != training_code_hash():
        raise RuntimeError("Training implementation changed after registration; audit/resume review required")
    return saved


def context():
    return load_torch(BASE / "artifacts/context.pt")


def audit():
    cfg = config()
    device = require_ppu()
    folder = Path(cfg["data_folder"])
    cache = folder / "save_tensor"
    required = [folder / "id.txt", folder / "Phenotypic_V1_0b_preprocessed1.csv"]
    required += [cache / name for name in ("up_triangles.pt", "labels.pt",
                 "ph_dict_SITE_ID_SEX_AGE_AT_SCAN.pt", "ph_data_SITE_ID_SEX_AGE_AT_SCAN.pt")]
    if not all(p.is_file() for p in required):
        raise FileNotFoundError("All original input caches must exist; loader is read-only")
    opts = make_options(911, BASE / "artifacts/audit_only")
    loader = MyDataloader(opts)
    raw_x, y, ph_dict, ph_data = loader.load_data(save=False)
    ids = loader.ids
    if len(set(ids)) != 871 or raw_x.shape != (871, 6670) or ph_data.shape != (871, 3):
        raise ValueError("Unexpected ABIDE input dimensions or duplicate subjects")
    fresh = get_upper_triangle_networks(ids, opts)
    if not np.array_equal(np.asarray(raw_x), fresh):
        raise ValueError("FC cache differs from exact original .mat transformation")
    if not np.isfinite(raw_x).all() or not np.isfinite(ph_data).all():
        raise FloatingPointError("Non-finite original inputs")
    with required[1].open() as stream:
        rows = list(csv.DictReader(stream))
    included = [r for r in rows if r["SUB_ID"] in set(ids)]
    if [r["SUB_ID"] for r in included] != list(ids):
        raise ValueError("Original CSV order and subject order differ; do not silently reorder")
    mapping = {int(cls): sorted({r["DX_GROUP"] for r, label in zip(included, y) if label == cls})
               for cls in np.unique(y)}
    if mapping != {0: ["1"], 1: ["2"]}:
        raise ValueError(f"Unexpected cached diagnostic mapping: {mapping}")
    label_counts = {int(cls): int((y == cls).sum()) for cls in np.unique(y)}
    if label_counts != {0: 403, 1: 468}:
        raise ValueError(label_counts)
    site_names = [r["SITE_ID"] for r in included]
    site_codes = np.asarray(ph_dict[opts.sites], dtype=np.int64)
    site_mapping = {}
    for code, name in zip(site_codes, site_names):
        previous = site_mapping.setdefault(int(code), name)
        if previous != name:
            raise ValueError("Phenotype site cache is not aligned with subject IDs")
    data_sha = {str(p.relative_to(folder)): sha_file(p) for p in required}
    data_sha.update({f"{s}_aal_correlation.mat": sha_file(folder / f"{s}_aal_correlation.mat") for s in ids})
    data_digest = hashlib.sha256(json.dumps(data_sha, sort_keys=True).encode()).hexdigest()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if git_commit != cfg["source_commit"]:
        raise RuntimeError(f"Unexpected local source commit: {git_commit}")
    import sklearn
    import torch_geometric
    info = {
        "time_utc": utc(), "config": cfg, "source_commit": git_commit,
        "source_sha256": {name: sha_file(ROOT / name) for name in SOURCE_FILES},
        "training_code_sha256": training_code_hash(),
        "data_sha256": data_sha, "data_digest": data_digest,
        "input_shapes": {"image": list(raw_x.shape), "phenotype": list(ph_data.shape)},
        "label_encoding": {"ASD": 0, "HC": 1}, "label_counts": label_counts,
        "site_mapping": site_mapping,
        "environment": {"python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "sklearn": sklearn.__version__,
            "pyg": torch_geometric.__version__, "backend_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_host_threads": torch.get_num_threads(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"),
            "host_thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "paper_audit": {
            "source": cfg["paper"], "figures_checked": [1, 2, 3],
            "sections_checked": ["3.2", "3.3", "3.4", "3.5", "4.1", "4.2", "4.3", "4.4"],
            "interpretation": "FC upper triangle and site/sex/age inputs; RFE and VAE create equal 500-D sizes, not demonstrated semantic alignment. Original shared embedding is the arithmetic mean of modality GTUNet outputs; specific branches remain in fusion. Training-only projection alignment is added at the two 16-D outputs. VAE, AMRS, graph, encoders, fusion and classifier stay fixed.",
            "paper_code_differences": "Paper: 300 classifier epochs, 8:1:1, correlation similarity and multi-head/LN diagram. Source/history: 500 epochs, 81:9:10, torch Euclidean similarity, one attention head and no explicit LN. Experiments preserve source/history, not silently the textual settings."
        },
        "preserved_source_behaviors": ["sequential overwritten RFE", "phenotype-loop early return",
            "shared RP conv", "Euclidean torch similarity", "TransformerConv heads=1",
            "stochastic eval feature dropout", "stochastic eval VAE",
            "detached torch.tensor contribution weights/value", "dropout_edge default probability",
            "np.var reporting", "unregistered RP weights", "shared repeated GTUNet convolutions"],
        "limitations": ["Inherited cross-fold RFE leakage; no clinical generalization claim",
                        "Same-site transductive CV is not leave-site-out validation",
                        "No explicit identifiability of private vs shared information",
                        "Other users' workloads may share both PPUs; do not stop them"]
    }
    atomic_torch({"ids": list(ids), "labels": np.asarray(y), "ph_dict": ph_dict,
                  "ph_data": ph_data, "site_codes": site_codes, "site_names": site_names,
                  "site_mapping": site_mapping, "data_digest": data_digest}, BASE / "artifacts/context.pt")
    info["context_sha256"] = sha_file(BASE / "artifacts/context.pt")
    atomic_json(info, BASE / "artifacts/audit.json")
    print("INITIAL AUDIT", json.dumps({k: info[k] for k in ("time_utc", "source_commit", "training_code_sha256",
          "input_shapes", "label_encoding", "environment", "paper_audit", "preserved_source_behaviors", "limitations")}), flush=True)
    return info


def fold_dir(method, seed, fold):
    return BASE / "outputs" / method / f"seed_{seed}" / f"fold_{fold}"


def anchor_dir(seed, fold):
    return BASE / "artifacts" / f"seed_{seed}" / f"fold_{fold}"


def completed(method, seed, fold, verify=True):
    out = fold_dir(method, seed, fold)
    try:
        status = json.loads((out / "status.json").read_text())
        result = json.loads((out / "metrics.json").read_text())
        if status["state"] != "complete" or result["training_code_sha256"] != training_code_hash():
            return False
        if result["method"] != method or result["seed"] != seed or result["fold"] != fold:
            return False
        for name in ("best.pt", "oof.csv", "embeddings.npz", "epochs.jsonl"):
            if not (out / name).is_file() or (out / name).stat().st_size == 0:
                return False
        if verify:
            if sha_file(out / "metrics.json") != status["metrics_sha256"]:
                return False
            for name, sha in result["output_sha256"].items():
                if sha_file(out / name) != sha:
                    return False
            with (out / "oof.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != result["split_sizes"]["test"] or len({r["subject_id"] for r in rows}) != len(rows):
                return False
        if method == "E0" and not (anchor_dir(seed, fold) / "end_rng.pt").is_file():
            return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def split_fingerprint(splits):
    return {name: array_hash(value) for name, value in zip(("train", "val", "test"), splits)}


def check_splits(splits, n=871):
    joined = np.concatenate(splits)
    if len(joined) != n or not np.array_equal(np.sort(joined), np.arange(n)):
        raise ValueError("Splits overlap, omit or duplicate subjects")


def recover_rfe_mask(raw, selected):
    raw32 = np.asarray(raw, dtype=np.float32)
    selected = selected.detach().cpu().numpy()
    lookup = {}
    for j in range(raw32.shape[1]):
        lookup.setdefault(array_hash(raw32[:, j]), []).append(j)
    mask = []
    for column in selected.T:
        matches = lookup.get(array_hash(column), [])
        if len(matches) != 1:
            raise ValueError("RFE support recovery is ambiguous; no guessed mask")
        mask.append(matches[0])
    if mask != sorted(mask) or len(set(mask)) != len(mask):
        raise ValueError("Unexpected RFE feature order")
    if not np.array_equal(raw32[:, mask], selected):
        raise ValueError("Recovered RFE mask is not exact")
    return np.asarray(mask, dtype=np.int64)


def binary_metrics(encoded_y, encoded_pred, asd_probability):
    y = np.asarray(encoded_y) == 0
    pred = np.asarray(encoded_pred) == 0
    tp = int(np.sum(y & pred)); fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred)); fp = int(np.sum(~y & pred))
    sen = tp / (tp + fn) if tp + fn else None
    spe = tn / (tn + fp) if tn + fp else None
    denom = 2 * tp + fp + fn
    return {"ACC": float(np.mean(y == pred)),
            "BA": (sen + spe) / 2 if sen is not None and spe is not None else None,
            "SEN": sen, "SPE": spe,
            "F1": 2 * tp / denom if denom else 0.0,
            "AUC": float(roc_auc_score(y, asd_probability)) if len(np.unique(y)) == 2 else None}


def norm_of_gradients(loss, tensors):
    if loss is None or not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    squared = sum(g.detach().square().sum() for g in grads if g is not None)
    return float(squared.sqrt()) if torch.is_tensor(squared) else 0.0


def gradient_norm(grads):
    squared = sum(g.detach().square().sum() for g in grads if g is not None)
    return float(squared.sqrt()) if torch.is_tensor(squared) else 0.0


def cosine_diagnostics(a, b):
    a, b = F.normalize(a.detach(), dim=1), F.normalize(b.detach(), dim=1)
    similarity = a @ b.T
    n = len(a)
    same = similarity.diag().mean()
    different = (similarity.sum() - similarity.diag().sum()) / max(n * (n - 1), 1)
    target = torch.arange(n, device=a.device)
    return {"same_subject_cosine": float(same), "different_subject_cosine": float(different),
            "cosine_gap": float(same - different),
            "R_at_1_img_to_ph": float((similarity.argmax(1) == target).float().mean()),
            "R_at_1_ph_to_img": float((similarity.argmax(0) == target).float().mean()),
            "variance_img": float(a.var(0, unbiased=False).mean()),
            "variance_ph": float(b.var(0, unbiased=False).mean()),
            "mean_shared_variance": float(((a + b) * 0.5).var(0, unbiased=False).mean())}


def prediction_outputs(core, hooks, auxiliary, x, ph, affinity, labels, splits, ctx, cfg, out, method):
    saved = load_torch(out / "best.pt")
    import_core(core, saved["core"])
    if auxiliary is not None:
        auxiliary.load_state_dict(saved["auxiliary"], strict=True)
        auxiliary.eval()
    core.eval()
    # The original evaluator has gradients enabled and samples VAE/dropout once.
    eval_rng = capture_rng()
    logits, graph_loss = core(x, ph, affinity)
    end_rng = capture_rng()
    logits_np = logits.detach().cpu().numpy()
    test = splits[2]
    probs = softmax(logits_np, axis=1)
    preds = logits_np.argmax(1)
    y = ctx["labels"]
    primary = binary_metrics(y[test], preds[test], probs[test, 0])
    sen, spe, f1 = metrics(logits_np[test], y[test])
    original = {"ACC": float(accuracy(logits_np[test], y[test])[1]), "SEN": float(sen),
                "SPE": float(spe), "BA": float((sen + spe) / 2),
                "AUC": float(auc(logits_np[test], y[test])), "F1": float(f1)}
    rows = []
    for idx in test:
        rows.append({"method": method, "seed": saved["seed"], "fold": saved["fold"],
            "index": int(idx), "subject_id": ctx["ids"][idx], "site": ctx["site_names"][idx],
            "label_original": int(y[idx]), "label_asd": int(y[idx] == 0),
            "pred_original": int(preds[idx]), "pred_asd": int(preds[idx] == 0),
            "prob_asd": float(probs[idx, 0]), "prob_hc": float(probs[idx, 1])})
    tmp = out / f"oof.csv.tmp.{os.getpid()}"
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, out / "oof.csv")
    diagnostics = {"raw": cosine_diagnostics(hooks.img, hooks.ph), "sites": {}}
    arrays = {"img_embed": hooks.img.detach().cpu().numpy(), "ph_embed": hooks.ph.detach().cpu().numpy(),
              "logits": logits_np, "indices_test": test}
    if auxiliary is not None:
        with torch.no_grad():
            value = auxiliary(hooks.img, hooks.ph)
            diagnostics["shared"] = cosine_diagnostics(value["si"], value["sp"])
            diagnostics["auxiliary"] = {}
            for name, indices in zip(("train", "val", "test"), splits):
                diagnostics["auxiliary"][name] = {}
                for modality, key in (("img", "logits_i"), ("ph", "logits_p")):
                    scores = value[key].detach().cpu().numpy()[indices]
                    diagnostics["auxiliary"][name][modality] = binary_metrics(
                        y[indices], scores.argmax(1), softmax(scores, axis=1)[:, 0])
            ti = torch.as_tensor(splits[0], dtype=torch.long, device=core.device)
            sites = torch.as_tensor(ctx["site_codes"], device=core.device)
            _, _, detail, _ = auxiliary.objective(hooks.img, hooks.ph, labels, sites, ti, method,
                                                  apply_shuffle=False)
            diagnostics["eval_train_gate_unshuffled"] = gate_diagnostics(detail, labels[ti], sites[ti])
            arrays.update({name: tensor.cpu().numpy() for name, tensor in value.items()})
    for site in sorted(set(ctx["site_names"])):
        idx = test[np.array([ctx["site_names"][j] == site for j in test])]
        counts = [int(np.sum(y[idx] == cls)) for cls in (0, 1)]
        record = {"n": len(idx), "n_asd": counts[0], "n_hc": counts[1], "eligible": False}
        if len(idx) >= cfg["minimum_site_metric_samples"] and min(counts) >= cfg["minimum_site_metric_per_class"]:
            record.update(binary_metrics(y[idx], preds[idx], probs[idx, 0])); record["eligible"] = True
        diagnostics["sites"][site] = record
    with (out / f"embeddings.npz.tmp.{os.getpid()}").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(out / f"embeddings.npz.tmp.{os.getpid()}", out / "embeddings.npz")
    return primary, original, diagnostics, eval_rng, end_rng


def train_fold(core, auxiliary, x, ph, affinity, splits, ctx, method, seed, fold, pairing):
    cfg = config()
    out = fold_dir(method, seed, fold)
    out.mkdir(parents=True, exist_ok=True)
    labels = torch.as_tensor(ctx["labels"], dtype=torch.long, device=core.device)
    sites = torch.as_tensor(ctx["site_codes"], dtype=torch.long, device=core.device)
    train_idx, val_idx = [torch.as_tensor(i, dtype=torch.long, device=core.device) for i in splits[:2]]
    hooks = EmbeddingHooks(core)
    params = list(core.parameters()) + (list(auxiliary.parameters()) if auxiliary is not None else [])
    optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    stopper = EarlyStopping(patience=cfg["early_stop"], verbose=True)
    shuffle_rng = torch.Generator(device="cpu").manual_seed(private_seed(seed, fold, "shuffle"))
    best_loss, best_epoch = 1e10, 0
    received_core, received_aux = set(), set()
    best_train, last_gate = {}, None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (out / "epochs.jsonl").open("w") as log:
        for epoch in range(cfg["epochs"]):
            core.train()
            if auxiliary is not None:
                auxiliary.train()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs, graph = core(x, ph, affinity)
                ce = F.cross_entropy(outputs[splits[0]], labels[splits[0]])
                official = ce + graph
                total = official
                aux = align = None
                detail = None
                if auxiliary is not None:
                    aux, align, detail, _ = auxiliary.objective(hooks.img, hooks.ph, labels, sites,
                                                                 train_idx, method, shuffle_rng)
                    total = total + cfg["lambda_aux"] * aux
                    if align is not None:
                        total = total + cfg["lambda_align"] * align
                finite(total, "total training loss")
            diagnostic = epoch % cfg["diagnostic_frequency"] == 0
            grad_record = None
            aux_grads = align_grads = (None, None)
            if diagnostic:
                embeddings = [hooks.img, hooks.ph]
                for embedding in embeddings:
                    embedding.retain_grad()
                if aux is not None:
                    aux_grads = torch.autograd.grad(cfg["lambda_aux"] * aux, embeddings,
                                                   retain_graph=True, allow_unused=True)
                if align is not None:
                    align_grads = torch.autograd.grad(cfg["lambda_align"] * align, embeddings,
                                                     retain_graph=True, allow_unused=True)
                grad_record = {"unit": "L2 norm at original GTUNet outputs; official gradient recovered from total minus new losses",
                    "weighted_aux": gradient_norm(aux_grads), "weighted_align": gradient_norm(align_grads)}
                if auxiliary is not None:
                    grad_record["weighted_aux_new_parameters"] = norm_of_gradients(cfg["lambda_aux"] * aux, list(auxiliary.parameters()))
                    grad_record["weighted_align_new_parameters"] = norm_of_gradients(cfg["lambda_align"] * align if align is not None else None, list(auxiliary.parameters()))
                for embedding in embeddings:
                    embedding.grad = None
            total.backward()
            if diagnostic:
                official_grads = []
                for embedding, ga, gl in zip(embeddings, aux_grads, align_grads):
                    gradient = embedding.grad.detach()
                    if ga is not None:
                        gradient = gradient - ga
                    if gl is not None:
                        gradient = gradient - gl
                    official_grads.append(gradient)
                grad_record["official"] = gradient_norm(official_grads)
            for name, param in core.named_parameters():
                if param.grad is not None:
                    if diagnostic:
                        finite(param.grad, f"core gradient {name}")
                    received_core.add(name)
            if auxiliary is not None:
                for name, param in auxiliary.named_parameters():
                    if param.grad is not None:
                        if diagnostic:
                            finite(param.grad, f"auxiliary gradient {name}")
                        received_aux.add(name)
                if not received_aux:
                    raise RuntimeError("No new-module gradients")
            optimizer.step()
            core.eval()
            with torch.set_grad_enabled(False):
                val_outputs, val_graph = core(x, ph, affinity)
            val_ce = F.cross_entropy(val_outputs[splits[1]], labels[splits[1]])
            val_loss = val_ce + val_graph
            finite(val_loss, "official validation loss")
            record = {"epoch": epoch, "train_ce": float(ce.detach()), "train_graph": float(graph.detach()),
                      "train_official": float(official.detach()), "train_total": float(total.detach()),
                      "train_aux": float(aux.detach()) if aux is not None else 0.0,
                      "train_align": float(align.detach()) if align is not None else 0.0,
                      "validation_ce": float(val_ce), "validation_graph": float(val_graph),
                      "validation_original_loss": float(val_loss),
                      "train_acc_original": float((outputs[train_idx].argmax(1) == labels[train_idx]).float().mean()),
                      "validation_acc_original": float((val_outputs[val_idx].argmax(1) == labels[val_idx]).float().mean()),
                      "gate_all_zero": bool(detail["all_zero"]) if detail is not None else False}
            if diagnostic:
                record["gradient_norms"] = grad_record
                if detail is not None:
                    record["gate"] = gate_diagnostics(detail, labels[train_idx], sites[train_idx])
                    last_gate = record["gate"]
                if auxiliary is not None:
                    record["new_parameter_total_grad_norm"] = float(torch.stack([
                        p.grad.detach().square().sum() for p in auxiliary.parameters() if p.grad is not None]).sum().sqrt())
                print(f"{method} seed={seed} fold={fold} epoch={epoch} train={record['train_total']:.5f} val={float(val_loss):.5f}", flush=True)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if best_loss > float(val_loss):
                best_loss, best_epoch = float(val_loss), epoch
                best_train = record.copy()
                checkpoint = {"method": method, "seed": seed, "fold": fold, "epoch": epoch,
                    "validation_original_loss": best_loss, "core": export_core(core),
                    "auxiliary": {k: v.detach().cpu().clone() for k, v in auxiliary.state_dict().items()} if auxiliary is not None else None,
                    "pairing": pairing, "training_code_sha256": training_code_hash()}
                atomic_torch(checkpoint, out / "best.pt")
            stopper(val_loss, core)
            if stopper.early_stop:
                break
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    primary, original, diag, eval_rng, end_rng = prediction_outputs(
        core, hooks, auxiliary, x, ph, affinity, labels, splits, ctx, cfg, out, method)
    result = {"method": method, "name": cfg["methods"][method]["name"], "seed": seed, "fold": fold,
        "time_utc": utc(), "config": cfg, "source_commit": cfg["source_commit"],
        "training_code_sha256": training_code_hash(), "data_digest": ctx["data_digest"],
        "pairing": pairing, "split_sizes": {name: len(value) for name, value in zip(("train", "val", "test"), splits)},
        "best_epoch": best_epoch, "epochs_run": epoch + 1, "best_validation_original_loss": best_loss,
        "metrics": primary, "official_metrics": original, "metric_positive_class": "ASD (original label 0)",
        "official_metric_positive_class": "HC (original label 1)", "best_train_losses": best_train,
        "last_sampled_gate": last_gate, "diagnostics": diag, "classifier_seconds": train_seconds,
        "peak_allocated_memory_bytes": peak, "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(0),
        "parameters": {"core_total": sum(p.numel() for p in core.parameters()),
             "core_receiving_gradients": sum(p.numel() for n, p in core.named_parameters() if n in received_core),
             "new_total": sum(p.numel() for p in auxiliary.parameters()) if auxiliary is not None else 0,
             "new_receiving_gradients": sum(p.numel() for n, p in auxiliary.named_parameters() if n in received_aux) if auxiliary is not None else 0},
        "test_forward_rng": rng_hash(eval_rng), "end_rng_fingerprint": rng_hash(end_rng)}
    result["output_sha256"] = {name: sha_file(out / name) for name in ("best.pt", "oof.csv", "embeddings.npz", "epochs.jsonl")}
    hooks.close()
    print(f"COMPLETE {method} seed={seed} fold={fold} ACC={primary['ACC']:.5f} AUC={primary['AUC']:.5f} best={best_epoch}", flush=True)
    return result, end_rng


def finish_result(result, method, seed, fold):
    out = fold_dir(method, seed, fold)
    atomic_json(result, out / "metrics.json")
    previous = json.loads((out / "status.json").read_text())
    atomic_json({**previous, "state": "complete", "completed_utc": utc(),
                 "metrics_sha256": sha_file(out / "metrics.json")}, out / "status.json")


def begin_status(method, seed, fold):
    out = fold_dir(method, seed, fold)
    previous = {}
    if (out / "status.json").exists():
        previous = json.loads((out / "status.json").read_text())
    if previous.get("state") == "complete":
        raise RuntimeError("An existing complete result failed validation; refuse to overwrite")
    atomic_json({"state": "running", "method": method, "seed": seed, "fold": fold,
                 "started_utc": utc(), "pid": os.getpid(), "attempt": previous.get("attempt", 0) + 1},
                out / "status.json")


def fail_status(method, seed, fold, exc):
    out = fold_dir(method, seed, fold)
    previous = json.loads((out / "status.json").read_text())
    atomic_json({**previous, "state": "failed", "failed_utc": utc(),
                 "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()},
                out / "status.json")


def anchors(seed):
    source_guard()
    ctx = context()
    seed_dir = BASE / "artifacts" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    opt = make_options(seed, seed_dir / "vae", initialize=True)
    loader = MyDataloader(opt)
    x, y, ph_dict, ph_data = loader.load_data(save=False)
    if not np.array_equal(y, ctx["labels"]) or not np.array_equal(ph_data, ctx["ph_data"]):
        raise ValueError("Anchor inputs changed since audit")
    splits_all = loader.data_split(config()["folds"], val_ratio=config()["val_ratio"])
    if not np.array_equal(np.sort(np.concatenate([s[2] for s in splits_all])), np.arange(871)):
        raise ValueError("CV test indices do not cover each subject once")
    features_path = seed_dir / "selected_features.pt"
    mask_path = seed_dir / "rfe_support.npy"
    for fold, splits in enumerate(splits_all):
        with file_lock(fold_dir("E0", seed, fold) / "worker.lock"):
            check_splits(splits)
            if completed("E0", seed, fold):
                x = load_torch(features_path)
                restore_rng(load_torch(anchor_dir(seed, fold) / "end_rng.pt"))
                print(f"VERIFIED SKIP E0 seed={seed} fold={fold}", flush=True)
                continue
            begin_status("E0", seed, fold)
            try:
                before_features = x
                x = feature_selection(x, y, splits[0], opt.node_dim)
                if fold == 0:
                    mask = recover_rfe_mask(before_features, x)
                    atomic_torch(x, features_path)
                    with mask_path.open("wb") as stream:
                        np.save(stream, mask)
                elif not torch.equal(x, load_torch(features_path)):
                    raise ValueError("Original repeated 500-to-500 RFE unexpectedly changed features")
                aff = torch.from_numpy(create_reward_penalty_graph(ph_dict, y, *splits, opt)).float()
                adir = anchor_dir(seed, fold)
                adir.mkdir(parents=True, exist_ok=True)
                pre_init_rng = capture_rng()
                if (adir / "initial.pt").exists():
                    previous = load_torch(adir / "initial.pt")
                    pre_init_rng = previous["pre_init_rng"]
                    restore_rng(pre_init_rng)
                core = MM_GTUNets(opt, fold).to(opt.device)
                initial = export_core(core)
                atomic_torch({"pre_init_rng": pre_init_rng, "core": initial}, adir / "initial.pt")
                # Constructing Adam is deterministic and matches the original order.
                original_optimizer = torch.optim.Adam(core.parameters(), lr=opt.lr, weight_decay=opt.wd)
                del original_optimizer
                pretrain_start = time.perf_counter()
                if (adir / "train_start.pt").exists():
                    start = load_torch(adir / "train_start.pt")
                    if start["initial_fingerprint"] != core_hash(initial):
                        raise RuntimeError("Retry did not reproduce the original core initialization")
                    import_core(core, start["core"])
                    restore_rng(start["rng"])
                    vae_seconds = start["vae_seconds"]
                else:
                    core.train_vae(torch.from_numpy(ph_data).float())
                    torch.cuda.synchronize()
                    vae_seconds = time.perf_counter() - pretrain_start
                    start = {"core": export_core(core), "rng": capture_rng(),
                        "initial_fingerprint": core_hash(initial), "vae_seconds": vae_seconds,
                        "vae_sha256": sha_file(core.vae_save_path)}
                    atomic_torch(start, adir / "train_start.pt")
                atomic_torch({"train": splits[0], "val": splits[1], "test": splits[2]}, adir / "splits.pt")
                pairing = {"split_fingerprints": split_fingerprint(splits),
                    "features_fingerprint": array_hash(x), "rfe_support_sha256": sha_file(mask_path),
                    "initial_core_fingerprint": core_hash(initial), "train_start_core_fingerprint": core_hash(start["core"]),
                    "vae_sha256": sha_file(core.vae_save_path), "classifier_start_rng": rng_hash(start["rng"])}
                atomic_json(pairing, adir / "fingerprints.json")
                restore_rng(start["rng"])
                result, end_rng = train_fold(core, None, x, torch.from_numpy(ph_data).float(), aff,
                                             splits, ctx, "E0", seed, fold, pairing)
                result["vae_pretrain_seconds"] = vae_seconds
                atomic_torch(end_rng, adir / "end_rng.pt")
                finish_result(result, "E0", seed, fold)
                restore_rng(end_rng)
                del core, aff
            except BaseException as exc:
                fail_status("E0", seed, fold, exc)
                raise


def variant(method, seed, fold):
    source_guard()
    if method == "E0":
        raise ValueError("E0 must run sequentially via anchors")
    out = fold_dir(method, seed, fold)
    with file_lock(out / "worker.lock"):
        if completed(method, seed, fold):
            print(f"VERIFIED SKIP {method} seed={seed} fold={fold}", flush=True)
            return
        begin_status(method, seed, fold)
        try:
            if not completed("E0", seed, fold):
                raise RuntimeError("Paired E0 anchor is incomplete")
            ctx = context()
            adir = anchor_dir(seed, fold)
            opt = make_options(seed, out)
            start = load_torch(adir / "train_start.pt")
            split = load_torch(adir / "splits.pt")
            splits = [split[k] for k in ("train", "val", "test")]
            check_splits(splits)
            x = load_torch(adir.parent / "selected_features.pt")
            pairing = json.loads((adir / "fingerprints.json").read_text())
            core = MM_GTUNets(opt, fold).to(opt.device)
            core.vae_save_path = str(adir.parent / "vae" / f"fold{fold}_pretrain.pth")
            import_core(core, start["core"])
            auxiliary = DRPA.isolated(config(), private_seed(seed, fold, "heads"), opt.device)
            pairing = {**pairing, "auxiliary_initial_fingerprint": state_hash(auxiliary.state_dict())}
            if core_hash(export_core(core)) != pairing["train_start_core_fingerprint"]:
                raise RuntimeError("Paired core weights mismatch")
            if sha_file(core.vae_save_path) != pairing["vae_sha256"] or array_hash(x) != pairing["features_fingerprint"]:
                raise RuntimeError("Paired features/VAE mismatch")
            if split_fingerprint(splits) != pairing["split_fingerprints"]:
                raise RuntimeError("Paired split mismatch")
            affinity = torch.from_numpy(create_reward_penalty_graph(ctx["ph_dict"], ctx["labels"], *splits, opt)).float()
            restore_rng(start["rng"])
            if rng_hash(capture_rng()) != pairing["classifier_start_rng"]:
                raise RuntimeError("Paired training-start RNG mismatch")
            result, _ = train_fold(core, auxiliary, x, torch.from_numpy(ctx["ph_data"]).float(), affinity,
                                    splits, ctx, method, seed, fold, pairing)
            result["vae_pretrain_seconds"] = 0.0
            result["vae_pretraining_reused"] = True
            finish_result(result, method, seed, fold)
        except BaseException as exc:
            fail_status(method, seed, fold, exc)
            raise


def history_check():
    cfg = config()
    historical_log = (Path(cfg["history_folder"]) / "train.log").read_text()
    pattern = r"Fold (\d+) test accuracy ([0-9.]+), AUC ([0-9.]+)"
    history = {int(f): (float(a), float(u)) for f, a, u in re.findall(pattern, historical_log)}
    if len(history) != 10 or not all(completed("E0", 911, f) for f in range(10)):
        raise RuntimeError("Ten replay/historical folds required for baseline verification")
    per_fold = []
    for fold in range(10):
        result = json.loads((fold_dir("E0", 911, fold) / "metrics.json").read_text())
        actual = result["official_metrics"]
        per_fold.append({"fold": fold, "historical_ACC": history[fold][0], "replay_ACC": actual["ACC"],
                         "delta_ACC": actual["ACC"] - history[fold][0], "historical_AUC": history[fold][1],
                         "replay_AUC": actual["AUC"], "delta_AUC": actual["AUC"] - history[fold][1]})
    delta_acc = float(np.mean([r["delta_ACC"] for r in per_fold]))
    delta_auc = float(np.mean([r["delta_AUC"] for r in per_fold]))
    max_delta = max(abs(r[k]) for r in per_fold for k in ("delta_ACC", "delta_AUC"))
    exact = all(round(r["historical_ACC"], 5) == round(r["replay_ACC"], 5)
                and round(r["historical_AUC"], 5) == round(r["replay_AUC"], 5) for r in per_fold)
    passed = (abs(delta_acc) <= cfg["history_mean_tolerance"] and abs(delta_auc) <= cfg["history_mean_tolerance"]
              and max_delta <= cfg["history_fold_tolerance"])
    value = {"passed": passed, "exact_at_original_output_precision": exact,
             "mean_delta_ACC": delta_acc, "mean_delta_AUC": delta_auc, "max_absolute_fold_delta": max_delta,
             "per_fold": per_fold,
             "interpretation": "Exact at printed precision" if exact else "Not exact; PPU parallel reductions/runtime contention must be investigated before claiming equivalence"}
    atomic_json(value, BASE / "artifacts/history_check.json")
    print("HISTORY CHECK", json.dumps(value), flush=True)
    if not passed:
        raise RuntimeError("Historical baseline tolerance exceeded; variants are blocked pending diagnosis")
    return value


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    p = sub.add_parser("anchors"); p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("variant"); p.add_argument("--method", choices=list(config()["methods"])[1:], required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--fold", type=int, required=True)
    sub.add_parser("history-check")
    args = parser.parse_args()
    if args.command == "audit": audit()
    elif args.command == "anchors": anchors(args.seed)
    elif args.command == "variant": variant(args.method, args.seed, args.fold)
    elif args.command == "history-check": history_check()


if __name__ == "__main__":
    main()
