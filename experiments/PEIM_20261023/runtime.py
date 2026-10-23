"""Frozen single-forward PEIM training runtime."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
BASE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/MM-GTUNets")
HISTORY = ROOT / "experiments/DRPA_20261014"
PCDCA = ROOT / "experiments/PCDCA_20261019"
CFG = json.loads((BASE / "config.json").read_text())
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from torch.nn import functional as F

from evidence import fit_reference, shuffle_conditions
from interaction import LowRankInteraction
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from opt import OptInit
from utils.tools import EarlyStopping

SUBSETS = ("train", "val", "test")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False).encode())


def array_hash(value):
    array = np.ascontiguousarray(value)
    return sha_bytes(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())


def payload_hash(value):
    h = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            h.update(b"tensor")
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            h.update(b"array")
            h.update(array_hash(item).encode())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(str(type(item)).encode())
            h.update(str(len(item)).encode())
            for part in item:
                visit(part)
        elif item is None or isinstance(item, (str, int, float, bool)):
            h.update(str(type(item)).encode())
            h.update(repr(item).encode())
        else:
            raise TypeError(type(item))

    visit(value)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(array_hash(state[name].detach().cpu().numpy()).encode())
    return h.hexdigest()


def anchor(fold):
    return HISTORY / "artifacts" / "seed_911" / f"fold_{fold}"


def result_path(group, fold):
    return BASE / "runs" / group / f"s911_f{fold}.pt"


def load_torch(path, artifact=False):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if artifact:
        if payload_hash(value["payload"]) != value["content_sha256"]:
            raise IOError(f"Artifact hash mismatch: {path}")
        return value["payload"]
    return value


def export_core(core):
    return {"state": {key: value.detach().cpu().clone() for key, value in core.state_dict().items()},
            "rp_weights": core.rp_attention.weights.detach().cpu().clone()}


def import_core(core, state):
    core.load_state_dict(state["state"], strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(state["rp_weights"].to(core.device))


def capture_rng():
    index = torch.cuda.current_device()
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(index).clone(),
            "backend": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], torch.cuda.current_device())
    backend = state["backend"]
    torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = backend["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = backend["cudnn_allow_tf32"]


def atomic_bytes(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(value, path):
    atomic_bytes((json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(), path)


def atomic_torch(value, path):
    stream = __import__("io").BytesIO()
    torch.save({"payload": value, "content_sha256": payload_hash(value)}, stream)
    atomic_bytes(stream.getvalue(), path)


def log(event, **fields):
    row = {"utc": utc(), "event": event, **fields}
    with (BASE / "run.log").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)


def metadata(ctx):
    csv_path = ROOT / CFG["data_folder"] / "Phenotypic_V1_0b_preprocessed1.csv"
    with csv_path.open(newline="") as stream:
        rows = {row["SUB_ID"]: row for row in csv.DictReader(stream)}
    ordered = [rows[subject_id] for subject_id in ctx["ids"]]

    def number(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    ages = np.asarray([number(row["AGE_AT_SCAN"]) for row in ordered], dtype=np.float64)
    sex = np.asarray([number(row["SEX"]) for row in ordered], dtype=np.float64)
    sites = np.asarray([row["SITE_ID"] for row in ordered]).astype(str)
    if not np.allclose(ages, ctx["ph_data"][:, 2], rtol=0, atol=2e-6):
        raise AssertionError("CSV age order does not match original model input")
    if not np.array_equal(sex - 1, ctx["ph_data"][:, 1]):
        raise AssertionError("CSV sex order does not match original model input")
    if not np.array_equal(sites, np.asarray(ctx["site_names"]).astype(str)):
        raise AssertionError("CSV site order does not match original context")
    return {"age": ages, "sex": sex, "site": sites}


def encode_conditions(meta, train):
    age = np.asarray(meta["age"], dtype=np.float64).copy()
    valid = np.isfinite(age) & (age > 0.0) & (age <= 120.0)
    train_valid = valid[np.asarray(train, dtype=np.int64)]
    valid_train_age = age[np.asarray(train, dtype=np.int64)][train_valid]
    fill = float(np.median(valid_train_age)) if len(valid_train_age) else 17.0
    age[~valid] = fill
    mu = float(age[train].mean())
    sd = max(float(age[train].std(ddof=0)), 1e-12)
    age_z = np.clip((age - mu) / sd, -3.0, 3.0)
    sex = np.asarray(meta["sex"], dtype=np.float64)
    sex_contrast = np.where(sex == 1, -1.0, np.where(sex == 2, 1.0, 0.0))
    raw = np.column_stack((age_z, sex_contrast))
    reference = raw[np.asarray(train, dtype=np.int64)].mean(0)
    centered = raw - reference
    return centered, {"age_fill_train_median": fill, "age_train_mean": mu, "age_train_std": sd,
                      "valid_age_count_train": int(train_valid.sum()),
                      "unknown_sex_count": int((~np.isin(sex, [1, 2])).sum()),
                      "d_ref": reference.tolist(), "heldout_stats_used": False}


def fixed_outer_permutation(split, sites, age, sex):
    groups = np.full(len(age), -1, dtype=np.int64)
    for group_id, subset in enumerate(SUBSETS):
        groups[np.asarray(split[subset], dtype=np.int64)] = group_id
    if np.any(groups < 0):
        raise AssertionError("Outer split does not cover every participant")
    indices = np.concatenate([split[key] for key in SUBSETS])
    perm, stats = shuffle_conditions(indices, sites, groups, age, sex, 911)
    stats["by_subset"] = {}
    for group_id, subset in enumerate(SUBSETS):
        rows = np.asarray(split[subset], dtype=np.int64)
        if np.any(groups[perm[rows]] != group_id) or not np.array_equal(sites[perm[rows]], sites[rows]):
            raise AssertionError("Condition permutation crossed outer split or site")
        age_changed = ~np.isclose(age[perm[rows]], age[rows], rtol=0.0, atol=0.0, equal_nan=True)
        sex_changed = ~np.isclose(sex[perm[rows]], sex[rows], rtol=0.0, atol=0.0, equal_nan=True)
        stats["by_subset"][subset] = {
            "n": int(len(rows)),
            "index_change_fraction": float(np.mean(perm[rows] != rows)),
            "age_change_fraction": float(np.mean(age_changed)),
            "sex_change_fraction": float(np.mean(sex_changed)),
            "joint_value_change_fraction": float(np.mean(age_changed | sex_changed)),
            "singletons": int(sum(np.sum((groups == group_id) & (sites == site)) == 1
                                   for site in np.unique(sites[rows])))}
    return perm, stats


def metrics(labels, probability, prediction):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.int64)
    return {"ACC": float(accuracy_score(labels, prediction)),
            "SEN-ASD": float(recall_score(labels, prediction, pos_label=0)),
            "SPE-TD": float(recall_score(labels, prediction, pos_label=1)),
            "probability_AUC": float(roc_auc_score(labels == 0, probability))}


def gate_summary(stats, n_train, selective):
    if not selective:
        return {"train_gate_mean": 1.0, "train_high_confidence_gate_mean": None,
                "train_low_confidence_gate_mean": None,
                "train_high_confidence_relative_l2": None,
                "train_low_confidence_relative_l2": None,
                "train_high_confidence_count": 0, "train_low_confidence_count": 0}
    if not stats or stats["gate"].numel() == 0:
        return {"train_gate_mean": None, "train_high_confidence_gate_mean": None,
                "train_low_confidence_gate_mean": None,
                "train_high_confidence_relative_l2": None,
                "train_low_confidence_relative_l2": None,
                "train_high_confidence_count": 0, "train_low_confidence_count": 0}
    gate = stats["gate"][:n_train]
    relative = stats["relative_l2"][:n_train]
    # A positive margin is only informative for selective groups; use gate <= 0.5
    # as the registered distance-contrast threshold (margin >= tau).
    high = gate <= 0.5
    low = ~high
    average = lambda values, mask: float(values[mask].mean()) if mask.any() else None
    return {"train_gate_mean": float(gate.mean()),
            "train_high_confidence_gate_mean": average(gate, high),
            "train_low_confidence_gate_mean": average(gate, low),
            "train_high_confidence_relative_l2": average(relative, high),
            "train_low_confidence_relative_l2": average(relative, low),
            "train_high_confidence_count": int(high.sum()),
            "train_low_confidence_count": int(low.sum())}


def historical_prototype_module():
    path = PCDCA / "prototype.py"
    spec = importlib.util.spec_from_file_location("peim_pcdca_prototype", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Loaded:
    def __init__(self, fold, physical_gpu, audit, train_only_labels=False):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Worker must see exactly one assigned PPU; CPU training is forbidden")
        torch.cuda.set_device(0)
        if "PPU" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError("Assigned device is not a PPU")
        torch.set_num_threads(CFG["host_threads"])
        self.fold = fold
        self.physical_gpu = physical_gpu
        self.device = torch.device("cuda:0")
        self.ctx = load_torch(HISTORY / "artifacts/context.pt")
        self.meta = metadata(self.ctx)
        self.split = load_torch(anchor(fold) / "splits.pt")
        self.start = load_torch(anchor(fold) / "train_start.pt")
        old_argv = sys.argv
        try:
            sys.argv = ["peim_runtime"]
            self.opt = OptInit(model="MM_GTUNets", dataset="ABIDE", atlas="aal").args
        finally:
            sys.argv = old_argv
        self.opt.seed = 911
        self.opt.device = self.device
        self.opt.data_folder = str(ROOT / CFG["data_folder"])
        self.opt.ckpt_path = str(anchor(fold).parent / "vae")
        self.opt.epoch = CFG["epochs"]
        self.opt.early_stop = CFG["patience"]
        self.opt.lr = CFG["learning_rate"]
        self.opt.wd = CFG["weight_decay"]
        self.opt.log_save = False
        self.opt.model_save = False
        self.core = MM_GTUNets(self.opt, fold).to(self.device)
        self.core.vae_save_path = str(anchor(fold).parent / "vae" / f"fold{fold}_pretrain.pth")
        import_core(self.core, self.start["core"])
        self.x = load_torch(anchor(fold).parent / "selected_features.pt").float().to(self.device)
        self.ph = torch.as_tensor(self.ctx["ph_data"], dtype=torch.float32, device=self.device)
        graph_labels = np.full_like(self.ctx["labels"], -1)
        graph_labels[self.split["train"]] = self.ctx["labels"][self.split["train"]]
        graph = create_reward_penalty_graph(self.ctx["ph_dict"], graph_labels,
                    self.split["train"], self.split["val"], self.split["test"], self.opt)
        self.affinity = torch.as_tensor(graph, dtype=torch.float32, device=self.device)
        self.diagnostic_labels = self.ctx["labels"].copy()
        if train_only_labels:
            self.diagnostic_labels.fill(-1)
            self.diagnostic_labels[self.split["train"]] = self.ctx["labels"][self.split["train"]]
        self.fit_labels = np.full_like(self.ctx["labels"], -1)
        self.fit_labels[self.split["train"]] = self.ctx["labels"][self.split["train"]]
        self.labels = torch.as_tensor(self.diagnostic_labels, dtype=torch.long, device=self.device)
        self.forward_count = 0
        cached_ph = load_torch(ROOT / CFG["data_folder"] / "save_tensor/ph_data_SITE_ID_SEX_AGE_AT_SCAN.pt")
        self.input_digest = payload_hash({"x": self.x.cpu(), "ph": self.ph.cpu(), "graph": self.affinity.cpu(),
                                          "train_labels": self.diagnostic_labels[self.split["train"]],
                                          "ids": self.ctx["ids"], "cached_ph": cached_ph})
        self.audit = audit

    def forward(self):
        self.forward_count += 1
        logits, graph = self.core(self.x, self.ph, self.affinity)
        if tuple(logits.shape) != (871, 2):
            raise AssertionError("Unexpected original classifier output shape")
        return logits, graph

    def close(self):
        del self.core
        torch.cuda.empty_cache()


class InteractionHooks:
    def __init__(self, loaded, module, group, d_true, d_used, private_group, cfg):
        self.loaded, self.module, self.group = loaded, module, group
        self.d_true, self.d_used, self.private_group, self.cfg = d_true, d_used, private_group, cfg
        self.epoch = 0
        self.raw_img = self.raw_ph = self.modified = self.stats = None
        self.evidence = []
        self.fit_seconds = 0.0
        self.h_img = loaded.core.img_unet.register_forward_hook(self.capture_image)
        self.h_ph = loaded.core.ph_unet.register_forward_hook(self.capture_ph)
        self.h_attn = loaded.core.mm_attention.register_forward_pre_hook(self.replace_attention_image)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.modified = self.stats = None
        self.evidence = []
        self.fit_seconds = 0.0

    def fit(self, z_cpu, labels, sites, reference, query):
        started = time.perf_counter()
        method = "main" if self.group == "E1" else "interaction"
        fit = fit_reference(z_cpu, reference, labels, self.d_true, sites, self.cfg, method)
        self.fit_seconds += time.perf_counter() - started
        if not fit["valid"]:
            fit["P"] = np.zeros((16, 16), dtype=np.float64)
        self.evidence.append({"rank": int(fit["rank"]), "valid": bool(fit["valid"]),
                              "reason": fit["reason"], "singular_values": fit["diagnostics"].get("singular_values", []),
                              "reference_n": int(len(reference)), "query_n": int(len(query))})
        return {"indices": query, "mean": fit["mean"], "std": fit["std"], "P": fit["P"],
                "prototypes": fit["prototypes"], "tau": fit["tau"]}

    def capture_image(self, _module, _args, output):
        self.raw_img = output
        z_cpu = output.detach().cpu().numpy()
        train = np.asarray(self.loaded.split["train"], dtype=np.int64)
        parts = []
        for group in range(3):
            query = np.flatnonzero(self.private_group == group)
            if query.size:
                reference = train[self.private_group[train] != group]
                parts.append(self.fit(z_cpu, self.loaded.fit_labels, self.loaded.meta["site"], reference, query))
        heldout = np.concatenate((self.loaded.split["val"], self.loaded.split["test"])).astype(np.int64)
        if heldout.size:
            parts.append(self.fit(z_cpu, self.loaded.fit_labels, self.loaded.meta["site"], train, heldout))
        cond = torch.as_tensor(self.d_used, dtype=output.dtype, device=output.device)
        self.modified, self.stats = self.module.apply(output, cond, parts, self.epoch, self.cfg,
                                                       selective=self.group in ("E3", "E4"))
        return output

    def capture_ph(self, _module, _args, output):
        self.raw_ph = output

    def replace_attention_image(self, _module, args):
        if self.modified is None:
            raise RuntimeError("PEIM image modification was not computed")
        return self.modified, args[1]

    def close(self):
        self.h_img.remove()
        self.h_ph.remove()
        self.h_attn.remove()


def private_group_map(prepared, n=871):
    rows = np.asarray(prepared["train"], dtype=np.int64)
    group = np.full(n, -1, dtype=np.int64)
    group[rows] = np.asarray(prepared["folds"], dtype=np.int64)
    if np.any(group[rows] < 0) or len(set(group[rows])) != 3:
        raise AssertionError("Invalid historical D4 private folds")
    return rows, group


def completed_result(path, audit):
    if not path.exists():
        return None
    saved = load_torch(path, artifact=True)
    if saved.get("state") != "complete" or saved.get("code_sha256") != audit["training_code_sha256"]:
        raise AssertionError(f"Invalid/incompatible saved task {path}")
    metric = metrics(saved["oof"]["labels"], saved["oof"]["probability"], saved["oof"]["prediction"])
    if max(abs(metric[k] - saved["metrics"][k]) for k in metric) > 1e-12:
        raise AssertionError(f"Metric checksum mismatch: {path}")
    split = load_torch(anchor(saved["fold"]) / "splits.pt")
    if not np.array_equal(saved["oof"]["indices"], split["test"]):
        raise AssertionError(f"OOF order mismatch: {path}")
    return saved


def train_one(group, fold, physical_gpu, audit):
    output = result_path(group, fold)
    existing = completed_result(output, audit)
    if existing is not None:
        log("skip_valid", group=group, fold=fold, gpu=physical_gpu)
        return existing
    for path, expected in audit["anchors"][str(fold)]["sha256"].items():
        if sha_file(path) != expected:
            raise AssertionError(f"Protected history SHA changed: {path}")
    loaded = Loaded(fold, physical_gpu, audit)
    hooks = None
    try:
        historical = historical_prototype_module()
        old_cfg = json.loads((PCDCA / "config.json").read_text())
        prepared = historical.prepare(loaded.fit_labels, loaded.meta, loaded.split["train"],
                                      911, fold, old_cfg)
        private_rows, private_group = private_group_map(prepared)
        d_true, condition_stats = encode_conditions(loaded.meta, loaded.split["train"])
        permutation, shuffle_stats = fixed_outer_permutation(
            loaded.split, loaded.meta["site"], loaded.meta["age"], loaded.meta["sex"])
        d_used = d_true if group != "E4" else d_true[permutation]
        objective = historical.PrototypeObjective(prepared, "D4", old_cfg, loaded.device)
        module = LowRankInteraction().to(loaded.device)
        initial_sha = state_hash(module.state_dict())
        if initial_sha != state_hash({"W_d": torch.zeros(16, 2)}) or sum(p.numel() for p in module.parameters()) != 32:
            raise AssertionError("Unexpected PEIM parameter initialization")
        optimizer = torch.optim.Adam(list(loaded.core.parameters()) + list(module.parameters()),
                                     lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
        stopper = EarlyStopping(patience=CFG["patience"], verbose=False, save=False)
        import_core(loaded.core, loaded.start["core"])
        restore_rng(loaded.start["rng"])
        if payload_hash(capture_rng()) != payload_hash(loaded.start["rng"]):
            raise AssertionError("Train-start full RNG mismatch")
        hooks = InteractionHooks(loaded, module, group, d_true, d_used, private_group, CFG)

        bn_start = int(loaded.core.clf[2].num_batches_tracked)
        best_loss, best, history = float("inf"), None, []
        loaded.core.train()
        module.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started, started_utc = time.perf_counter(), utc()
        loaded.forward_count = 0
        for epoch in range(CFG["epochs"]):
            loaded.core.train()
            module.train()
            hooks.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            logits, graph = loaded.forward()
            if loaded.forward_count != 2 * epoch + 1:
                raise AssertionError("Training forward count mismatch")
            z_img, z_ph = hooks.raw_img, hooks.raw_ph
            if z_img is None or z_ph is None:
                raise AssertionError("Raw modality embeddings were not captured")
            train_stats = hooks.stats
            train_evidence = list(hooks.evidence)
            train_evidence_seconds = hooks.fit_seconds
            train_img_weight = float(loaded.core.img_weight.detach())
            train_ph_weight = float(loaded.core.ph_weight.detach())
            native_ce = F.cross_entropy(logits[loaded.split["train"]], loaded.labels[loaded.split["train"]])
            native = native_ce + graph
            d4_loss, _ = objective(z_img)
            lambda_d4, scale_stats = historical.gradient_scale(native, d4_loss, z_img, epoch, old_cfg)
            native_norm = z_img.new_tensor(scale_stats["native_z_grad"])
            d4_norm = z_img.new_tensor(scale_stats["aux_z_grad"])
            total = native if float(lambda_d4) == 0.0 else native + lambda_d4 * d4_loss
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite native+D4 total loss")
            total.backward()
            if module.W_d.grad is None or not torch.isfinite(module.W_d.grad).all():
                raise FloatingPointError("PEIM gradient is absent or nonfinite")
            for name, param in loaded.core.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Nonfinite core gradient: {name}")
            optimizer.step()
            if int(loaded.core.clf[2].num_batches_tracked) != bn_start + epoch + 1:
                raise AssertionError("Unexpected train BN update count")

            loaded.core.eval()
            module.eval()
            hooks.set_epoch(epoch)
            with torch.set_grad_enabled(False):
                val_logits, val_graph = loaded.forward()
            if loaded.forward_count != 2 * epoch + 2:
                raise AssertionError("Validation forward count mismatch")
            val_ce = F.cross_entropy(val_logits[loaded.split["val"]], loaded.labels[loaded.split["val"]])
            val_loss = val_ce + val_graph
            if not torch.isfinite(val_loss):
                raise FloatingPointError("Nonfinite native validation loss")
            if int(loaded.core.clf[2].num_batches_tracked) != bn_start + epoch + 1:
                raise AssertionError("Validation modified classifier BatchNorm")
            stats = train_stats
            ranks = [item["rank"] for item in train_evidence]
            gate_info = gate_summary(stats, len(loaded.split["train"]), group in ("E3", "E4"))
            record = {"epoch": epoch, "train_native_ce": float(native_ce.detach()),
                      "train_graph": float(graph.detach()), "train_d4": float(d4_loss.detach()),
                      "train_img_weight": train_img_weight, "train_ph_weight": train_ph_weight,
                      "lambda_d4": float(lambda_d4), "native_image_gradient": float(native_norm),
                      "d4_image_gradient": float(d4_norm),
                      "d4_weighted_gradient_ratio": scale_stats["weighted_z_gradient_ratio"],
                      "val_ce": float(val_ce), "val_graph": float(val_graph), "val_loss": float(val_loss),
                      "w_norm": float(module.W_d.detach().norm()),
                      "w_gradient_norm": float(module.W_d.grad.detach().norm()),
                      "evidence_rank_mean": float(np.mean(ranks)),
                      "evidence_inactive_fraction": float(np.mean([r == 0 for r in ranks])),
                      "evidence_seconds": float(train_evidence_seconds),
                      "modulation_relative_l2_mean": float(stats["relative_l2"].mean()) if stats["relative_l2"].numel() else 0.0,
                      "gate_mean": float(stats["gate"].mean()) if stats["gate"].numel() else 1.0,
                      **gate_info,
                      "cap_fraction": float(stats["cap_hit"].float().mean()) if stats["cap_hit"].numel() else 0.0,
                      "evidence": train_evidence}
            history.append(record)
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best = {"core": export_core(loaded.core),
                        "interaction": {key: value.detach().cpu().clone()
                                        for key, value in module.state_dict().items()},
                        "epoch": epoch, "val_loss": best_loss, "record": record.copy()}
            stopper(val_loss, loaded.core)
            if epoch % 20 == 0:
                log("progress", group=group, fold=fold, gpu=physical_gpu, epoch=epoch,
                    val_loss=float(val_loss), rank=record["evidence_rank_mean"], w_norm=record["w_norm"])
            if stopper.early_stop:
                break

        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        train_end_rng = capture_rng()
        if best is None:
            raise RuntimeError("No validation-selected checkpoint")
        import_core(loaded.core, best["core"])
        module.load_state_dict(best["interaction"], strict=True)
        loaded.core.eval()
        module.eval()
        if payload_hash(capture_rng()) != payload_hash(train_end_rng):
            raise AssertionError("Best checkpoint restore changed RNG")
        hooks.set_epoch(best["epoch"])
        loaded.forward_count = 0
        with torch.set_grad_enabled(True):
            test_logits, test_graph = loaded.forward()
        if loaded.forward_count != 1:
            raise AssertionError("Formal testing must use one native forward")
        test_rng = capture_rng()
        idx = np.asarray(loaded.split["test"], dtype=np.int64)
        logits = test_logits.detach().cpu().numpy()[idx]
        probability = torch.softmax(test_logits.detach(), dim=1).cpu().numpy()[idx, 0]
        prediction = logits.argmax(1)
        labels = loaded.ctx["labels"][idx]
        formal_metrics = metrics(labels, probability, prediction)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise FloatingPointError("Nonfinite formal test probability")
        record = {"state": "complete", "group": group, "seed": 911, "fold": fold,
                  "physical_gpu": physical_gpu, "started_utc": started_utc, "completed_utc": utc(),
                  "code_sha256": audit["training_code_sha256"], "input_sha256": audit["anchors"][str(fold)]["sha256"],
                  "input_digest": loaded.input_digest, "train_start_sha256": audit["anchors"][str(fold)]["train_start_sha256"],
                  "train_start_rng_sha256": audit["anchors"][str(fold)]["train_start_rng_sha256"],
                  "initial_W_d_sha256": initial_sha,
                  "core": best["core"], "interaction": best["interaction"],
                  "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                  "best_record": best["record"], "epochs": history, "epochs_run": len(history),
                  "training_seconds": seconds, "train_end_rng": train_end_rng,
                  "test_rng_before": train_end_rng, "test_rng_after": test_rng,
                  "formal_test_forwards": 1, "test_logits": logits,
                  "oof": {"indices": idx, "ids": [loaded.ctx["ids"][int(i)] for i in idx],
                          "labels": labels, "logits": logits, "probability": probability, "prediction": prediction},
                  "metrics": formal_metrics, "private_fold_sizes": np.bincount(
                      np.asarray(prepared["folds"]), minlength=3).tolist(),
                  "condition_stats": condition_stats, "shuffle_stats": shuffle_stats,
                  "trainable_added_parameters": 32,
                  "W_d_norm": float(module.W_d.detach().norm()),
                  "test_modulation": {"relative_l2_mean": float(hooks.stats["relative_l2"].mean()) if hooks.stats["relative_l2"].numel() else 0.0,
                                      "gate_mean": float(hooks.stats["gate"].mean()) if hooks.stats["gate"].numel() else 1.0,
                                      "cap_fraction": float(hooks.stats["cap_hit"].float().mean()) if hooks.stats["cap_hit"].numel() else 0.0,
                                      "train_only_gate_summary": gate_summary(
                                          hooks.stats, len(loaded.split["train"]), group in ("E3", "E4")),
                                      "w_norm": float(module.W_d.detach().norm()), "evidence": hooks.evidence},
                  "test_graph_loss": float(test_graph.detach()),
                  "test_img_weight": float(loaded.core.img_weight.detach()),
                  "test_ph_weight": float(loaded.core.ph_weight.detach()),
                  "test_raw_image_sha256": array_hash(hooks.raw_img.detach().cpu().numpy()),
                  "test_raw_ph_sha256": array_hash(hooks.raw_ph.detach().cpu().numpy()),
                  "bn_start": bn_start, "bn_end": int(loaded.core.clf[2].num_batches_tracked),
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                  "train_validation_forwards": len(history) * 2}
        atomic_torch(record, output)
        checked = completed_result(output, audit)
        log("complete", group=group, fold=fold, gpu=physical_gpu, epochs=len(history),
            best_epoch=best["epoch"], seconds=seconds, metrics=formal_metrics)
        return checked
    except BaseException as exc:
        log("failed", group=group, fold=fold, gpu=physical_gpu, error=repr(exc),
            traceback=traceback.format_exc())
        raise
    finally:
        if hooks is not None:
            hooks.close()
        loaded.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=CFG["groups"], required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    audit = json.loads((BASE / "audit.json").read_text())
    if not audit.get("formal_authorized") or not audit.get("manifest_sha256"):
        raise RuntimeError("Formal training is not authorized by the frozen audit")
    from audit import code_hashes, hash_json, source_hashes
    if hash_json(code_hashes()) != audit["training_code_sha256"]:
        raise AssertionError("Frozen experiment source changed after authorization")
    if sha_file(BASE / "manifest.json") != audit["manifest_sha256"]:
        raise AssertionError("Frozen manifest SHA mismatch")
    if source_hashes() != audit["original_source_sha256"]:
        raise AssertionError("Original model source changed after authorization")
    task_lock = result_path(args.group, args.fold).with_suffix(".task.lock")
    task_lock.parent.mkdir(parents=True, exist_ok=True)
    with task_lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        train_one(args.group, args.fold, args.gpu, audit)
        fcntl.flock(stream, fcntl.LOCK_UN)

