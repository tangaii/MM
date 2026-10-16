"""Independent anchored PPU runner. Historical scripts are never imported."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / "experiments/DRPA_20261014"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.nn import functional as F
from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
from pcci import PCCI, ResidualBridge, encode_conditions, private_seed

KEYS = ("ACC", "BA", "AUC", "SEN", "SPE", "F1")
SUBSETS = ("train", "val", "test")


def utc():
    return datetime.now(timezone.utc).isoformat()


def config():
    return json.loads((BASE / "config.json").read_text())


def emit(kind, **values):
    print(json.dumps({"kind": kind, "time_utc": utc(), **values}, allow_nan=False), flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(array_hash(state[key]).encode())
    return h.hexdigest()


def code_hash():
    return hashlib.sha256("".join(sha_file(BASE / name) for name in ("run.py", "pcci.py", "config.json")).encode()).hexdigest()


def atomic_json(value, path=None):
    target = Path(path or BASE / "summary.json")
    tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(value, tmp)
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_torch(path, mmap=False):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)


def export_core(core):
    return {"state": {k: v.detach().cpu().clone() for k, v in core.state_dict().items()},
            "rp_weights": core.rp_attention.weights.detach().cpu().clone()}


def import_core(core, saved):
    core.load_state_dict(saved["state"], strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(saved["rp_weights"].to(core.device))


def core_hash(saved):
    return state_hash({**saved["state"], "unregistered_rp_weights": saved["rp_weights"]})


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(0).clone(), "backend": {
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}


def restore_rng(saved):
    random.setstate(saved["python"]); np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["cpu"]); torch.cuda.set_rng_state(saved["cuda"], 0)
    backend = saved["backend"]
    torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = backend["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = backend["cudnn_allow_tf32"]


def rng_hash(saved):
    return {"python": hashlib.sha256(repr(saved["python"]).encode()).hexdigest(),
            "numpy": hashlib.sha256(repr(saved["numpy"][0]).encode() + saved["numpy"][1].tobytes()
                                     + repr(saved["numpy"][2:]).encode()).hexdigest(),
            "cpu": array_hash(saved["cpu"]), "cuda": array_hash(saved["cuda"])}


def draw_rng(seed, fold, draw):
    value = private_seed(seed, fold, f"PCCI-repeat-{config()['repeat_seed_bases'][draw]}") % (2**32)
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    return value


def require_ppu():
    if not torch.cuda.is_available():
        raise RuntimeError("CPU training is forbidden; no accelerator available")
    torch.cuda.set_device(0)
    if "PPU" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("This experiment expects the audited PPU backend")
    return torch.device("cuda:0")


def options(seed, fold):
    argv = sys.argv
    try:
        sys.argv = ["official_options"]
        settings = OptInit(model="MM_GTUNets", dataset="ABIDE", atlas="aal")
    finally:
        sys.argv = argv
    cfg, args = config(), settings.args
    args.seed, args.device = seed, require_ppu()
    args.data_folder = cfg["data_folder"]
    args.ckpt_path = str(OLD / f"artifacts/seed_{seed}/vae")
    args.epoch, args.early_stop, args.lr, args.wd = cfg["epochs"], cfg["early_stop"], cfg["lr"], cfg["weight_decay"]
    return args


def anchor_dir(seed, fold):
    return OLD / f"artifacts/seed_{seed}/fold_{fold}"


def result_path(method, seed, fold):
    return BASE / f"runs/{method}/s{seed}_f{fold}.pt"


def split_hash(split):
    return {name: array_hash(split[name]) for name in SUBSETS}


def check_splits(split):
    joined = np.concatenate([split[name] for name in SUBSETS])
    if not np.array_equal(np.sort(joined), np.arange(871)):
        raise ValueError("Split overlap, omission or duplicate")


def binary_metrics(labels, pred, probability):
    y, predicted = np.asarray(labels) == 0, np.asarray(pred) == 0
    tp, fn = int(np.sum(y & predicted)), int(np.sum(y & ~predicted))
    tn, fp = int(np.sum(~y & ~predicted)), int(np.sum(~y & predicted))
    sen, spe = tp / (tp + fn) if tp + fn else None, tn / (tn + fp) if tn + fp else None
    return {"ACC": float(np.mean(y == predicted)), "BA": (sen + spe) / 2 if sen is not None and spe is not None else None,
            "AUC": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "SEN": sen, "SPE": spe, "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def distribution(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0}
    return {"n": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            **{f"p{p}": float(np.quantile(values, p / 100)) for p in (10, 50, 90)},
            "above_half_fraction": float(np.mean(values > 0.5))}


def gate_statistics(value, indices, ctx):
    gate = value["gate"].detach().cpu().numpy()[indices]
    residual = value["delta"].detach().cpu().numpy()[indices]
    labels, sites = ctx["labels"][indices], np.asarray(ctx["site_names"])[indices]
    return {"gate": distribution(gate), "residual_l2": distribution(np.linalg.norm(residual, axis=1)),
            "classes": {str(c): distribution(gate[labels == c]) for c in (0, 1)},
            "sites": {site: distribution(gate[sites == site]) for site in sorted(set(sites))}}


def raw_metadata(ctx):
    with (Path(config()["data_folder"]) / "Phenotypic_V1_0b_preprocessed1.csv").open() as stream:
        lookup = {r["SUB_ID"]: r for r in csv.DictReader(stream)}
    rows = [lookup[str(subject)] for subject in ctx["ids"]]
    return {"age": np.asarray([float(r["AGE_AT_SCAN"]) for r in rows]),
            "sex": np.asarray([r["SEX"] for r in rows]), "site": np.asarray([r["SITE_ID"] for r in rows]), "rows": rows}


def source_guard():
    state = json.loads((BASE / "summary.json").read_text())
    if state["training_code_sha256"] != code_hash():
        raise RuntimeError("Registered PCCI training implementation changed")
    for name, saved in state["audit"]["source_sha256"].items():
        if sha_file(ROOT / name) != saved:
            raise RuntimeError(f"Protected source changed: {name}")
    return state


def audit():
    cfg = config()
    old = json.loads((OLD / "artifacts/audit.json").read_text())
    expected = cfg["historical_training_sha256"]
    old_hash = hashlib.sha256("".join(sha_file(OLD / name) for name in ("run.py", "drpa.py", "config.json")).encode()).hexdigest()
    if old_hash != expected or old["training_code_sha256"] != expected:
        raise RuntimeError("Historical training source hash mismatch")
    manifest = {}
    def remember(path, expected_sha=None):
        path = Path(path)
        value = sha_file(path)
        if expected_sha is not None and value != expected_sha:
            raise RuntimeError(f"Read-only input hash mismatch: {path}")
        manifest[str(path.relative_to(ROOT))] = {"sha256": value, "bytes": path.stat().st_size}
        return value
    for name, saved in old["source_sha256"].items():
        remember(ROOT / name, saved)
    for relative, saved in old["data_sha256"].items():
        remember(Path(cfg["data_folder"]) / relative, saved)
    for name in ("run.py", "drpa.py", "launch.py", "analyze.py", "config.json", "README.md", "outputs/final_report.txt", "outputs/final_summary.json", "artifacts/audit.json"):
        remember(OLD / name)
    for name in ("README.md", "report.md", "diagnose.py", "replay.py", "summary.json"):
        remember(ROOT / "experiments/DRPA_Mechanism_20261015" / name)
    remember(OLD / "artifacts/context.pt", old["context_sha256"])
    ctx = load_torch(OLD / "artifacts/context.pt")
    metadata = raw_metadata(ctx)
    if len(set(ctx["ids"])) != 871 or list(metadata["site"]) != ctx["site_names"]:
        raise ValueError("Subject/phenotype order mismatch")
    for j, key in enumerate(("SITE_ID", "SEX", "AGE_AT_SCAN")):
        if not np.array_equal(ctx["ph_data"][:, j], ctx["ph_dict"][key]):
            raise ValueError("Original ph_data column mapping mismatch")
    if not np.allclose(metadata["age"], ctx["ph_data"][:, 2], atol=1e-5, rtol=0):
        raise ValueError("CSV age does not match the original age column")
    sex_mapping = {}
    for raw, encoded in zip(metadata["sex"], ctx["ph_data"][:, 1]):
        if sex_mapping.setdefault(raw, int(encoded)) != int(encoded):
            raise ValueError("Original sex ordinal encoding inconsistent")
    expected_labels = np.array([int(r["DX_GROUP"]) - 1 for r in metadata["rows"]])
    if not np.array_equal(expected_labels, ctx["labels"]) or np.bincount(expected_labels).tolist() != [403, 468]:
        raise ValueError("Original ASD=0/HC=1 labels changed")
    anchors, baseline = {}, {}
    for seed in cfg["seeds"]:
        selected = OLD / f"artifacts/seed_{seed}/selected_features.pt"
        mask_path = selected.parent / "rfe_support.npy"
        remember(selected); remember(mask_path)
        features, mask = load_torch(selected), np.load(mask_path, allow_pickle=False)
        raw = load_torch(Path(cfg["data_folder"]) / "save_tensor/up_triangles.pt")
        if not np.array_equal(np.asarray(raw, dtype=np.float32)[:, mask], features.numpy()):
            raise ValueError("Saved RFE features changed")
        outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for fold, (train_val, test) in enumerate(outer.split(np.zeros((871, 1)), ctx["labels"])):
            train, val = train_test_split(train_val, test_size=cfg["val_ratio"], random_state=seed, stratify=ctx["labels"][train_val])
            adir, out = anchor_dir(seed, fold), OLD / f"outputs/E0/seed_{seed}/fold_{fold}"
            for name in ("initial.pt", "train_start.pt", "splits.pt", "fingerprints.json", "end_rng.pt"):
                remember(adir / name)
            vae = adir.parent / "vae" / f"fold{fold}_pretrain.pth"
            remember(vae)
            split = load_torch(adir / "splits.pt"); check_splits(split)
            if not all(np.array_equal(split[name], value) for name, value in zip(SUBSETS, (train, val, test))):
                raise ValueError("Saved split does not match original stratified protocol")
            pairing = json.loads((adir / "fingerprints.json").read_text())
            initial, start = load_torch(adir / "initial.pt"), load_torch(adir / "train_start.pt")
            if core_hash(initial["core"]) != pairing["initial_core_fingerprint"] or core_hash(start["core"]) != pairing["train_start_core_fingerprint"]:
                raise ValueError("Historical initial core fingerprint mismatch")
            if start["initial_fingerprint"] != pairing["initial_core_fingerprint"]:
                raise ValueError("VAE anchor lost initial model identity")
            if split_hash(split) != pairing["split_fingerprints"] or array_hash(features) != pairing["features_fingerprint"] or sha_file(vae) != pairing["vae_sha256"]:
                raise ValueError("Historical feature/split/VAE pairing mismatch")
            if rng_hash(start["rng"]) != pairing["classifier_start_rng"]:
                raise ValueError("Historical training RNG anchor mismatch")
            m = json.loads((out / "metrics.json").read_text()); status = json.loads((out / "status.json").read_text())
            remember(out / "metrics.json", status["metrics_sha256"]); remember(out / "status.json")
            if status["state"] != "complete" or m["training_code_sha256"] != expected or m["pairing"] != pairing:
                raise ValueError("Historical E0 fit is not complete or paired")
            for name, saved in m["output_sha256"].items():
                remember(out / name, saved)
            best = load_torch(out / "best.pt", mmap=True)
            if best["training_code_sha256"] != expected or best["epoch"] != m["best_epoch"]:
                raise ValueError("E0 checkpoint identity mismatch")
            if any(not torch.isfinite(t).all() for t in best["core"]["state"].values()):
                raise ValueError("Nonfinite E0 parameters")
            with (out / "oof.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            idx = np.array([int(r["index"]) for r in rows]); labels = np.array([int(r["label_original"]) for r in rows])
            pred = np.array([int(r["pred_original"]) for r in rows]); probs = np.array([float(r["prob_asd"]) for r in rows])
            if not np.array_equal(idx, test) or not np.array_equal(labels, ctx["labels"][test]):
                raise ValueError("E0 OOF index/label mismatch")
            if [r["subject_id"] for r in rows] != [ctx["ids"][i] for i in test] or [r["site"] for r in rows] != list(metadata["site"][test]):
                raise ValueError("E0 OOF ID/site mismatch")
            recalculated = binary_metrics(labels, pred, probs)
            if any(abs(recalculated[k] - m["metrics"][k]) > 1e-10 for k in KEYS):
                raise ValueError("E0 OOF metrics do not reproduce")
            ident = f"{seed}/{fold}"
            _, _, encoding = encode_conditions(metadata, train, seed, fold, cfg)
            anchors[ident] = {"pairing": pairing, "encoding_digest": digest(encoding),
                              "input_sha256": {str(p.relative_to(ROOT)): manifest[str(p.relative_to(ROOT))]["sha256"]
                                                for p in (selected, mask_path, adir / "initial.pt", adir / "train_start.pt", adir / "splits.pt", vae)}}
            baseline[ident] = {"method": "C0", "seed": seed, "fold": fold, "metrics": recalculated,
                               "indices": idx.tolist(), "labels": labels.tolist(), "pred": pred.tolist(), "prob_asd": probs.tolist(),
                               "ids": [r["subject_id"] for r in rows], "sites": [r["site"] for r in rows],
                               "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"],
                               "classifier_seconds": m["classifier_seconds"], "vae_pretrain_seconds": m["vae_pretrain_seconds"],
                               "parameters": m["parameters"], "best_reference": str((out / "best.pt").relative_to(ROOT)),
                               "best_sha256": sha_file(out / "best.pt")}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True, timeout=30).split()[0]
    if head != cfg["source_commit"] or remote != cfg["source_commit"]:
        raise RuntimeError("Local/remote reference version changed; inspect before training")
    return {"time_utc": utc(), "head": head, "remote_main": remote, "source_sha256": old["source_sha256"],
            "manifest": manifest, "data_digest": ctx["data_digest"], "anchors": anchors,
            "baseline": baseline, "historical_training_sha256": expected,
            "phenotype_columns_original": ["SITE_ID", "SEX", "AGE_AT_SCAN"],
            "sex_original_to_cache": sex_mapping, "age_raw": distribution(metadata["age"]),
            "label_encoding": {"ASD": 0, "HC": 1}, "baseline_verified": 30,
            "prior_diagnosis": {"H1": "partial", "H2": "supported", "H3": "partial", "H4": "partial; phenotype encoder gradient about 7.27%"},
            "paper_audit": {"MM_GTUNets": "https://arxiv.org/html/2406.14455v3",
                "sections": ["Fig1", "3.2", "3.3", "3.4", "3.5", "4.4.5/Table5"],
                "facts": ["Phenotypes already affect population graph and both GTUNets", "Joint 16-D fusion already contains imaging and phenotype features", "PCCI measures incremental conditional value on this existing fusion", "C1 retains native phenotype branch/graph", "Paper/source differences and inherited RFE/dropout behavior are preserved"],
                "related": {"PhenoBrain": "https://proceedings.mlr.press/v306/meng26f.html", "SMGD": "https://proceedings.mlr.press/v306/shi26ag.html", "MGDR": "https://papers.miccai.org/miccai-2024/514-Paper2494.html", "ToAlign": "https://ojs.aaai.org/index.php/AAAI/article/view/39248"},
                "novelty_boundary": "FiLM, gating and conditional residuals are established; require paired non-additive diagnostic evidence beyond added capacity"}}


class Loaded:
    def __init__(self, seed, fold, registered, metadata, ctx, cache):
        self.seed, self.fold, self.ctx, self.metadata = seed, fold, ctx, metadata
        self.cfg, self.identity = config(), f"{seed}/{fold}"
        expected = registered["audit"]["anchors"][self.identity]
        for relative, saved in expected["input_sha256"].items():
            if sha_file(ROOT / relative) != saved:
                raise RuntimeError(f"Read-only anchor changed: {relative}")
        adir = anchor_dir(seed, fold)
        self.start = load_torch(adir / "train_start.pt")
        self.split = load_torch(adir / "splits.pt"); check_splits(self.split)
        self.pairing = expected["pairing"]
        self.opt = options(seed, fold)
        self.core = MM_GTUNets(self.opt, fold).to(self.opt.device)
        self.core.vae_save_path = str(adir.parent / "vae" / f"fold{fold}_pretrain.pth")
        import_core(self.core, self.start["core"])
        if core_hash(export_core(self.core)) != self.pairing["train_start_core_fingerprint"]:
            raise RuntimeError("Core construction/import changed initial weights")
        if seed not in cache:
            cache[seed] = load_torch(adir.parent / "selected_features.pt")
        x = cache[seed]
        affinity = create_reward_penalty_graph(ctx["ph_dict"], ctx["labels"], *[self.split[k] for k in SUBSETS], self.opt)
        self.x = x.to(self.opt.device)
        self.ph = torch.as_tensor(ctx["ph_data"], dtype=torch.float32, device=self.opt.device)
        self.affinity = torch.as_tensor(affinity, dtype=torch.float32, device=self.opt.device)
        condition, reference, self.encoding = encode_conditions(metadata, self.split["train"], seed, fold, self.cfg)
        if digest(self.encoding) != expected["encoding_digest"]:
            raise RuntimeError("Pre-registered condition encoding changed")
        self.conditions, self.reference = condition.to(self.opt.device), reference.to(self.opt.device)
        self.permutation = torch.as_tensor(self.encoding["permutation"], dtype=torch.long, device=self.opt.device)
        self.labels = torch.as_tensor(ctx["labels"], dtype=torch.long, device=self.opt.device)
        self.train = torch.as_tensor(self.split["train"], dtype=torch.long, device=self.opt.device)
        self.module = self.bridge = None

    def attach(self, method):
        self.module = PCCI.isolated(self.cfg, private_seed(self.seed, self.fold, "PCCI-initialization"), self.opt.device)
        self.bridge = ResidualBridge(self.core, self.module, self.conditions, self.reference, self.permutation, method)

    def forward(self):
        before = self.bridge.calls if self.bridge is not None else 0
        result = self.core(self.x, self.ph, self.affinity)
        if self.bridge is not None and self.bridge.calls != before + 1:
            raise RuntimeError("Original classifier must execute exactly once")
        return result

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
        self.bridge = self.module = self.core = None
        self.x = self.ph = self.affinity = self.labels = self.conditions = self.reference = None
        torch.cuda.empty_cache()


@contextmanager
def task_lock(method, seed, fold):
    offset = int(method[1:]) * 30 + config()["seeds"].index(seed) * 10 + fold
    with (BASE / "run.py").open("r+") as stream:
        fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset)
        try:
            yield
        finally:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_UN, 1, offset)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}")


def oof_record(loaded, logits):
    test, ctx = loaded.split["test"], loaded.ctx
    scores = logits.detach().cpu().numpy()
    probability, prediction = softmax(scores, axis=1)[:, 0], scores.argmax(1)
    return {"indices": np.asarray(test), "labels": np.asarray(ctx["labels"])[test],
            "pred": prediction[test], "prob_asd": probability[test],
            "ids": [ctx["ids"][i] for i in test], "sites": [ctx["site_names"][i] for i in test],
            "metrics": binary_metrics(ctx["labels"][test], prediction[test], probability[test])}


def verify_result(method, seed, fold, registered, require_aux=False):
    path = result_path(method, seed, fold)
    if not path.exists():
        return False
    saved = load_torch(path, mmap=True)
    anchor = registered["audit"]["anchors"][f"{seed}/{fold}"]
    expected = {"method": method, "seed": seed, "fold": fold,
                "training_code_sha256": registered["training_code_sha256"],
                "config_digest": digest(config()), "input_sha256": anchor["input_sha256"],
                "encoding_digest": anchor["encoding_digest"]}
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError(f"Artifact mismatch {path}: {key}")
    if saved.get("state") not in ("trained", "complete"):
        raise RuntimeError(f"Incomplete artifact: {path}")
    for name, tensor in {**saved["core"]["state"], **{f"pcci.{k}": v for k, v in saved["pcci"].items()},
                         "rp": saved["core"]["rp_weights"]}.items():
        finite(tensor, name)
    if core_hash(saved["core"]) != saved["best_core_sha256"] or state_hash(saved["pcci"]) != saved["best_pcci_sha256"]:
        raise RuntimeError("Checkpoint state fingerprint mismatch")
    oof = saved["oof"]
    split = load_torch(anchor_dir(seed, fold) / "splits.pt")
    ctx = load_torch(OLD / "artifacts/context.pt")
    idx = split["test"]
    if not np.array_equal(oof["indices"], idx) or not np.array_equal(oof["labels"], ctx["labels"][idx]):
        raise RuntimeError("OOF split/label mismatch")
    if oof["ids"] != [ctx["ids"][i] for i in idx] or oof["sites"] != [ctx["site_names"][i] for i in idx]:
        raise RuntimeError("OOF ID/site mismatch")
    probability = np.asarray(oof["prob_asd"])
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise RuntimeError("Invalid OOF probability")
    recalculated = binary_metrics(oof["labels"], oof["pred"], probability)
    if any(abs(recalculated[k] - oof["metrics"][k]) > 1e-12 for k in KEYS):
        raise RuntimeError("OOF metrics failed independent recomputation")
    if require_aux:
        aux = saved.get("frozen")
        if saved["state"] != "complete" or aux is None or np.asarray(aux["test_prob_asd"]).shape != (5, len(idx)):
            return False
        for d in range(5):
            metrics = binary_metrics(oof["labels"], aux["test_pred"][d], aux["test_prob_asd"][d])
            if any(abs(metrics[k] - aux["draw_metrics"][d][k]) > 1e-12 for k in KEYS):
                raise RuntimeError("Frozen-draw OOF metrics mismatch")
        if method in ("C2", "C3", "C4", "C5"):
            for name in config()["counterfactuals"]:
                p = np.asarray(aux["counterfactual_prob_asd"][name])
                if p.shape != (5, 871) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise RuntimeError("Invalid counterfactual probabilities")
    return True


def gradient_sample(core, module, received, nonzero):
    answer = {}
    for prefix, model in (("core", core), ("pcci", module)):
        norm = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            finite(parameter.grad, f"{prefix}.{name} gradient")
            received[prefix].add(name)
            norm.append(parameter.grad.detach().square().sum())
            if prefix == "pcci":
                mask = parameter.grad.detach().ne(0).cpu()
                nonzero[name] = mask if name not in nonzero else nonzero[name] | mask
        answer[prefix] = float(torch.stack(norm).sum().sqrt()) if norm else 0.0
    return answer


def train_one(method, seed, fold, registered, metadata, ctx, cache):
    if verify_result(method, seed, fold, registered):
        return {"skipped": True, "path": str(result_path(method, seed, fold).relative_to(BASE))}
    cfg = config()
    loaded = Loaded(seed, fold, registered, metadata, ctx, cache)
    try:
        loaded.attach(method)
        core, module = loaded.core, loaded.module
        initial_module = state_hash(module.state_dict())
        optimizer = torch.optim.Adam(list(core.parameters()) + list(module.parameters()),
                                     lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        stopper = EarlyStopping(patience=cfg["early_stop"], verbose=False)
        received, nonzero = {"core": set(), "pcci": set()}, {}
        best, history, best_loss = None, [], float("inf")
        restore_rng(loaded.start["rng"])
        if rng_hash(capture_rng()) != loaded.pairing["classifier_start_rng"]:
            raise RuntimeError("Training-start RNG mismatch")
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        for epoch in range(cfg["epochs"]):
            core.train(); module.train(); optimizer.zero_grad()
            outputs, graph = loaded.forward()
            ce = F.cross_entropy(outputs[loaded.train], loaded.labels[loaded.train])
            select = (cfg["select_weight"] * loaded.bridge.last["gate"][loaded.train].mean()
                      if cfg["methods"][method]["select"] else ce.new_zeros(()))
            total = ce + graph + select
            finite(total, "training objective")
            total.backward()
            diagnostic = epoch % cfg["diagnostic_frequency"] == 0
            if diagnostic:
                gradients = gradient_sample(core, module, received, nonzero)
                train_gate = gate_statistics(loaded.bridge.last, loaded.split["train"], ctx)
            optimizer.step()
            core.eval(); module.eval()
            with torch.no_grad():
                val_output, val_graph = loaded.forward()
                val_ce = F.cross_entropy(val_output[loaded.split["val"]], loaded.labels[loaded.split["val"]])
                val_loss = val_ce + val_graph
            finite(val_loss, "validation objective")
            record = {"epoch": epoch, "train_ce": float(ce.detach()), "train_graph": float(graph.detach()),
                      "train_select": float(select.detach()), "train_total": float(total.detach()),
                      "val_ce": float(val_ce), "val_graph": float(val_graph), "val_loss": float(val_loss)}
            if diagnostic:
                record.update(gradient_norms=gradients, train_gate=train_gate)
                history.append(record)
                emit("progress", method=method, seed=seed, fold=fold, epoch=epoch,
                     val_loss=float(val_loss), train_loss=float(total.detach()))
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best = {"core": export_core(core), "pcci": cpu_state(module), "epoch": epoch,
                        "validation_loss": best_loss, "training_record": record}
            stopper(val_loss, core)
            if stopper.early_stop:
                break
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        import_core(core, best["core"]); module.load_state_dict(best["pcci"])
        core.eval(); module.eval()
        test_rng = capture_rng()
        with torch.set_grad_enabled(True):
            logits, graph = loaded.forward()
        finite(logits, "formal test logits")
        oof = oof_record(loaded, logits)
        test_gate = {name: gate_statistics(loaded.bridge.last, loaded.split[name], ctx) for name in SUBSETS}
        parameters = {"core_total": sum(p.numel() for p in core.parameters()),
                      "core_requires_grad": sum(p.numel() for p in core.parameters() if p.requires_grad),
                      "core_receiving_gradients": sum(p.numel() for n, p in core.named_parameters() if n in received["core"]),
                      "new_total": sum(p.numel() for p in module.parameters()),
                      "new_receiving_gradients": sum(p.numel() for n, p in module.named_parameters() if n in received["pcci"]),
                      "new_nonzero_scalar_gradients_sampled_union": sum(int(m.sum()) for m in nonzero.values()),
                      "new_nonzero_scalar_gradients_by_parameter": {n: int(m.sum()) for n, m in nonzero.items()},
                      "gradient_sampling": f"epochs 0,{cfg['diagnostic_frequency']},...; union is a lower bound"}
        result = {"state": "trained", "method": method, "seed": seed, "fold": fold, "completed_utc": utc(),
                  "training_code_sha256": code_hash(), "config_digest": digest(cfg),
                  "input_sha256": registered["audit"]["anchors"][f"{seed}/{fold}"]["input_sha256"],
                  "pairing": loaded.pairing, "encoding": loaded.encoding, "encoding_digest": digest(loaded.encoding),
                  "config": cfg, "core": best["core"], "pcci": best["pcci"],
                  "best_core_sha256": core_hash(best["core"]), "best_pcci_sha256": state_hash(best["pcci"]),
                  "initial_pcci_sha256": initial_module, "best_epoch": best["epoch"], "best_validation_loss": best_loss,
                  "best_training_record": best["training_record"], "epochs_run": epoch + 1,
                  "classifier_seconds": seconds, "vae_pretrain_seconds": 0.0, "vae_pretraining_reused": True,
                  "parameters": parameters, "sampled_epochs": history, "oof": oof, "gate": test_gate,
                  "formal_rng_before": rng_hash(test_rng), "formal_rng_after": rng_hash(capture_rng()),
                  "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]), "device": torch.cuda.get_device_name(0),
                  "peak_memory_bytes": torch.cuda.max_memory_allocated()}
        atomic_torch(result, result_path(method, seed, fold))
        verify_result(method, seed, fold, registered)
        return {"metrics": oof["metrics"], "seconds": seconds, "best_epoch": best["epoch"],
                "epochs_run": epoch + 1, "path": str(result_path(method, seed, fold).relative_to(BASE))}
    finally:
        loaded.close()


def compare_tensors(left, right, name, atol=2e-6, rtol=2e-5):
    if left is None or right is None:
        if left is not None or right is not None:
            raise AssertionError(f"Gradient connectivity mismatch: {name}")
        return 0.0
    finite(left, name); finite(right, name)
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        raise AssertionError(f"Numerical mismatch {name}: {float((left-right).abs().max())}")
    return float((left - right).abs().max())


def check_one(seed, fold, registered, metadata, ctx, cache):
    loaded = Loaded(seed, fold, registered, metadata, ctx, cache)
    try:
        core = loaded.core
        # Deterministic arithmetic is diagnostic only; native training remains unchanged.
        def single(mode):
            import_core(core, loaded.start["core"])
            restore_rng(loaded.start["rng"])
            torch.use_deterministic_algorithms(True)
            core.train(); core.zero_grad(set_to_none=True)
            if loaded.module is not None:
                loaded.module.zero_grad(set_to_none=True)
                loaded.bridge.enabled = mode != "disabled"
            before = int(core.clf[2].num_batches_tracked)
            logits, graph = loaded.forward()
            (F.cross_entropy(logits[loaded.train], loaded.labels[loaded.train]) + graph).backward()
            if int(core.clf[2].num_batches_tracked) != before + 1:
                raise AssertionError("Classifier BatchNorm updated more than once")
            gradients = {n: p.grad.detach().cpu().clone() if p.grad is not None else None for n, p in core.named_parameters()}
            return logits.detach().cpu(), graph.detach().cpu(), gradients
        reference = single("reference")
        loaded.attach("C3")
        disabled, zero = single("disabled"), single("zero")
        max_gradient = max(compare_tensors(reference[2][n], candidate[2][n], n)
                           for candidate in (disabled, zero) for n in reference[2])
        max_output = max(compare_tensors(reference[i], candidate[i], f"output{i}")
                         for candidate in (disabled, zero) for i in (0, 1))
        if torch.count_nonzero(loaded.bridge.last["delta"]).item():
            raise AssertionError("Zero-initialized residual is not zero")
        cfg = config()
        image = loaded.bridge.imaging.detach()
        for method in cfg["methods"]:
            value = loaded.module.components(image, loaded.conditions, loaded.reference, method,
                                              permutation=loaded.permutation)
            if method == "C1" and not bool((value["condition"] == value["condition"][0]).all()):
                raise AssertionError("C1 contains personal condition variation")
            if method == "C2" and not torch.allclose(value["gamma"], value["gamma"][:1].expand_as(value["gamma"])):
                raise AssertionError("C2 gamma varies by patient")
            if method != "C4" and torch.count_nonzero(value["condition"][:, 3:]).item():
                raise AssertionError("Site leaked into a non-C4 bypass")
            if method == "C4" and not torch.all(value["condition"][:, 3:].sum(1) == 1):
                raise AssertionError("C4 site one-hot invalid")
            if method == "C6" and not torch.equal(value["condition"][:, :3], loaded.conditions[loaded.permutation, :3]):
                raise AssertionError("C6 shuffle is inconsistent")
        gate = loaded.bridge.last["gate"].detach().clone().requires_grad_(True)
        penalty = cfg["select_weight"] * gate[loaded.train].mean()
        derivative, = torch.autograd.grad(penalty, gate)
        heldout = np.concatenate((loaded.split["val"], loaded.split["test"]))
        if torch.count_nonzero(derivative[heldout]).item():
            raise AssertionError("C5 regularizes held-out subjects")
        synthetic = PCCI.isolated(cfg, private_seed(seed, fold, "PCCI-gradient-check"), loaded.opt.device)
        optimizer = torch.optim.Adam(synthetic.parameters(), lr=cfg["lr"])
        gamma_norms = []
        for step in range(2):
            optimizer.zero_grad()
            value = synthetic.components(image, loaded.conditions, loaded.reference, "C3", permutation=loaded.permutation)
            loss = (value["delta"] * image).square().mean() + value["delta"].sum() / len(image)
            loss.backward()
            gamma_norms.append(float(torch.stack([p.grad.square().sum() for p in synthetic.gamma.parameters()]).sum().sqrt()))
            for name, p in synthetic.named_parameters():
                if p.grad is not None:
                    finite(p.grad, name)
            optimizer.step()
        if gamma_norms[1] <= 0:
            raise AssertionError("Multiplicative gamma has no real gradient after output projection starts learning")
        buffer = io.BytesIO()
        torch.save({"core": export_core(core), "pcci": cpu_state(loaded.module)}, buffer)
        buffer.seek(0)
        roundtrip = torch.load(buffer, map_location="cpu", weights_only=False)
        if core_hash(roundtrip["core"]) != core_hash(export_core(core)) or state_hash(roundtrip["pcci"]) != state_hash(loaded.module.state_dict()):
            raise AssertionError("Checkpoint roundtrip changed parameters")
        saved_core, saved_module, rng = export_core(core), cpu_state(loaded.module), capture_rng()
        loaded.bridge.enabled = True; core.eval(); loaded.module.eval()
        with torch.no_grad():
            restore_rng(rng); first = loaded.forward()[0].detach().clone()
            import_core(core, roundtrip["core"]); loaded.module.load_state_dict(roundtrip["pcci"])
            restore_rng(rng); second = loaded.forward()[0].detach().clone()
        roundtrip_error = compare_tensors(first, second, "checkpoint restored forward")
        import_core(core, saved_core); loaded.module.load_state_dict(saved_module)
        return {"seed": seed, "fold": fold, "passed": True, "max_output_error": max_output,
                "max_core_gradient_error": max_gradient, "checkpoint_output_error": roundtrip_error,
                "gamma_grad_norm_step0_step1": gamma_norms, "classifier_calls_per_core_forward": 1,
                "batchnorm_updates_per_train_forward": 1, "vae_frozen_after_original_forward": all(not p.requires_grad for p in core.vae.parameters()),
                "new_parameters": sum(p.numel() for p in loaded.module.parameters()),
                "condition_checks": ["train-only age statistics", "C1 constant", "C2 constant gamma", "C4-only site", "C5 train-only", "C6 bypass-only joint within-site shuffle"]}
    finally:
        loaded.close()


def frozen_one(method, seed, fold, registered, metadata, ctx, cache):
    if method != "C0" and verify_result(method, seed, fold, registered, require_aux=True):
        return {"skipped": True}
    loaded = Loaded(seed, fold, registered, metadata, ctx, cache)
    try:
        saved = None
        if method == "C0":
            reference = registered["audit"]["baseline"][f"{seed}/{fold}"]
            path = ROOT / reference["best_reference"]
            if sha_file(path) != reference["best_sha256"]:
                raise RuntimeError("C0 checkpoint changed")
            checkpoint = load_torch(path, mmap=True)
            core_state = checkpoint["core"]
        else:
            verify_result(method, seed, fold, registered)
            saved = load_torch(result_path(method, seed, fold))
            core_state = saved["core"]
            loaded.attach(method); loaded.module.load_state_dict(saved["pcci"]); loaded.module.eval()
        core, test = loaded.core, loaded.split["test"]
        core.eval()
        probabilities, predictions, metrics, seeds, gate_values, residual_values = [], [], [], [], [], []
        counterfactual = {name: [] for name in config()["counterfactuals"]}
        counterfactual_residual = {name: [] for name in config()["counterfactuals"]}
        for draw in range(5):
            import_core(core, core_state)
            if loaded.module is not None:
                loaded.module.load_state_dict(saved["pcci"])
                loaded.bridge.enabled = True
            restore_rng(loaded.start["rng"])
            seeds.append(draw_rng(seed, fold, draw))
            with torch.no_grad():
                logits, _ = loaded.forward()
                scores = logits.cpu().numpy()
                p, pred = softmax(scores, axis=1)[:, 0], scores.argmax(1)
                probabilities.append(p[test]); predictions.append(pred[test])
                metrics.append(binary_metrics(ctx["labels"][test], pred[test], p[test]))
                if loaded.module is not None:
                    imaging, base = loaded.bridge.imaging, loaded.bridge.base
                    gate_values.append(loaded.bridge.last["gate"].cpu().numpy())
                    residual_values.append(loaded.bridge.last["delta"].norm(dim=1).cpu().numpy())
                    if method in ("C2", "C3", "C4", "C5"):
                        counterfactual["true"].append(p)
                        counterfactual_residual["true"].append(residual_values[-1])
                        loaded.bridge.enabled = False
                        for intervention in ("shuffled", "reference", "off"):
                            value = loaded.module.components(imaging, loaded.conditions, loaded.reference, method,
                                                             intervention=intervention, permutation=loaded.permutation)
                            alternative = core.clf(base + value["delta"]).cpu().numpy()
                            counterfactual[intervention].append(softmax(alternative, axis=1)[:, 0])
                            counterfactual_residual[intervention].append(value["delta"].norm(dim=1).cpu().numpy())
                        loaded.bridge.enabled = True
        probability = np.stack(probabilities)
        prediction = np.stack(predictions)
        mean_probability = probability.mean(0)
        result = {"policy": config()["repeat_policy"], "draw_seeds": seeds,
                  "test_prob_asd": probability, "test_pred": prediction, "draw_metrics": metrics,
                  "mean_probability_metrics": binary_metrics(ctx["labels"][test], np.where(mean_probability >= .5, 0, 1), mean_probability),
                  "metric_distribution": {k: distribution([m[k] for m in metrics]) for k in KEYS},
                  "test_prediction_instability_fraction": float(np.mean(np.any(prediction != prediction[0], axis=0))),
                  "gate": np.stack(gate_values) if gate_values else None,
                  "residual_l2": np.stack(residual_values) if residual_values else None,
                  "counterfactual_prob_asd": {k: np.stack(v) for k, v in counterfactual.items() if v},
                  "counterfactual_residual_l2": {k: np.stack(v) for k, v in counterfactual_residual.items() if v},
                  "same_encoder_realization_for_all_counterfactuals": True,
                  "residual_off_is_current_trained_core_not_C0": True, "completed_utc": utc()}
        if method == "C0":
            return json.loads(json.dumps(result, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v))
        saved["frozen"], saved["state"] = result, "complete"
        atomic_torch(saved, result_path(method, seed, fold))
        verify_result(method, seed, fold, registered, require_aux=True)
        return {"mean_probability_metrics": result["mean_probability_metrics"],
                "instability": result["test_prediction_instability_fraction"]}
    finally:
        loaded.close()


def worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("checks", "train", "evaluate"), required=True)
    parser.add_argument("--method", default="C0", choices=tuple(config()["methods"]))
    args = parser.parse_args()
    torch.set_num_threads(config()["host_threads"])
    require_ppu()
    if args.phase == "checks":
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
    registered = source_guard()
    ctx = load_torch(OLD / "artifacts/context.pt")
    metadata, cache = raw_metadata(ctx), {}
    jobs = json.load(sys.stdin)
    errors = 0
    for job in jobs:
        seed, fold = job["seed"], job["fold"]
        identity = {"phase": args.phase, "method": args.method, "seed": seed, "fold": fold,
                    "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"])}
        emit("start", **identity)
        try:
            with task_lock(args.method, seed, fold):
                if args.phase == "checks":
                    value = check_one(seed, fold, registered, metadata, ctx, cache)
                elif args.phase == "train":
                    value = train_one(args.method, seed, fold, registered, metadata, ctx, cache)
                else:
                    value = frozen_one(args.method, seed, fold, registered, metadata, ctx, cache)
            emit("done", **identity, result=value)
        except Exception as exc:
            errors += 1
            emit("error", **identity, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        torch.cuda.empty_cache()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(worker())
