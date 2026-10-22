"""Audited PC-RPA fold training, preflight, and result persistence."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

BASE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/MM-GTUNets")
HISTORY = ROOT / "experiments/DRPA_20261014"
PCDCA = ROOT / "experiments/PCDCA_20261019"
P1_HISTORY = ROOT / "experiments/PCDPACM_20261021"
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from torch.nn import functional as F

from residual import (CFG, HISTORICAL_CFG, ResidualPrototypeObjective,
                      gradient_budget, residual_diagnostics, _kernel)
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from opt import OptInit
from prototype import PrototypeObjective, prepare
from utils.tools import EarlyStopping


SOURCE = (
    "opt.py", "train_mm_gtunets.py", "model/mm_gtunets.py", "model/gtunet.py",
    "model/rp_graph.py", "utils/mydataloader.py", "utils/tools.py", "utils/metrics.py",
)
TRAIN_CODE = ("config.json", "prototype.py", "residual.py", "run.py", "launch.py", "analyze.py")
SUBSETS = ("train", "val", "test")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path | str) -> str:
    return sha_bytes(Path(path).read_bytes())


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def json_digest(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False).encode())


def array_hash(value) -> str:
    arr = np.ascontiguousarray(value)
    return sha_bytes(str(arr.dtype).encode() + str(arr.shape).encode() + arr.tobytes())


def payload_hash(value) -> str:
    h = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            h.update(b"tensor")
            add(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            h.update(b"array")
            h.update(array_hash(item).encode())
        elif isinstance(item, np.generic):
            add(item.item())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(str(type(item)).encode())
            h.update(str(len(item)).encode())
            for child in item:
                add(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            h.update(str(type(item)).encode())
            h.update(repr(item).encode())
        else:
            raise TypeError(f"Unsupported payload type: {type(item)}")

    add(value)
    return h.hexdigest()


def state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else value
        h.update(key.encode())
        h.update(array_hash(arr).encode())
    return h.hexdigest()


def export_core(core) -> dict:
    return {"state": {key: value.detach().cpu().clone() for key, value in core.state_dict().items()},
            "rp_weights": core.rp_attention.weights.detach().cpu().clone()}


def core_hash(saved: dict) -> str:
    return state_hash({**saved["state"], "unregistered_rp_weights": saved["rp_weights"]})


def core_change(start: dict, best: dict) -> dict:
    components = {}
    for key, value in start["state"].items():
        other = best["state"][key]
        name = key.split(".")[0]
        slot = components.setdefault(name, [0.0, 0.0])
        slot[0] += float((value.float() - other.float()).square().sum())
        slot[1] += float(value.float().square().sum())
    for key in ("rp_weights",):
        value, other = start[key].float(), best[key].float()
        components[key] = [float((value - other).square().sum()), float(value.square().sum())]
    relative = {name: float(np.sqrt(delta / max(norm, 1e-30)))
                for name, (delta, norm) in components.items()}
    total_delta = sum(values[0] for values in components.values())
    total_norm = sum(values[1] for values in components.values())
    return {"relative_l2": float(np.sqrt(total_delta / max(total_norm, 1e-30))),
            "per_component_relative_l2": relative}


def import_core(core, saved: dict):
    core.load_state_dict(saved["state"], strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(saved["rp_weights"].to(core.device))


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone(),
            "backend": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}


def restore_rng(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], 0)
    backend = state["backend"]
    torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = backend["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = backend["cudnn_allow_tf32"]


def rng_hash(state: dict) -> dict:
    np_state = state["numpy"]
    return {"python": sha_bytes(repr(state["python"]).encode()),
            "numpy": sha_bytes(repr(np_state[0]).encode() + np_state[1].tobytes()
                               + repr(np_state[2:]).encode()),
            "cpu": array_hash(state["cpu"].numpy()), "cuda": array_hash(state["cuda"].numpy())}


def load_torch(path: Path | str):
    return torch.load(path, map_location="cpu", weights_only=False)


def atomic_bytes(data: bytes, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(value, path: Path | str):
    atomic_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n", path)


def atomic_torch(value, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_log(event: str, **fields):
    line = json.dumps({"utc": utc(), "event": event, **fields}, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")) + "\n"
    with (BASE / "run.lock").open("a+") as lock, (BASE / "run.log").open("a", encoding="utf-8") as stream:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(lock, fcntl.LOCK_UN)


def anchor(seed: int, fold: int) -> Path:
    return HISTORY / "artifacts" / f"seed_{seed}" / f"fold_{fold}"


def result_path(group: str, seed: int, fold: int) -> Path:
    return BASE / "runs" / group / f"s{seed}_f{fold}.pt"


def code_hash() -> str:
    return json_digest({name: sha_file(BASE / name) for name in TRAIN_CODE})


def source_hashes() -> dict:
    return {name: sha_file(ROOT / name) for name in SOURCE}


def runtime_versions() -> dict:
    versions = {"python": sys.version, "torch": str(torch.__version__)}
    for package, key in (("torch-geometric", "torch_geometric"), ("numpy", "numpy"),
                         ("scipy", "scipy"), ("pandas", "pandas"),
                         ("scikit-learn", "scikit_learn"), ("nilearn", "nilearn"),
                         ("nibabel", "nibabel"), ("torch-scatter", "torch_scatter"),
                         ("torch-sparse", "torch_sparse")):
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def read_metadata(ctx: dict) -> dict:
    csv_path = ROOT / CFG["data_folder"] / "Phenotypic_V1_0b_preprocessed1.csv"
    with csv_path.open(newline="") as stream:
        rows = {row["SUB_ID"]: row for row in csv.DictReader(stream)}
    ordered = [rows[subject_id] for subject_id in ctx["ids"]]
    labels = np.asarray([int(row["DX_GROUP"]) - 1 for row in ordered], dtype=np.int64)

    def number(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    age = np.asarray([number(row["AGE_AT_SCAN"]) for row in ordered])
    sex = np.asarray([number(row["SEX"]) for row in ordered])
    site = np.asarray([row["SITE_ID"] for row in ordered])
    if not np.array_equal(labels, ctx["labels"]):
        raise AssertionError("CSV and historical context label order differ")
    if not np.array_equal(np.bincount(labels), [403, 468]):
        raise AssertionError("Unexpected ASD/TD cohort class counts")
    if not np.allclose(age, ctx["ph_data"][:, 2], rtol=0, atol=2e-6):
        raise AssertionError("CSV and graph age values differ")
    if not np.array_equal(sex - 1, ctx["ph_data"][:, 1]):
        raise AssertionError("CSV and graph sex values differ")
    if not np.array_equal(site, ctx["site_names"]) or len(set(site)) != 20:
        raise AssertionError("CSV and graph site order differ")
    if [(int(np.sum((labels == cls) & (sex == 2))), int(np.sum((labels == cls) & (sex == 1))))
            for cls in (0, 1)] != [(54, 349), (90, 378)]:
        raise AssertionError("Sex coding differs from the validated cohort counts")
    return {"age": age, "sex": sex, "site": site}


def audit() -> dict:
    path = BASE / "audit.json"
    if path.exists():
        saved = read_json(path)
        if saved["training_code_sha256"] != code_hash():
            if saved.get("formal_authorized") or saved.get("manifest_sha256"):
                raise AssertionError("Training files changed after formal freeze")
            saved["training_code_files_sha256"] = {name: sha_file(BASE / name) for name in TRAIN_CODE}
            saved["training_code_sha256"] = json_digest(saved["training_code_files_sha256"])
            saved["runtime_environment"] = runtime_versions()
            if "historical_read_only_artifacts" not in saved:
                refs = {}
                for label, directory, group in (("D4", PCDCA, "D4"),
                                                ("M3", ROOT / "experiments/PCDPAX_20261021", "M3"),
                                                ("P1", P1_HISTORY, "P1")):
                    refs[label] = {}
                    for fold in range(10):
                        artifact = directory / "runs" / group / f"s911_f{fold}.pt"
                        if not artifact.is_file():
                            raise FileNotFoundError(f"Required read-only historical result missing: {artifact}")
                        refs[label][str(fold)] = {"path": str(artifact), "sha256": sha_file(artifact)}
                saved["historical_read_only_artifacts"] = refs
            if "historical_read_only_documents_sha256" not in saved:
                docs = {}
                for directory in (PCDCA, ROOT / "experiments/PCDPAX_20261021", P1_HISTORY):
                    for name in ("audit.json", "summary.json", "report.md"):
                        item = directory / name
                        if item.is_file():
                            docs[str(item)] = sha_file(item)
                saved["historical_read_only_documents_sha256"] = docs
            for attempt in saved.get("preflight_attempts", []):
                error_text = attempt.get("error", "")
                if "Missing key(s)" in error_text and ".select.weight" in error_text:
                    attempt["core_forwards"] = 0
                    attempt["core_forward_count_note"] = "Loaded construction failed before the single preflight forward."
            saved["mathematical_tests"] = None
            saved["preflight"] = None
            atomic_json(saved, path)
        return saved

    historical_audit = read_json(PCDCA / "audit.json")
    current_sources = source_hashes()
    for name, expected in historical_audit["source_sha256"].items():
        if current_sources.get(name) != expected:
            raise AssertionError(f"Original model source changed since PCDCA audit: {name}")
    historical_proto_hash = sha_file(PCDCA / "prototype.py")
    historical_code_hash = json_digest({name: sha_file(PCDCA / name) for name in
                                        ("config.json", "prototype.py", "run.py", "launch.py", "analyze.py")})
    if historical_audit["code_sha256"] != historical_code_hash:
        raise AssertionError("Historical PCDCA files changed since their accepted audit")

    ctx_path = HISTORY / "artifacts/context.pt"
    ctx = load_torch(ctx_path)
    metadata = read_metadata(ctx)
    if len(ctx["ids"]) != 871 or tuple(ctx["ph_data"].shape) != (871, 3):
        raise AssertionError("Unexpected cohort size or non-imaging input dimension")
    features_path = anchor(911, 0).parent / "selected_features.pt"
    support_path = anchor(911, 0).parent / "rfe_support.npy"
    features, support = load_torch(features_path), np.load(support_path)
    raw_path = ROOT / CFG["data_folder"] / "save_tensor/up_triangles.pt"
    raw = load_torch(raw_path)
    if tuple(features.shape) != (871, 500) or support.shape != (500,):
        raise AssertionError("Historical RFE output has an unexpected shape")
    if not torch.equal(torch.as_tensor(raw[:, support], dtype=features.dtype), features):
        raise AssertionError("Cached RFE features do not match selected indices")

    counts = np.zeros(871, dtype=np.int64)
    anchor_records = {}
    for fold in range(10):
        directory = anchor(911, fold)
        split_path, start_path = directory / "splits.pt", directory / "train_start.pt"
        fingerprint_path = directory / "fingerprints.json"
        vae_path = directory.parent / "vae" / f"fold{fold}_pretrain.pth"
        split, start, fp = load_torch(split_path), load_torch(start_path), read_json(fingerprint_path)
        union = np.concatenate([split[key] for key in SUBSETS])
        if len(union) != 871 or not np.array_equal(np.sort(union), np.arange(871)):
            raise AssertionError(f"Invalid historical split at fold {fold}")
        counts[split["test"]] += 1
        for subset in SUBSETS:
            if array_hash(split[subset]) != fp["split_fingerprints"][subset]:
                raise AssertionError(f"Split fingerprint mismatch at fold {fold}/{subset}")
        if core_hash(start["core"]) != fp["train_start_core_fingerprint"]:
            raise AssertionError(f"train_start core mismatch at fold {fold}")
        if rng_hash(start["rng"]) != fp["classifier_start_rng"]:
            raise AssertionError(f"train_start RNG mismatch at fold {fold}")
        if sha_file(vae_path) != start["vae_sha256"] or sha_file(vae_path) != fp["vae_sha256"]:
            raise AssertionError(f"VAE checkpoint mismatch at fold {fold}")
        if array_hash(features.numpy()) != fp["features_fingerprint"]:
            raise AssertionError(f"RFE feature fingerprint mismatch at fold {fold}")
        prepared = prepare(ctx["labels"], metadata, split["train"], 911, fold, CFG["prototype"])
        expected = historical_audit["anchors"][f"911/{fold}"]
        if json_digest(prepared["info"]) != json_digest(expected["prototype"]):
            raise AssertionError(f"D4 private three-fold preparation differs at fold {fold}")
        files = [split_path, start_path, fingerprint_path, features_path, support_path, vae_path]
        hashes = {str(item): sha_file(item) for item in files}
        for item, digest in hashes.items():
            if expected["input_sha256"].get(item) != digest:
                raise AssertionError(f"Historical input SHA mismatch: {item}")
        anchor_records[str(fold)] = {
            "paths": {"split": str(split_path), "train_start": str(start_path),
                      "fingerprints": str(fingerprint_path), "selected_features": str(features_path),
                      "rfe_support": str(support_path), "vae": str(vae_path)},
            "sha256": hashes, "train_start_core_sha256": core_hash(start["core"]),
            "train_start_rng_sha256": rng_hash(start["rng"]),
            "split_fingerprints": fp["split_fingerprints"],
            "prototype_preparation_sha256": json_digest(prepared["info"]), "gpu": fold % 2}
    if not np.all(counts == 1):
        raise AssertionError("Ten-fold test splits do not cover each subject exactly once")

    historical_refs = {}
    for label, directory, group in (
            ("D4", PCDCA, "D4"),
            ("M3", ROOT / "experiments/PCDPAX_20261021", "M3"),
            ("P1", P1_HISTORY, "P1")):
        historical_refs[label] = {}
        for fold in range(10):
            artifact = directory / "runs" / group / f"s911_f{fold}.pt"
            if not artifact.is_file():
                raise FileNotFoundError(f"Required read-only historical result missing: {artifact}")
            historical_refs[label][str(fold)] = {"path": str(artifact), "sha256": sha_file(artifact)}
    historical_documents = {}
    for directory in (PCDCA, ROOT / "experiments/PCDPAX_20261021", P1_HISTORY):
        for name in ("audit.json", "summary.json", "report.md"):
            item = directory / name
            if item.is_file():
                historical_documents[str(item)] = sha_file(item)

    handoff = ROOT / "handoff.md"
    code_files = {name: sha_file(BASE / name) for name in TRAIN_CODE}
    record = {
        "experiment": CFG["experiment"], "created_utc": utc(),
        "training_code_sha256": json_digest(code_files), "training_code_files_sha256": code_files,
        "original_source_sha256": current_sources, "historical_pcdca_prototype_sha256": historical_proto_hash,
        "historical_pcdca_code_sha256": historical_audit["code_sha256"],
        "historical_pcdca_audit_sha256": sha_file(PCDCA / "audit.json"),
        "context_sha256": sha_file(ctx_path), "raw_feature_tensor_sha256": sha_file(raw_path),
        "cohort": {"subjects": 871, "ASD_label": 0, "TD_label": 1, "class_counts": [403, 468],
                   "non_imaging_fields": ["SITE_ID", "SEX", "AGE_AT_SCAN"],
                   "non_imaging_tensor_shape": [871, 3], "RFE_shape": [871, 500], "sites": 20},
        "metadata": {"age_range": [float(np.nanmin(metadata["age"])), float(np.nanmax(metadata["age"]))],
                     "csv_sex_coding": "1=male, 2=female; verified against cohort counts"},
        "runtime_environment": runtime_versions(),
        "anchors": anchor_records,
        "historical_read_only_artifacts": historical_refs,
        "historical_read_only_documents_sha256": historical_documents,
        "history_policy": {"D0_retrained": False, "D4_retrained": False,
                           "historical_experiments_retrained": False, "RFE_refit": False,
                           "VAE_pretrained": False, "split_created": False, "historical_weights_read_only": True},
        "handoff_before": {"path": str(handoff), "bytes": handoff.stat().st_size, "sha256": sha_file(handoff)},
        "preflight_attempts": [], "paper": "https://arxiv.org/html/2406.14455v3",
        "repository": "https://github.com/tangaii/MM",
        "git_commit_at_audit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "formal_authorized": False, "preflight": None, "manifest_sha256": None,
        "mathematical_tests": None,
        "execution": {"tasks": 40, "gpu0": 20, "gpu1": 20, "completed": 0, "failures": [],
                      "retries": 0, "queue_wall_seconds": None, "worker_seconds": None}}
    atomic_json(record, path)
    append_log("audit_passed", anchors=10, code_sha256=record["training_code_sha256"])
    return record


def guard(require_manifest: bool = True) -> dict:
    record = read_json(BASE / "audit.json")
    if record["training_code_sha256"] != code_hash():
        raise AssertionError("Frozen training code fingerprint changed")
    for name, digest in record["original_source_sha256"].items():
        if sha_file(ROOT / name) != digest:
            raise AssertionError(f"Protected original source changed: {name}")
    for item in record["anchors"].values():
        for path, digest in item["sha256"].items():
            if sha_file(path) != digest:
                raise AssertionError(f"Protected historical input changed: {path}")
    if require_manifest:
        manifest_path = BASE / "manifest.json"
        if not manifest_path.exists() or sha_file(manifest_path) != record["manifest_sha256"]:
            raise AssertionError("Frozen manifest is missing or changed")
        manifest = read_json(manifest_path)
        if manifest["training_code_sha256"] != code_hash() or not manifest["preflight_passed"]:
            raise AssertionError("Frozen manifest does not match validated training code")
    return record


class Loaded:
    def __init__(self, seed: int, fold: int, audit_info: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("PPU training is required; CUDA/PPU is unavailable")
        torch.cuda.set_device(0)
        torch.set_num_threads(CFG["host_threads"])
        if "PPU" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError(f"Expected PPU; found {torch.cuda.get_device_name(0)}")
        self.seed, self.fold = seed, fold
        self.ctx = load_torch(HISTORY / "artifacts/context.pt")
        self.meta = read_metadata(self.ctx)
        self.split = load_torch(anchor(seed, fold) / "splits.pt")
        self.start = load_torch(anchor(seed, fold) / "train_start.pt")
        self.anchor_record = audit_info["anchors"][str(fold)]
        for path, digest in self.anchor_record["sha256"].items():
            if sha_file(path) != digest:
                raise AssertionError(f"Input changed after audit: {path}")
        saved_argv = sys.argv
        try:
            sys.argv = ["pcdpacm_options"]
            options = OptInit(model="MM_GTUNets", dataset="ABIDE", atlas="aal").args
        finally:
            sys.argv = saved_argv
        options.seed = seed
        options.device = torch.device("cuda:0")
        options.data_folder = str(ROOT / CFG["data_folder"])
        options.ckpt_path = str(anchor(seed, fold).parent / "vae")
        options.epoch = CFG["epochs"]
        options.early_stop = CFG["early_stop"]
        options.lr = CFG["lr"]
        options.wd = CFG["weight_decay"]
        options.log_save = False
        options.model_save = False
        self.options = options
        self.core = MM_GTUNets(options, fold).to(options.device)
        self.core.vae_save_path = str(anchor(seed, fold).parent / "vae" / f"fold{fold}_pretrain.pth")
        import_core(self.core, self.start["core"])
        if core_hash(export_core(self.core)) != self.anchor_record["train_start_core_sha256"]:
            raise AssertionError("Loaded core differs from historical train_start")
        self.x_cpu = load_torch(anchor(seed, fold).parent / "selected_features.pt").float()
        self.ph_cpu = torch.as_tensor(self.ctx["ph_data"], dtype=torch.float32)
        affinity = create_reward_penalty_graph(self.ctx["ph_dict"], self.ctx["labels"],
                                               self.split["train"], self.split["val"], self.split["test"], options)
        self.affinity_cpu = torch.as_tensor(affinity, dtype=torch.float32)
        self.x, self.ph, self.affinity = [value.to(options.device) for value in
                                          (self.x_cpu, self.ph_cpu, self.affinity_cpu)]
        self.y = torch.as_tensor(self.ctx["labels"], dtype=torch.long, device=options.device)
        self.train = torch.as_tensor(self.split["train"], dtype=torch.long, device=options.device)
        self.val = torch.as_tensor(self.split["val"], dtype=torch.long, device=options.device)
        self.test = torch.as_tensor(self.split["test"], dtype=torch.long, device=options.device)
        self.prepared = prepare(self.ctx["labels"], self.meta, self.split["train"], seed, fold,
                                CFG["prototype"])
        if json_digest(self.prepared["info"]) != self.anchor_record["prototype_preparation_sha256"]:
            raise AssertionError("Loaded D4 preparation differs from historical audit")
        self.z_img = self.z_ph = None
        self.forward_count = 0
        self.img_hook = self.core.img_unet.register_forward_hook(self.capture_img)
        self.ph_hook = self.core.ph_unet.register_forward_hook(self.capture_ph)
        self.input_digest = payload_hash({"x": self.x_cpu, "ph": self.ph_cpu, "affinity": self.affinity_cpu,
                                          "labels": self.ctx["labels"], "ids": self.ctx["ids"]})

    def capture_img(self, module, inputs, output):
        if tuple(output.shape) != (871, 16):
            raise AssertionError(f"img_unet output shape is {tuple(output.shape)}")
        self.z_img = output

    def capture_ph(self, module, inputs, output):
        if tuple(output.shape) != (871, 16):
            raise AssertionError(f"ph_unet output shape is {tuple(output.shape)}")
        self.z_ph = output

    def optimizer(self):
        params = list(self.core.parameters())
        if any(param is self.core.rp_attention.weights for param in params):
            raise AssertionError("Historical unregistered RP_Attention weights became registered")
        return torch.optim.Adam(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    def reset(self):
        import_core(self.core, self.start["core"])
        for param in self.core.parameters():
            param.grad = None
        restore_rng(self.start["rng"])
        torch.use_deterministic_algorithms(False)

    def forward(self):
        self.forward_count += 1
        return self.core(self.x, self.ph, self.affinity)

    def close(self):
        for name in ("img_hook", "ph_hook"):
            hook = getattr(self, name, None)
            if hook is not None:
                hook.remove()
        if hasattr(self, "core"):
            del self.core
        torch.cuda.empty_cache()


def formal_metrics(labels, probability, prediction) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.int64)
    return {"ACC": float(accuracy_score(labels, prediction)),
            "SEN-ASD": float(recall_score(labels == 0, prediction == 0, pos_label=True)),
            "SPE-TD": float(recall_score(labels == 1, prediction == 1, pos_label=True)),
            "probability_AUC": float(roc_auc_score(labels == 0, probability))}


def valid_result(group: str, seed: int, fold: int, audit_info: dict, required=False):
    path = result_path(group, seed, fold)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    value = load_torch(path)
    if value.get("state") != "complete" or (value.get("group"), value.get("seed"), value.get("fold")) != (group, seed, fold):
        raise AssertionError(f"Incomplete or mismatched artifact: {path}")
    if value.get("training_code_sha256") != audit_info["training_code_sha256"]:
        raise AssertionError(f"Frozen code fingerprint differs: {path}")
    if value.get("manifest_sha256") != audit_info.get("manifest_sha256"):
        raise AssertionError(f"Manifest fingerprint differs: {path}")
    if value.get("input_sha256") != audit_info["anchors"][str(fold)]["sha256"]:
        raise AssertionError(f"Historical input fingerprint differs: {path}")
    if value.get("best_core_sha256") != core_hash(value["best_model_state"]):
        raise AssertionError(f"Best core checksum differs: {path}")
    if (value.get("formal_test_forwards"), value.get("formal_test_grad_enabled"),
            value.get("uses_test_labels_for_training_or_selection")) != (1, True, False):
        raise AssertionError(f"Formal test protocol is invalid: {path}")
    if value.get("new_trainable_parameters") != 0 or value.get("vae_pretrain_count") != 0 \
            or value.get("rfe_refit_count") != 0:
        raise AssertionError(f"Forbidden refit recorded: {path}")
    if value.get("native_core_forwards") != 2 * value.get("epochs_run", -1) + 1:
        raise AssertionError(f"Unexpected core forward count: {path}")
    if not value.get("start_rng_fingerprint") or not value.get("test_rng_fingerprint"):
        raise AssertionError(f"Missing RNG fingerprints: {path}")
    split = load_torch(anchor(seed, fold) / "splits.pt")
    oof = value["oof"]
    if not np.array_equal(oof["indices"], split["test"]):
        raise AssertionError(f"OOF indices differ from historical split: {path}")
    cpu_probability = torch.softmax(torch.as_tensor(oof["logits"]), dim=1).numpy()[:, 0]
    max_diff = float(np.max(np.abs(cpu_probability - oof["probability"])))
    if max_diff > 2e-6:
        raise AssertionError(f"Stored PPU probability differs from CPU softmax by {max_diff}: {path}")
    if not np.array_equal(torch.as_tensor(oof["logits"]).argmax(1).numpy(), oof["prediction"]):
        raise AssertionError(f"Stored predictions differ from logits argmax: {path}")
    metrics = formal_metrics(oof["labels"], oof["probability"], oof["prediction"])
    if max(abs(metrics[key] - value["metrics"][key]) for key in metrics) > 1e-12:
        raise AssertionError(f"Stored metrics do not recompute: {path}")
    value["cpu_softmax_max_abs_diff"] = max_diff
    return value


def finite_scalar(value: torch.Tensor, name: str):
    if not bool(torch.isfinite(value)):
        raise FloatingPointError(f"Nonfinite {name}")


def math_tests() -> dict:
    rng = np.random.default_rng(20261021)
    n, n_train = 72, 60
    labels = np.asarray([0, 1] * (n // 2), dtype=np.int64)
    sites = np.asarray([f"S{i % 6}" for i in range(n)])
    age = 8 + (np.arange(n) * 7 % 31).astype(float)
    age[[3, 41]] = [np.nan, 0.0]
    sex = np.where(np.arange(n) % 3 == 0, 1.0, 2.0)
    sex[[8, 36]] = -1
    meta = {"age": age, "sex": sex, "site": sites}
    train = np.arange(n_train, dtype=np.int64)
    prepared = prepare(labels, meta, train, 911, 0, CFG["prototype"])
    z_img = torch.tensor(rng.normal(size=(n, 16)), dtype=torch.float32, requires_grad=True)
    z_ph = torch.tensor(rng.normal(size=(n, 16)) * 1.7 + .3, dtype=torch.float32, requires_grad=True)
    objective = CoupledPrototypeObjective(prepared, torch.device("cpu"))
    values = {group: objective(z_img, z_ph, group) for group in CFG["groups"]}
    checks = {}

    # Direct cubic oracle checks every ASD/TD triplet against the production marginals.
    k = objective.k_real.numpy().astype(np.float64)
    pf, y = prepared["folds"], prepared["y"]
    ref = (pf[:, None] != pf[None, :]).astype(np.float64)
    for group, value in values.items():
        c = value["c"].detach().numpy().astype(np.float64)
        h = k if group == "P1" else k * c
        oracle0 = np.zeros((n_train, int((y == 0).sum())), dtype=np.float64)
        oracle1 = np.zeros((n_train, int((y == 1).sum())), dtype=np.float64)
        i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        for i in range(n_train):
            for jpos, j in enumerate(i0):
                if ref[i, j] == 0:
                    continue
                for kpos, other in enumerate(i1):
                    if ref[i, other] == 0:
                        continue
                    weight = ref[i, j] * ref[i, other] * h[i, j] * h[i, other] * h[j, other]
                    oracle0[i, jpos] += weight
                    oracle1[i, kpos] += weight
        checks[f"{group}_matrix_marginals_equal_cubic_oracle"] = (
            np.allclose(value["pair_detail"]["m0"].numpy(), oracle0, atol=3e-5, rtol=3e-5)
            and np.allclose(value["pair_detail"]["m1"].numpy(), oracle1, atol=3e-5, rtol=3e-5))
        checks[f"{group}_class_total_masses_equal"] = torch.allclose(
            value["pair_detail"]["mass0"], value["pair_detail"]["mass1"], atol=2e-5, rtol=2e-5)
        for key in ("w0", "w1"):
            weights = value["pair_detail"][key]
            positive = weights.sum(1) > 0
            checks[f"{group}_{key}_normalizes_supported_rows"] = torch.allclose(
                weights[positive].sum(1), torch.ones(int(positive.sum())), atol=2e-6, rtol=0)

    # The copied accepted D4 implementation must remain bitwise identical.
    import importlib.util
    spec = importlib.util.spec_from_file_location("pcdca_math_oracle", PCDCA / "prototype.py")
    oracle_module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(oracle_module)
    original_prepared = oracle_module.prepare(labels, meta, train, 911, 0, CFG["prototype"])
    old_obj = oracle_module.PrototypeObjective(original_prepared, "D4", HISTORICAL_CFG, torch.device("cpu"))
    copied_obj = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, torch.device("cpu"))
    old_loss, old_detail = old_obj(z_img)
    copied_loss, copied_detail = copied_obj(z_img)
    checks["D4_preparation_info_exact"] = json_digest(original_prepared["info"]) == json_digest(prepared["info"])
    checks["D4_prototype_ce_logits_means_bitwise_exact"] = (
        torch.equal(old_loss, copied_loss) and torch.equal(old_detail["logits"], copied_detail["logits"])
        and torch.equal(old_detail["mu"], copied_detail["mu"]))
    age32 = torch.as_tensor(prepared["age"], dtype=torch.float32)
    sex64 = torch.as_tensor(prepared["sex"], dtype=torch.long)
    expected_age = torch.exp(-0.5 * ((age32[:, None] - age32[None, :]) / prepared["bandwidth"]).square())
    known = (sex64[:, None] > 0) & (sex64[None, :] > 0)
    expected_sex = torch.where(known, torch.where(sex64[:, None] == sex64[None, :], 1.0, 0.5), 0.75)
    checks["D4_age_sex_kernel_exact"] = torch.equal(objective.k_real, expected_age * expected_sex)
    checks["D4_private_folds_and_reference_mask"] = (
        set(np.unique(prepared["folds"])) == {0, 1, 2}
        and not np.diag(ref).any()
        and np.all(prepared["folds"][np.where(ref)[0]] != prepared["folds"][np.where(ref)[1]]))
    checks["train_only_reference_contains_no_heldout"] = objective.reference.shape == (n_train, n_train)

    checks["C_is_bounded_and_detached"] = all(
        (0.5 <= values[group]["c"]).all() and (values[group]["c"] <= 1.0).all()
        and not values[group]["c"].requires_grad for group in ("P2", "P3", "P4"))
    checks["P1_uses_real_K_without_phenotype_coupling"] = torch.equal(values["P1"]["h"], objective.k_real)
    checks["P2_P3_real_coupling_is_identical"] = (
        torch.equal(values["P2"]["c"], values["P3"]["c"])
        and torch.equal(values["P2"]["pair_loss"], values["P3"]["pair_loss"]))
    checks["P4_changes_only_coupling_embedding"] = (
        torch.equal(values["P3"]["kernel"], values["P4"]["kernel"])
        and torch.equal(values["P3"]["d4_detail"]["mu"], values["P4"]["d4_detail"]["mu"])
        and np.array_equal(prepared["permutation"], objective.permutation.numpy()))
    checks["no_own_or_same_private_fold_reference"] = not objective.reference.diagonal().any() and bool(
        torch.all(objective.pf[:, None].expand_as(objective.reference)[objective.reference]
                  != objective.pf[None, :].expand_as(objective.reference)[objective.reference]))
    checks["P1_P2_have_phCE_P3_P4_do_not"] = (
        values["P1"]["ph_ce"] is not None and values["P2"]["ph_ce"] is not None
        and values["P3"]["ph_ce"] is None and values["P4"]["ph_ce"] is None)
    checks["all_groups_share_exact_D4_CE_and_pair_support"] = all(
        torch.equal(values["P1"]["image_d4"], values[group]["image_d4"])
        and torch.equal(values["P1"]["d4_detail"]["mu"], values[group]["d4_detail"]["mu"])
        and torch.equal(values["P1"]["pair_detail"]["valid"], values[group]["pair_detail"]["valid"])
        for group in CFG["groups"])
    checks["no_KL_or_new_trainable_parameters"] = all(
        "align_loss" not in value and not isinstance(objective, torch.nn.Module) for value in values.values())

    grad_img, grad_ph = torch.autograd.grad(values["P3"]["pair_loss"], (z_img, z_ph),
                                             retain_graph=True, allow_unused=True)
    checks["pair_loss_updates_img_query_and_has_no_direct_ph_gradient"] = (
        grad_img is not None and float(grad_img.norm()) > 0 and grad_ph is None)
    checks["phCE_updates_real_ph_embedding"] = all(
        torch.autograd.grad(values[group]["ph_ce"], z_ph, retain_graph=True, allow_unused=True)[0] is not None
        for group in ("P1", "P2"))

    altered_labels = labels.copy()
    altered_labels[n_train:] = 1 - altered_labels[n_train:]
    altered_meta = {key: value.copy() for key, value in meta.items()}
    altered_meta["age"][n_train:] = 99
    altered_meta["sex"][n_train:] = -1
    altered_meta["site"][n_train:] = "CHANGED"
    altered_prepared = prepare(altered_labels, altered_meta, train, 911, 0, CFG["prototype"])
    changed_z_img, changed_z_ph = z_img.detach().clone(), z_ph.detach().clone()
    changed_z_img[n_train:] = 10000
    changed_z_ph[n_train:] = -10000
    changed_objective = CoupledPrototypeObjective(altered_prepared, torch.device("cpu"))
    changed_values = {group: changed_objective(changed_z_img, changed_z_ph, group) for group in CFG["groups"]}
    checks["heldout_labels_metadata_embeddings_do_not_change_auxiliary_losses"] = all(
        torch.equal(values[group]["image_d4"].detach(), changed_values[group]["image_d4"].detach())
        and torch.equal(values[group]["pair_loss"].detach(), changed_values[group]["pair_loss"].detach())
        for group in CFG["groups"])

    degenerate_meta = {"age": np.full(n, 20.0), "sex": np.full(n, -1.0),
                       "site": np.asarray(["S0"] * n)}
    degenerate = prepare(labels, degenerate_meta, train, 911, 0, CFG["prototype"])
    z_constant_img = torch.ones((n, 16), requires_grad=True)
    z_constant_ph = torch.ones((n, 16), requires_grad=True)
    deg_obj = CoupledPrototypeObjective(degenerate, torch.device("cpu"))
    deg_values = {group: deg_obj(z_constant_img, z_constant_ph, group) for group in CFG["groups"]}
    checks["constant_embeddings_age_unknown_sex_are_finite"] = all(
        torch.isfinite(value["pair_loss"]) and torch.isfinite(value["image_d4"])
        and torch.isfinite(value["pair_detail"]["beta"]).all()
        for value in deg_values.values())
    no_ref = dict(degenerate)
    no_ref["folds"] = np.zeros(n_train, dtype=np.int64)
    no_ref_obj = CoupledPrototypeObjective(no_ref, torch.device("cpu"))
    empty = no_ref_obj(z_constant_img, z_constant_ph, "P3")
    checks["empty_reference_uses_finite_D4_fallback"] = (
        not empty["pair_detail"]["support"].any()
        and torch.equal(empty["pair_detail"]["beta"], torch.zeros_like(empty["pair_detail"]["beta"]))
        and torch.isfinite(empty["pair_loss"]))
    checks["40_unique_registered_tasks"] = len({(g, 911, f) for g in CFG["groups"] for f in range(10)}) == 40

    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    result = {"passed": not failed, "checks": checks, "failed": failed,
              "cubic_oracle": "all train query/ASD/TD triplets; CPU-only synthetic data",
              "max_coupling_C_spread": {group: float(values[group]["c"].max() - values[group]["c"].min())
                                        for group in ("P2", "P3", "P4")}}
    if failed:
        raise AssertionError(f"PCDPA-CM mathematical tests failed: {failed}")
    return result


def preflight_one(seed: int, fold: int, audit_info: dict) -> dict:
    try:
        loaded = Loaded(seed, fold, audit_info)
    except Exception as error:
        error.core_forwards = 0
        raise
    try:
        loaded.reset()
        parameter_ids = {id(param) for param in loaded.core.parameters()}
        input_hash = payload_hash({"x": loaded.x, "ph": loaded.ph, "affinity": loaded.affinity})
        loaded.core.train()
        with torch.set_grad_enabled(True):
            logits, graph = loaded.forward()
        if loaded.forward_count != 1:
            raise AssertionError("Real preflight must execute exactly one native forward")
        if tuple(logits.shape) != (871, 2) or tuple(loaded.z_img.shape) != (871, 16) \
                or tuple(loaded.z_ph.shape) != (871, 16):
            raise AssertionError("Unexpected logits or branch embedding dimensions")
        if payload_hash({"x": loaded.x, "ph": loaded.ph, "affinity": loaded.affinity}) != input_hash:
            raise AssertionError("Native forward mutated model inputs")
        native = F.cross_entropy(logits[loaded.train], loaded.y[loaded.train]) + graph
        objective = CoupledPrototypeObjective(loaded.prepared, loaded.options.device)
        outputs = {group: objective(loaded.z_img, loaded.z_ph, group) for group in CFG["groups"]}
        for group, value in outputs.items():
            for key in ("image_d4", "pair_loss", "image_aux"):
                finite_scalar(value[key], f"{group}/{key}")
            if value["ph_ce"] is not None:
                finite_scalar(value["ph_ce"], f"{group}/phCE")
        if not torch.isfinite(native):
            raise FloatingPointError("Nonfinite native loss in preflight")
        for group in CFG["groups"][1:]:
            if not torch.equal(outputs["P1"]["image_d4"], outputs[group]["image_d4"]):
                raise AssertionError("D4 image CE differs across groups")
            if not torch.equal(outputs["P1"]["d4_detail"]["mu"], outputs[group]["d4_detail"]["mu"]):
                raise AssertionError("D4 prototype differs across groups")
        if not torch.equal(outputs["P2"]["pair_loss"], outputs["P3"]["pair_loss"]):
            raise AssertionError("P2/P3 pair loss must be identical for the same forward")
        if not torch.equal(outputs["P1"]["kernel"], outputs["P2"]["kernel"]):
            raise AssertionError("P1/P2 original age-sex kernel changed")
        if not torch.equal(outputs["P3"]["kernel"], outputs["P4"]["kernel"]):
            raise AssertionError("P3/P4 original age-sex kernel changed")
        if not torch.equal(outputs["P3"]["d4_detail"]["mu"], outputs["P4"]["d4_detail"]["mu"]):
            raise AssertionError("P3/P4 historical D4 prototype changed")
        grad_img, grad_ph = torch.autograd.grad(outputs["P3"]["pair_loss"],
                                                 (loaded.z_img, loaded.z_ph), retain_graph=True,
                                                 allow_unused=True)
        if grad_img is None or float(grad_img.norm()) <= 0 or grad_ph is not None:
            raise AssertionError("Pair loss gradient path must reach image queries only")
        native_img, native_ph = torch.autograd.grad(native, (loaded.z_img, loaded.z_ph),
                                                     retain_graph=True, allow_unused=True)
        if native_img is None or native_ph is None or float(native_ph.norm()) == 0:
            raise AssertionError("Original native loss no longer updates the phenotype branch")
        if len({id(p) for p in loaded.core.parameters()}) != len(parameter_ids):
            raise AssertionError("Objective unexpectedly changed trainable model parameter set")

        heldout = np.setdiff1d(np.arange(871), loaded.split["train"])
        altered_labels = loaded.ctx["labels"].copy()
        altered_labels[heldout] = 1 - altered_labels[heldout]
        altered_meta = {key: value.copy() for key, value in loaded.meta.items()}
        altered_meta["age"][heldout] = 99
        altered_meta["sex"][heldout] = -1
        altered_meta["site"][heldout] = "HELDOUT_CHANGED"
        changed_prepared = prepare(altered_labels, altered_meta, loaded.split["train"], seed, fold,
                                   CFG["prototype"])
        changed_z_img, changed_z_ph = loaded.z_img.detach().clone(), loaded.z_ph.detach().clone()
        changed_z_img[heldout] = 10000
        changed_z_ph[heldout] = -10000
        changed_objective = CoupledPrototypeObjective(changed_prepared, loaded.options.device)
        changed = {group: changed_objective(changed_z_img, changed_z_ph, group) for group in CFG["groups"]}
        heldout_invariant = all(
            torch.equal(outputs[group]["image_d4"].detach(), changed[group]["image_d4"].detach())
            and torch.equal(outputs[group]["pair_loss"].detach(), changed[group]["pair_loss"].detach())
            for group in CFG["groups"])
        if not heldout_invariant:
            raise AssertionError("Auxiliary objective depends on heldout labels, metadata, or embeddings")

        group_records = {}
        for group, value in outputs.items():
            group_records[group] = {
                "image_d4_ce": float(value["image_d4"].detach()),
                "pair_ce": float(value["pair_loss"].detach()),
                "image_aux": float(value["image_aux"].detach()),
                "ph_ce": None if value["ph_ce"] is None else float(value["ph_ce"].detach()),
                "image_d4_logits_sha256": payload_hash(value["d4_detail"]["logits"]),
                "pair_logits_sha256": payload_hash(value["pair_detail"]["logits"]),
                "coupling": coupling_summary(value, loaded.prepared),
            }
        def matrix_diff(a, b):
            return {"mean_abs": float((a - b).abs().mean()), "max_abs": float((a - b).abs().max()),
                    "nonzero_fraction": float(((a - b).abs() > 1e-7).float().mean())}

        return {
            "passed": True, "seed": seed, "fold": fold, "timestamp_utc": utc(),
            "single_core_forward": True, "forward_count": loaded.forward_count,
            "native_logits_shape": list(logits.shape), "z_img_shape": list(loaded.z_img.shape),
            "z_ph_shape": list(loaded.z_ph.shape), "native_loss": float(native.detach()),
            "groups": group_records,
            "pair_loss_differences": {"P1-P2": matrix_diff(outputs["P1"]["pair_detail"]["w0"], outputs["P2"]["pair_detail"]["w0"]),
                                      "P3-P4": matrix_diff(outputs["P3"]["pair_detail"]["w0"], outputs["P4"]["pair_detail"]["w0"])},
            "gradient_path": {"pair_to_z_img_norm": float(grad_img.norm()), "pair_to_z_ph_norm": 0.0,
                              "native_to_z_img_norm": float(native_img.norm()),
                              "native_to_z_ph_norm": float(native_ph.norm())},
            "input_unchanged": True, "heldout_auxiliary_invariance": True,
            "parameter_count_unchanged": True,
            "limitations": ["Native 871-node population graph remains transductive, matching historical protocol."],
            "physical_gpu": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
            "device": torch.cuda.get_device_name(0)}
    except Exception as error:
        error.core_forwards = loaded.forward_count
        raise
    finally:
        loaded.close()


def math_tests() -> dict:
    """Independent CPU tests for conditional means, cross-fit references, and residual CE."""
    rng = np.random.default_rng(20261022)
    n = 72
    labels = np.tile(np.array([0, 1]), n // 2).astype(np.int64)
    sites = np.asarray([f"S{i % 6}" for i in range(n)])
    age = (9 + (np.arange(n) * 7 % 29)).astype(float)
    age[[3, 41]] = [np.nan, 0.0]
    sex = np.where(np.arange(n) % 3 == 0, 1.0, 2.0)
    sex[[8, 36]] = -1
    meta = {"age": age, "sex": sex, "site": sites}
    train = np.arange(n, dtype=np.int64)
    prepared = prepare(labels, meta, train, 911, 0, CFG["prototype"])
    z_img = torch.tensor(rng.normal(size=(n, 16)), dtype=torch.float32, requires_grad=True)
    z_ph = torch.tensor(rng.normal(size=(n, 16)), dtype=torch.float32, requires_grad=True)
    obj = ResidualPrototypeObjective(prepared, torch.device("cpu"))
    values = {g: obj(z_img, z_ph, g) for g in CFG["groups"]}
    checks = {}
    checks["labels_ASD0_TD1"] = bool(np.array_equal(np.unique(prepared["y"]), [0, 1]))
    checks["private_folds_are_exactly_three"] = set(np.unique(prepared["folds"])) == {0, 1, 2}
    checks["train_only_preparation"] = np.array_equal(prepared["train"], train)
    checks["age_fill_and_unknown_sex_finite"] = np.isfinite(prepared["age"]).all() and np.isin(prepared["sex"], [-1, 1, 2]).all()
    checks["d4_references_exclude_self"] = not bool(obj.d4.reference.diagonal().any())
    checks["d4_weights_exclude_same_private_fold"] = all(
        not bool((obj.d4.weights[q][:, :, s] != 0).logical_and(
            (obj.pf[s][None, None, :] == obj.pf[q][:, None, None])).any())
        for g in range(3) for q, s in [(obj.groups[g]["variants"]["R2"]["q"], obj.groups[g]["variants"]["R2"]["s"])])
    # Compare the copied D4 implementation to the accepted read-only module.
    import importlib.util
    spec = importlib.util.spec_from_file_location("historical_pcdca_prototype", PCDCA / "prototype.py")
    oracle_module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(oracle_module)
    old_prep = oracle_module.prepare(labels, meta, train, 911, 0, CFG["prototype"])
    old_obj = oracle_module.PrototypeObjective(old_prep, "D4", HISTORICAL_CFG, torch.device("cpu"))
    new_obj = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, torch.device("cpu"))
    old_ce, old_detail = old_obj(z_img)
    new_ce, new_detail = new_obj(z_img)
    checks["D4_prepare_equal_historical"] = json_digest(old_prep["info"]) == json_digest(prepared["info"])
    checks["D4_CE_equal_historical"] = torch.equal(old_ce, new_ce)
    checks["D4_prototypes_equal_historical"] = torch.equal(old_detail["mu"], new_detail["mu"])
    checks["D4_tau_equal_historical"] = torch.equal(old_detail["tau"], new_detail["tau"])
    checks["all_groups_share_D4_CE"] = all(torch.equal(values["R1"]["image_d4"], values[g]["image_d4"]) for g in CFG["groups"])
    checks["all_groups_share_phCE"] = all(torch.equal(values["R1"]["ph_ce"], values[g]["ph_ce"]) for g in CFG["groups"])
    checks["all_groups_share_real_pair_weights"] = all(
        all(torch.equal(values["R1"]["pair_weights"][g][k], values[group]["pair_weights"][g][k])
            for g in range(3) for k in (0, 1)) for group in CFG["groups"][1:])
    checks["R1_mixed_weights_normalize"] = all(torch.allclose(
        obj.groups[g]["variants"]["R1"]["wq"].sum(1), torch.ones_like(obj.groups[g]["variants"]["R1"]["okq"], dtype=torch.float32))
        for g in range(3))
    checks["R2_equal_class_weights"] = all(torch.allclose(
        obj.groups[g]["variants"]["R2"]["wq"] @ (obj.y[obj.groups[g]["variants"]["R2"]["s"]] == c).float(),
        torch.full_like(obj.groups[g]["variants"]["R2"]["okq"].float(), .5), atol=2e-6)
        for g in range(3) for c in (0, 1))
    checks["R1_R2_are_distinct"] = any(not torch.allclose(
        obj.groups[g]["variants"]["R1"]["wq"], obj.groups[g]["variants"]["R2"]["wq"]) for g in range(3))
    checks["R3_beta_bounded"] = all(bool(((obj.groups[g]["variants"]["R3"]["beta_q"] >= 0) &
                                                  (obj.groups[g]["variants"]["R3"]["beta_q"] <= 1)).all()) for g in range(3))
    low_k = torch.ones((1, 4), dtype=torch.float32)
    low_y = torch.tensor([0, 0, 1, 1])
    low_site = torch.tensor([-1])
    ref_site = torch.tensor([0, 0, 0, 0])
    r2_w, _, _ = obj._condition(low_k, low_y, low_site, ref_site, "R2")
    r3_w, _, r3_b = obj._condition(low_k, low_y, low_site, ref_site, "R3")
    checks["R3_low_support_falls_back_R2"] = bool(torch.equal(r3_b, torch.zeros_like(r3_b)) and torch.equal(r2_w, r3_w))
    checks["R3_site_beta_formula_finite"] = all(bool(torch.isfinite(obj.groups[g]["variants"]["R3"]["beta_q"]).all()) for g in range(3))
    checks["R4_permutation_is_within_site"] = np.array_equal(prepared["site"], prepared["site"][prepared["permutation"]])
    checks["R4_changes_b_on_synthetic_data"] = not torch.allclose(
        values["R3"]["residual_detail"]["b"], values["R4"]["residual_detail"]["b"], atol=1e-7, rtol=0)
    checks["R4_keeps_P1_pair_weights"] = torch.equal(obj.groups[0]["w0"], obj.groups[0]["w1"] * 0 + obj.groups[0]["w0"])
    checks["condition_weights_detached"] = all(not state["wq"].requires_grad and not state["wr"].requires_grad
        for g in range(3) for state in obj.groups[g]["variants"].values())
    checks["b_vectors_are_detached"] = not values["R3"]["residual_detail"]["b"].requires_grad
    grad_img, grad_ph = torch.autograd.grad(values["R3"]["residual_ce"], (z_img, z_ph), retain_graph=True, allow_unused=True)
    checks["Lres_has_image_gradient"] = grad_img is not None and float(grad_img.norm()) > 0
    checks["Lres_has_no_direct_ph_gradient"] = grad_ph is None
    checks["residual_query_keeps_gradient"] = values["R3"]["residual_detail"]["query"].requires_grad
    checks["residual_losses_finite"] = all(bool(torch.isfinite(values[g]["residual_ce"])) for g in CFG["groups"])
    checks["residual_logits_use_D4_tau"] = all(torch.equal(values[g]["d4_detail"]["tau"], new_detail["tau"]) for g in CFG["groups"])
    # Independent cubic P1 oracle over every query and every opposite-class pair.
    p1_ok = True
    for g in range(3):
        state = obj.groups[g]
        q, s = state["variants"]["R1"]["q"].numpy(), state["variants"]["R1"]["s"].numpy()
        k = obj.k_real.detach().numpy()
        ys = prepared["y"][s]
        i0, i1 = np.flatnonzero(ys == 0), np.flatnonzero(ys == 1)
        m0 = np.zeros((len(q), len(i0)), dtype=np.float64)
        m1 = np.zeros((len(q), len(i1)), dtype=np.float64)
        for qi, i in enumerate(q):
            for a, jp in enumerate(i0):
                j = s[jp]
                for b, kp in enumerate(i1):
                    kk = s[kp]
                    w = k[i, j] * k[i, kk] * k[j, kk]
                    m0[qi, a] += w
                    m1[qi, b] += w
        p1_ok &= np.allclose(m0, state["m0"].numpy(), atol=3e-5, rtol=3e-5)
        p1_ok &= np.allclose(m1, state["m1"].numpy(), atol=3e-5, rtol=3e-5)
    checks["P1_marginals_equal_independent_cubic_oracle"] = p1_ok
    checks["query_fold_not_in_reference_pool"] = all(
        not bool(torch.isin(state["s"], state["q"]).any()) for g in range(3) for state in obj.groups[g]["variants"].values())
    checks["reference_self_excluded_from_b"] = all(
        bool((torch.diagonal(state["wr"]) == 0).all()) for g in range(3) for state in obj.groups[g]["variants"].values())
    # Query-fold labels cannot affect that fold's b weights; mutate only those labels.
    altered = dict(prepared)
    altered["y"] = prepared["y"].copy()
    altered["y"][prepared["folds"] == 0] = 1 - altered["y"][prepared["folds"] == 0]
    altered_obj = ResidualPrototypeObjective(altered, torch.device("cpu"))
    checks["query_fold_labels_excluded_from_b_estimation"] = torch.equal(
        obj.groups[0]["variants"]["R3"]["wq"], altered_obj.groups[0]["variants"]["R3"]["wq"])
    checks["R4_does_not_change_D4_or_phCE"] = torch.equal(values["R3"]["image_d4"], values["R4"]["image_d4"]) and torch.equal(values["R3"]["ph_ce"], values["R4"]["ph_ce"])
    checks["R4_does_not_change_P1_weights"] = all(
        torch.equal(values["R3"]["pair_weights"][g][k], values["R4"]["pair_weights"][g][k])
        for g in range(3) for k in (0, 1))
    checks["D4_residual_fallback_in_reference_coordinate_space"] = values["R3"]["residual_detail"]["mu"].shape == (n, 2, 16)
    # Explicit invariance identity with shared b, plus a nonconstant-b counterexample.
    raw_q, raw_mu = torch.randn(4, 16), torch.randn(4, 2, 16)
    shared = torch.randn(1, 16)
    checks["shared_b_preserves_pair_distances"] = torch.allclose(
        (raw_q[:, None] - raw_mu).square().sum(-1),
        ((raw_q - shared)[:, None] - (raw_mu - shared)).square().sum(-1), atol=1e-6)
    checks["constant_b_translation_invariance"] = checks["shared_b_preserves_pair_distances"]
    varying = torch.randn(4, 16)
    checks["varying_b_can_change_pair_distances"] = not torch.allclose(
        (raw_q[:, None] - raw_mu).square().sum(-1),
        ((raw_q - varying)[:, None] - raw_mu).square().sum(-1), atol=1e-6)
    checks["no_new_trainable_parameters_or_KL"] = not isinstance(obj, torch.nn.Module) and all(
        "kl" not in key.lower() for key in values["R1"])
    checks["no_N_cubed_production_tensor"] = all(
        tensor.ndim <= 2 for tensor in [obj.k_real, obj.groups[0]["m0"], obj.groups[0]["m1"]])
    checks["all_40_tasks_unique"] = len({(g, 911, f) for g in CFG["groups"] for f in range(10)}) == 40
    # Extra guards for edge-case inputs and loss support.
    checks["unknown_sex_kernel_fallback"] = float(_kernel(torch.tensor([20.]), torch.tensor([-1]), torch.tensor([20.]), torch.tensor([2]), prepared["bandwidth"])[0, 0]) == 0.75
    checks["missing_age_imputed_from_train"] = bool(prepared["info"]["age_missing_count"] == 2)
    checks["empty_residual_loss_is_finite"] = all(bool(torch.isfinite(values[g]["residual_ce"])) for g in CFG["groups"])
    checks = {k: bool(v) for k, v in checks.items()}
    failed = [k for k, v in checks.items() if not v]
    result = {"passed": not failed, "checks": checks, "check_count": len(checks), "failed": failed,
              "independent_oracle": "nested Python loops over query x ASD reference x TD reference"}
    if failed:
        raise AssertionError(f"PC-RPA mathematical tests failed: {failed}")
    return result


def preflight_one(seed: int, fold: int, audit_info: dict) -> dict:
    loaded = Loaded(seed, fold, audit_info)
    try:
        loaded.reset()
        parameter_ids = {id(p) for p in loaded.core.parameters()}
        input_hash = payload_hash({"x": loaded.x, "ph": loaded.ph, "affinity": loaded.affinity})
        loaded.core.train()
        with torch.set_grad_enabled(True):
            logits, graph = loaded.forward()
        if loaded.forward_count != 1:
            raise AssertionError("Preflight requires exactly one native forward")
        native = F.cross_entropy(logits[loaded.train], loaded.y[loaded.train]) + graph
        objective = ResidualPrototypeObjective(loaded.prepared, loaded.options.device)
        values = {g: objective(loaded.z_img, loaded.z_ph, g) for g in CFG["groups"]}
        for g, v in values.items():
            for key in ("image_d4", "residual_ce", "image_aux", "ph_ce"):
                finite_scalar(v[key], f"{g}/{key}")
        if tuple(logits.shape) != (871, 2) or tuple(loaded.z_img.shape) != (871, 16) or tuple(loaded.z_ph.shape) != (871, 16):
            raise AssertionError("Unexpected native tensor dimensions")
        if payload_hash({"x": loaded.x, "ph": loaded.ph, "affinity": loaded.affinity}) != input_hash:
            raise AssertionError("Preflight mutated native inputs")
        for group in CFG["groups"][1:]:
            if not torch.equal(values["R1"]["image_d4"], values[group]["image_d4"]):
                raise AssertionError("D4 loss differs by group")
            if not torch.equal(values["R1"]["ph_ce"], values[group]["ph_ce"]):
                raise AssertionError("True phenotype CE differs by group")
        grad_img, grad_ph = torch.autograd.grad(values["R3"]["residual_ce"], (loaded.z_img, loaded.z_ph), retain_graph=True, allow_unused=True)
        if grad_img is None or float(grad_img.norm()) == 0 or grad_ph is not None:
            raise AssertionError("Residual CE gradient must reach image branch only")
        native_img, native_ph = torch.autograd.grad(native, (loaded.z_img, loaded.z_ph), retain_graph=True, allow_unused=True)
        if native_img is None or native_ph is None or float(native_ph.norm()) == 0:
            raise AssertionError("Native graph/CE must preserve phenotype branch gradient")
        if {id(p) for p in loaded.core.parameters()} != parameter_ids:
            raise AssertionError("Objective changed trainable parameter set")
        # R1-R4 share the exact pair weights; only conditional common-component estimates differ.
        pair_same = all(
            all(torch.equal(values["R1"]["pair_weights"][g][k], values[group]["pair_weights"][g][k])
                for g in range(3) for k in (0, 1))
            for group in CFG["groups"][1:])
        if not pair_same:
            raise AssertionError("P1 true age-sex pair weights differ by group")
        p1_detail = objective.raw_p1_pair_detail(loaded.z_img, values["R3"]["d4_detail"])
        p1_grad = torch.autograd.grad(p1_detail["loss"], loaded.z_img, retain_graph=True)[0]
        d4_grad = torch.autograd.grad(values["R3"]["image_d4"], loaded.z_img, retain_graph=True)[0]
        def grad_cosine(a, b):
            return float((a.detach() * b.detach()).sum() /
                         (a.detach().norm() * b.detach().norm()).clamp_min(CFG["gradient_epsilon"]))
        b = {g: values[g]["residual_detail"]["b"].detach() for g in CFG["groups"]}
        diffs = {"R1-R2": float((b["R1"] - b["R2"]).norm(dim=1).mean()),
                 "R2-R3": float((b["R2"] - b["R3"]).norm(dim=1).mean()),
                 "R3-R4": float((b["R3"] - b["R4"]).norm(dim=1).mean())}
        records = {g: {"d4_ce": float(v["image_d4"].detach()), "ph_ce": float(v["ph_ce"].detach()),
                       "residual_ce": float(v["residual_ce"].detach()),
                       "b_sha256": payload_hash(v["residual_detail"]["b"]),
                       "b_norm_mean": float(v["residual_detail"]["b"].norm(dim=1).mean()),
                       "site_beta_zero_fraction": float((v["residual_detail"]["beta_site"] == 0).float().mean())}
                   for g, v in values.items()}
        return {"passed": True, "seed": seed, "fold": fold, "single_core_forward": True,
                "forward_count": loaded.forward_count, "native_logits_shape": list(logits.shape),
                "z_img_shape": list(loaded.z_img.shape), "z_ph_shape": list(loaded.z_ph.shape),
                "native_loss": float(native.detach()), "groups": records, "b_differences": diffs,
                "p1_pair_ce": float(p1_detail["loss"].detach()),
                "residual_vs_p1_ce_delta": float(values["R3"]["residual_ce"].detach() - p1_detail["loss"].detach()),
                "residual_vs_p1_prototype_mean_l2": float((values["R3"]["residual_detail"]["mu"] - p1_detail["mu"].detach()).norm(dim=-1).mean()),
                "residual_vs_p1_gradient_cosine": grad_cosine(grad_img, p1_grad),
                "residual_vs_d4_gradient_cosine": grad_cosine(grad_img, d4_grad),
                "pair_weights_identical_all_groups": pair_same,
                "gradient_path": {"residual_to_z_img_norm": float(grad_img.norm()), "residual_to_z_ph_norm": 0.0,
                                  "native_to_z_img_norm": float(native_img.norm()), "native_to_z_ph_norm": float(native_ph.norm())},
                "input_unchanged": True, "trainable_parameter_set_unchanged": True,
                "physical_gpu": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")), "device": torch.cuda.get_device_name(0)}
    except Exception as error:
        error.core_forwards = loaded.forward_count
        raise
    finally:
        loaded.close()


def freeze_manifest(audit_info: dict) -> dict:
    if not audit_info.get("mathematical_tests", {}).get("passed"):
        raise AssertionError("Mathematical tests must pass before freezing")
    if not (audit_info.get("preflight") or {}).get("passed"):
        raise AssertionError("Single-forward preflight must pass before freezing")
    files = {name: sha_file(BASE / name) for name in TRAIN_CODE}
    manifest = {
        "experiment": CFG["experiment"], "name": CFG["name"], "frozen_utc": utc(),
        "training_code_sha256": json_digest(files), "training_code_files_sha256": files,
        "config_sha256": sha_file(BASE / "config.json"), "residual_definition": CFG["residual_pair"],
        "groups_definition": CFG["conditions"], "preflight_passed": True,
        "preflight_sha256": json_digest(audit_info["preflight"]),
        "math_tests_sha256": json_digest(audit_info["mathematical_tests"]),
        "seed": 911, "folds": list(range(10)), "groups": list(CFG["groups"]), "tasks": 40,
        "gpu0_folds": [0, 2, 4, 6, 8], "gpu1_folds": [1, 3, 5, 7, 9],
        "optimizer": {"name": "Adam", "lr": CFG["lr"], "weight_decay": CFG["weight_decay"]},
        "historical_model_source_sha256": audit_info["original_source_sha256"],
        "historical_inputs_sha256": {fold: item["sha256"] for fold, item in audit_info["anchors"].items()},
        "read_only_prototype_source_sha256": audit_info["historical_pcdca_prototype_sha256"],
        "no_git": True}
    atomic_json(manifest, BASE / "manifest.json")
    audit_info["manifest_sha256"] = sha_file(BASE / "manifest.json")
    audit_info["formal_authorized"] = True
    atomic_json(audit_info, BASE / "audit.json")
    append_log("training_code_frozen", manifest_sha256=audit_info["manifest_sha256"],
               training_code_sha256=audit_info["training_code_sha256"])
    return manifest


def valid_or_skip(group: str, seed: int, fold: int, audit_info: dict) -> bool:
    result = valid_result(group, seed, fold, audit_info)
    if result is None:
        return False
    append_log("skip_valid", group=group, seed=seed, fold=fold,
               artifact_sha256=sha_file(result_path(group, seed, fold)))
    return True


def train_one(group: str, seed: int, fold: int, audit_info: dict):
    if valid_or_skip(group, seed, fold, audit_info):
        return
    if not audit_info.get("formal_authorized"):
        raise AssertionError("Formal training is not authorized by passed preflight")
    loaded = Loaded(seed, fold, audit_info)
    path = result_path(group, seed, fold)
    status_path, worker_log = path.with_suffix(".status.json"), path.with_suffix(".jsonl")
    task_started, task_started_utc = time.perf_counter(), utc()

    def event(name, **fields):
        line = json.dumps({"utc": utc(), "event": name, "group": group,
                           "seed": seed, "fold": fold, **fields}, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")) + "\n"
        with worker_log.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        append_log(name, group=group, seed=seed, fold=fold, **fields)

    try:
        atomic_json({"state": "running", "group": group, "seed": seed, "fold": fold,
                     "started_utc": task_started_utc, "pid": os.getpid()}, status_path)
        optimizer = loaded.optimizer()
        objective = ResidualPrototypeObjective(loaded.prepared, loaded.options.device)
        stopper = EarlyStopping(patience=CFG["early_stop"], save=False)
        loaded.reset()
        if rng_hash(capture_rng()) != audit_info["anchors"][str(fold)]["train_start_rng_sha256"]:
            raise AssertionError("Training RNG differs from historical train_start")
        image_params = [p for p in loaded.core.img_unet.parameters() if p.requires_grad]
        ph_params = [p for p in loaded.core.ph_unet.parameters() if p.requires_grad]
        rp_params = [p for p in loaded.core.rp_attention.parameters() if p.requires_grad]
        mm_params = [p for p in loaded.core.mm_attention.parameters() if p.requires_grad]
        classifier_params = [p for p in loaded.core.clf.parameters() if p.requires_grad]
        parameter_names = list(dict(loaded.core.named_parameters()))
        parameter_count = sum(p.numel() for p in loaded.core.parameters())
        initial_bn_batches = int(loaded.core.clf[2].num_batches_tracked)
        history, best, best_loss = [], None, float("inf")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        train_start = time.perf_counter()

        for epoch in range(CFG["epochs"]):
            loaded.core.train()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits, graph_loss = loaded.forward()
            native_ce = F.cross_entropy(logits[loaded.train], loaded.y[loaded.train])
            native = native_ce + graph_loss
            value = objective(loaded.z_img, loaded.z_ph, group)
            image_scale, image_budget = gradient_budget(native, value["image_aux"],
                                                        loaded.z_img, epoch, CFG["image_gradient_budget"])
            if value["ph_ce"] is not None:
                ph_scale, ph_budget = gradient_budget(native, value["ph_ce"], loaded.z_ph,
                                                       epoch, CFG["phenotype_gradient_budget"])
            else:
                ph_scale = native.new_zeros(())
                ph_budget = {"native_gradient_norm": None, "auxiliary_gradient_norm": None,
                             "lambda": 0.0, "weighted_gradient_ratio": 0.0,
                             "gradient_cosine": None, "lambda_cap_hit": False,
                             "native_gradient_near_zero": False}
            total = native + image_scale * value["image_aux"]
            if value["ph_ce"] is not None:
                total = total + ph_scale * value["ph_ce"]
            for name, item in (("native", native), ("image_d4", value["image_d4"]),
                               ("pair", value["pair_loss"]), ("image_aux", value["image_aux"]),
                               ("total", total)):
                finite_scalar(item, name)
            if value["ph_ce"] is not None:
                finite_scalar(value["ph_ce"], "phCE")

            sample = epoch < 3 or epoch % CFG["diagnostic_frequency"] == 0
            record = {"epoch": epoch, "native_ce": float(native_ce.detach()),
                      "native_graph": float(graph_loss.detach()), "native_total": float(native.detach()),
                      "image_d4_ce": float(value["image_d4"].detach()),
                      "residual_ce": float(value["pair_loss"].detach()),
                      "pair_ce": float(value["pair_loss"].detach()),
                      "image_aux": float(value["image_aux"].detach()),
                      "phenotype_real_ce": None if value["ph_ce"] is None else float(value["ph_ce"].detach()),
                      "lambda_img": float(image_scale), "lambda_ph": float(ph_scale),
                      "image_gradient_budget": image_budget, "phenotype_gradient_budget": ph_budget,
                      "train_total": float(total.detach())}
            if sample:
                record["residual_mechanism"] = residual_diagnostics(value, loaded.prepared)
                g_native_img = torch.autograd.grad(native, loaded.z_img, retain_graph=True, allow_unused=True)[0]
                g_d4 = torch.autograd.grad(value["image_d4"], loaded.z_img, retain_graph=True, allow_unused=True)[0]
                g_pair = torch.autograd.grad(value["pair_loss"], loaded.z_img, retain_graph=True, allow_unused=True)[0]
                g_aux = torch.autograd.grad(value["image_aux"], loaded.z_img, retain_graph=True, allow_unused=True)[0]
                g_native_ph = torch.autograd.grad(native, loaded.z_ph, retain_graph=True, allow_unused=True)[0]
                g_phce = None if value["ph_ce"] is None else torch.autograd.grad(
                    value["ph_ce"], loaded.z_ph, retain_graph=True, allow_unused=True)[0]
                def norm_or_zero(grad):
                    return 0.0 if grad is None else float(grad.detach().norm())
                def cosine(a, b):
                    if a is None or b is None:
                        return None
                    denom = a.detach().norm() * b.detach().norm()
                    return float((a.detach() * b.detach()).sum() / denom.clamp_min(CFG["gradient_epsilon"]))
                record["embedding_gradient_diagnostics"] = {
                    "img_native_norm": norm_or_zero(g_native_img), "img_d4_norm": norm_or_zero(g_d4),
                    "img_pair_norm": norm_or_zero(g_pair), "img_combined_aux_norm": norm_or_zero(g_aux),
                    "img_d4_residual_cosine": cosine(g_d4, g_pair),
                    "img_native_d4_cosine": cosine(g_native_img, g_d4),
                    "img_native_pair_cosine": cosine(g_native_img, g_pair),
                    "img_native_aux_cosine": cosine(g_native_img, g_aux),
                    "ph_native_norm": norm_or_zero(g_native_ph), "ph_ce_norm": norm_or_zero(g_phce),
                    "pair_direct_ph_gradient_norm": 0.0}
            total.backward()
            if sample:
                def parameter_grad_norm(params):
                    squares = [p.grad.detach().square().sum() for p in params if p.grad is not None]
                    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0
                record["total_parameter_gradient_norms"] = {
                    "img_unet": parameter_grad_norm(image_params),
                    "ph_unet": parameter_grad_norm(ph_params),
                    "rp_attention": parameter_grad_norm(rp_params),
                    "mm_attention": parameter_grad_norm(mm_params),
                    "classifier": parameter_grad_norm(classifier_params)}
                for name, param in loaded.core.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise FloatingPointError(f"Nonfinite model gradient: {name}")
            optimizer.step()
            if int(loaded.core.clf[2].num_batches_tracked) != initial_bn_batches + epoch + 1:
                raise AssertionError("Native classifier BatchNorm update count changed")
            loaded.core.eval()
            with torch.set_grad_enabled(False):
                val_logits, val_graph = loaded.forward()
            val_ce = F.cross_entropy(val_logits[loaded.val], loaded.y[loaded.val])
            val_loss = val_ce + val_graph
            finite_scalar(val_loss, "validation loss")
            if int(loaded.core.clf[2].num_batches_tracked) != initial_bn_batches + epoch + 1:
                raise AssertionError("Validation changed classifier BatchNorm state")
            record.update({"val_ce": float(val_ce), "val_graph": float(val_graph),
                           "val_loss": float(val_loss)})
            history.append(record)
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best = {"core": export_core(loaded.core), "epoch": epoch,
                        "val_loss": best_loss, "record": record.copy()}
            stopper(val_loss, loaded.core)
            if epoch % 50 == 0:
                event("progress", epoch=epoch, val_loss=float(val_loss),
                      lambda_img=float(image_scale), lambda_ph=float(ph_scale))
            if stopper.early_stop:
                break

        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - train_start
        if best is None:
            raise AssertionError("Training completed without a valid checkpoint")
        train_end_rng = capture_rng()
        import_core(loaded.core, best["core"])
        loaded.core.eval()
        if payload_hash(capture_rng()) != payload_hash(train_end_rng):
            raise AssertionError("Restoring best core changed train-end RNG")
        test_start_rng = capture_rng()
        with torch.set_grad_enabled(True):
            test_logits, _ = loaded.forward()
        test_end_rng = capture_rng()
        if loaded.forward_count != 2 * len(history) + 1:
            raise AssertionError("Unexpected native core forward count")
        indices = loaded.split["test"]
        native_logits = test_logits.detach()
        raw_logits = native_logits.cpu().numpy()[indices]
        probability = torch.softmax(test_logits.detach(), dim=1).cpu().numpy()[indices, 0]
        prediction = native_logits.argmax(dim=1).cpu().numpy()[indices]
        labels = loaded.ctx["labels"][indices]
        oof = {"indices": indices.copy(), "ids": [loaded.ctx["ids"][int(i)] for i in indices],
               "labels": labels.copy(), "logits": raw_logits,
               "probability": probability, "prediction": prediction}
        result = {
            "state": "complete", "group": group, "seed": seed, "fold": fold,
            "started_utc": task_started_utc, "completed_utc": utc(),
            "training_code_sha256": audit_info["training_code_sha256"],
            "manifest_sha256": audit_info["manifest_sha256"],
            "input_sha256": audit_info["anchors"][str(fold)]["sha256"],
            "input_digest": loaded.input_digest, "prototype_preparation": loaded.prepared["info"],
            "initialization_sha256": loaded.anchor_record["train_start_core_sha256"],
            "initial_core_sha256": core_hash(loaded.start["core"]),
            "best_core_change_from_start": core_change(loaded.start["core"], best["core"]),
            "start_rng_fingerprint": rng_hash(loaded.start["rng"]),
            "training_rng_fingerprint": rng_hash(train_end_rng),
            "test_rng_fingerprint": rng_hash(test_start_rng),
            "rng_before_test": rng_hash(test_start_rng), "rng_after_test": rng_hash(test_end_rng),
            "best_epoch": best["epoch"], "best_val_loss": best_loss,
            "best_model_state": best["core"], "best_core_sha256": core_hash(best["core"]),
            "best_record": best["record"], "epochs": history, "epochs_run": len(history),
            "oof": oof, "metrics": formal_metrics(labels, probability, prediction),
            "probability_cpu_recalc_max_abs_diff": float(np.max(np.abs(
                torch.softmax(torch.as_tensor(raw_logits), dim=1).numpy()[:, 0] - probability))),
            "training_seconds": training_seconds, "task_seconds": time.perf_counter() - task_started,
            "native_core_forwards": loaded.forward_count, "formal_test_forwards": 1,
            "formal_test_grad_enabled": True, "new_trainable_parameters": 0,
            "rfe_refit_count": 0, "vae_pretrain_count": 0,
            "physical_gpu": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_parameter_names": parameter_names, "initial_parameter_count": parameter_count,
            "best_bn_batches": int(best["core"]["state"]["clf.2.num_batches_tracked"]),
            "forward_path": "native fusion classifier only; no auxiliary objective at validation or test",
            "uses_test_labels_for_training_or_selection": False}
        atomic_torch(result, path)
        valid_result(group, seed, fold, audit_info, required=True)
        atomic_json({"state": "complete", "group": group, "seed": seed, "fold": fold,
                     "completed_utc": utc(), "artifact_sha256": sha_file(path),
                     "training_seconds": training_seconds, "best_epoch": best["epoch"]}, status_path)
        event("complete", epochs=len(history), best_epoch=best["epoch"],
              training_seconds=training_seconds, artifact_sha256=sha_file(path))
    except Exception as error:
        atomic_json({"state": "failed", "group": group, "seed": seed, "fold": fold,
                     "failed_utc": utc(), "error": repr(error), "traceback": traceback.format_exc()}, status_path)
        append_log("worker_failed", group=group, seed=seed, fold=fold,
                   error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        loaded.close()


def math_action():
    record = audit()
    result = math_tests()
    record["mathematical_tests"] = result
    atomic_json(record, BASE / "audit.json")
    append_log("mathematical_tests_passed", checks=len(result["checks"]))
    return result


def preflight_action(seed: int, fold: int):
    record = audit()
    if record.get("mathematical_tests", {}).get("passed") is not True:
        raise AssertionError("Pure mathematical tests must pass before real-data preflight")
    if (record.get("preflight") or {}).get("passed") is True:
        return record["preflight"]
    if seed != 911 or fold != 0:
        raise ValueError("Registered real-data preflight is seed 911 fold 0 only")
    try:
        result = preflight_one(seed, fold, record)
    except Exception as error:
        record.setdefault("preflight_attempts", []).append({"recorded_utc": utc(),
            "status": "failed_before_formal_training", "error": repr(error),
            "traceback": traceback.format_exc(), "core_forwards": getattr(error, "core_forwards", None),
            "formal_tasks_started": False})
        atomic_json(record, BASE / "audit.json")
        append_log("real_forward_preflight_failed", seed=seed, fold=fold,
                   error=repr(error), formal_tasks_started=False)
        raise
    record["preflight"] = result
    atomic_json(record, BASE / "audit.json")
    append_log("real_forward_preflight_passed", seed=seed, fold=fold,
               groups=list(result["groups"]), forward_count=result["forward_count"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "math-test", "preflight", "freeze", "train"))
    parser.add_argument("--group", choices=list(CFG["groups"]))
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.action == "audit":
        audit()
        return
    if args.action == "math-test":
        math_action()
        return
    if args.action == "preflight":
        preflight_action(args.seed, args.fold)
        return
    if args.action == "freeze":
        freeze_manifest(audit())
        return
    if args.group is None or args.fold is None or not 0 <= args.fold < 10:
        parser.error("train requires --group and --fold in [0,9]")
    record = guard(require_manifest=True)
    lock_path = result_path(args.group, args.seed, args.fold).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Task already has an active worker: {args.group}/s{args.seed}_f{args.fold}") from error
        try:
            train_one(args.group, args.seed, args.fold, record)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
