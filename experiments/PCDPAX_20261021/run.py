"""Audited PCDPA-X training, one-forward preflight, and result persistence."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
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
ROOT = BASE.parents[1]
HISTORY = ROOT / "experiments/DRPA_20261014"
PCDCA = ROOT / "experiments/PCDCA_20261019"
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from torch.nn import functional as F

from alignment import CFG, HISTORICAL_CFG, PCDPAXObjective, PrototypeObjective, prepare
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from opt import OptInit
from utils.tools import EarlyStopping


SOURCE = (
    "opt.py", "train_mm_gtunets.py", "model/mm_gtunets.py", "model/gtunet.py",
    "model/rp_graph.py", "utils/mydataloader.py", "utils/tools.py", "utils/metrics.py",
)
TRAIN_CODE = ("config.json", "alignment.py", "run.py", "launch.py")
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
    return {
        "state": {key: value.detach().cpu().clone() for key, value in core.state_dict().items()},
        "rp_weights": core.rp_attention.weights.detach().cpu().clone(),
    }


def core_hash(saved: dict) -> str:
    return state_hash({**saved["state"], "unregistered_rp_weights": saved["rp_weights"]})


def import_core(core, saved: dict):
    core.load_state_dict(saved["state"], strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(saved["rp_weights"].to(core.device))


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state(0).clone(),
        "backend": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


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
    return {
        "python": sha_bytes(repr(state["python"]).encode()),
        "numpy": sha_bytes(repr(np_state[0]).encode() + np_state[1].tobytes()
                            + repr(np_state[2:]).encode()),
        "cpu": array_hash(state["cpu"].numpy()),
        "cuda": array_hash(state["cuda"].numpy()),
    }


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
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    atomic_bytes(data, path)


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
    record = {"utc": utc(), "event": event, **fields}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    path = BASE / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with (BASE / "run.lock").open("a+") as lock, path.open("a", encoding="utf-8") as stream:
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
        raise AssertionError("Unexpected ASD/TD class counts")
    if not np.allclose(age, ctx["ph_data"][:, 2], rtol=0, atol=2e-6):
        raise AssertionError("CSV and graph age values differ")
    if not np.array_equal(sex - 1, ctx["ph_data"][:, 1]):
        raise AssertionError("CSV and graph sex values differ")
    if not np.array_equal(site, ctx["site_names"]) or len(set(site)) != 20:
        raise AssertionError("CSV and graph site order differ")
    if [(int(np.sum((labels == cls) & (sex == 2))), int(np.sum((labels == cls) & (sex == 1))))
            for cls in (0, 1)] != [(54, 349), (90, 378)]:
        raise AssertionError("Sex coding does not match the validated cohort counts")
    return {"age": age, "sex": sex, "site": site}


def audit() -> dict:
    path = BASE / "audit.json"
    if path.exists():
        saved = read_json(path)
        if saved["training_code_sha256"] != code_hash():
            if saved.get("formal_authorized") or saved.get("manifest_sha256"):
                raise AssertionError("Training files changed after formal freeze")
            saved.setdefault("preflight_attempts", [])
            if not saved["preflight_attempts"] and saved.get("preflight") is None:
                saved["preflight_attempts"].append({
                    "recorded_utc": utc(), "status": "failed_before_formal_training",
                    "error": "NameError in PCDPAXObjective lazy construction of M2 global prototype; corrected in alignment.py",
                    "core_forwards": 1, "formal_tasks_started": False,
                })
            if isinstance(saved.get("preflight"), dict) and saved["preflight"].get("passed"):
                saved["preflight_attempts"].append({
                    "recorded_utc": utc(), "status": "passed_superseded_by_training_code_change",
                    "training_code_sha256": saved["training_code_sha256"],
                    "core_forwards": saved["preflight"].get("forward_count"),
                    "formal_tasks_started": False,
                })
            code_files = {name: sha_file(BASE / name) for name in TRAIN_CODE}
            saved["training_code_sha256"] = json_digest(code_files)
            saved["training_code_files_sha256"] = code_files
            saved["mathematical_tests"] = None
            saved["preflight"] = None
            atomic_json(saved, path)
        return saved

    historical_audit = read_json(PCDCA / "audit.json")
    current_sources = source_hashes()
    for name, expected in historical_audit["source_sha256"].items():
        if current_sources.get(name) != expected:
            raise AssertionError(f"Original model source changed relative to accepted PCDCA audit: {name}")
    historical_proto_hash = sha_file(PCDCA / "prototype.py")
    if historical_audit["code_sha256"] != json_digest({
            name: sha_file(PCDCA / name)
            for name in ("config.json", "prototype.py", "run.py", "launch.py", "analyze.py")}):
        raise AssertionError("Historical PCDCA files changed since their accepted audit")

    ctx_path = HISTORY / "artifacts/context.pt"
    ctx = load_torch(ctx_path)
    metadata = read_metadata(ctx)
    if len(ctx["ids"]) != 871 or tuple(ctx["ph_data"].shape) != (871, 3):
        raise AssertionError("Unexpected cohort size or non-imaging input dimension")
    features_path = anchor(911, 0).parent / "selected_features.pt"
    features = load_torch(features_path)
    support_path = anchor(911, 0).parent / "rfe_support.npy"
    support = np.load(support_path)
    raw_path = ROOT / CFG["data_folder"] / "save_tensor/up_triangles.pt"
    raw = load_torch(raw_path)
    if tuple(features.shape) != (871, 500) or support.shape != (500,):
        raise AssertionError("Historical RFE output has an unexpected shape")
    if not torch.equal(torch.as_tensor(raw[:, support], dtype=features.dtype), features):
        raise AssertionError("Cached historical RFE features do not match selected indices")

    counts = np.zeros(871, dtype=np.int64)
    anchor_records = {}
    for fold in range(10):
        directory = anchor(911, fold)
        split_path = directory / "splits.pt"
        start_path = directory / "train_start.pt"
        fingerprint_path = directory / "fingerprints.json"
        vae_path = directory.parent / "vae" / f"fold{fold}_pretrain.pth"
        split = load_torch(split_path)
        start = load_torch(start_path)
        fp = read_json(fingerprint_path)
        union = np.concatenate([split[key] for key in SUBSETS])
        if len(union) != 871 or not np.array_equal(np.sort(union), np.arange(871)):
            raise AssertionError(f"Invalid historical split at fold {fold}")
        counts[split["test"]] += 1
        for subset in SUBSETS:
            if array_hash(split[subset]) != fp["split_fingerprints"][subset]:
                raise AssertionError(f"Split fingerprint mismatch at fold {fold}/{subset}")
        if core_hash(start["core"]) != fp["train_start_core_fingerprint"]:
            raise AssertionError(f"train_start core fingerprint mismatch at fold {fold}")
        if rng_hash(start["rng"]) != fp["classifier_start_rng"]:
            raise AssertionError(f"train_start RNG fingerprint mismatch at fold {fold}")
        if sha_file(vae_path) != start["vae_sha256"] or sha_file(vae_path) != fp["vae_sha256"]:
            raise AssertionError(f"VAE checkpoint mismatch at fold {fold}")
        if array_hash(features.numpy()) != fp["features_fingerprint"]:
            raise AssertionError(f"Selected feature fingerprint mismatch at fold {fold}")
        prep = prepare(ctx["labels"], metadata, split["train"], 911, fold, CFG["prototype"])
        expected = historical_audit["anchors"][f"911/{fold}"]
        if json_digest(prep["info"]) != json_digest(expected["prototype"]):
            raise AssertionError(f"Private 3-fold prototype preparation differs at fold {fold}")
        files = [split_path, start_path, fingerprint_path, features_path, support_path, vae_path]
        hashes = {str(item): sha_file(item) for item in files}
        for item, digest in hashes.items():
            if expected["input_sha256"].get(item) != digest:
                raise AssertionError(f"Historical input SHA mismatch: {item}")
        anchor_records[str(fold)] = {
            "paths": {"split": str(split_path), "train_start": str(start_path),
                      "fingerprints": str(fingerprint_path), "selected_features": str(features_path),
                      "rfe_support": str(support_path), "vae": str(vae_path)},
            "sha256": hashes,
            "train_start_core_sha256": core_hash(start["core"]),
            "train_start_rng_sha256": rng_hash(start["rng"]),
            "split_fingerprints": fp["split_fingerprints"],
            "prototype_preparation_sha256": json_digest(prep["info"]),
            "gpu": fold % 2,
        }
    if not np.all(counts == 1):
        raise AssertionError("Historical ten-fold test splits do not cover each subject exactly once")

    handoff = ROOT / "handoff.md"
    code_files = {name: sha_file(BASE / name) for name in TRAIN_CODE}
    audit_record = {
        "experiment": CFG["experiment"], "created_utc": utc(),
        "training_code_sha256": json_digest(code_files), "training_code_files_sha256": code_files,
        "original_source_sha256": current_sources,
        "historical_pcdca_prototype_sha256": historical_proto_hash,
        "historical_pcdca_code_sha256": historical_audit["code_sha256"],
        "historical_pcdca_audit_sha256": sha_file(PCDCA / "audit.json"),
        "context_sha256": sha_file(ctx_path), "raw_feature_tensor_sha256": sha_file(raw_path),
        "cohort": {"subjects": 871, "ASD_label": 0, "TD_label": 1,
                   "class_counts": [403, 468], "non_imaging_fields": ["SITE_ID", "SEX", "AGE_AT_SCAN"],
                   "non_imaging_tensor_shape": [871, 3], "RFE_shape": [871, 500], "sites": 20},
        "metadata": {"age_range": [float(np.nanmin(metadata["age"])), float(np.nanmax(metadata["age"]))],
                     "csv_sex_coding": "1=male, 2=female; verified against cohort counts"},
        "anchors": anchor_records,
        "history_policy": {"D0_retrained": False, "D4_retrained": False,
                           "other_historical_experiments_retrained": False,
                           "RFE_refit": False, "VAE_pretrained": False,
                           "split_created": False, "historical_weights_read_only": True},
        "handoff_before": {"path": str(handoff), "bytes": handoff.stat().st_size,
                           "sha256": sha_file(handoff)},
        "preflight_attempts": [],
        "paper": "https://arxiv.org/html/2406.14455v3",
        "repository": "https://github.com/tangaii/MM",
        "git_commit_at_audit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "formal_authorized": False, "preflight": None, "manifest_sha256": None,
        "execution": {"tasks": 40, "gpu0": 20, "gpu1": 20, "completed": 0, "failures": [],
                      "retries": 0, "queue_wall_seconds": None, "worker_seconds": None},
    }
    atomic_json(audit_record, path)
    append_log("audit_passed", anchors=10, code_sha256=audit_record["training_code_sha256"])
    return audit_record


def guard(require_manifest: bool = True) -> dict:
    record = read_json(BASE / "audit.json")
    if record["training_code_sha256"] != code_hash():
        raise AssertionError("Frozen training code fingerprint changed")
    for name, digest in record["original_source_sha256"].items():
        if sha_file(ROOT / name) != digest:
            raise AssertionError(f"Protected original source changed: {name}")
    for fold, item in record["anchors"].items():
        for path, digest in item["sha256"].items():
            if sha_file(path) != digest:
                raise AssertionError(f"Protected historical input changed: {path}")
    if require_manifest:
        manifest_path = BASE / "manifest.json"
        if not manifest_path.exists() or sha_file(manifest_path) != record["manifest_sha256"]:
            raise AssertionError("Frozen manifest is missing or changed")
        manifest = read_json(manifest_path)
        if manifest["training_code_sha256"] != code_hash() or not manifest["preflight_passed"]:
            raise AssertionError("Frozen manifest does not match the validated training code")
    return record


def math_tests() -> dict:
    rng = np.random.default_rng(20261021)
    n = 72
    labels = np.asarray([0, 1] * (n // 2), dtype=np.int64)
    site = np.asarray([f"S{i % 6}" for i in range(n)])
    age = 8 + (np.arange(n) * 7 % 31).astype(float)
    sex = np.where(np.arange(n) % 3 == 0, 1.0, 2.0)
    age[[3, 41]] = np.nan
    sex[[8, 36]] = -1
    meta = {"age": age, "sex": sex, "site": site}
    train = np.arange(60, dtype=np.int64)
    prepared = prepare(labels, meta, train, 911, 0, CFG["prototype"])
    device = torch.device("cpu")
    z_img = torch.tensor(rng.normal(size=(n, 16)), dtype=torch.float32, requires_grad=True)
    z_ph = torch.tensor(rng.normal(size=(n, 16)) * 1.7 + .3, dtype=torch.float32, requires_grad=True)

    d4 = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
    d1 = PrototypeObjective(prepared, "D1", HISTORICAL_CFG, device)
    d5 = PrototypeObjective(prepared, "D5", HISTORICAL_CFG, device)
    age_tensor = torch.as_tensor(prepared["age"], dtype=torch.float32)
    expected_age = torch.exp(-.5 * ((age_tensor[:, None]
                                     - age_tensor) / prepared["bandwidth"]).square())
    sex_tensor = torch.as_tensor(prepared["sex"])
    known = (sex_tensor[:, None] > 0) & (sex_tensor > 0)
    expected_sex = torch.where(known, torch.where(sex_tensor[:, None] == sex_tensor, 1., .5), .75)
    checks = {
        "ASD_TD_class_order": [0, 1],
        "age_kernel_elementwise": bool(torch.allclose(d4.kernel.cpu(), (expected_age * expected_sex).float(), atol=0, rtol=0)),
        "sex_kernel_elementwise": bool(torch.allclose((d4.kernel / expected_age).cpu(), expected_sex.float(), atol=1e-6, rtol=1e-6)),
        "global_kernel_is_unconditioned": bool(torch.equal(d1.kernel, torch.ones_like(d1.kernel))),
        "private_three_fold_mapping": bool(np.array_equal(prepared["folds"],
              prepare(labels, meta, train, 911, 0, CFG["prototype"])["folds"])),
        "no_self_reference": bool(not d4.reference.diagonal().any()),
        "no_same_private_fold_reference": bool(not (d4.reference & (d4.pf[:, None] == d4.pf[None, :])).any()),
        "heldout_not_reference": bool(torch.equal(d4.reference[:, 60:], torch.zeros_like(d4.reference[:, 60:]))),
    }
    loss_img, logits_img, detail_img, _ = __import__("alignment").image_d4_objective(z_img, prepared, device)
    loss_ph, logits_ph, detail_ph, _ = __import__("alignment").phenotype_prototype_objective(z_ph, prepared, "real", device)
    _, detail_global = d1(z_ph)
    _, detail_shuffle = d5(z_ph)
    global_logits = __import__("alignment").live_logits(z_ph, detail_global, d1.train)
    shuffle_logits = __import__("alignment").live_logits(z_ph, detail_shuffle, d5.train)
    y = torch.as_tensor(prepared["y"], dtype=torch.long)
    checks.update({
        "D4_CE_equals_historical_CE": bool(torch.allclose(loss_img, F.cross_entropy(logits_img, y), atol=1e-7, rtol=1e-6)),
        "phenotype_uses_own_mean_std_tau": bool(torch.allclose(detail_ph["mean"], z_ph[torch.as_tensor(train)].detach().mean(0))),
        "phenotype_tau_independent": bool(torch.allclose(detail_ph["tau"], ((detail_ph["teacher"] - detail_ph["teacher"].mean(0)).square().sum(-1).mean()).clamp_min(1.0))),
        "global_prototype_unconditioned": bool(torch.allclose(detail_global["mu"], detail_global["global_mu"])),
        "real_vs_shuffled_prototype_differs": bool(float((detail_ph["mu"] - detail_shuffle["mu"]).abs().max()) > 1e-6),
        "student_global_logits_class_order": tuple(global_logits.shape) == (60, 2),
        "student_shuffle_logits_class_order": tuple(shuffle_logits.shape) == (60, 2),
        "all_prototype_statistics_stop_gradient": all(not item.requires_grad for item in
              (detail_img["mean"], detail_img["std"], detail_img["tau"], detail_img["mu"],
               detail_ph["mean"], detail_ph["std"], detail_ph["tau"], detail_ph["mu"])),
    })
    align = __import__("alignment").semantic_kl(logits_ph, logits_img)
    g_img, g_ph = torch.autograd.grad(align, (z_img, z_ph), retain_graph=True, allow_unused=True)
    same = __import__("alignment").semantic_kl(logits_img, logits_img)
    checks.update({
        "KL_nonnegative": bool(float(align.detach()) >= -1e-7),
        "KL_identical_distributions_zero": bool(float(same.detach().abs()) < 1e-7),
        "KL_teacher_stopped": g_img is None or bool(torch.count_nonzero(g_img) == 0),
        "KL_student_gradient_nonzero": g_ph is not None and float(g_ph.norm()) > 1e-8,
    })
    combined = loss_ph + CFG["gamma"] * align
    weight, stats = __import__("alignment").gradient_budget(
        loss_img + loss_ph, combined, z_ph, epoch=9, budget=CFG["phenotype_gradient_budget"])
    checks.update({
        "gradient_scale_finite_and_capped": bool(torch.isfinite(weight) and 0 <= float(weight) <= CFG["lambda_cap"]),
        "gradient_scale_within_budget_for_test_case": bool(stats["weighted_auxiliary_z_gradient_ratio"] <= .050001),
        "labels_are_ASD0_TD1": bool(labels[0] == 0 and labels[1] == 1),
    })

    altered_labels = labels.copy()
    altered_labels[60:] = 1 - altered_labels[60:]
    altered_meta = {key: value.copy() for key, value in meta.items()}
    altered_meta["age"][60:] = 99
    altered_meta["sex"][60:] = -1
    altered_meta["site"][60:] = "HELDOUT_CHANGED"
    changed = prepare(altered_labels, altered_meta, train, 911, 0, CFG["prototype"])
    changed_obj = PrototypeObjective(changed, "D4", HISTORICAL_CFG, device)
    old_loss, old_detail = d4(z_img)
    new_z = z_img.detach().clone()
    new_z[60:] = 10000
    new_loss, new_detail = changed_obj(new_z)
    checks["heldout_metadata_labels_and_embeddings_invariant"] = (
        payload_hash(old_loss.detach()) == payload_hash(new_loss.detach())
        and payload_hash(old_detail["mu"]) == payload_hash(new_detail["mu"])
    )
    failed = [key for key, value in checks.items() if value is False]
    result = {"passed": not failed, "checks": checks, "failed": failed,
              "synthetic_KL": float(align.detach()), "gradient_scale": stats}
    if failed:
        raise AssertionError(f"Mathematical preflight failed: {failed}")
    return result


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
            sys.argv = ["pcdpa_x_options"]
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
            raise AssertionError("Loaded core differs from the historical train_start")
        self.x_cpu = load_torch(anchor(seed, fold).parent / "selected_features.pt").float()
        self.ph_cpu = torch.as_tensor(self.ctx["ph_data"], dtype=torch.float32)
        affinity = create_reward_penalty_graph(
            self.ctx["ph_dict"], self.ctx["labels"],
            self.split["train"], self.split["val"], self.split["test"], options,
        )
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
            raise AssertionError("Loaded D4 prototype preparation differs from historical audit")
        self.z_img = None
        self.z_ph = None
        self.forward_count = 0
        self.img_hook = self.core.img_unet.register_forward_hook(self.capture_img)
        self.ph_hook = self.core.ph_unet.register_forward_hook(self.capture_ph)
        self.input_digest = payload_hash({"x": self.x_cpu, "ph": self.ph_cpu,
                                          "affinity": self.affinity_cpu, "labels": self.ctx["labels"],
                                          "ids": self.ctx["ids"]})

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
            raise AssertionError("Historical RP_Attention weights unexpectedly became registered")
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


def gradient_norm_for_params(loss: torch.Tensor, params) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    square = [grad.detach().square().sum() for grad in grads if grad is not None]
    return float(torch.stack(square).sum().sqrt()) if square else 0.0


def model_gradient_norm(params) -> float:
    square = [param.grad.detach().square().sum() for param in params if param.grad is not None]
    return float(torch.stack(square).sum().sqrt()) if square else 0.0


def formal_metrics(labels, probability, prediction) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.int64)
    return {
        "ACC": float(accuracy_score(labels, prediction)),
        "SEN-ASD": float(recall_score(labels == 0, prediction == 0, pos_label=True)),
        "SPE-TD": float(recall_score(labels == 1, prediction == 1, pos_label=True)),
        "probability_AUC": float(roc_auc_score(labels == 0, probability)),
    }


def valid_result(group: str, seed: int, fold: int, audit_info: dict, required=False):
    path = result_path(group, seed, fold)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    value = load_torch(path)
    if value.get("state") != "complete" or (value.get("group"), value.get("seed"), value.get("fold")) != (group, seed, fold):
        raise AssertionError(f"Incomplete or mismatched result artifact: {path}")
    if value["training_code_sha256"] != audit_info["training_code_sha256"]:
        raise AssertionError(f"Frozen code fingerprint differs: {path}")
    if value["input_sha256"] != audit_info["anchors"][str(fold)]["sha256"]:
        raise AssertionError(f"Historical input fingerprint differs: {path}")
    if value["best_core_sha256"] != core_hash(value["best_model_state"]):
        raise AssertionError(f"Best core fingerprint differs: {path}")
    split = load_torch(anchor(seed, fold) / "splits.pt")
    oof = value["oof"]
    if not np.array_equal(oof["indices"], split["test"]):
        raise AssertionError(f"OOF test indices differ from frozen split: {path}")
    expected_probability = torch.softmax(torch.as_tensor(oof["logits"]), dim=1).numpy()[:, 0]
    if not np.allclose(expected_probability, oof["probability"], rtol=0, atol=1e-7):
        raise AssertionError(f"Stored ASD probabilities do not match logits: {path}")
    metrics = formal_metrics(oof["labels"], oof["probability"], oof["prediction"])
    if max(abs(metrics[key] - value["metrics"][key]) for key in metrics) > 1e-12:
        raise AssertionError(f"Stored four metrics do not recompute: {path}")
    return value


def _finite_scalar(value, name):
    if not bool(torch.isfinite(value)):
        raise FloatingPointError(f"Nonfinite {name}")


def summarize_prototype(detail: dict, logits: torch.Tensor, labels: torch.Tensor) -> dict:
    ess_min = detail["ess"].min(-1).values.detach().float()
    beta = detail["beta"].detach().float()
    delta = (detail["mu"][:, 0] - detail["mu"][:, 1]).detach()
    probs = torch.softmax(logits.detach(), dim=1)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(1)
    return {
        "tau": float(detail["tau"]),
        "ess_min_mean": float(ess_min.mean()),
        "ess_min_p10": float(torch.quantile(ess_min, .1)),
        "beta_mean": float(beta.mean()),
        "beta_zero_fraction": float((beta == 0).float().mean()),
        "valid_fraction": float(detail["valid"].float().mean()),
        "class_prototype_distance_mean": float(delta.norm(dim=1).mean()),
        "class_delta_query_sd_l2": float(delta.std(0, unbiased=False).norm()),
        "prototype_train_accuracy": float((logits.detach().argmax(1) == labels).float().mean()),
        "evidence_entropy_mean": float(entropy.mean()),
    }


def alignment_diagnostics(values: dict, labels: torch.Tensor) -> dict:
    teacher = torch.softmax(values["image_logits"].detach(), dim=1)
    student = torch.softmax(values["student_logits"].detach(), dim=1)
    y = labels
    teacher_entropy = -(teacher.clamp_min(1e-12) * teacher.clamp_min(1e-12).log()).sum(1)
    student_entropy = -(student.clamp_min(1e-12) * student.clamp_min(1e-12).log()).sum(1)
    return {
        "kl": float(values["align_loss"].detach()),
        "teacher_entropy": float(teacher_entropy.mean()),
        "student_entropy": float(student_entropy.mean()),
        "teacher_student_argmax_agreement": float((teacher.argmax(1) == student.argmax(1)).float().mean()),
        "teacher_train_accuracy": float((teacher.argmax(1) == y).float().mean()),
        "student_train_accuracy": float((student.argmax(1) == y).float().mean()),
        "teacher_asd_probability_mean": float(teacher[:, 0].mean()),
        "student_asd_probability_mean": float(student[:, 0].mean()),
    }


def preflight_one(seed: int, fold: int, audit_info: dict) -> dict:
    loaded = Loaded(seed, fold, audit_info)
    try:
        loaded.reset()
        initial_params = {id(param) for param in loaded.core.parameters()}
        input_hash_before = payload_hash({"x": loaded.x, "ph": loaded.ph, "affinity": loaded.affinity})
        loaded.core.train()
        with torch.set_grad_enabled(True):
            logits, graph = loaded.forward()
        if loaded.forward_count != 1:
            raise AssertionError("Real-data preflight must perform exactly one core forward")
        if tuple(logits.shape) != (871, 2) or tuple(loaded.z_img.shape) != (871, 16) or tuple(loaded.z_ph.shape) != (871, 16):
            raise AssertionError("Unexpected logits or branch embedding dimensions")
        if payload_hash({"x": loaded.x, "ph": loaded.ph, "affinity": loaded.affinity}) != input_hash_before:
            raise AssertionError("Native forward mutated model inputs")
        native = F.cross_entropy(logits[loaded.train], loaded.y[loaded.train]) + graph
        objective = PCDPAXObjective(loaded.prepared, loaded.options.device)
        records = {}
        outputs = {}
        for group in CFG["groups"]:
            value = objective(loaded.z_img, loaded.z_ph, group)
            outputs[group] = value
            for name in ("image_loss", "ph_ce", "ph_aux"):
                _finite_scalar(value[name], f"{group}/{name}")
            if value["align_loss"] is not None:
                _finite_scalar(value["align_loss"], f"{group}/align_loss")
                if float(value["align_loss"].detach()) < -1e-7:
                    raise AssertionError("KL is negative beyond floating point tolerance")
                g_img, g_ph = torch.autograd.grad(
                    value["align_loss"], (loaded.z_img, loaded.z_ph), retain_graph=True, allow_unused=True)
                image_norm = 0.0 if g_img is None else float(g_img.norm())
                ph_norm = 0.0 if g_ph is None else float(g_ph.norm())
                if image_norm != 0 or ph_norm <= 1e-10:
                    raise AssertionError("KL teacher/student gradient direction is incorrect")
                ph_parameter_norm = gradient_norm_for_params(
                    value["align_loss"], [p for p in loaded.core.ph_unet.parameters() if p.requires_grad])
                image_parameter_norm = gradient_norm_for_params(
                    value["align_loss"], [p for p in loaded.core.img_unet.parameters() if p.requires_grad])
                if ph_parameter_norm <= 1e-10 or image_parameter_norm != 0:
                    raise AssertionError("KL parameter-gradient path does not match the branch contract")
            else:
                g_img = g_ph = None
                image_norm = ph_norm = ph_parameter_norm = image_parameter_norm = None
            records[group] = {
                "image_d4_loss": float(value["image_loss"].detach()),
                "phenotype_real_ce": float(value["ph_ce"].detach()),
                "alignment_kl": None if value["align_loss"] is None else float(value["align_loss"].detach()),
                "teacher_logits_sha256": payload_hash(value["image_logits"].detach()),
                "student_logits_sha256": None if value["student_logits"] is None else payload_hash(value["student_logits"].detach()),
                "kl_img_embedding_gradient_norm": image_norm,
                "kl_ph_embedding_gradient_norm": ph_norm,
                "kl_img_unet_parameter_gradient_norm": image_parameter_norm,
                "kl_ph_unet_parameter_gradient_norm": ph_parameter_norm,
            }
        for group in ("M2", "M3", "M4"):
            if not torch.equal(outputs["M1"]["image_logits"].detach(), outputs[group]["image_logits"].detach()):
                raise AssertionError(f"Teacher differs between M1 and {group}")
            if not torch.allclose(outputs["M1"]["ph_ce"].detach(), outputs[group]["ph_ce"].detach(), atol=1e-7, rtol=1e-7):
                raise AssertionError(f"Shared phenotype CE differs between M1 and {group}")
            if not torch.allclose(outputs["M1"]["image_loss"].detach(), outputs[group]["image_loss"].detach(), atol=1e-7, rtol=1e-7):
                raise AssertionError(f"Shared D4 image objective differs between M1 and {group}")
        if outputs["M3"]["student_logits"].data_ptr() == outputs["M3"]["image_logits"].data_ptr():
            raise AssertionError("M3 teacher and phenotype student must be independently constructed")
        if torch.allclose(outputs["M3"]["student_detail"]["mu"], outputs["M4"]["student_detail"]["mu"], atol=1e-7, rtol=1e-7):
            raise AssertionError("M4 condition shuffle did not change phenotype prototypes")
        if not torch.equal(outputs["M3"]["image_logits"].detach(), outputs["M4"]["image_logits"].detach()):
            raise AssertionError("M3/M4 teacher should be identical")
        if len({id(param) for param in loaded.core.parameters()}) != len(initial_params):
            raise AssertionError("PCDPA-X added trainable model parameters")
        if any(not torch.isfinite(param).all() for param in loaded.core.parameters()):
            raise FloatingPointError("Nonfinite model parameter after preflight forward")
        test_indices = np.setdiff1d(np.arange(871), loaded.split["train"])
        altered_labels = loaded.ctx["labels"].copy()
        altered_labels[test_indices] = 1 - altered_labels[test_indices]
        altered_meta = {key: value.copy() for key, value in loaded.meta.items()}
        altered_meta["age"][test_indices] = 99
        altered_meta["sex"][test_indices] = -1
        altered_meta["site"][test_indices] = "HELDOUT_CHANGED"
        altered_prepared = prepare(altered_labels, altered_meta, loaded.split["train"], seed, fold,
                                   CFG["prototype"])
        if json_digest(altered_prepared["info"]) != json_digest(loaded.prepared["info"]):
            raise AssertionError("Auxiliary preparation depends on heldout labels or metadata")
        altered_z = {key: value.detach().clone() for key, value in
                     (("img", loaded.z_img), ("ph", loaded.z_ph))}
        altered_z["img"][test_indices] = 10000
        altered_z["ph"][test_indices] = -10000
        before_values = objective(loaded.z_img, loaded.z_ph, "M3")
        altered_obj = PCDPAXObjective(altered_prepared, loaded.options.device)
        after_values = altered_obj(altered_z["img"], altered_z["ph"], "M3")
        for key in ("image_loss", "ph_ce", "align_loss"):
            if not torch.equal(before_values[key].detach(), after_values[key].detach()):
                raise AssertionError(f"Auxiliary loss depends on heldout labels/metadata/embeddings: {key}")
        return {
            "passed": True, "seed": seed, "fold": fold, "timestamp_utc": utc(),
            "single_core_forward": True, "forward_count": loaded.forward_count,
            "native_logits_shape": list(logits.shape), "z_img_shape": list(loaded.z_img.shape),
            "z_ph_shape": list(loaded.z_ph.shape), "native_loss": float(native.detach()),
            "groups": records,
            "m3_m4_student_prototype_max_abs_difference": float(
                (outputs["M3"]["student_detail"]["mu"] - outputs["M4"]["student_detail"]["mu"]).abs().max()),
            "input_unchanged": True, "heldout_auxiliary_invariance": True,
            "parameter_count_unchanged": True,
            "original_fusion_path_unchanged": "No model/module/parameter modifications; read-only hooks capture both native GTUNet outputs.",
            "limitations": ["The historical population graph is transductive and includes heldout nodes as in the original protocol."],
            "physical_gpu": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
            "device": torch.cuda.get_device_name(0),
        }
    finally:
        loaded.close()


def freeze_manifest(audit_info: dict) -> dict:
    if not audit_info.get("mathematical_tests", {}).get("passed"):
        raise AssertionError("Mathematical tests must pass before freezing training")
    if not audit_info.get("preflight", {}).get("passed"):
        raise AssertionError("Single-forward real-data preflight must pass before freezing training")
    code_files = {name: sha_file(BASE / name) for name in TRAIN_CODE}
    manifest = {
        "experiment": CFG["experiment"], "name": CFG["name"], "frozen_utc": utc(),
        "training_code_sha256": json_digest(code_files), "training_code_files_sha256": code_files,
        "preflight_passed": True, "preflight_sha256": json_digest(audit_info["preflight"]),
        "math_tests_sha256": json_digest(audit_info["mathematical_tests"]),
        "seed": 911, "folds": list(range(10)), "groups": list(CFG["groups"]),
        "tasks": 40, "gpu0_folds": [0, 2, 4, 6, 8], "gpu1_folds": [1, 3, 5, 7, 9],
        "optimizer": {"name": "Adam", "lr": CFG["lr"], "weight_decay": CFG["weight_decay"]},
        "historical_model_source_sha256": audit_info["original_source_sha256"],
        "historical_inputs_sha256": {fold: item["sha256"] for fold, item in audit_info["anchors"].items()},
        "read_only_prototype_source_sha256": audit_info["historical_pcdca_prototype_sha256"],
        "no_git": True,
    }
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
        raise AssertionError("Formal training is not authorized by a passed preflight")
    loaded = Loaded(seed, fold, audit_info)
    status_path = result_path(group, seed, fold).with_suffix(".status.json")
    worker_log = result_path(group, seed, fold).with_suffix(".jsonl")
    task_started = time.perf_counter()
    task_started_utc = utc()

    def worker_event(event, **fields):
        line = json.dumps({"utc": utc(), "event": event, "group": group,
                           "seed": seed, "fold": fold, **fields}, allow_nan=False,
                          ensure_ascii=False, separators=(",", ":")) + "\n"
        with worker_log.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        append_log(event, group=group, seed=seed, fold=fold, **fields)

    try:
        atomic_json({"state": "running", "group": group, "seed": seed, "fold": fold,
                     "started_utc": task_started_utc, "pid": os.getpid()}, status_path)
        optimizer = loaded.optimizer()
        objective = PCDPAXObjective(loaded.prepared, loaded.options.device)
        stopper = EarlyStopping(patience=CFG["early_stop"], save=False)
        loaded.reset()
        expected_rng = audit_info["anchors"][str(fold)]["train_start_rng_sha256"]
        if rng_hash(capture_rng()) != expected_rng:
            raise AssertionError("Training RNG does not match historical train_start")
        named = dict(loaded.core.named_parameters())
        img_params = [p for p in loaded.core.img_unet.parameters() if p.requires_grad]
        ph_params = [p for p in loaded.core.ph_unet.parameters() if p.requires_grad]
        rp_params = [p for p in loaded.core.rp_attention.parameters() if p.requires_grad]
        mm_params = [p for p in loaded.core.mm_attention.parameters() if p.requires_grad]
        clf_params = [p for p in loaded.core.clf.parameters() if p.requires_grad]
        initial_param_names = list(named)
        initial_parameter_count = sum(param.numel() for param in loaded.core.parameters())
        initial_bn_batches = int(loaded.core.clf[2].num_batches_tracked)
        history = []
        best = None
        best_loss = float("inf")
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
            values = objective(loaded.z_img, loaded.z_ph, group)
            image_scale, image_budget = __import__("alignment").gradient_budget(
                native, values["image_loss"], loaded.z_img, epoch, CFG["image_gradient_budget"])
            ph_scale, ph_budget = __import__("alignment").gradient_budget(
                native, values["ph_aux"], loaded.z_ph, epoch, CFG["phenotype_gradient_budget"])
            total = native + image_scale * values["image_loss"] + ph_scale * values["ph_aux"]
            for name, value in (("native", native), ("image_loss", values["image_loss"]),
                                ("ph_ce", values["ph_ce"]), ("ph_aux", values["ph_aux"]),
                                ("total", total)):
                _finite_scalar(value, name)
            if values["align_loss"] is not None:
                _finite_scalar(values["align_loss"], "alignment KL")
            sample = epoch < 3 or epoch % CFG["diagnostic_frequency"] == 0
            record = {
                "epoch": epoch, "native_ce": float(native_ce.detach()),
                "native_graph": float(graph_loss.detach()), "native_total": float(native.detach()),
                "image_d4_ce": float(values["image_loss"].detach()),
                "phenotype_real_ce": float(values["ph_ce"].detach()),
                "alignment_kl": None if values["align_loss"] is None else float(values["align_loss"].detach()),
                "ph_auxiliary_total": float(values["ph_aux"].detach()),
                "lambda_img": float(image_scale), "lambda_ph": float(ph_scale),
                "image_gradient_budget": image_budget, "phenotype_gradient_budget": ph_budget,
                "train_total": float(total.detach()),
            }
            if sample:
                record["image_prototype"] = summarize_prototype(
                    values["image_detail"], values["image_logits"][...], loaded.y[loaded.train])
                record["phenotype_prototype"] = summarize_prototype(
                    values["ph_detail_real"], values["ph_logits_real"], loaded.y[loaded.train])
                if values["align_loss"] is not None:
                    record["alignment"] = alignment_diagnostics(values, loaded.y[loaded.train])
                    record["alignment_student_prototype"] = summarize_prototype(
                        values["student_detail"], values["student_logits"], loaded.y[loaded.train])
                    kl_grad_img, kl_grad_ph = torch.autograd.grad(
                        values["align_loss"], (loaded.z_img, loaded.z_ph), retain_graph=True, allow_unused=True)
                    record["kl_gradient_path"] = {
                        "img_embedding_norm": 0.0 if kl_grad_img is None else float(kl_grad_img.norm()),
                        "ph_embedding_norm": 0.0 if kl_grad_ph is None else float(kl_grad_ph.norm()),
                        "img_unet_parameter_norm": gradient_norm_for_params(values["align_loss"], img_params),
                        "ph_unet_parameter_norm": gradient_norm_for_params(values["align_loss"], ph_params),
                        "rp_attention_parameter_norm": gradient_norm_for_params(values["align_loss"], rp_params),
                        "mm_attention_parameter_norm": gradient_norm_for_params(values["align_loss"], mm_params),
                    }
                record["native_parameter_gradient_norms"] = {
                    "img_unet": gradient_norm_for_params(native, img_params),
                    "ph_unet": gradient_norm_for_params(native, ph_params),
                }
            total.backward()
            if sample:
                record["total_parameter_gradient_norms"] = {
                    "img_unet": model_gradient_norm(img_params),
                    "ph_unet": model_gradient_norm(ph_params),
                    "rp_attention": model_gradient_norm(rp_params),
                    "mm_attention": model_gradient_norm(mm_params),
                    "classifier": model_gradient_norm(clf_params),
                }
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
            _finite_scalar(val_loss, "validation loss")
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
                worker_event("progress", epoch=epoch, val_loss=float(val_loss),
                             lambda_img=float(image_scale), lambda_ph=float(ph_scale))
            if stopper.early_stop:
                break
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - train_start
        if best is None:
            raise AssertionError("Training completed without a valid checkpoint")
        training_end_rng = capture_rng()
        import_core(loaded.core, best["core"])
        loaded.core.eval()
        if payload_hash(capture_rng()) != payload_hash(training_end_rng):
            raise AssertionError("Restoring the selected checkpoint changed the training-end RNG")
        test_start_rng = capture_rng()
        with torch.set_grad_enabled(True):
            test_logits, _ = loaded.forward()
        test_end_rng = capture_rng()
        if loaded.forward_count != 2 * len(history) + 1:
            raise AssertionError("Unexpected train/validation/test core-forward count")
        indices = loaded.split["test"]
        raw_logits = test_logits.detach().cpu().numpy()[indices]
        probability = torch.softmax(test_logits.detach(), dim=1).cpu().numpy()[indices, 0]
        prediction = raw_logits.argmax(1)
        labels = loaded.ctx["labels"][indices]
        oof = {
            "indices": indices.copy(), "ids": [loaded.ctx["ids"][int(index)] for index in indices],
            "labels": labels.copy(), "logits": raw_logits,
            "probability": probability, "prediction": prediction,
        }
        result = {
            "state": "complete", "group": group, "seed": seed, "fold": fold,
            "started_utc": task_started_utc, "completed_utc": utc(),
            "training_code_sha256": audit_info["training_code_sha256"],
            "manifest_sha256": audit_info["manifest_sha256"],
            "input_sha256": audit_info["anchors"][str(fold)]["sha256"],
            "input_digest": loaded.input_digest,
            "prototype_preparation": loaded.prepared["info"],
            "initialization_sha256": loaded.anchor_record["train_start_core_sha256"],
            "training_rng_fingerprint": rng_hash(training_end_rng),
            "test_rng_fingerprint": rng_hash(test_start_rng),
            "rng_before_test": rng_hash(test_start_rng), "rng_after_test": rng_hash(test_end_rng),
            "best_epoch": best["epoch"], "best_val_loss": best_loss,
            "best_model_state": best["core"], "best_core_sha256": core_hash(best["core"]),
            "best_record": best["record"], "epochs": history, "epochs_run": len(history),
            "oof": oof, "metrics": formal_metrics(labels, probability, prediction),
            "training_seconds": training_seconds, "task_seconds": time.perf_counter() - task_started,
            "native_core_forwards": loaded.forward_count, "formal_test_forwards": 1,
            "formal_test_grad_enabled": True, "new_trainable_parameters": 0,
            "rfe_refit_count": 0, "vae_pretrain_count": 0,
            "physical_gpu": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_parameter_names": initial_param_names,
            "initial_parameter_count": initial_parameter_count,
            "best_bn_batches": int(best["core"]["state"]["clf.2.num_batches_tracked"]),
            "forward_path": "native fusion classifier only; no auxiliary objective at validation or test",
            "uses_test_labels_for_training_or_selection": False,
        }
        path = result_path(group, seed, fold)
        atomic_torch(result, path)
        valid_result(group, seed, fold, audit_info, required=True)
        atomic_json({"state": "complete", "group": group, "seed": seed, "fold": fold,
                     "completed_utc": utc(), "artifact_sha256": sha_file(path),
                     "training_seconds": training_seconds, "best_epoch": best["epoch"]}, status_path)
        worker_event("complete", epochs=len(history), best_epoch=best["epoch"],
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
    if isinstance(record.get("preflight"), dict) and record["preflight"].get("passed") is True:
        return record["preflight"]
    if seed != 911 or fold != 0:
        raise ValueError("The registered real-data preflight is seed 911 fold 0 only")
    try:
        result = preflight_one(seed, fold, record)
    except Exception as error:
        record.setdefault("preflight_attempts", []).append({
            "recorded_utc": utc(), "status": "failed_before_formal_training",
            "error": repr(error), "traceback": traceback.format_exc(),
            "core_forwards": 1, "formal_tasks_started": False,
        })
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
    train_one(args.group, args.seed, args.fold, record)


if __name__ == "__main__":
    main()
