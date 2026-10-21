"""Fixed PCDPA-X2 group teachers and the shared M3 prototype losses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prototype import PrototypeObjective, prepare


BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text())
HISTORICAL_CFG = {**CFG["prototype"], "groups": {
    "D4": {"name": "Age-sex disease prototype", "kernel": "age_sex",
           "reliable": False, "shuffled": False}
}}


def live_logits(z: torch.Tensor, detail: dict, train: torch.Tensor) -> torch.Tensor:
    query = z[train]
    normalized = (query - detail["mean"]) / detail["std"]
    logits = -(normalized[:, None] - detail["mu"]).square().sum(-1) / detail["tau"]
    if not torch.allclose(logits.detach(), detail["logits"], atol=1e-6, rtol=1e-6):
        raise AssertionError("Live prototype logits differ from the accepted D4 objective")
    return logits


def build_same_private_fold_reference(prepared: dict) -> np.ndarray:
    pf = np.asarray(prepared["folds"], dtype=np.int64)
    reference = (pf[:, None] == pf[None, :])
    np.fill_diagonal(reference, False)
    return reference


def _normalize(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    den = weights.sum(axis=1)
    valid = den > 0
    normalized = np.zeros_like(weights)
    normalized[valid] = weights[valid] / den[valid, None]
    squared = np.square(weights).sum(axis=1)
    ess = np.zeros(len(weights), dtype=np.float64)
    ok = squared > 0
    ess[ok] = np.square(den[ok]) / squared[ok]
    return normalized, valid, ess


def build_teacher_weights(prepared: dict) -> dict:
    """Build fixed outer-train-only N1-N4 matrices and a shared validity mask."""
    reference = build_same_private_fold_reference(prepared)
    age = np.asarray(prepared["age"], dtype=np.float64)
    sex = np.asarray(prepared["sex"], dtype=np.int64)
    bandwidth = float(prepared["bandwidth"])
    age_kernel = np.exp(-0.5 * np.square((age[:, None] - age[None, :]) / bandwidth))

    def sex_kernel(values: np.ndarray) -> np.ndarray:
        known = (values[:, None] > 0) & (values[None, :] > 0)
        return np.where(known, np.where(values[:, None] == values[None, :], 1.0, 0.5), 0.75)

    permutation = np.asarray(prepared["permutation"], dtype=np.int64)
    age_shuffled = age[permutation]
    sex_shuffled = sex[permutation]
    shuffle_age_kernel = np.exp(-0.5 * np.square(
        (age_shuffled[:, None] - age_shuffled[None, :]) / bandwidth))

    raw = {
        "N1": reference.astype(np.float64),
        "N2": reference.astype(np.float64),
        "N3": reference * age_kernel * sex_kernel(sex),
        "N4": reference * shuffle_age_kernel * sex_kernel(sex_shuffled),
    }
    normalized, row_valid, ess = {}, {}, {}
    for group, matrix in raw.items():
        normalized[group], row_valid[group], ess[group] = _normalize(matrix)
    common_valid = np.logical_and.reduce([row_valid[group] for group in CFG["groups"]])
    for group in CFG["groups"]:
        normalized[group][~common_valid] = 0.0
        row_valid[group] = common_valid.copy()
        if not np.array_equal(normalized[group][~reference], np.zeros_like(normalized[group][~reference])):
            raise AssertionError(f"{group} assigns weight outside the same-fold non-self reference set")

    changed_age = age != age_shuffled
    changed_sex = sex != sex_shuffled
    changed_joint = changed_age | changed_sex
    return {
        "reference": reference,
        "weights": {key: torch.as_tensor(value, dtype=torch.float32) for key, value in normalized.items()},
        "valid": torch.as_tensor(common_valid, dtype=torch.bool),
        "ess": {key: torch.as_tensor(value, dtype=torch.float32) for key, value in ess.items()},
        "raw": raw,
        "info": {
            "n_train": int(len(age)),
            "n_valid_queries": int(common_valid.sum()),
            "n_empty_reference_rows": int((~common_valid).sum()),
            "reference_count_by_query": reference.sum(axis=1).astype(int).tolist(),
            "same_private_fold_only": True,
            "self_reference_count": int(np.diag(reference).sum()),
            "age_bandwidth": bandwidth,
            "shuffle_moved_subject_fraction": float(np.mean(permutation != np.arange(len(age)))),
            "shuffle_age_changed_fraction": float(changed_age.mean()),
            "shuffle_sex_changed_fraction": float(changed_sex.mean()),
            "shuffle_joint_condition_changed_fraction": float(changed_joint.mean()),
            "shuffle_kernel_mean_abs_difference": float(np.mean(np.abs(
                (age_kernel * sex_kernel(sex)) - (shuffle_age_kernel * sex_kernel(sex_shuffled))))),
            "shuffle_sources_global_indices": np.asarray(prepared["train"])[permutation].tolist(),
            "ess": {key: {"mean": float(value[row_valid[key]].mean()) if row_valid[key].any() else 0.0,
                          "min": float(value[row_valid[key]].min()) if row_valid[key].any() else 0.0,
                          "max": float(value[row_valid[key]].max()) if row_valid[key].any() else 0.0}
                    for key, value in ess.items()},
        },
    }


def group_teacher_target(group: str, q_img: torch.Tensor, labels: torch.Tensor,
                         teacher_weights: dict, device: torch.device) -> torch.Tensor:
    weights = teacher_weights["weights"][group].to(device=device, dtype=torch.float32)
    if group == "N1":
        source = F.one_hot(labels, num_classes=2).to(dtype=torch.float32)
    else:
        source = F.softmax(q_img.detach() / CFG["semantic_temperature"], dim=1).detach()
    return torch.matmul(weights, source).detach()


def semantic_kl(student_logits: torch.Tensor, teacher: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(device=student_logits.device)
    if not bool(valid.any()):
        return student_logits.sum() * 0.0
    teacher = teacher[valid].detach()
    return F.kl_div(F.log_softmax(student_logits[valid], dim=1), teacher,
                    reduction="batchmean")


class PCDPAX2Objective:
    def __init__(self, prepared: dict, device: torch.device):
        self.prepared = prepared
        self.device = device
        self.image = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
        self.ph_real = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
        self.teacher = build_teacher_weights(prepared)
        self.train = self.image.train

    def __call__(self, z_img: torch.Tensor, z_ph: torch.Tensor, group: str) -> dict:
        if group not in CFG["groups"]:
            raise ValueError(f"Unknown teacher group: {group}")
        image_loss, image_detail = self.image(z_img)
        ph_ce, ph_detail = self.ph_real(z_ph)
        q_img = live_logits(z_img, image_detail, self.train)
        q_ph = live_logits(z_ph, ph_detail, self.train)
        target = group_teacher_target(group, q_img, self.image.y, self.teacher, self.device)
        valid = self.teacher["valid"].to(device=self.device)
        kl = semantic_kl(q_ph, target, valid)
        return {
            "image_loss": image_loss,
            "image_logits": q_img,
            "image_detail": image_detail,
            "ph_ce": ph_ce,
            "ph_logits_real": q_ph,
            "ph_detail_real": ph_detail,
            "student_logits": q_ph,
            "student_detail": ph_detail,
            "teacher_target": target,
            "teacher_weights": self.teacher["weights"][group].to(self.device),
            "teacher_valid": valid,
            "align_loss": kl,
            "ph_aux": ph_ce + CFG["gamma"] * kl,
        }


def gradient_budget(native: torch.Tensor, auxiliary: torch.Tensor, z: torch.Tensor,
                    epoch: int, budget: float) -> tuple[torch.Tensor, dict]:
    g_native = torch.autograd.grad(native, z, retain_graph=True, allow_unused=True)[0]
    g_aux = torch.autograd.grad(auxiliary, z, retain_graph=True, allow_unused=True)[0]
    n = z.new_zeros(()) if g_native is None else g_native.detach().norm()
    a = z.new_zeros(()) if g_aux is None else g_aux.detach().norm()
    epsilon = CFG["gradient_epsilon"]
    raw = (budget * n / a.clamp_min(epsilon)).clamp(0, CFG["lambda_cap"])
    scale = raw.detach() * min(max(epoch, 0) / CFG["warmup_last_epoch"], 1.0)
    cosine = (g_native.detach() * g_aux.detach()).sum() / (n * a).clamp_min(epsilon) \
        if g_native is not None and g_aux is not None else z.new_zeros(())
    actual = float((scale * a) / n.clamp_min(epsilon)) if float(n) > 0 else 0.0
    return scale, {
        "native_z_gradient_norm": float(n),
        "auxiliary_z_gradient_norm": float(a),
        "lambda": float(scale),
        "weighted_auxiliary_z_gradient_ratio": actual,
        "native_auxiliary_gradient_cosine": float(cosine),
        "lambda_cap_hit": bool(float(budget * n / a.clamp_min(epsilon)) > CFG["lambda_cap"]),
        "native_gradient_near_zero": bool(float(n) <= epsilon),
    }
