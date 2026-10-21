"""PCDPA-X objectives with the accepted PCDCA prototype math loaded read-only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
CFG = __import__("json").loads((BASE / "config.json").read_text())
HISTORY_PROTOTYPE = ROOT / CFG["historical_prototype"]
_spec = importlib.util.spec_from_file_location("pcdca_historical_prototype", HISTORY_PROTOTYPE)
_historical = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_historical)

prepare = _historical.prepare
PrototypeObjective = _historical.PrototypeObjective
private_seed = _historical.private_seed
HISTORICAL_CFG = {**CFG["prototype"], "groups": CFG["historical_groups"]}


def live_logits(z: torch.Tensor, detail: dict, train: torch.Tensor) -> torch.Tensor:
    """Reconstruct the exact accepted distance logits with query-only gradients."""
    query = z[train]
    normalized = (query - detail["mean"]) / detail["std"]
    logits = -(normalized[:, None] - detail["mu"]).square().sum(-1) / detail["tau"]
    if not torch.allclose(logits.detach(), detail["logits"], atol=1e-6, rtol=1e-6):
        raise AssertionError("Live logits differ from the historical objective logits")
    return logits


def image_d4_objective(z_img: torch.Tensor, prepared: dict, device: torch.device):
    objective = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
    loss, detail = objective(z_img)
    logits = live_logits(z_img, detail, objective.train)
    return loss, logits, detail, objective


def phenotype_prototype_objective(z_ph: torch.Tensor, prepared: dict, kind: str,
                                  device: torch.device):
    groups = {"global": "D1", "real": "D4", "shuffled": "D5"}
    objective = PrototypeObjective(prepared, groups[kind], HISTORICAL_CFG, device)
    loss, detail = objective(z_ph)
    logits = live_logits(z_ph, detail, objective.train)
    return loss, logits, detail, objective


def semantic_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher = F.softmax(teacher_logits.detach() / CFG["semantic_temperature"], dim=1)
    student_log = F.log_softmax(student_logits / CFG["semantic_temperature"], dim=1)
    return F.kl_div(student_log, teacher, reduction="batchmean")


class PCDPAXObjective:
    def __init__(self, prepared: dict, device: torch.device):
        self.prepared = prepared
        self.device = device
        self.image = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
        self.ph_real = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
        self.ph_global = None
        self.ph_shuffled = None

    def __call__(self, z_img: torch.Tensor, z_ph: torch.Tensor, group: str) -> dict:
        if group not in CFG["groups"]:
            raise KeyError(group)
        image_loss, image_detail = self.image(z_img)
        image_logits = live_logits(z_img, image_detail, self.image.train)
        ph_ce, ph_detail = self.ph_real(z_ph)
        ph_logits_real = live_logits(z_ph, ph_detail, self.ph_real.train)
        align_kind = CFG["groups"][group]["alignment"]
        align_loss = None
        student_logits = None
        student_detail = None
        if align_kind == "global":
            if self.ph_global is None:
                self.ph_global = PrototypeObjective(self.prepared, "D1", HISTORICAL_CFG, self.device)
            _, student_detail = self.ph_global(z_ph)
            student_logits = live_logits(z_ph, student_detail, self.ph_global.train)
        elif align_kind == "real":
            student_logits, student_detail = ph_logits_real, ph_detail
        elif align_kind == "shuffled":
            if self.ph_shuffled is None:
                self.ph_shuffled = PrototypeObjective(self.prepared, "D5", HISTORICAL_CFG, self.device)
            _, student_detail = self.ph_shuffled(z_ph)
            student_logits = live_logits(z_ph, student_detail, self.ph_shuffled.train)
        elif align_kind is not None:
            raise ValueError(f"Unknown alignment mode: {align_kind}")
        if student_logits is not None:
            align_loss = semantic_kl(student_logits, image_logits)
        return {
            "image_loss": image_loss,
            "image_logits": image_logits,
            "image_detail": image_detail,
            "ph_ce": ph_ce,
            "ph_logits_real": ph_logits_real,
            "ph_detail_real": ph_detail,
            "student_logits": student_logits,
            "student_detail": student_detail,
            "align_loss": align_loss,
            "ph_aux": ph_ce if align_loss is None else ph_ce + CFG["gamma"] * align_loss,
        }


def gradient_norm(loss: torch.Tensor, z: torch.Tensor, retain_graph: bool = True) -> torch.Tensor:
    grad = torch.autograd.grad(loss, z, retain_graph=retain_graph, allow_unused=True)[0]
    if grad is None:
        return z.new_zeros(())
    return grad.detach().norm()


def gradient_budget(native: torch.Tensor, auxiliary: torch.Tensor, z: torch.Tensor,
                    epoch: int, budget: float | None = None) -> tuple[torch.Tensor, dict]:
    g_native = torch.autograd.grad(native, z, retain_graph=True, allow_unused=True)[0]
    g_aux = torch.autograd.grad(auxiliary, z, retain_graph=True, allow_unused=True)[0]
    n = z.new_zeros(()) if g_native is None else g_native.detach().norm()
    a = z.new_zeros(()) if g_aux is None else g_aux.detach().norm()
    cap = CFG["lambda_cap"]
    ratio = CFG["image_gradient_budget"] if budget is None else budget
    raw = ratio * n / a.clamp_min(CFG["gradient_epsilon"])
    scale = raw.clamp(0, cap).detach() * min(max(epoch, 0) / CFG["warmup_last_epoch"], 1.0)
    actual = float((scale * a) / n.clamp_min(CFG["gradient_epsilon"])) if float(n) > 0 else 0.0
    return scale, {
        "native_z_gradient_norm": float(n),
        "auxiliary_z_gradient_norm": float(a),
        "lambda": float(scale),
        "weighted_auxiliary_z_gradient_ratio": actual,
        "lambda_cap_hit": bool(float(raw) > cap),
        "native_gradient_near_zero": bool(float(n) <= CFG["gradient_epsilon"]),
    }
