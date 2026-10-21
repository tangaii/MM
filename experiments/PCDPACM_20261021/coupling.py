"""PCDPA-CM objectives with exact matrix-marginalized triplet coupling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prototype import PrototypeObjective


BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text())
HISTORICAL_CFG = {**CFG["prototype"], "groups": {
    "D4": {"name": "Age-sex disease prototype", "kernel": "age_sex",
           "reliable": False, "shuffled": False}
}}


def _normalize_rows(weights: torch.Tensor, epsilon: float):
    total = weights.sum(dim=1)
    valid = total > epsilon
    normalized = torch.zeros_like(weights)
    normalized[valid] = weights[valid] / total[valid, None]
    square = weights.square().sum(dim=1)
    ess = torch.zeros_like(total)
    has_weight = square > epsilon
    ess[has_weight] = total[has_weight].square() / square[has_weight]
    return normalized, valid, ess, total


class CoupledPrototypeObjective:
    """Keep D4 unchanged and add class-paired prototypes to image queries."""

    def __init__(self, prepared: dict, device: torch.device):
        self.prepared = prepared
        self.device = device
        self.d4 = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
        self.train = self.d4.train
        self.y = self.d4.y
        self.pf = self.d4.pf
        self.epsilon = float(CFG["prototype"]["epsilon"])
        age = torch.as_tensor(prepared["age"], dtype=torch.float32, device=device)
        sex = torch.as_tensor(prepared["sex"], dtype=torch.long, device=device)
        bandwidth = float(prepared["bandwidth"])
        k_age = torch.exp(-0.5 * ((age[:, None] - age[None, :]) / bandwidth).square())
        known = (sex[:, None] > 0) & (sex[None, :] > 0)
        k_sex = torch.where(known, torch.where(sex[:, None] == sex[None, :], 1.0, 0.5), 0.75)
        self.k_real = (k_age * k_sex).detach()
        self.reference = (self.pf[:, None] != self.pf[None, :]).detach()
        self.class0 = (self.y == 0)
        self.class1 = (self.y == 1)
        self.permutation = torch.as_tensor(prepared["permutation"], dtype=torch.long, device=device)
        self.age = age
        self.sex = sex

    def _coupling(self, z_ph: torch.Tensor, shuffled: bool):
        z = z_ph[self.train].detach()
        if shuffled:
            z = z[self.permutation]
        mean = z.mean(0)
        raw_std = z.std(0, unbiased=False)
        std = raw_std.clamp_min(CFG["coupling"]["embedding_std_floor"])
        normalized = (z - mean) / std
        norms = normalized.norm(dim=1, keepdim=True)
        unit = torch.where(norms > self.epsilon, normalized / norms.clamp_min(self.epsilon),
                           torch.zeros_like(normalized))
        cosine = (unit @ unit.T).clamp(-1.0, 1.0)
        c = (0.75 + 0.25 * cosine).clamp(0.5, 1.0)
        return c.detach(), {"mean": mean.detach(), "std": std.detach(), "raw_std": raw_std.detach(),
                             "standardized": normalized.detach(), "unit": unit.detach()}

    def _pair(self, teacher: torch.Tensor, query: torch.Tensor, d4_detail: dict,
              h: torch.Tensor):
        with torch.no_grad():
            ref = self.reference.to(dtype=teacher.dtype)
            a0 = ref * h * self.class0[None, :].to(dtype=teacher.dtype)
            a1 = ref * h * self.class1[None, :].to(dtype=teacher.dtype)
            i0, i1 = self.class0, self.class1
            h01 = h[i0][:, i1]
            m0 = a0[:, i0] * (a1[:, i1] @ h01.T)
            m1 = a1[:, i1] * (a0[:, i0] @ h01)
            den0, den1 = m0.sum(1), m1.sum(1)
            z_tol = 2e-5 * torch.maximum(torch.ones_like(den0), den0.abs())
            if not torch.all((den0 - den1).abs() <= z_tol):
                raise AssertionError("Triplet ASD/TD marginal total masses differ")
            w0, valid0, ess0, _ = _normalize_rows(m0, self.epsilon)
            w1, valid1, ess1, _ = _normalize_rows(m1, self.epsilon)
            support = valid0 & valid1
            min_ess = torch.minimum(ess0, ess1)
            beta = ((min_ess - CFG["coupling"]["pair_ess_offset"])
                    / CFG["coupling"]["pair_ess_range"]).clamp(0, 1)
            beta = torch.where(support, beta, torch.zeros_like(beta))
            t0, t1 = teacher[i0], teacher[i1]
            raw0 = w0 @ t0
            raw1 = w1 @ t1
            d4_mu = d4_detail["mu"]
            pair_mu = torch.stack((raw0, raw1), dim=1)
            mu = beta[:, None, None] * pair_mu + (1 - beta[:, None, None]) * d4_mu
        u = (query - d4_detail["mean"]) / d4_detail["std"]
        logits = -(u[:, None] - mu).square().sum(-1) / d4_detail["tau"]
        valid = d4_detail["valid"]
        ce = F.cross_entropy(logits, self.y, reduction="none")
        loss = (ce * valid.to(ce.dtype)).sum() / valid.sum().clamp_min(1)
        return loss, {
            "m0": m0.detach(), "m1": m1.detach(), "w0": w0.detach(), "w1": w1.detach(),
            "mass0": den0.detach(), "mass1": den1.detach(), "ess0": ess0.detach(),
            "ess1": ess1.detach(), "beta": beta.detach(), "support": support.detach(),
            "mu": mu.detach(), "raw_mu": pair_mu.detach(), "logits": logits.detach(),
            "ce": ce.detach(), "valid": valid.detach(), "h": h.detach(),
        }

    def __call__(self, z_img: torch.Tensor, z_ph: torch.Tensor, group: str) -> dict:
        if group not in CFG["groups"]:
            raise ValueError(f"Unknown PCDPA-CM group: {group}")
        image_d4, d4_detail = self.d4(z_img)
        query_img = z_img[self.train]
        with torch.no_grad():
            teacher = d4_detail["teacher"]
            if group == "P1":
                c = torch.ones_like(self.k_real)
                coupling_info = None
            else:
                c, coupling_info = self._coupling(z_ph, shuffled=(group == "P4"))
            h = self.k_real if group == "P1" else self.k_real * c
        pair_loss, pair_detail = self._pair(teacher, query_img, d4_detail, h)
        image_aux = 0.5 * (image_d4 + pair_loss)
        ph_ce = None
        ph_detail = None
        if group in ("P1", "P2"):
            ph_ce, ph_detail = self.d4(z_ph)
        return {
            "image_d4": image_d4, "d4_detail": d4_detail,
            "pair_loss": pair_loss, "pair_detail": pair_detail,
            "image_aux": image_aux, "ph_ce": ph_ce, "ph_detail": ph_detail,
            "c": c.detach(), "coupling_info": coupling_info,
            "kernel": self.k_real, "reference": self.reference,
            "d4_weights": self.d4.weights.detach(), "labels": self.y.detach(), "h": h.detach(),
        }


def gradient_budget(native: torch.Tensor, auxiliary: torch.Tensor, z: torch.Tensor,
                    epoch: int, budget: float) -> tuple[torch.Tensor, dict]:
    gn = torch.autograd.grad(native, z, retain_graph=True, allow_unused=True)[0]
    ga = torch.autograd.grad(auxiliary, z, retain_graph=True, allow_unused=True)[0]
    gn = z.new_zeros(()) if gn is None else gn.detach()
    ga = z.new_zeros(()) if ga is None else ga.detach()
    n, a = gn.norm(), ga.norm()
    eps = CFG["gradient_epsilon"]
    raw = (budget * n / a.clamp_min(eps)).clamp(0, CFG["lambda_cap"])
    scale = (raw.detach() * min(max(epoch, 0) / CFG["warmup_last_epoch"], 1.0)).detach()
    cosine = (gn * ga).sum() / (n * a).clamp_min(eps)
    ratio = (scale * a) / n.clamp_min(eps) if float(n) > eps else z.new_zeros(())
    return scale, {"native_gradient_norm": float(n), "auxiliary_gradient_norm": float(a),
                   "lambda": float(scale), "weighted_gradient_ratio": float(ratio),
                   "gradient_cosine": float(cosine),
                   "lambda_cap_hit": bool(float(budget * n / a.clamp_min(eps)) > CFG["lambda_cap"]),
                   "native_gradient_near_zero": bool(float(n) <= eps)}


def coupling_summary(value: dict, prepared: dict) -> dict:
    detail = value["pair_detail"]
    w0, w1 = detail["w0"].detach(), detail["w1"].detach()
    y = torch.as_tensor(prepared["y"], dtype=torch.long, device=w0.device)
    age = torch.as_tensor(prepared["age"], dtype=w0.dtype, device=w0.device)
    sex = torch.as_tensor(prepared["sex"], dtype=torch.long, device=w0.device)
    site = np.asarray(prepared["site"], dtype=str)
    valid_t = detail["valid"].detach()
    valid = valid_t.cpu().numpy()

    def distribution(array):
        if torch.is_tensor(array):
            array = array.detach().cpu().numpy()
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        return {"mean": float(array.mean()) if array.size else None,
                "sd": float(array.std()) if array.size else None,
                "p10": float(np.quantile(array, .1)) if array.size else None,
                "p50": float(np.quantile(array, .5)) if array.size else None,
                "p90": float(np.quantile(array, .9)) if array.size else None,
                "min": float(array.min()) if array.size else None,
                "max": float(array.max()) if array.size else None}

    age0 = w0 @ age[y == 0]
    age1 = w1 @ age[y == 1]
    sex0 = w0 @ (sex[y == 0] == 2).to(w0.dtype)
    sex1 = w1 @ (sex[y == 1] == 2).to(w1.dtype)
    site_codes = np.searchsorted(np.unique(site), site)
    site_basis = torch.as_tensor(np.eye(len(np.unique(site)))[site_codes], dtype=w0.dtype, device=w0.device)
    site_counts0 = w0 @ site_basis[y == 0]
    site_counts1 = w1 @ site_basis[y == 1]
    beta = detail["beta"].detach()
    ess0 = detail["ess0"].detach()
    ess1 = detail["ess1"].detach()
    support = detail["support"].detach()
    d4 = value["d4_weights"]
    d4w0, d4w1 = d4[:, 0, :][:, y == 0], d4[:, 1, :][:, y == 1]
    d4w0 = d4w0 / d4w0.sum(1, keepdim=True).clamp_min(1e-12)
    d4w1 = d4w1 / d4w1.sum(1, keepdim=True).clamp_min(1e-12)
    d4age0, d4age1 = d4w0 @ age[y == 0], d4w1 @ age[y == 1]
    d4sex0 = d4w0 @ (sex[y == 0] == 2).to(w0.dtype)
    d4sex1 = d4w1 @ (sex[y == 1] == 2).to(w1.dtype)
    d4site0, d4site1 = d4w0 @ site_basis[y == 0], d4w1 @ site_basis[y == 1]
    pair_l1_d4 = (0.5 * ((w0 - d4w0).abs().sum(1) + (w1 - d4w1).abs().sum(1)))
    ref = value["reference"].to(w0.dtype)
    k = value["kernel"]
    class0, class1 = y == 0, y == 1
    a0 = ref * k * class0[None, :]
    a1 = ref * k * class1[None, :]
    h01 = k[class0][:, class1]
    p1m0 = a0[:, class0] * (a1[:, class1] @ h01.T)
    p1m1 = a1[:, class1] * (a0[:, class0] @ h01)
    p1w0 = p1m0 / p1m0.sum(1, keepdim=True).clamp_min(1e-12)
    p1w1 = p1m1 / p1m1.sum(1, keepdim=True).clamp_min(1e-12)
    pair_l1_p1 = 0.5 * ((w0 - p1w0).abs().sum(1) + (w1 - p1w1).abs().sum(1))
    pair_d4_mu_l2 = (detail["mu"] - value["d4_detail"]["mu"]).norm(dim=-1).mean(-1)
    ess_min = torch.minimum(ess0, ess1)
    valid_ess = ess_min[valid_t]
    median_ess = valid_ess.median() if valid_ess.numel() else ess_min.new_zeros(())
    low_ess = valid_t & (ess_min <= median_ess)
    high_ess = valid_t & (ess_min > median_ess)
    pair_correct = detail["logits"].argmax(1) == value["labels"]
    d4_correct = value["d4_detail"]["logits"].argmax(1) == value["labels"]
    result = {
        "pair_ess_asd": distribution(ess0[valid_t]), "pair_ess_td": distribution(ess1[valid_t]),
        "pair_min_ess": distribution(torch.minimum(ess0, ess1)[valid_t]),
        "pair_beta": distribution(beta[valid_t]),
        "pair_beta_zero_fraction": float((beta[valid_t] == 0).float().mean()) if valid.any() else None,
        "pair_fallback_fraction": float((~support[valid_t]).float().mean()) if valid.any() else None,
        "pair_supported_queries": int((support & valid_t).sum()),
        "d4_valid_queries": int(valid.sum()),
        "pair_age_group_mean_abs_gap": distribution((age0 - age1).abs()[valid_t]),
        "pair_sex_female_fraction_abs_gap": distribution((sex0 - sex1).abs()[valid_t]),
        "pair_site_total_variation": distribution(0.5 * (site_counts0 - site_counts1).abs().sum(1)[valid_t]),
        "d4_independent_age_group_mean_abs_gap": distribution((d4age0 - d4age1).abs()[valid_t]),
        "d4_independent_sex_female_fraction_abs_gap": distribution((d4sex0 - d4sex1).abs()[valid_t]),
        "d4_independent_site_total_variation": distribution(0.5 * (d4site0 - d4site1).abs().sum(1)[valid_t]),
        "pair_weight_mean_l1_vs_independent_d4": distribution(pair_l1_d4[valid_t]),
        "pair_weight_mean_l1_vs_p1_triplet": distribution(pair_l1_p1[valid_t]),
        "pair_prototype_mean_l2_vs_d4": distribution(pair_d4_mu_l2[valid_t]),
        "pair_ce_low_ess": distribution(detail["ce"][low_ess]),
        "pair_ce_high_ess": distribution(detail["ce"][high_ess]),
        "pair_accuracy_low_ess": float(pair_correct[low_ess].float().mean()) if low_ess.any() else None,
        "pair_accuracy_high_ess": float(pair_correct[high_ess].float().mean()) if high_ess.any() else None,
        "d4_ce_low_ess": distribution(value["d4_detail"]["ce"][low_ess]),
        "d4_ce_high_ess": distribution(value["d4_detail"]["ce"][high_ess]),
        "ess_median_split": float(median_ess),
        "pair_class_prototype_distance": distribution(
            detail["mu"][:, 0].detach().sub(detail["mu"][:, 1].detach()).norm(dim=1).cpu().numpy()[valid]),
    }
    if value["coupling_info"] is not None:
        info = value["coupling_info"]
        c = value["c"].detach()
        triu = torch.triu_indices(c.shape[0], c.shape[1], offset=1, device=c.device)
        c_values = c[triu[0], triu[1]].cpu().numpy()
        k_values = value["kernel"][triu[0], triu[1]].cpu().numpy()
        pair_i, pair_j = triu[0].cpu().numpy(), triu[1].cpu().numpy()
        same_site = (site[pair_i] == site[pair_j]).astype(float)
        c_corr_age = np.corrcoef(c_values, k_values)[0, 1] if np.std(c_values) > 0 and np.std(k_values) > 0 else 0.0
        c_corr_site = np.corrcoef(c_values, same_site)[0, 1] if np.std(c_values) > 0 and np.std(same_site) > 0 else 0.0
        result["phenotype_embedding"] = {
            "per_dimension_variance_mean": float(info["standardized"].var(0, unbiased=False).mean()),
            "raw_per_dimension_variance": distribution(info["raw_std"].square()),
            "raw_per_dimension_std": distribution(info["raw_std"]),
            "row_norm": distribution(info["standardized"].norm(dim=1).cpu().numpy()),
            "cosine": distribution((2 * c_values - 1)), "C": distribution(c_values),
            "C_vs_age_sex_kernel_correlation": float(c_corr_age),
            "C_vs_same_site_correlation": float(c_corr_site),
            "C_std": float(np.std(c_values)),
        }
    return result
