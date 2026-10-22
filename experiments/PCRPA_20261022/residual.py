"""PC-RPA residual prototype objective; all conditions are train-only and detached."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prototype import PrototypeObjective

BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text())
HISTORICAL_CFG = CFG["prototype"]


def _row_normalize(weight: torch.Tensor, eps: float):
    den = weight.sum(1)
    valid = den > eps
    out = torch.zeros_like(weight)
    out[valid] = weight[valid] / den[valid, None]
    sq = weight.square().sum(1)
    ess = torch.zeros_like(den)
    mask = sq > eps
    ess[mask] = den[mask].square() / sq[mask]
    return out, valid, ess


def _kernel(age_q, sex_q, age_r, sex_r, bandwidth):
    age = torch.exp(-0.5 * ((age_q[:, None] - age_r[None, :]) / bandwidth).square())
    known = (sex_q[:, None] > 0) & (sex_r[None, :] > 0)
    sex = torch.where(known, torch.where(sex_q[:, None] == sex_r[None, :], 1.0, 0.5), 0.75)
    return age * sex


class ResidualPrototypeObjective:
    """Use three-fold cross-fit references for b(v), pair weights, and residual CE."""

    def __init__(self, prepared: dict, device: torch.device):
        self.prepared = prepared
        self.device = device
        self.d4 = PrototypeObjective(prepared, "D4", HISTORICAL_CFG, device)
        self.train, self.y, self.pf = self.d4.train, self.d4.y, self.d4.pf
        self.n = len(prepared["train"])
        self.eps = float(CFG["prototype"]["epsilon"])
        age = torch.as_tensor(prepared["age"], dtype=torch.float32, device=device)
        sex = torch.as_tensor(prepared["sex"], dtype=torch.long, device=device)
        site = np.asarray(prepared["site"], dtype=str)
        self.age, self.sex = age, sex
        self.site = site
        valid_sites = sorted(set(site) - {"__MISSING_SITE__"})
        site_code = {name: index for index, name in enumerate(valid_sites)}
        self.site_codes = np.asarray([site_code.get(name, -1) for name in site], dtype=np.int64)
        self.k_real = _kernel(age, sex, age, sex, prepared["bandwidth"]).detach()
        self.permutation = torch.as_tensor(prepared["permutation"], dtype=torch.long, device=device)
        self.groups = {}
        for g in range(3):
            q = torch.nonzero(self.pf == g, as_tuple=False).flatten()
            s = torch.nonzero(self.pf != g, as_tuple=False).flatten()
            self.groups[g] = self._prepare_fold(q, s)

    def _condition(self, kernel, y_ref, site_q, site_ref, mode):
        """Return row weights for b and site ESS beta; weights are static per fold."""
        n_q, n_s = kernel.shape
        y0, y1 = y_ref == 0, y_ref == 1
        mixed, mixed_ok, _ = _row_normalize(kernel, self.eps)
        w0, ok0, ess0 = _row_normalize(kernel * y0[None], self.eps)
        w1, ok1, ess1 = _row_normalize(kernel * y1[None], self.eps)
        balanced = 0.5 * (w0 + w1)
        balanced_ok = ok0 & ok1
        if mode == "R1":
            return mixed, mixed_ok, torch.zeros(n_q, device=self.device)
        base = balanced
        base_ok = balanced_ok
        if mode == "R2":
            return base, base_ok, torch.zeros(n_q, device=self.device)
        same = (site_q[:, None] == site_ref[None, :])
        valid_site = site_q[:, None] >= 0
        site_kernel = kernel * same * valid_site
        sw0, sok0, sess0 = _row_normalize(site_kernel * y0[None], self.eps)
        sw1, sok1, sess1 = _row_normalize(site_kernel * y1[None], self.eps)
        site_balanced = 0.5 * (sw0 + sw1)
        site_ok = sok0 & sok1
        beta = ((torch.minimum(sess0, sess1) - 8.0) / 32.0).clamp(0.0, 1.0)
        beta = torch.where(site_ok & base_ok, beta, torch.zeros_like(beta))
        weights = (1.0 - beta[:, None]) * base + beta[:, None] * site_balanced
        return weights, base_ok, beta

    def _prepare_fold(self, q, s):
        y_s = self.y[s]
        q_site = self.site_codes[q.cpu().numpy()]
        s_site = self.site_codes[s.cpu().numpy()]
        s_site_t = torch.as_tensor(s_site, dtype=torch.long, device=self.device)
        q_site_t = torch.as_tensor(q_site, dtype=torch.long, device=self.device)
        age, sex = self.age, self.sex
        variants = {}
        for group in ("R1", "R2", "R3", "R4"):
            if group == "R4":
                age_v, sex_v = age[self.permutation], sex[self.permutation]
            else:
                age_v, sex_v = age, sex
            k_qs = _kernel(age_v[q], sex_v[q], age_v[s], sex_v[s], self.prepared["bandwidth"])
            # Reference b_j is built from S_g\{j}; no query-fold row can enter it.
            k_ss = _kernel(age_v[s], sex_v[s], age_v[s], sex_v[s], self.prepared["bandwidth"])
            k_ss.fill_diagonal_(0.0)
            wq, okq, beta_q = self._condition(k_qs, y_s, q_site_t, s_site_t, group)
            wr, okr, beta_r = self._condition(k_ss, y_s, s_site_t, s_site_t, group)
            variants[group] = {"q": q, "s": s, "wq": wq.detach(), "wr": wr.detach(),
                               "okq": okq.detach(), "okr": okr.detach(),
                               "beta_q": beta_q.detach(), "beta_r": beta_r.detach()}

        # P1 pair marginals use real age/sex for every group, including R4.
        k_qs = self.k_real[q][:, s]
        k_ss = self.k_real[s][:, s]
        y0, y1 = y_s == 0, y_s == 1
        a0, a1 = k_qs * y0[None], k_qs * y1[None]
        h01 = k_ss[y0][:, y1]
        m0 = a0[:, y0] * (a1[:, y1] @ h01.T)
        m1 = a1[:, y1] * (a0[:, y0] @ h01)
        w0, v0, e0 = _row_normalize(m0, self.eps)
        w1, v1, e1 = _row_normalize(m1, self.eps)
        pair_ok = v0 & v1
        beta_pair = ((torch.minimum(e0, e1) - 8.0) / 32.0).clamp(0.0, 1.0)
        beta_pair = torch.where(pair_ok, beta_pair, torch.zeros_like(beta_pair))
        return {"variants": variants, "w0": w0.detach(), "w1": w1.detach(),
                "beta_pair": beta_pair.detach(), "pair_ok": pair_ok.detach(),
                "ess0": e0.detach(), "ess1": e1.detach(), "m0": m0.detach(), "m1": m1.detach()}

    def __call__(self, z_img: torch.Tensor, z_ph: torch.Tensor, group: str) -> dict:
        if group not in CFG["groups"]:
            raise ValueError(group)
        image_d4, d4_detail = self.d4(z_img)
        ph_ce, ph_detail = self.d4(z_ph)
        query = z_img[self.train]
        teacher = d4_detail["teacher"]
        q_residual, raw_query, query_indices, mu_by_group, logits_by_group = [], [], [], [], []
        valid_rows, b_means, beta_site = [], [], []
        pair_betas, pair_esses = [], []
        for g in range(3):
            data = self.groups[g]
            state = data["variants"][group]
            q, s = state["q"], state["s"]
            bq = state["wq"] @ teacher[s]
            br = state["wr"] @ teacher[s]
            uq = (query[q] - d4_detail["mean"]) / d4_detail["std"]
            rq = uq - bq.detach()
            rr = (teacher[s] - br.detach()).detach()
            y_s = self.y[s]
            d4_weights = self.d4.weights[q][:, :, s]
            d4_global_weights = self.d4.global_weights[q][:, :, s]
            d4_mu = []
            d4_ok = torch.ones(len(q), dtype=torch.bool, device=self.device)
            for c in (0, 1):
                wc = d4_weights[:, c]
                gwc = d4_global_weights[:, c]
                den, gden = wc.sum(1), gwc.sum(1)
                ok = gden > self.eps
                d4_ok &= ok
                local_mu = (wc @ rr) / den.clamp_min(self.eps)[:, None]
                global_mu = (gwc @ rr) / gden.clamp_min(self.eps)[:, None]
                beta = d4_detail["beta"][q]
                if self.d4.spec["kernel"] == "global":
                    beta = torch.zeros_like(beta)
                beta = torch.where(den > self.eps, beta, torch.zeros_like(beta))
                d4_mu.append(beta[:, None] * local_mu + (1.0 - beta[:, None]) * global_mu)
            d4_res = torch.stack(d4_mu, dim=1)
            pair0 = data["w0"] @ rr[y_s == 0]
            pair1 = data["w1"] @ rr[y_s == 1]
            pair_mu = torch.stack((pair0, pair1), dim=1)
            beta = data["beta_pair"]
            mu = beta[:, None, None] * pair_mu + (1.0 - beta[:, None, None]) * d4_res
            supported = d4_ok & state["okq"] & state["okr"].all().expand(len(q))
            # Reference rows that lack their own legal condition estimate cannot be used.
            if not bool(state["okr"].all()):
                ref_keep = state["okr"].to(rr.dtype)
                rr = rr * ref_keep[:, None]
                # In this cohort all three folds contain both classes; fail closed otherwise.
                raise AssertionError("A reference row lacks legal condition support")
            logits = -(rq[:, None] - mu).square().sum(-1) / d4_detail["tau"]
            q_residual.append(rq)
            raw_query.append(uq)
            query_indices.append(q)
            mu_by_group.append(mu)
            logits_by_group.append(logits)
            valid_rows.append(supported & d4_detail["valid"][q])
            b_means.append(bq.detach())
            beta_site.append(state["beta_q"])
            pair_betas.append(beta)
            pair_esses.append(torch.minimum(data["ess0"], data["ess1"]))
        logits = torch.cat(logits_by_group, dim=0)
        valid = torch.cat(valid_rows, dim=0)
        ce = F.cross_entropy(logits, self.y, reduction="none")
        loss = (ce * valid.to(ce.dtype)).sum() / valid.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite residual prototype CE")
        return {"image_d4": image_d4, "d4_detail": d4_detail,
                "residual_ce": loss, "pair_loss": loss,
                "image_aux": 0.5 * image_d4 + 0.5 * loss,
                "ph_ce": ph_ce, "ph_detail": ph_detail,
                "pair_weights": [(item["w0"], item["w1"]) for item in self.groups.values()],
                "residual_detail": {"query": torch.cat(q_residual), "mu": torch.cat(mu_by_group),
                                    "raw_query": torch.cat(raw_query),
                                    "query_indices": torch.cat(query_indices),
                                    "logits": logits.detach(), "ce": ce.detach(), "valid": valid.detach(),
                                    "b": torch.cat(b_means), "beta_site": torch.cat(beta_site),
                                    "beta_pair": torch.cat(pair_betas), "pair_ess": torch.cat(pair_esses),
                                    "groups": self.groups}}

    def raw_p1_pair_detail(self, z_img: torch.Tensor, d4_detail: dict) -> dict:
        """Preflight-only P1 pair CE in the original standardized image coordinates."""
        query = z_img[self.train]
        teacher = d4_detail["teacher"]
        all_logits, all_valid, all_mu = [], [], []
        for g in range(3):
            data = self.groups[g]
            q, s = data["variants"]["R1"]["q"], data["variants"]["R1"]["s"]
            y_s = self.y[s]
            pair_mu = torch.stack((data["w0"] @ teacher[s][y_s == 0],
                                   data["w1"] @ teacher[s][y_s == 1]), dim=1)
            beta = data["beta_pair"]
            mu = beta[:, None, None] * pair_mu + (1.0 - beta[:, None, None]) * d4_detail["mu"][q]
            u = (query[q] - d4_detail["mean"]) / d4_detail["std"]
            all_logits.append(-(u[:, None] - mu).square().sum(-1) / d4_detail["tau"])
            all_valid.append(d4_detail["valid"][q])
            all_mu.append(mu)
        logits, valid = torch.cat(all_logits), torch.cat(all_valid)
        ce = F.cross_entropy(logits, self.y, reduction="none")
        return {"loss": (ce * valid.to(ce.dtype)).sum() / valid.sum().clamp_min(1),
                "mu": torch.cat(all_mu), "logits": logits, "valid": valid}


def residual_diagnostics(value: dict, prepared: dict | None = None) -> dict:
    d = value["residual_detail"]
    valid = d["valid"]
    def stats(x):
        x = x.detach().float().flatten()
        return {"n": int(x.numel()), "mean": float(x.mean()) if x.numel() else None,
                "sd": float(x.std(unbiased=False)) if x.numel() else None,
                "min": float(x.min()) if x.numel() else None,
                "max": float(x.max()) if x.numel() else None}
    result = {"valid_fraction": float(valid.float().mean()),
            "site_beta": stats(d["beta_site"][valid]),
            "site_beta_zero_fraction": float((d["beta_site"][valid] == 0).float().mean()),
            "pair_beta": stats(d["beta_pair"][valid]),
            "pair_min_ess": stats(d["pair_ess"][valid]),
            "b_norm": stats(d["b"].norm(dim=1)),
            "residual_norm": stats(d["query"].norm(dim=1)),
            "residual_ce": float(value["residual_ce"].detach()),
            "d4_ce": float(value["image_d4"].detach()),
            "ph_ce": float(value["ph_ce"].detach())}
    gap = (d["mu"][:, 0] - d["mu"][:, 1]).norm(dim=1)
    result["residual_class_prototype_gap"] = stats(gap[valid])
    result["residual_prototype_train_accuracy"] = float((d["logits"].argmax(1)[valid] ==
                                                           torch.as_tensor(prepared["y"], device=valid.device)[valid]).float().mean()) if valid.any() and prepared else None
    if prepared:
        index = d["query_indices"].detach().cpu().numpy()
        age = np.asarray(prepared["age"])[index]
        sex = np.asarray(prepared["sex"])[index]
        site = np.asarray(prepared["site"])[index]
        label = np.asarray(prepared["y"])[index]
        def association(vector):
            arr = vector.detach().float().cpu().numpy()
            age_corr = []
            for k in range(arr.shape[1]):
                if np.std(arr[:, k]) > 1e-12 and np.std(age) > 1e-12:
                    age_corr.append(abs(float(np.corrcoef(arr[:, k], age)[0, 1])))
            known = np.isin(sex, [1, 2])
            sex_gap = float(np.linalg.norm(arr[known & (sex == 1)].mean(0) - arr[known & (sex == 2)].mean(0))) if (known & (sex == 1)).any() and (known & (sex == 2)).any() else None
            grand = arr.mean(0)
            total = float(((arr - grand) ** 2).sum())
            between = 0.0
            for item in np.unique(site):
                mask = site == item
                if mask.any():
                    between += int(mask.sum()) * float(((arr[mask].mean(0) - grand) ** 2).sum())
            site_eta = between / total if total > 1e-12 else 0.0
            class_gap = float(np.linalg.norm(arr[label == 0].mean(0) - arr[label == 1].mean(0))) if (label == 0).any() and (label == 1).any() else None
            return {"mean_abs_age_pearson_by_dimension": float(np.mean(age_corr)) if age_corr else None,
                    "sex_mean_gap_l2": sex_gap, "site_between_variance_fraction": site_eta,
                    "class_mean_gap_l2": class_gap}
        result["training_only_associations"] = {"raw": association(d["raw_query"]),
                                                "common_component": association(d["b"]),
                                                "residual": association(d["query"])}
    return result


def gradient_budget(native: torch.Tensor, auxiliary: torch.Tensor, z: torch.Tensor,
                    epoch: int, budget: float):
    gn = torch.autograd.grad(native, z, retain_graph=True, allow_unused=True)[0]
    ga = torch.autograd.grad(auxiliary, z, retain_graph=True, allow_unused=True)[0]
    gn = z.new_zeros(()) if gn is None else gn.detach()
    ga = z.new_zeros(()) if ga is None else ga.detach()
    n, a = gn.norm(), ga.norm()
    eps = CFG["gradient_epsilon"]
    raw = (budget * n / a.clamp_min(eps)).clamp(0.0, CFG["lambda_cap"])
    warmup = min(max(epoch, 0) / CFG["warmup_last_epoch"], 1.0)
    scale = (raw.detach() * warmup).detach()
    cosine = (gn * ga).sum() / (n * a).clamp_min(eps)
    ratio = (scale * a) / n.clamp_min(eps) if float(n) > eps else z.new_zeros(())
    return scale, {"native_gradient_norm": float(n), "auxiliary_gradient_norm": float(a),
                   "lambda": float(scale), "weighted_gradient_ratio": float(ratio),
                   "gradient_cosine": float(cosine),
                   "lambda_cap_hit": bool(float(budget * n / a.clamp_min(eps)) > CFG["lambda_cap"]),
                   "native_gradient_near_zero": bool(float(n) <= eps)}

