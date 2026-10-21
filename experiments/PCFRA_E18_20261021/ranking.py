"""PCFRA-E18 train-only ranking objective and historical D4 prototype teacher."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


GROUP_SPECS = {
    "J1": {"prototype": False, "margin_scale": 0.0, "site_balanced": True},
    "J2": {"prototype": True, "margin_scale": 0.5, "site_balanced": False},
    "J3": {"prototype": True, "margin_scale": 0.25, "site_balanced": False},
    "J4": {"prototype": True, "margin_scale": 0.25, "site_balanced": True},
}


def private_seed(seed, fold, purpose):
    return int(hashlib.sha256(f"{seed}/{fold}/PCDCA/{purpose}".encode()).hexdigest()[:15], 16)


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def prepare(labels, meta, train, seed, fold, cfg):
    """Reuse PCFRA G3/D4 train-only split assignment, preprocessing, and RNG keys."""
    tr = np.asarray(train, np.int64)
    y = np.asarray(labels, np.int64)[tr].copy()
    age = np.asarray(meta["age"], float)[tr].copy()
    sex = np.asarray(meta["sex"], float)[tr].copy()
    site = np.asarray(meta["site"], str)[tr].copy()
    bad_age = ~np.isfinite(age) | (age <= 0) | (age > cfg["age_valid_range"][1])
    median = float(np.median(age[~bad_age])) if (~bad_age).any() else 17.0
    age[bad_age] = median
    sex = np.where(np.isin(sex, [1, 2]), sex, -1).astype(np.int64)
    site = np.where(np.isin(site, ["", "nan", "None", "-1"]), "__MISSING_SITE__", site)
    bandwidth = max(cfg["age_bandwidth_floor"], float(np.subtract(*np.percentile(age, [75, 25]))) / 2)
    nf = cfg["prototype_folds"]
    if len(tr) < nf or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid outer-training subjects/classes")

    rng = np.random.default_rng(private_seed(seed, fold, "prototype-folds"))
    pf = np.full(len(tr), -1, np.int64)
    class_count = np.zeros((nf, 2), int)
    site_count = {s: np.zeros(nf, int) for s in np.unique(site)}
    for k in (0, 1):
        sites = sorted(np.unique(site[y == k]), key=lambda s: (-np.sum((site == s) & (y == k)), s))
        for s in sites:
            for i in rng.permutation(np.flatnonzero((site == s) & (y == k))):
                choices = np.flatnonzero(class_count[:, k] == class_count[:, k].min())
                choices = choices[site_count[s][choices] == site_count[s][choices].min()]
                g = int(rng.choice(choices))
                pf[i] = g
                class_count[g, k] += 1
                site_count[s][g] += 1

    p, singles = np.arange(len(tr)), []
    for s in sorted(set(site)):
        members = np.flatnonzero(site == s)
        srng = np.random.default_rng(private_seed(seed, fold, f"condition-shuffle/{s}"))
        order = srng.permutation(members)
        if len(order) > 1:
            p[order] = np.roll(order, 1)
            assert np.all(p[members] != members)
        else:
            singles.extend(members.tolist())
    assert np.array_equal(np.sort(p), np.arange(len(tr))) and np.array_equal(site, site[p])

    halves = np.full((nf, len(tr)), -1, np.int64)
    half_records = []
    for g in range(nf):
        ref = pf != g
        hrng = np.random.default_rng(private_seed(seed, fold, f"site-halves/{g}"))
        order = list(np.unique(site[ref]))
        tie = {s: float(hrng.random()) for s in order}
        order.sort(key=lambda s: (-np.sum(ref & (site == s)), tie[s]))
        counts = np.zeros((2, 2), int)
        assignment = {}
        for s in order:
            mask = ref & (site == s)
            cnt = np.bincount(y[mask], minlength=2)
            scores = []
            for h in (0, 1):
                trial = counts.copy()
                trial[h] += cnt
                scores.append(float(np.square((trial[0] - trial[1]) /
                                              np.maximum(np.bincount(y[ref], minlength=2), 1)).sum()))
            best = np.flatnonzero(np.asarray(scores) == min(scores))
            h = int(hrng.choice(best))
            counts[h] += cnt
            halves[g, mask] = h
            assignment[str(s)] = h
        assert not (set(site[halves[g] == 0]) & set(site[halves[g] == 1]))
        half_records.append({"sites": assignment, "class_counts": counts.tolist()})

    changed = (age != age[p]) | (sex != sex[p])

    info = {
        "train_indices": tr.tolist(),
        "train_label_sha256": array_hash(y),
        "prototype_fold": pf.tolist(),
        "prototype_fold_sha256": array_hash(pf),
        "prototype_class_counts": class_count.tolist(),
        "site_halves": half_records,
        "site_halves_sha256": array_hash(halves),
        "age_bandwidth": bandwidth,
        "age_fill_train_median": median,
        "age_missing_count": int(bad_age.sum()),
        "age_all_missing_fallback": bool(bad_age.all()),
        "unknown_sex_count": int((sex == -1).sum()),
        "shuffle_sources": tr[p].tolist(),
        "shuffle_sha256": array_hash(p),
        "shuffle_moved_fraction": float(np.mean(p != np.arange(len(tr)))),
        "shuffle_changed_fraction": float(changed.mean()),
        "singleton_subjects": tr[singles].tolist(),
        "mean_abs_age_change": float(np.abs(age - age[p]).mean()),
        "sex_changed_fraction": float(np.mean(sex != sex[p])),
        "per_site_changes": {s: {"n": int((site == s).sum()),
                                 "changed_fraction": float(changed[site == s].mean())}
                             for s in sorted(set(site))},
        "uses_heldout_labels_or_covariates": False,
    }
    return {"train": tr, "y": y, "age": age, "sex": sex, "site": site,
            "folds": pf, "halves": halves, "permutation": p, "bandwidth": bandwidth, "info": info}


class PCFRA_E18_Objective:
    def __init__(self, prepared, group, device):
        if group not in GROUP_SPECS:
            raise ValueError(group)
        self.group = group
        self.spec = GROUP_SPECS[group]
        self.eps = 1e-12
        self.train = torch.as_tensor(prepared["train"], dtype=torch.long, device=device)
        self.y = torch.as_tensor(prepared["y"], dtype=torch.long, device=device)
        self.asd = torch.where(self.y == 0)[0]
        self.td = torch.where(self.y == 1)[0]
        if not self.asd.numel() or not self.td.numel():
            raise ValueError("Outer train must contain ASD and TD")

        age = torch.as_tensor(prepared["age"], dtype=torch.float32, device=device)
        sex = torch.as_tensor(prepared["sex"], dtype=torch.long, device=device)
        sites = np.asarray(prepared["site"], str)
        site_equal = torch.as_tensor(sites[:, None] == sites[None, :], dtype=torch.bool, device=device)
        self.site = sites
        self.site_equal = site_equal.index_select(0, self.asd).index_select(1, self.td)
        ka = torch.exp(-0.5 * ((age[:, None] - age[None, :]) / float(prepared["bandwidth"])).square())
        known = (sex[:, None] > 0) & (sex[None, :] > 0)
        ks = torch.where(known, torch.where(sex[:, None] == sex[None, :], 1.0, 0.5), 0.75)
        self.kernel = (ka * ks).detach()
        self.w = (self.kernel * site_equal.to(self.kernel.dtype)).index_select(0, self.asd).index_select(1, self.td).detach()

        self.reference = (torch.as_tensor(prepared["folds"], device=device)[:, None] !=
                          torch.as_tensor(prepared["folds"], device=device)[None, :]).detach()
        if bool(self.reference.diagonal().any()):
            raise AssertionError("Prototype teacher must exclude query private fold")
        self.local_w = torch.stack([
            self.kernel * self.reference.to(self.kernel.dtype) *
            (self.y == c)[None, :].to(self.kernel.dtype) for c in (0, 1)
        ], dim=1).detach()
        self.global_w = torch.stack([
            self.reference.to(self.kernel.dtype) *
            (self.y == c)[None, :].to(self.kernel.dtype) for c in (0, 1)
        ], dim=1).detach()

        self.sites = []
        counts = {}
        for s in sorted(set(sites.tolist())):
            mask = torch.as_tensor(sites == s, device=device)
            ai = torch.where(mask & (self.y == 0))[0]
            ti = torch.where(mask & (self.y == 1))[0]
            counts[s] = {"n": int(mask.sum()), "n_asd": int(ai.numel()), "n_td": int(ti.numel())}
            if ai.numel() and ti.numel():
                rows_mask = torch.as_tensor(sites[self.asd.cpu().numpy()] == s, device=device)
                cols_mask = torch.as_tensor(sites[self.td.cpu().numpy()] == s, device=device)
                rows = torch.where(rows_mask)[0]
                cols = torch.where(cols_mask)[0]
                n_s = int(ai.numel() + ti.numel())
                self.sites.append({"site": s, "rows": rows, "cols": cols, "n": n_s,
                                   "weight_raw": float(np.sqrt(n_s))})
        denom = sum(x["weight_raw"] for x in self.sites)
        for x in self.sites:
            x["weight"] = x["weight_raw"] / denom if denom else 0.0
        self.site_weights = {x["site"]: x["weight"] for x in self.sites}
        self.site_counts = counts
        self.row_valid = self.w.sum(dim=1) > 0
        self.col_valid = self.w.sum(dim=0) > 0
        valid_rows = max(int(self.row_valid.sum()), 1)
        valid_cols = max(int(self.col_valid.sum()), 1)
        self.anchor_site_weights = {}
        asd_sites = self.site[self.asd.detach().cpu().numpy()]
        td_sites = self.site[self.td.detach().cpu().numpy()]
        for s in sorted(counts):
            row_mask = torch.as_tensor(asd_sites == s, device=device) & self.row_valid
            col_mask = torch.as_tensor(td_sites == s, device=device) & self.col_valid
            self.anchor_site_weights[s] = 0.5 * float(row_mask.sum()) / valid_rows + \
                0.5 * float(col_mask.sum()) / valid_cols
        self.static_info = {
            "group": group,
            "n_train": int(len(prepared["train"])),
            "n_asd": int(self.asd.numel()),
            "n_td": int(self.td.numel()),
            "n_all_pairs": int(self.asd.numel() * self.td.numel()),
            "n_conditional_pairs": int((self.w > 0).sum()),
            "conditional_asd_anchor_fraction": float(self.row_valid.float().mean()),
            "conditional_td_anchor_fraction": float(self.col_valid.float().mean()),
            "site_counts": counts,
            "effective_sites": [x["site"] for x in self.sites],
            "site_weight_shares": self.site_weights,
            "original_anchor_weight_shares": self.anchor_site_weights,
            "site_weight_change_vs_anchor": {
                s: self.site_weights.get(s, 0.0) - self.anchor_site_weights[s] for s in counts
            },
            "site_weight_min": min(self.site_weights.values()) if self.site_weights else None,
            "site_weight_max": max(self.site_weights.values()) if self.site_weights else None,
            "nyu_condition_weight_share": float(self.site_weights.get("NYU", 0.0)),
        }
        self.prototype_calls = 0
        self.last_info = {}
        self.last_tensors = {}
        self.last_site_losses = {}

    @staticmethod
    def _means(weights, features, eps=1e-12):
        den = weights.sum(dim=-1)
        mu = torch.matmul(weights, features) / den.clamp_min(eps)[..., None]
        ess = den.square() / weights.square().sum(dim=-1).clamp_min(eps)
        return mu, ess, den

    @torch.no_grad()
    def original_d4_prototype_evidence(self, z_img):
        self.prototype_calls += 1
        query = z_img.index_select(0, self.train).detach()
        if query.ndim != 2 or query.shape[1] != 16 or not bool(torch.isfinite(query).all()):
            raise ValueError("Expected finite [871,16] Imaging GTUNet output")
        mean = query.mean(dim=0)
        std = query.std(dim=0, unbiased=False).clamp_min(0.05)
        teacher = (query - mean) / std
        tau = (teacher - teacher.mean(0)).square().sum(-1).mean().clamp_min(1.0)
        global_mu, _, global_den = self._means(self.global_w, teacher)
        local_mu, ess, local_den = self._means(self.local_w, teacher)
        valid = (global_den > 0).all(-1)
        beta = ((ess.min(-1).values - 8.0) / 32.0).clamp(0.0, 1.0)
        beta = torch.where((local_den > self.eps).all(-1), beta, torch.zeros_like(beta))
        mu = beta[:, None, None] * local_mu + (1.0 - beta[:, None, None]) * global_mu
        q = -(teacher[:, None, :] - mu).square().sum(-1) / tau
        evidence = (q[:, 0] - q[:, 1]).detach()
        if not bool(valid.all()) or not bool(torch.isfinite(evidence).all()) or evidence.requires_grad:
            raise FloatingPointError("Invalid stop-gradient historical D4 teacher")
        return evidence

    def centered_margin(self, evidence):
        e_asd = evidence.index_select(0, self.asd)
        e_td = evidence.index_select(0, self.td)
        d = (e_asd[:, None] - e_td[None, :]).detach()
        support = self.w > 0
        temperature = d[support].abs().median().clamp_min(1.0).detach() if bool(support.any()) else d.new_tensor(1.0)
        r = torch.tanh(d / temperature)
        c = torch.zeros_like(r)
        mu_by_site = {}
        for item in self.sites:
            rows, cols, s = item["rows"], item["cols"], item["site"]
            ww = self.w.index_select(0, rows).index_select(1, cols)
            rr = r.index_select(0, rows).index_select(1, cols)
            mu = ((ww * rr).sum() / ww.sum().clamp_min(self.eps)).detach()
            cc = (rr - mu) / (1.0 + mu.abs())
            c[rows[:, None], cols[None, :]] = cc
            mu_by_site[s] = float(mu)
            centered = (ww * cc).sum() / ww.sum().clamp_min(self.eps)
            if abs(float(centered)) > 1e-5:
                raise AssertionError(f"Site {s} margin is not centered")
        if float(c.abs().max()) > 1.0 + 1e-6:
            raise AssertionError("Centered margin residual exceeds [-1,1]")
        margin = 1.0 + self.spec["margin_scale"] * c
        margin_means = {}
        for item in self.sites:
            rows, cols, s = item["rows"], item["cols"], item["site"]
            ww = self.w.index_select(0, rows).index_select(1, cols)
            mm = margin.index_select(0, rows).index_select(1, cols)
            mmean = float((ww * mm).sum() / ww.sum().clamp_min(self.eps))
            margin_means[s] = mmean
            if abs(mmean - 1.0) > 1e-5:
                raise AssertionError(f"Site {s} weighted margin mean={mmean}")
        self.last_tensors = {"evidence": evidence, "D": d, "temperature": temperature,
                             "mu_by_site": mu_by_site, "C": c, "margin": margin}
        return margin, {"evidence": evidence, "D": d, "temperature": temperature, "mu_by_site": mu_by_site,
                        "C": c, "margin": margin, "weighted_mean_by_site": margin_means}

    def global_ranking_loss(self, delta):
        return F.softplus(1.0 - delta).mean()

    def original_conditional_loss(self, pair_loss):
        row_den = self.w.sum(dim=1)
        col_den = self.w.sum(dim=0)
        row = (self.w * pair_loss).sum(dim=1) / row_den.clamp_min(self.eps)
        col = (self.w * pair_loss).sum(dim=0) / col_den.clamp_min(self.eps)
        row_loss = row[self.row_valid].mean() if bool(self.row_valid.any()) else pair_loss.sum() * 0.0
        col_loss = col[self.col_valid].mean() if bool(self.col_valid.any()) else pair_loss.sum() * 0.0
        return 0.5 * (row_loss + col_loss)

    def original_site_contributions(self, pair_loss):
        row_den = self.w.sum(dim=1)
        col_den = self.w.sum(dim=0)
        row = (self.w * pair_loss).sum(dim=1) / row_den.clamp_min(self.eps)
        col = (self.w * pair_loss).sum(dim=0) / col_den.clamp_min(self.eps)
        pieces = {}
        asd_sites = self.site[self.asd.detach().cpu().numpy()]
        td_sites = self.site[self.td.detach().cpu().numpy()]
        for s in sorted(self.site_counts):
            row_mask = torch.as_tensor(asd_sites == s, device=pair_loss.device) & self.row_valid
            col_mask = torch.as_tensor(td_sites == s, device=pair_loss.device) & self.col_valid
            row_piece = 0.5 * row[row_mask].sum() / self.row_valid.sum().clamp_min(1)
            col_piece = 0.5 * col[col_mask].sum() / self.col_valid.sum().clamp_min(1)
            pieces[s] = row_piece + col_piece
        total = sum(pieces.values(), pair_loss.sum() * 0.0)
        return {s: float((v / total.clamp_min(self.eps)).detach()) for s, v in pieces.items()}

    def site_balanced_conditional_loss(self, pair_loss):
        losses = {}
        weighted = pair_loss.sum() * 0.0
        for item in self.sites:
            rows, cols = item["rows"], item["cols"]
            w = self.w.index_select(0, rows).index_select(1, cols)
            ell = pair_loss.index_select(0, rows).index_select(1, cols)
            row_den = w.sum(1)
            col_den = w.sum(0)
            row = (w * ell).sum(1) / row_den.clamp_min(self.eps)
            col = (w * ell).sum(0) / col_den.clamp_min(self.eps)
            if not bool((row_den > 0).all()) or not bool((col_den > 0).all()):
                raise AssertionError(f"Effective site {item['site']} contains an unsupported anchor")
            site_loss = 0.5 * row.mean() + 0.5 * col.mean()
            losses[item["site"]] = site_loss
            weighted = weighted + item["weight"] * site_loss
        return weighted, losses

    def forward(self, logits, z_img):
        if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] <= int(self.train.max()):
            raise ValueError("Expected all-subject logits [N,2] covering outer train")
        train_logits = logits.index_select(0, self.train)
        score = train_logits[:, 0] - train_logits[:, 1]
        delta = score.index_select(0, self.asd)[:, None] - score.index_select(0, self.td)[None, :]
        global_loss = self.global_ranking_loss(delta)
        prototype_info = None
        if self.spec["prototype"]:
            evidence = self.original_d4_prototype_evidence(z_img)
            margin, prototype_info = self.centered_margin(evidence)
        else:
            margin = torch.ones_like(delta)
        pair_loss = F.softplus(margin - delta)
        if self.spec["site_balanced"]:
            conditional, per_site = self.site_balanced_conditional_loss(pair_loss)
            self.last_site_losses = per_site
            contribution_den = conditional.detach().clamp_min(self.eps)
            site_contributions = {
                s: float(self.site_weights[s] * loss.detach() / contribution_den)
                for s, loss in per_site.items()
            }
        else:
            conditional = self.original_conditional_loss(pair_loss)
            per_site = None
            self.last_site_losses = {}
            site_contributions = self.original_site_contributions(pair_loss)
        auxiliary = 0.5 * global_loss + 0.5 * conditional
        if not bool(torch.isfinite(auxiliary)) or not auxiliary.requires_grad:
            raise FloatingPointError("Invalid PCFRA-E18 auxiliary loss")
        self.last_info = {
            **self.static_info,
            "global_loss": float(global_loss.detach()),
            "conditional_loss": float(conditional.detach()),
            "per_site_conditional_loss": ({k: float(v.detach()) for k, v in per_site.items()}
                                           if per_site is not None else None),
            "site_conditional_contribution_shares": site_contributions,
            "prototype_used": self.spec["prototype"],
            "margin_scale": self.spec["margin_scale"],
            "margin_min": float(margin.detach().min()),
            "margin_max": float(margin.detach().max()),
            "margin_mean": float(margin.detach().mean()),
            "margin_abs_deviation": float((margin.detach() - 1.0).abs().mean()),
            "temperature": float(prototype_info["temperature"]) if prototype_info else None,
            "teacher_evidence_std": float(prototype_info["evidence"].std(unbiased=False))
            if prototype_info else None,
            "margin_std": float(prototype_info["margin"].std(unbiased=False)) if prototype_info else None,
            "site_margin_weighted_mean": prototype_info["weighted_mean_by_site"] if prototype_info else None,
            "site_center_mu": prototype_info["mu_by_site"] if prototype_info else None,
        }
        return auxiliary

    __call__ = forward
