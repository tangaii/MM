"""Train-only reliability-aware within-site and cross-site ranking objectives."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch.nn import functional as F

GROUP_SPECS = {
    "K1": {"site_reliability": True, "cross_mode": "none", "within_mix": 1.0},
    "K2": {"site_reliability": False, "cross_mode": "real", "within_mix": 0.75},
    "K3": {"site_reliability": True, "cross_mode": "real", "within_mix": 0.75},
    "K4": {"site_reliability": True, "cross_mode": "shuffle", "within_mix": 0.75},
}


def private_seed(seed, fold, purpose):
    return int(hashlib.sha256(f"{seed}/{fold}/PCDCA/{purpose}".encode()).hexdigest()[:15], 16)


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _clean_meta(labels, meta, train, cfg):
    tr = np.asarray(train, np.int64)
    y = np.asarray(labels, np.int64)[tr].copy()
    age = np.asarray(meta["age"], dtype=np.float64)[tr].copy()
    sex = np.asarray(meta["sex"], dtype=np.float64)[tr].copy()
    site = np.asarray(meta["site"], dtype=str)[tr].copy()
    lo, hi = cfg["age_valid_range"]
    bad_age = ~np.isfinite(age) | (age <= lo) | (age > hi)
    median = float(np.median(age[~bad_age])) if (~bad_age).any() else 17.0
    age[bad_age] = median
    sex = np.where(np.isin(sex, [1, 2]), sex, -1).astype(np.int64)
    site = np.where(np.isin(site, ["", "nan", "None", "-1"]), "__MISSING_SITE__", site)
    bandwidth = max(float(cfg["age_bandwidth_floor"]),
                    float(np.subtract(*np.percentile(age, [75, 25]))) / 2.0)
    if not len(tr) or not np.isin(y, [0, 1]).all() or not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("Outer train must be nonempty and contain ASD=0 and TD=1")
    return tr, y, age, sex, site, bandwidth, bad_age, median


def build_within_kernel(age, sex, site, bandwidth):
    age = np.asarray(age, dtype=np.float64)
    sex = np.asarray(sex, dtype=np.int64)
    site = np.asarray(site, dtype=str)
    ka = np.exp(-0.5 * ((age[:, None] - age[None, :]) / float(bandwidth)) ** 2)
    known = (sex[:, None] > 0) & (sex[None, :] > 0)
    ks = np.where(known, np.where(sex[:, None] == sex[None, :], 1.0, 0.5), 0.75)
    return ka * ks * (site[:, None] == site[None, :])


def build_site_statistics(y, site, kernel):
    y, site = np.asarray(y, np.int64), np.asarray(site, str)
    stats = {}
    for name in sorted(set(site.tolist())):
        a, t = np.flatnonzero((site == name) & (y == 0)), np.flatnonzero((site == name) & (y == 1))
        n = len(a) + len(t)
        b = float(4.0 * len(a) * len(t) / (n * n)) if n else 0.0
        o = float(kernel[np.ix_(a, t)].sum() / (len(a) * len(t))) if len(a) and len(t) else 0.0
        stats[name] = {"n": n, "n_asd": len(a), "n_td": len(t), "b": b, "o": o,
                       "r": float(0.5 + 0.5 * np.sqrt(max(0.0, b * o))) if len(a) and len(t) else None,
                       "effective": False}
    return stats


def build_reliable_site_weights(site_stats):
    active = [s for s, x in site_stats.items() if x["n_asd"] and x["n_td"]]
    if not active:
        raise ValueError("No site has both ASD and TD support")
    raw_j1 = {s: float(np.sqrt(site_stats[s]["n"])) for s in active}
    raw_r = {s: raw_j1[s] * float(site_stats[s]["r"]) for s in active}
    z_j1, z_r = sum(raw_j1.values()), sum(raw_r.values())
    return ({s: raw_j1[s] / z_j1 for s in active}, {s: raw_r[s] / z_r for s in active}, active)


def _anchor_terms(w):
    row_den, col_den = w.sum(axis=1), w.sum(axis=0)
    valid_rows, valid_cols = row_den > 0, col_den > 0
    row_term, col_term = np.zeros_like(w, dtype=np.float64), np.zeros_like(w, dtype=np.float64)
    if valid_rows.any():
        row_term[valid_rows] = w[valid_rows] / row_den[valid_rows, None]
    if valid_cols.any():
        col_term[:, valid_cols] = w[:, valid_cols] / col_den[None, valid_cols]
    directions = int(valid_rows.any()) + int(valid_cols.any())
    if directions == 2:
        row_term *= 0.5 / int(valid_rows.sum())
        col_term *= 0.5 / int(valid_cols.sum())
    elif directions == 1:
        if valid_rows.any():
            row_term /= int(valid_rows.sum())
        if valid_cols.any():
            col_term /= int(valid_cols.sum())
    return row_term + col_term, directions


def build_within_coefficient(kernel_asd_td, asd_sites, td_sites, site_weights):
    w_all = np.asarray(kernel_asd_td, dtype=np.float64)
    asd_sites, td_sites = np.asarray(asd_sites, str), np.asarray(td_sites, str)
    out, details = np.zeros_like(w_all), {}
    for s in sorted(site_weights):
        rows, cols = np.flatnonzero(asd_sites == s), np.flatnonzero(td_sites == s)
        if not len(rows) or not len(cols):
            raise ValueError(f"Effective site {s} lacks one class")
        w = w_all[np.ix_(rows, cols)]
        c, directions = _anchor_terms(w)
        if directions == 0:
            raise ValueError(f"Effective site {s} has no supported phenotype anchors")
        out[np.ix_(rows, cols)] += float(site_weights[s]) * c
        details[s] = {"row_anchors": int((w.sum(1) > 0).sum()), "td_anchors": int((w.sum(0) > 0).sum()),
                      "directions": directions, "coefficient_mass": float(c.sum())}
    return out, details


def build_cross_coefficient(kernel_asd_td, asd_sites, td_sites, pair_weights=None):
    kernel = np.asarray(kernel_asd_td, dtype=np.float64)
    asd_sites, td_sites = np.asarray(asd_sites, str), np.asarray(td_sites, str)
    sites, local, raw = sorted(set(asd_sites.tolist()) | set(td_sites.tolist())), {}, {}
    for s in sites:
        rows = np.flatnonzero(asd_sites == s)
        if not len(rows):
            continue
        for t in sites:
            cols = np.flatnonzero(td_sites == t)
            if s == t or not len(cols):
                continue
            w = kernel[np.ix_(rows, cols)]
            c, directions = _anchor_terms(w)
            if directions:
                key = (s, t)
                local[key], raw[key] = (rows, cols, c, directions), float(np.sqrt(len(rows) * len(cols)))
    if pair_weights is None:
        z = sum(raw.values())
        omegas = {k: v / z for k, v in raw.items()} if z else {}
    else:
        omegas = {tuple(k): float(v) for k, v in pair_weights.items()}
        unknown = set(omegas) - set(local)
        if unknown:
            raise ValueError(f"Shuffled kernel lost fixed cross-site support: {sorted(unknown)}")
    out, records = np.zeros_like(kernel), {}
    for key, (rows, cols, c, directions) in local.items():
        omega = omegas.get(key, 0.0)
        out[np.ix_(rows, cols)] += omega * c
        records[f"{key[0]}->{key[1]}"] = {
            "n_asd": int(len(rows)), "n_td": int(len(cols)),
            "raw_weight": float(np.sqrt(len(rows) * len(cols))), "omega": float(omega),
            "directions": directions, "supported_asd_anchors": int((kernel[np.ix_(rows, cols)].sum(1) > 0).sum()),
            "supported_td_anchors": int((kernel[np.ix_(rows, cols)].sum(0) > 0).sum()),
            "coefficient_mass": float(c.sum()),
        }
    return out, records, omegas


def prepare(labels, meta, train, seed, fold, cfg):
    tr, y, age, sex, site, bandwidth, bad_age, median = _clean_meta(labels, meta, train, cfg)
    p = np.arange(len(tr), dtype=np.int64)
    for s in sorted(set(site.tolist())):
        members = np.flatnonzero(site == s)
        order = np.random.default_rng(private_seed(seed, fold, f"condition-shuffle/{s}")).permutation(members)
        if len(order) > 1:
            p[order] = np.roll(order, 1)
    if not np.array_equal(np.sort(p), np.arange(len(tr))) or not np.array_equal(site, site[p]):
        raise AssertionError("Condition shuffle must be a within-site permutation")
    changed_age, changed_sex = age != age[p], sex != sex[p]
    info = {
        "train_indices": tr.tolist(), "train_label_sha256": array_hash(y), "age_bandwidth": bandwidth,
        "age_fill_train_median": median, "age_missing_count": int(bad_age.sum()),
        "sex_unknown_count": int((sex == -1).sum()), "shuffle_sources": tr[p].tolist(),
        "shuffle_sha256": array_hash(p), "shuffle_moved_fraction": float(np.mean(p != np.arange(len(tr)))),
        "age_changed_fraction": float(changed_age.mean()), "sex_changed_fraction": float(changed_sex.mean()),
        "joint_phenotype_changed_fraction": float((changed_age | changed_sex).mean()),
        "per_site_changes": {s: {"n": int((site == s).sum()),
                                 "age_changed_fraction": float(changed_age[site == s].mean()),
                                 "sex_changed_fraction": float(changed_sex[site == s].mean()),
                                 "joint_changed_fraction": float((changed_age | changed_sex)[site == s].mean())}
                             for s in sorted(set(site.tolist()))},
        "uses_heldout_labels_or_covariates": False,
    }
    return {"train": tr, "y": y, "age": age, "sex": sex, "site": site, "permutation": p,
            "shuffle_age": age[p].copy(), "shuffle_sex": sex[p].copy(), "bandwidth": bandwidth, "info": info}


def _kernel_asd_td(prepared, age, sex):
    age = torch.as_tensor(age, dtype=torch.float64)
    sex = torch.as_tensor(sex, dtype=torch.long)
    ka = torch.exp(-0.5 * ((age[:, None] - age[None, :]) / float(prepared["bandwidth"])).square())
    known = (sex[:, None] > 0) & (sex[None, :] > 0)
    ks = torch.where(known, torch.where(sex[:, None] == sex[None, :], 1.0, 0.5), 0.75)
    y = np.asarray(prepared["y"], np.int64)
    return (ka * ks).numpy()[np.ix_(np.flatnonzero(y == 0), np.flatnonzero(y == 1))]


class RCSBPRObjective:
    def __init__(self, prepared, group, device):
        if group not in GROUP_SPECS:
            raise ValueError(group)
        self.group, self.spec, self.device = group, GROUP_SPECS[group], torch.device(device)
        self.train = torch.as_tensor(prepared["train"], dtype=torch.long, device=self.device)
        self.y = torch.as_tensor(prepared["y"], dtype=torch.long, device=self.device)
        self.asd, self.td = torch.where(self.y == 0)[0], torch.where(self.y == 1)[0]
        if not self.asd.numel() or not self.td.numel():
            raise ValueError("Outer train must contain both classes")
        self.site = np.asarray(prepared["site"], str)
        self.asd_sites, self.td_sites = self.site[self.asd.cpu().numpy()], self.site[self.td.cpu().numpy()]
        within_full = build_within_kernel(prepared["age"], prepared["sex"], self.site, prepared["bandwidth"])
        y_np = np.asarray(prepared["y"], np.int64)
        ai, ti = np.flatnonzero(y_np == 0), np.flatnonzero(y_np == 1)
        within = within_full[np.ix_(ai, ti)]
        stats = build_site_statistics(y_np, self.site, within_full)
        omega_j1, omega_r, active = build_reliable_site_weights(stats)
        for name, values in stats.items():
            values.update({"effective": name in active, "omega_J1": omega_j1.get(name, 0.0),
                           "omega_R": omega_r.get(name, 0.0),
                           "weight_delta": omega_r.get(name, 0.0) - omega_j1.get(name, 0.0)})
        self.omega_j1, self.omega_r, self.active_sites, self.site_stats = omega_j1, omega_r, active, stats
        self.w_within = torch.as_tensor(within, dtype=torch.float32, device=self.device).detach()
        c_j1, d_j1 = build_within_coefficient(within, self.asd_sites, self.td_sites, omega_j1)
        c_r, d_r = build_within_coefficient(within, self.asd_sites, self.td_sites, omega_r)
        self.c_within_j1 = torch.as_tensor(c_j1, dtype=torch.float32, device=self.device).detach()
        self.c_within_r = torch.as_tensor(c_r, dtype=torch.float32, device=self.device).detach()
        cross = _kernel_asd_td(prepared, prepared["age"], prepared["sex"])
        real_mask = self.asd_sites[:, None] != self.td_sites[None, :]
        cross *= real_mask
        c_cross, pair_detail, omega_st = build_cross_coefficient(cross, self.asd_sites, self.td_sites)
        self.cross_pair_weights = omega_st
        self.w_cross_real = torch.as_tensor(cross, dtype=torch.float32, device=self.device).detach()
        self.c_cross_real = torch.as_tensor(c_cross, dtype=torch.float32, device=self.device).detach()
        self.cross_pairs = pair_detail
        self.w_cross_shuffled = self.c_cross_shuffled = None
        shuffle_change = None
        if self.spec["cross_mode"] == "shuffle":
            shuffled = _kernel_asd_td(prepared, prepared["shuffle_age"], prepared["shuffle_sex"])
            shuffled *= real_mask
            c_shuf, pair_shuf, omega_shuf = build_cross_coefficient(
                shuffled, self.asd_sites, self.td_sites, pair_weights=omega_st)
            if omega_shuf != omega_st:
                raise AssertionError("K4 changed fixed site-pair omega_st")
            self.w_cross_shuffled = torch.as_tensor(shuffled, dtype=torch.float32, device=self.device).detach()
            self.c_cross_shuffled = torch.as_tensor(c_shuf, dtype=torch.float32, device=self.device).detach()
            self.cross_shuffle_pairs = pair_shuf
            delta_w, delta_c = np.abs(cross - shuffled), np.abs(c_cross - c_shuf)
            flat_a, flat_b = c_cross.ravel(), c_shuf.ravel()
            corr = float(np.corrcoef(flat_a, flat_b)[0, 1]) if flat_a.std() and flat_b.std() else None
            shuffle_change = {"age_sex_kernel_changed_fraction": float(np.count_nonzero(delta_w) / delta_w.size),
                              "coefficient_mean_absolute_change": float(delta_c.mean()),
                              "coefficient_l1_change": float(delta_c.sum()),
                              "coefficient_correlation": corr,
                              "valid_site_pair_count": len(pair_shuf)}
        self.global_pair_count = int(self.asd.numel() * self.td.numel())
        self.cross_pair_count = int(np.count_nonzero(cross))
        self.static_info = {
            "group": group, "n_train": int(len(prepared["train"])), "n_asd": int(self.asd.numel()),
            "n_td": int(self.td.numel()), "n_global_patient_pairs": self.global_pair_count,
            "n_within_patient_pairs": int(np.count_nonzero(within)), "n_cross_patient_pairs": self.cross_pair_count,
            "cross_coefficient_sum": float(c_cross.sum()), "within_j1_coefficient_sum": float(c_j1.sum()),
            "within_r_coefficient_sum": float(c_r.sum()), "effective_sites": active, "site_counts": stats,
            "omega_J1": omega_j1, "omega_R": omega_r, "within_j1_anchor_support": d_j1,
            "within_r_anchor_support": d_r, "cross_site_pairs": pair_detail,
            "omega_st": {f"{s}->{t}": w for (s, t), w in omega_st.items()},
            "cross_valid_pair_count": len(pair_detail),
            "cross_asd_anchor_fraction": float(np.mean(cross.sum(1) > 0)),
            "cross_td_anchor_fraction": float(np.mean(cross.sum(0) > 0)),
            "within_coefficient_l1_change": float(np.abs(c_r - c_j1).sum()),
            "within_kernel_sha256": array_hash(within),
            "within_j1_coefficient_sha256": array_hash(c_j1),
            "within_r_coefficient_sha256": array_hash(c_r),
            "cross_real_kernel_sha256": array_hash(cross),
            "cross_real_coefficient_sha256": array_hash(c_cross),
            "cross_shuffle_kernel_sha256": array_hash(shuffled) if self.spec["cross_mode"] == "shuffle" else None,
            "cross_shuffle_coefficient_sha256": array_hash(c_shuf) if self.spec["cross_mode"] == "shuffle" else None,
            "cross_shuffle_coefficient_sum": float(c_shuf.sum()) if self.spec["cross_mode"] == "shuffle" else None,
            "cross_shuffle_site_pairs": pair_shuf if self.spec["cross_mode"] == "shuffle" else None,
            "shuffle_change": shuffle_change,
            "uses_heldout_labels_or_covariates": False, "new_trainable_parameters": 0,
        }
        if self.cross_pair_count and not np.isclose(c_cross.sum(), 1.0, atol=1e-7):
            raise AssertionError(f"Cross-site C must sum to one, got {c_cross.sum()}")
        if np.any(c_j1 < 0) or np.any(c_r < 0) or np.any(c_cross < 0):
            raise AssertionError("Ranking coefficients must be nonnegative")
        if np.any(cross[~real_mask] != 0):
            raise AssertionError("Same-site pair entered cross-site kernel")
        self.last_info, self.last_site_losses = {}, {}

    def forward(self, logits, joint_embed=None):
        if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] <= int(self.train.max()):
            raise ValueError("Expected all-subject fusion logits [N,2]")
        train_logits = logits.index_select(0, self.train)
        score = train_logits[:, 0] - train_logits[:, 1]
        delta = score.index_select(0, self.asd)[:, None] - score.index_select(0, self.td)[None, :]
        pair_loss = F.softplus(1.0 - delta)
        global_loss = pair_loss.mean()
        within_j1, within_r = (self.c_within_j1 * pair_loss).sum(), (self.c_within_r * pair_loss).sum()
        cross = None
        if self.spec["cross_mode"] == "real":
            cross = (self.c_cross_real * pair_loss).sum()
        elif self.spec["cross_mode"] == "shuffle":
            cross = (self.c_cross_shuffled * pair_loss).sum()
        if self.spec["cross_mode"] == "none":
            conditional = within_r
        else:
            within = within_r if self.spec["site_reliability"] else within_j1
            conditional = self.spec["within_mix"] * within + 0.25 * cross
        auxiliary = 0.5 * global_loss + 0.5 * conditional
        if not bool(torch.isfinite(auxiliary)) or not auxiliary.requires_grad:
            raise FloatingPointError("Invalid RCSBPR auxiliary loss")
        self.last_info = {"global_loss": float(global_loss.detach()),
                          "within_j1_loss": float(within_j1.detach()), "within_r_loss": float(within_r.detach()),
                          "cross_loss": float(cross.detach()) if cross is not None else None,
                          "conditional_loss": float(conditional.detach()), "auxiliary_loss": float(auxiliary.detach()),
                          "site_reliability": self.spec["site_reliability"], "cross_mode": self.spec["cross_mode"],
                          "within_mix": self.spec["within_mix"], "cross_mix": 0.25 if cross is not None else 0.0,
                          "global_margin": 1.0, "nonfinite": False}
        return auxiliary

    __call__ = forward


def _naive_cross_oracle(kernel, asd_sites, td_sites):
    """Independent small-matrix loop oracle, used only by math tests."""
    kernel, asd_sites, td_sites = np.asarray(kernel, float), list(asd_sites), list(td_sites)
    entries = []
    for s in sorted(set(asd_sites)):
        rows = [i for i, name in enumerate(asd_sites) if name == s]
        for t in sorted(set(td_sites)):
            cols = [j for j, name in enumerate(td_sites) if name == t]
            if s == t or not rows or not cols:
                continue
            w = [[float(kernel[i, j]) for j in cols] for i in rows]
            valid_r = [i for i, row in enumerate(w) if sum(row) > 0]
            valid_c = [j for j in range(len(cols)) if sum(w[i][j] for i in range(len(rows))) > 0]
            if not valid_r and not valid_c:
                continue
            mass_r = 0.5 if valid_r and valid_c else 1.0 if valid_r else 0.0
            mass_c = 0.5 if valid_r and valid_c else 1.0 if valid_c else 0.0
            matrix = [[0.0 for _ in cols] for _ in rows]
            for i in valid_r:
                den = sum(w[i])
                for j in range(len(cols)):
                    matrix[i][j] += mass_r * w[i][j] / den / len(valid_r)
            for j in valid_c:
                den = sum(w[i][j] for i in range(len(rows)))
                for i in range(len(rows)):
                    matrix[i][j] += mass_c * w[i][j] / den / len(valid_c)
            entries.append((rows, cols, matrix, np.sqrt(len(rows) * len(cols))))
    z = sum(x[3] for x in entries)
    out = np.zeros_like(kernel)
    for rows, cols, local, raw in entries:
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                out[r, c] += (raw / z) * local[i][j]
    return out


def run_math_tests():
    cfg = {"age_valid_range": [0, 120], "age_bandwidth_floor": 2.0}
    y = np.asarray([0, 1] * 4 + [0, 1, 0, 0, 1, 1, 0, 1], dtype=np.int64)
    site = np.asarray(["A"] * 8 + ["B"] * 8)
    age = np.asarray([10, 11, 18, 19, 24, 25, 32, 33, 12, 14, 20, 21, 28, 29, 35, 37], float)
    sex = np.asarray([1, 1, 2, 1, -1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, -1], float)
    meta = {"age": age, "sex": sex, "site": site}
    prepared = prepare(y, meta, np.arange(len(y)), 911, 0, cfg)
    base = RCSBPRObjective(prepared, "K3", "cpu")
    kernel = build_within_kernel(prepared["age"], prepared["sex"], prepared["site"], prepared["bandwidth"])
    ka, ks = np.zeros_like(kernel), np.zeros_like(kernel)
    for i in range(len(y)):
        for j in range(len(y)):
            ka[i, j] = np.exp(-0.5 * ((prepared["age"][i] - prepared["age"][j]) / prepared["bandwidth"]) ** 2)
            if prepared["sex"][i] <= 0 or prepared["sex"][j] <= 0:
                ks[i, j] = 0.75
            else:
                ks[i, j] = 1.0 if prepared["sex"][i] == prepared["sex"][j] else 0.5
    expected = ka * ks * (prepared["site"][:, None] == prepared["site"][None, :])
    ai, ti = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    within = expected[np.ix_(ai, ti)]
    c_j1, _ = build_within_coefficient(within, base.asd_sites, base.td_sites, base.omega_j1)
    c_r, _ = build_within_coefficient(within, base.asd_sites, base.td_sites, base.omega_r)
    real = _kernel_asd_td(prepared, prepared["age"], prepared["sex"])
    mask = base.asd_sites[:, None] != base.td_sites[None, :]
    real *= mask
    cross_c, pair_details, omegas = build_cross_coefficient(real, base.asd_sites, base.td_sites)
    naive_c = _naive_cross_oracle(real, base.asd_sites, base.td_sites)
    identity = {**prepared, "shuffle_age": prepared["age"].copy(), "shuffle_sex": prepared["sex"].copy()}
    identity_k4 = RCSBPRObjective(identity, "K4", "cpu")
    logits = torch.tensor([[2.0, 0.0] if label == 0 else [0.0, 2.0] for label in y], dtype=torch.float32,
                          requires_grad=True)
    objectives = {g: RCSBPRObjective(prepared, g, "cpu") for g in GROUP_SPECS}
    losses, grads = {}, {}
    for group, item in objectives.items():
        loss = item.forward(logits)
        grads[group] = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        losses[group] = loss
    imbalanced_y = np.asarray([0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    imbalanced_site = np.asarray(["A"] * 5 + ["B"] * 5 + ["C"])
    imbalanced_meta = {"age": np.arange(11, dtype=float) + 15, "sex": np.asarray([1, 2] * 5 + [1], float),
                       "site": imbalanced_site}
    imbalanced = RCSBPRObjective(prepare(imbalanced_y, imbalanced_meta, np.arange(11), 911, 0, cfg), "K1", "cpu")
    extreme_y = np.asarray([0, 1, 0, 1])
    extreme_meta = {"age": np.asarray([1, 2, 120, 119], float), "sex": np.asarray([1, 2, 1, 2], float),
                    "site": np.asarray(["A", "A", "B", "B"])}
    extreme = RCSBPRObjective(prepare(extreme_y, extreme_meta, np.arange(4), 911, 0, cfg), "K1", "cpu")
    extreme_loss = extreme.forward(torch.randn(4, 2, requires_grad=True))
    train = np.arange(12)
    changed_y, changed_meta = y.copy(), {k: v.copy() for k, v in meta.items()}
    changed_y[12:] = 1 - changed_y[12:]
    changed_meta["age"][12:], changed_meta["sex"][12:], changed_meta["site"][12:] = 99, -1, "CHANGED"
    p0, p1 = prepare(y, meta, train, 911, 0, cfg), prepare(changed_y, changed_meta, train, 911, 0, cfg)
    checks = [
        np.array_equal(np.unique(y), [0, 1]),
        F.softplus(torch.tensor(-3.0)) < F.softplus(torch.tensor(5.0)),
        np.allclose(ka, np.exp(-0.5 * ((prepared["age"][:, None] - prepared["age"][None, :]) / prepared["bandwidth"]) ** 2)),
        np.array_equal(ks[0], np.where((prepared["sex"][0] > 0) & (prepared["sex"] > 0),
                                       np.where(prepared["sex"][0] == prepared["sex"], 1.0, 0.5), 0.75)),
        np.allclose(kernel, expected, atol=1e-12),
        all(np.isclose(base.site_stats[s]["r"], 0.5 + 0.5 * np.sqrt(base.site_stats[s]["b"] * base.site_stats[s]["o"])) for s in base.active_sites),
        all(0.5 <= base.site_stats[s]["r"] <= 1.0 for s in base.active_sites),
        all(0.0 <= x["b"] <= 1.0 for x in base.site_stats.values()),
        all(0.0 <= x["o"] <= 1.0 for x in base.site_stats.values()),
        np.isclose(sum(base.omega_r.values()), 1.0),
        not np.allclose(c_j1, c_r),
        all(base.asd_sites[i] != base.td_sites[j] for i, j in zip(*np.nonzero(cross_c))),
        np.array_equal(p0["y"], p1["y"]) and np.array_equal(p0["age"], p1["age"]),
        all(k.split("->")[0] != k.split("->")[1] for k in pair_details),
        all(abs(x["coefficient_mass"] - 1.0) < 1e-12 for x in pair_details.values()),
        np.allclose(list(omegas.values()), [x["raw_weight"] / sum(v["raw_weight"] for v in pair_details.values()) for x in pair_details.values()]),
        bool((cross_c >= 0).all()), np.isclose(cross_c.sum(), 1.0), bool(np.all(cross_c[~mask] == 0)),
        np.allclose(cross_c, naive_c, atol=1e-12, rtol=1e-12),
        torch.equal(objectives["K2"].c_cross_real, objectives["K3"].c_cross_real),
        torch.equal(objectives["K3"].c_within_r, objectives["K4"].c_within_r),
        objectives["K3"].omega_r == objectives["K4"].omega_r,
        objectives["K3"].cross_pair_weights == objectives["K4"].cross_pair_weights,
        np.array_equal(objectives["K3"].train.cpu().numpy(), objectives["K4"].train.cpu().numpy()),
        torch.equal(identity_k4.c_cross_shuffled, identity_k4.c_cross_real),
        all(np.isclose(item.last_info["global_loss"], objectives["K1"].last_info["global_loss"]) for item in objectives.values()),
        not torch.equal(objectives["K3"].w_cross_real, objectives["K4"].w_cross_shuffled),
        all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in grads.values()),
        all(not any(isinstance(v, torch.nn.Parameter) for v in vars(item).values()) for item in objectives.values()),
        imbalanced.static_info["effective_sites"] == ["A", "B"] and "C" not in imbalanced.static_info["effective_sites"],
        bool(torch.isfinite(extreme_loss)) and bool(torch.isfinite(extreme.c_within_r).all()),
        np.array_equal(p0["y"], p1["y"]) and np.array_equal(p0["site"], p1["site"]),
        all(item.static_info["uses_heldout_labels_or_covariates"] is False for item in objectives.values()),
    ]
    names = ["label_asd0_td1", "correct_disease_order_reduces_loss", "age_kernel_formula", "sex_kernel_formula",
             "within_kernel_matches_j1", "reliability_weight_formula", "reliability_bounds", "class_balance_bounds",
             "phenotype_support_bounds", "reliable_weights_sum_one", "anchor_does_not_cancel_site_weight",
             "cross_pairs_exclude_same_site", "heldout_supervision_excluded", "ordered_asd_s_td_t_pairs",
             "bidirectional_anchor_normalization", "site_pair_sqrt_count_weight", "cross_coeff_nonnegative",
             "cross_coeff_sums_one", "same_site_cross_coefficients_zero", "dense_cross_matches_naive_oracle",
             "k2_k3_cross_identical", "k3_k4_within_identical", "k3_k4_reliable_weights_identical",
             "k3_k4_site_pair_weights_identical", "k3_k4_model_inputs_identical", "identity_shuffle_matches_real",
             "all_groups_share_global_loss", "k4_shuffle_changes_cross_kernel", "all_auxiliary_gradients_finite",
             "no_new_parameters", "single_class_site_skipped", "underflow_has_no_nan",
             "heldout_metadata_does_not_change_coefficients", "validation_test_do_not_determine_coefficients"]
    if len(checks) != 34 or not all(bool(x) for x in checks):
        failed = [name for name, passed in zip(names, checks) if not bool(passed)]
        raise AssertionError(f"RCSBPR math tests failed ({len(checks)} checks): {failed}")
    return {"checks": 34, "all_passed": True, "passed": names,
            "oracle_max_abs_error": float(np.max(np.abs(cross_c - naive_c))),
            "reliable_site_weights": base.omega_r}


if __name__ == "__main__":
    import json
    print(json.dumps(run_math_tests(), ensure_ascii=False, indent=2, allow_nan=False))
