"""Training-internal raw-phenotype feasibility probes. No neural model code."""
from __future__ import annotations
import numpy as np
from common import CFG, array_hash, private_seed

SC = CFG["stage_a"]
DC = SC["shared_direction"]


def finite(a):
    if not np.isfinite(a).all():
        raise FloatingPointError("NaN/Inf in Stage A statistics")


def canonical_site(site):
    site = np.asarray(site, str)
    return np.where(np.isin(site, ["", "nan", "None", "-1", "N/A"]), "__MISSING_SITE__", site)


class PhenotypeEncoding:
    """Fit only on caller-supplied outer-train rows; no label argument."""
    def __init__(self, raw):
        age = np.asarray(raw["age"], float)
        valid = np.isfinite(age) & (age > 0) & (age <= SC["age_valid_range"][1])
        self.median = float(np.median(age[valid])) if valid.any() else 17.0
        filled = np.where(valid, age, self.median)
        self.mean = float(filled.mean())
        sd = float(filled.std(ddof=0))
        self.std = sd if sd > 1e-12 else 1.0
        self.sites = np.unique(canonical_site(raw["site"]))
        sex = np.asarray(raw["sex"], float)
        self.info = {"fitted_rows": len(age), "age_median": self.median,
                     "age_mean": self.mean, "age_std": self.std,
                     "age_missing": int((~valid).sum()), "all_age_missing": bool(not valid.any()),
                     "unknown_sex": int((~np.isin(sex, [1, 2])).sum()),
                     "site_vocabulary": self.sites.tolist()}

    def transform(self, raw):
        age, sex = np.asarray(raw["age"], float), np.asarray(raw["sex"], float)
        valid = np.isfinite(age) & (age > 0) & (age <= SC["age_valid_range"][1])
        age = (np.where(valid, age, self.median) - self.mean) / self.std
        sex_code = np.where(sex == 1, 0, np.where(sex == 2, 1, 2)).astype(int)
        sx = np.eye(3)[sex_code]
        s = canonical_site(raw["site"])
        sites = (s[:, None] == self.sites[None, :]).astype(float)
        phenotype = np.column_stack((age, sx))
        finite(phenotype)
        return sites, phenotype


def safe_stats(x, floor=None):
    mean = x.mean(0)
    sd = x.std(0, ddof=0)
    sd = np.maximum(sd, floor) if floor is not None else np.where(sd > 1e-12, sd, 1.0)
    return mean, sd


def ridge_predict(x_ref, y_ref, x_query, alpha=1.0):
    """Reference-only standardization; unpenalized target intercept."""
    finite(x_ref); finite(y_ref); finite(x_query)
    xm, xs = safe_stats(x_ref)
    ym = y_ref.mean(0)
    x = (x_ref - xm) / xs
    xq = (x_query - xm) / xs
    a = x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float64)
    w = np.linalg.solve(a, x.T @ (y_ref - ym))
    pred = xq @ w + ym
    finite(pred)
    return pred


def oof_predictions(z, site, phenotype, pf, full):
    """Each patient is predicted exactly once, never fitted to itself."""
    z = np.asarray(z, np.float64)
    x = np.column_stack((site, phenotype)) if full else site
    oof = np.empty_like(z)
    coverage = np.zeros(len(z), int)
    splits = []
    for g in range(3):
        q, r = np.flatnonzero(pf == g), np.flatnonzero(pf != g)
        assert len(q) and len(r) and not np.intersect1d(q, r).size
        oof[q] = ridge_predict(x[r], z[r], x[q], SC["ridge_alpha"])
        coverage[q] += 1
        splits.append({"g": g, "query_n": len(q), "reference_n": len(r),
                       "query_sha256": array_hash(q), "reference_sha256": array_hash(r)})
    assert np.all(coverage == 1)
    return oof, splits


def mismatch(pf, raw_site, fold, replicate):
    """No diagnosis input. Joint variables are moved by the returned permutation."""
    rng = np.random.default_rng(private_seed(fold, f"stage-a-null/{replicate}"))
    site = np.asarray(raw_site, str)
    p = np.arange(len(pf))
    singleton_n = 0
    for g in range(3):
        for s in np.unique(site[pf == g]):
            members = np.flatnonzero((pf == g) & (site == s))
            if len(members) >= 2:
                order = rng.permutation(members)
                p[order] = np.roll(order, 1)
                assert np.all(p[members] != members)
            else:
                singleton_n += len(members)
    assert np.array_equal(np.sort(p), np.arange(len(p)))
    assert np.array_equal(pf[p], pf) and np.array_equal(site[p], site)
    return p, singleton_n


def residualize_site(x, s):
    """Reference-only ridge removal, centered site dummy design, alpha=1."""
    xc, sc = x - x.mean(0), s - s.mean(0)
    b = np.linalg.solve(sc.T @ sc + SC["ridge_alpha"] * np.eye(sc.shape[1]), sc.T @ xc)
    residual = xc - sc @ b
    finite(residual)
    return residual


def shared_directions(z, site, phenotype, pf, train_labels):
    """u is label-free; d alone uses labels from the legal reference."""
    out = []
    vectors = []
    for g in range(3):
        ref = np.flatnonzero(pf != g)
        im, sd = safe_stats(z[ref], CFG["stage_b_preregistered"]["image_standard_deviation_floor"])
        zi = (z[ref] - im) / sd
        pm, ps = safe_stats(phenotype[ref])
        p = (phenotype[ref] - pm) / ps
        rz = residualize_site(zi, site[ref])
        rp = residualize_site(p, site[ref])
        c = rz.T @ rp / len(ref)
        finite(c)
        left, sv, _ = np.linalg.svd(c, full_matrices=False)
        threshold = max(DC["singular_absolute_floor"],
                        DC["singular_relative_floor"] * float(sv[0]) if len(sv) else 0.)
        rank = int((sv > threshold).sum())
        u = left[:, 0] if rank else np.zeros(16)
        pv = float(np.var(zi @ u, ddof=0)) if rank else 0.0
        residual_pv = float(np.var(rz @ u, ddof=0)) if rank else 0.0
        valid = bool(rank > 0 and pv > DC["projection_variance_floor"]
                     and residual_pv > DC["projection_variance_floor"])
        y = train_labels[ref]
        d = zi[y == 0].mean(0) - zi[y == 1].mean(0)
        dn = float(np.linalg.norm(d))
        disease_valid = bool(np.isin([0, 1], y).all() and dn > 1e-12)
        if disease_valid:
            d /= dn
        else:
            d = np.zeros(16)
        dot = float(np.dot(u, d)) if valid and disease_valid else None
        pn2 = float(np.square(u - np.dot(u, d) * d).sum()) if dot is not None else 0.0
        share_var = float(np.square(rz @ u).sum() / max(float(np.square(rz).sum()), 1e-12)) if valid else 0.0
        # Common coordinates for direction stability (raw output perturbation axes).
        raw_u = u / sd
        raw_u /= max(float(np.linalg.norm(raw_u)), 1e-12)
        vectors.append(raw_u if valid else np.zeros(16))
        out.append({"g": g, "reference_n": len(ref), "reference_sha256": array_hash(ref),
                    "singular_values": sv.tolist(), "singular_threshold": threshold,
                    "effective_rank": rank, "direction_valid": valid,
                    "u": u.tolist(), "u_unit_norm": float(np.linalg.norm(u)),
                    "projection_variance": pv, "site_residual_projection_variance": residual_pv,
                    "shared_direction_variance_fraction": share_var,
                    "reference_disease_direction_valid": disease_valid, "disease_delta_norm": dn,
                    "absolute_u_dot_d": None if dot is None else abs(dot),
                    "acute_angle_degrees": None if dot is None else float(np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))),
                    "protected_norm_squared": pn2,
                    "protection_pass": bool(valid and disease_valid and pn2 >= 0.20),
                    "u_does_not_use_labels": True})
    stability = {f"{i}-{j}": abs(float(vectors[i] @ vectors[j]))
                 for i in range(3) for j in range(i + 1, 3)}
    return out, stability


def study(z, raw_train, pf, fold):
    """A1-A4 probe path deliberately has no labels argument."""
    z = np.asarray(z, np.float64)
    finite(z)
    assert z.shape == (len(pf), 16)
    enc = PhenotypeEncoding(raw_train)
    site, ph = enc.transform(raw_train)
    pred_site, partitions = oof_predictions(z, site, ph, pf, False)
    pred_full, full_partitions = oof_predictions(z, site, ph, pf, True)
    assert partitions == full_partitions
    sst = float(np.square(z - z.mean(0)).sum())
    sse_site = float(np.square(z - pred_site).sum())
    sse_full = float(np.square(z - pred_full).sum())
    if sst <= 1e-12:
        return {"degenerate_target": True, "SST": sst, "encoding": enc.info}, (site, ph)
    real = (sse_site - sse_full) / sst
    nulls, changes = [], []
    permutations = set()
    for b in range(SC["null_replicates"]):
        perm, singles = mismatch(pf, raw_train["site"], fold, b)
        psha = array_hash(perm)
        assert psha not in permutations, "Null replicates must not duplicate permutations"
        permutations.add(psha)
        pn, n_partitions = oof_predictions(z, site, ph[perm], pf, True)
        assert n_partitions == partitions
        sse = float(np.square(z - pn).sum())
        nulls.append((sse_site - sse) / sst)
        age_changed = ph[:, 0] != ph[perm, 0]
        sex_changed = np.any(ph[:, 1:] != ph[perm, 1:], axis=1)
        changes.append({"replicate": b, "seed": private_seed(fold, f"stage-a-null/{b}"),
                        "permutation_sha256": psha, "index_changed_fraction": float(np.mean(perm != np.arange(len(perm)))),
                        "age_changed_fraction": float(age_changed.mean()),
                        "sex_changed_fraction": float(sex_changed.mean()),
                        "actual_joint_changed_fraction": float((age_changed | sex_changed).mean()),
                        "singleton_rows": singles,
                        "mean_absolute_age_change_years": float(np.mean(np.abs(ph[:, 0] - ph[perm, 0])) * enc.std),
                        "null_SSE_full": sse, "delta_R2": nulls[-1]})
    p95 = float(np.quantile(nulls, .95, method="linear"))
    return {"degenerate_target": False, "SST": sst, "SSE_site": sse_site, "SSE_full": sse_full,
            "real_delta_R2": real, "site_R2": 1 - sse_site / sst,
            "full_R2_descriptive_only": 1 - sse_full / sst,
            "null_mean": float(np.mean(nulls)), "null_p95": p95,
            "real_minus_null_mean": real - float(np.mean(nulls)), "real_minus_null_p95": real - p95,
            "real_positive": bool(real > 0), "real_above_null_p95": bool(real > p95),
            "null_delta_R2": nulls, "nulls": changes, "encoding": enc.info,
            "partitions": partitions, "probe_uses_labels": False,
            "site_prediction_sha256": array_hash(pred_site), "full_prediction_sha256": array_hash(pred_full),
            "oof_self_fit_count": 0, "oof_coverage": len(z)}, (site, ph)


def decide(records):
    assert len(records) == 10 and sorted(r["fold"] for r in records) == list(range(10))
    ok = [not r["probe"]["degenerate_target"] for r in records]
    vals = [r["probe"].get("real_delta_R2", 0.) for r in records]
    g = SC["gates"]
    positive = sum(o and r["probe"]["real_positive"] for o, r in zip(ok, records))
    above = sum(o and r["probe"]["real_above_null_p95"] for o, r in zip(ok, records))
    rank = sum(all(s["direction_valid"] for s in r["shared_directions"]) for r in records)
    protected = sum(all(s["protection_pass"] for s in r["shared_directions"]) for r in records)
    gates = {"positive_delta": {"observed": positive, "required": ">=7/10", "passed": positive >= 7},
             "above_null_p95": {"observed": above, "required": ">=7/10 strictly", "passed": above >= 7},
             "median_delta": {"observed": float(np.median(vals)), "required": ">0.002", "passed": all(ok) and float(np.median(vals)) > .002},
             "effective_rank": {"observed": rank, "required": ">=7/10 all three reference sets valid", "passed": rank >= 7},
             "protectability": {"observed": protected, "required": ">=7/10 all three squared norms>=0.20", "passed": protected >= 7}}
    go = all(v["passed"] for v in gates.values())
    return {"passed": go, "status": "STAGE_A_PASSED" if go else "NO_GO_STAGE_A", "gates": gates,
            "failed_gates": [k for k, v in gates.items() if not v["passed"]],
            "median_delta_R2": float(np.median(vals)), "formal_tasks_started": 0,
            "formal_preflight_forbidden": not go}
