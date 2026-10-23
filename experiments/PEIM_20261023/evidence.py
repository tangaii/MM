"""Reference-only low-rank disease and phenotype-interaction evidence."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def private_seed(seed: int, fold: int, purpose: str) -> int:
    return int(hashlib.sha256(f"{seed}/{fold}/PEIM/{purpose}".encode()).hexdigest()[:15], 16)


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def diagnosis_target(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Align ASD=0 / TD=1 regression targets to an explicit query order."""
    target = np.asarray(labels, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Disease targets must be available only for legal reference/query rows")
    return np.where(target == 0, 1.0, -1.0)


def _site_design(ref_site: np.ndarray, query_site: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vocab = sorted({str(x) for x in ref_site})
    lut = {site: j for j, site in enumerate(vocab)}

    def encode(values: np.ndarray) -> np.ndarray:
        out = np.zeros((len(values), len(vocab)), dtype=np.float64)
        for i, value in enumerate(values):
            j = lut.get(str(value))
            if j is not None:
                out[i, j] = 1.0
        return out

    return encode(ref_site), encode(query_site), vocab


def _singular_rank(values: np.ndarray, cfg: dict) -> int:
    if values.size == 0 or not np.isfinite(values).all():
        return 0
    threshold = max(float(cfg["singular_abs_floor"]),
                    float(values.max(initial=0.0)) * float(cfg["singular_relative_floor"]))
    return int(np.sum(values > threshold))


def _projector_from_basis(basis: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        return np.zeros((16, 16), dtype=np.float64)
    u = np.asarray(basis[:, :rank], dtype=np.float64)
    q, _ = np.linalg.qr(u)
    q = q[:, :rank]
    return q @ q.T


def interaction_projector(z_ref: np.ndarray, d_ref: np.ndarray, y_ref: np.ndarray,
                           site_ref: np.ndarray, cfg: dict) -> dict:
    """Residualized z-by-phenotype disease evidence, fitted only on ref."""
    z_ref = np.asarray(z_ref, dtype=np.float64)
    d_ref = np.asarray(d_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.int64)
    site_ref = np.asarray(site_ref).astype(str)
    if len(np.unique(y_ref)) != 2 or not np.isin(y_ref, [0, 1]).all():
        return {"P": np.zeros((16, 16)), "B_int": np.zeros((16, 2)), "rank": 0,
                "singular_values": [], "valid": False, "reason": "reference_class_support"}

    mean = z_ref.mean(0)
    std = np.maximum(z_ref.std(0, ddof=0), float(cfg["image_std_floor"]))
    z = (z_ref - mean) / std
    sites, _, vocabulary = _site_design(site_ref, site_ref)
    x0 = np.column_stack((np.ones(len(z)), z, d_ref, sites))
    t = np.where(y_ref == 0, 1.0, -1.0)
    h = (z[:, :, None] * d_ref[:, None, :]).reshape(len(z), 32)
    beta_t = _ridge(x0, t, cfg["ridge_alpha"])
    beta_h = _ridge(x0, h, cfg["ridge_alpha"])
    t_res = t - x0 @ beta_t
    h_res = h - x0 @ beta_h
    b_vec = _ridge(h_res, t_res, cfg["ridge_alpha"]).reshape(16, 2)
    u, singular, _ = np.linalg.svd(b_vec, full_matrices=False)
    rank = min(int(cfg["rank"]), _singular_rank(singular, cfg))
    p = _projector_from_basis(u, rank)
    return {"P": p, "B_int": b_vec, "rank": rank, "singular_values": singular.tolist(),
            "valid": rank > 0, "reason": None if rank > 0 else "numerically_degenerate",
            "mean": mean, "std": std, "beta_t": beta_t, "beta_h": beta_h,
            "vocabulary": vocabulary, "u": u[:, :rank]}


def main_effect_projector(z_ref: np.ndarray, d_ref: np.ndarray, y_ref: np.ndarray,
                          site_ref: np.ndarray, cfg: dict) -> dict:
    """Two-dimensional ordinary disease-direction control, fitted only on ref."""
    z_ref = np.asarray(z_ref, dtype=np.float64)
    d_ref = np.asarray(d_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.int64)
    site_ref = np.asarray(site_ref).astype(str)
    if not np.isin(y_ref, [0, 1]).all() or min(np.sum(y_ref == k) for k in (0, 1)) < 2:
        return {"P": np.zeros((16, 16)), "rank": 0, "valid": False,
                "reason": "reference_class_support", "u1": np.zeros(16)}

    mean = z_ref.mean(0)
    std = np.maximum(z_ref.std(0, ddof=0), float(cfg["image_std_floor"]))
    z = (z_ref - mean) / std
    sites, _, vocabulary = _site_design(site_ref, site_ref)
    confounds = np.column_stack((np.ones(len(z)), d_ref, sites))
    z_res = z - confounds @ _ridge(confounds, z, cfg["ridge_alpha"])
    u1 = z_res[y_ref == 0].mean(0) - z_res[y_ref == 1].mean(0)
    n1 = np.linalg.norm(u1)
    scale = max(float(np.linalg.norm(z_res, axis=1).mean()), 1.0)
    bases = []
    if n1 > max(float(cfg["singular_abs_floor"]), scale * float(cfg["singular_relative_floor"])):
        u1 = u1 / n1
        bases.append(u1)
    else:
        return {"P": np.zeros((16, 16)), "rank": 0, "valid": False,
                "reason": "degenerate_disease_main_direction", "u1": np.zeros(16),
                "u2": np.zeros(16), "mean": mean, "std": std,
                "vocabulary": vocabulary, "eigenvalues": []}

    centered = z_res.copy()
    for label in (0, 1):
        mask = y_ref == label
        centered[mask] -= centered[mask].mean(0)
    cov = centered.T @ centered / max(len(centered), 1)
    projector = np.eye(16) - np.outer(u1, u1)
    cov_orth = projector @ cov @ projector
    eigval, eigvec = np.linalg.eigh((cov_orth + cov_orth.T) / 2)
    order = np.argsort(eigval)[::-1]
    u2 = eigvec[:, order[0]]
    u2 = projector @ u2
    n2 = np.linalg.norm(u2)
    eig_scale = max(float(np.max(eigval, initial=0.0)), 0.0)
    eig_floor = max(float(cfg["singular_abs_floor"]),
                    eig_scale * float(cfg["singular_relative_floor"]))
    if n2 > float(cfg["singular_abs_floor"]) and eigval[order[0]] > eig_floor:
        u2 /= n2
        if not bases or abs(float(np.dot(u1, u2))) < 1e-8:
            bases.append(u2)
    basis = np.column_stack(bases) if bases else np.zeros((16, 0))
    rank = min(int(cfg["rank"]), basis.shape[1])
    p = _projector_from_basis(basis, rank)
    return {"P": p, "rank": rank, "valid": rank > 0,
            "reason": None if rank > 0 else "numerically_degenerate", "u1": u1,
            "u2": u2 if rank > (1 if n1 > 0 else 0) else np.zeros(16),
            "mean": mean, "std": std, "vocabulary": vocabulary,
            "eigenvalues": eigval[order].tolist()}


def fit_reference(z: np.ndarray, reference: np.ndarray, labels: np.ndarray, d: np.ndarray,
                  sites: np.ndarray, cfg: dict, method: str) -> dict:
    """Fit one legal reference object's statistics and the chosen projector."""
    ref = np.asarray(reference, dtype=np.int64)
    zr = np.asarray(z[ref], dtype=np.float64)
    yr = np.asarray(labels[ref], dtype=np.int64)
    dr = np.asarray(d[ref], dtype=np.float64)
    sr = np.asarray(sites[ref]).astype(str)
    if len(np.unique(yr)) < 2:
        raise ValueError("Evidence reference does not contain both disease classes")
    if method == "main":
        proj = main_effect_projector(zr, dr, yr, sr, cfg)
    elif method == "interaction":
        proj = interaction_projector(zr, dr, yr, sr, cfg)
    else:
        raise ValueError(method)
    mean = zr.mean(0)
    std = np.maximum(zr.std(0, ddof=0), float(cfg["image_std_floor"]))
    zn = (zr - mean) / std
    prototypes = np.stack([zn[yr == k].mean(0) for k in (0, 1)])
    margin = np.abs(np.sum((zn - prototypes[0]) ** 2, axis=1) -
                    np.sum((zn - prototypes[1]) ** 2, axis=1))
    tau = max(float(np.median(margin)), 1e-12)
    return {"P": proj["P"],
            "mean": mean, "std": std, "prototypes": prototypes, "tau": tau,
            "rank": proj["rank"], "valid": proj["valid"], "reason": proj["reason"],
            "diagnostics": proj, "reference_n": int(len(ref)),
            "class_counts": np.bincount(yr, minlength=2).tolist()}


def predict_reference_delta(z: np.ndarray, reference: np.ndarray, query: np.ndarray,
                            labels: np.ndarray, d: np.ndarray, sites: np.ndarray,
                            cfg: dict) -> dict:
    """OOF base-vs-interaction ridge probe, fit and evaluated on disjoint rows."""
    ref = np.asarray(reference, dtype=np.int64)
    qry = np.asarray(query, dtype=np.int64)
    zr = np.asarray(z[ref], dtype=np.float64)
    zq = np.asarray(z[qry], dtype=np.float64)
    dr = np.asarray(d[ref], dtype=np.float64)
    dq = np.asarray(d[qry], dtype=np.float64)
    sr = np.asarray(sites[ref]).astype(str)
    sq = np.asarray(sites[qry]).astype(str)
    yr = np.asarray(labels[ref], dtype=np.int64)
    yq = np.asarray(labels[qry], dtype=np.int64)
    if len(np.unique(yr)) != 2:
        raise ValueError("Private ridge reference lacks a class")
    mean = zr.mean(0)
    std = np.maximum(zr.std(0, ddof=0), float(cfg["image_std_floor"]))
    zr = (zr - mean) / std
    zq = (zq - mean) / std
    sr_oh, sq_oh, _ = _site_design(sr, sq)
    x0r = np.column_stack((np.ones(len(ref)), zr, dr, sr_oh))
    x0q = np.column_stack((np.ones(len(qry)), zq, dq, sq_oh))
    hr = (zr[:, :, None] * dr[:, None, :]).reshape(len(ref), 32)
    hq = (zq[:, :, None] * dq[:, None, :]).reshape(len(qry), 32)
    target_r = diagnosis_target(labels, ref)
    target_q = diagnosis_target(labels, qry)
    beta_t = _ridge(x0r, target_r, cfg["ridge_alpha"])
    beta_h = _ridge(x0r, hr, cfg["ridge_alpha"])
    hres_r = hr - x0r @ beta_h
    hres_q = hq - x0q @ beta_h
    b_int = _ridge(hres_r, target_r - x0r @ beta_t, cfg["ridge_alpha"])
    pred_base = x0q @ beta_t
    pred_interaction = pred_base + hres_q @ b_int
    mse_base = float(np.mean((target_q - pred_base) ** 2))
    mse_interaction = float(np.mean((target_q - pred_interaction) ** 2))
    bmat = b_int.reshape(16, 2)
    u, singular, _ = np.linalg.svd(bmat, full_matrices=False)
    rank = min(int(cfg["rank"]), _singular_rank(singular, cfg))
    p = _projector_from_basis(u, rank)
    return {"mse_base": mse_base, "mse_interaction": mse_interaction,
            "delta_mse": mse_base - mse_interaction, "rank": rank,
            "singular_values": singular.tolist(), "projector": p,
            "b_int": bmat, "pred_base": pred_base, "pred_interaction": pred_interaction}


def shuffle_conditions(indices: np.ndarray, sites: np.ndarray, groups: np.ndarray,
                       age: np.ndarray, sex: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """No-label joint age/sex derangement inside group-by-site buckets."""
    indices = np.asarray(indices, dtype=np.int64)
    perm = np.arange(len(age), dtype=np.int64)
    moved = 0
    buckets = {}
    for idx in indices:
        buckets.setdefault((int(groups[idx]), str(sites[idx])), []).append(int(idx))
    for key in sorted(buckets):
        members = np.asarray(buckets[key], dtype=np.int64)
        if len(members) < 2:
            continue
        rng = np.random.default_rng(private_seed(seed, int(key[0]), f"shuffle/{key[1]}"))
        order = rng.permutation(members)
        recipients = np.roll(order, 1)
        perm[recipients] = order
        moved += len(members)
    age_values = np.asarray(age, dtype=np.float64)
    sex_values = np.asarray(sex, dtype=np.float64)
    changed_age = ~np.isclose(age_values[perm], age_values, rtol=0.0, atol=0.0, equal_nan=True)
    changed_sex = ~np.isclose(sex_values[perm], sex_values, rtol=0.0, atol=0.0, equal_nan=True)
    changed_joint = changed_age | changed_sex
    return perm, {"index_change_fraction": moved / max(len(indices), 1),
                  "age_change_fraction": float(changed_age[indices].mean()) if len(indices) else 0.0,
                  "sex_change_fraction": float(changed_sex[indices].mean()) if len(indices) else 0.0,
                  "joint_value_change_fraction": float(changed_joint[indices].mean()) if len(indices) else 0.0,
                  "singletons": sum(len(v) == 1 for v in buckets.values()),
                  "bucket_count": len(buckets)}


def projector_properties(p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=np.float64)
    eig = np.linalg.eigvalsh((p + p.T) / 2)
    return {"symmetry_max_abs": float(np.max(np.abs(p - p.T))),
            "idempotence_frobenius": float(np.linalg.norm(p @ p - p)),
            "minimum_eigenvalue": float(eig.min()),
            "rank": int(np.linalg.matrix_rank(p, tol=1e-6)),
            "trace": float(np.trace(p))}

