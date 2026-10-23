"""Train-reference anchored, bounded FiLM and audited native-forward hooks."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn


class IdentityAnchoredFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 32, bias=False))
        nn.init.zeros_(self.net[-1].weight)

    @staticmethod
    def rho(epoch: int) -> float:
        return 0.1 * min(max(float(epoch) / 9.0, 0.0), 1.0)

    def modulate(self, h: torch.Tensor, d: torch.Tensor, d_ref: torch.Tensor,
                 epoch: int, enabled: bool = True):
        if not enabled:
            return h, None
        delta = self.net(d) - self.net(d_ref.reshape(1, 3))
        dg, db = delta.chunk(2, dim=-1)
        rho = self.rho(epoch)
        gamma = 1.0 + rho * torch.tanh(dg)
        beta = rho * torch.tanh(db)
        return gamma * h + beta, {"gamma": gamma, "beta": beta, "delta": delta, "rho": rho}


def fit_encode_conditions(age, sex, train_indices):
    age = np.asarray(age, dtype=np.float64)
    sex = np.asarray(sex, dtype=np.float64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    valid = np.isfinite(age) & (age >= 0) & (age <= 120)
    train_valid = train_indices[valid[train_indices]]
    median = float(np.median(age[train_valid])) if len(train_valid) else 17.0
    filled = np.where(valid, age, median)
    mean = float(filled[train_indices].mean())
    sd = float(filled[train_indices].std(ddof=0))
    if sd <= 1e-12:
        sd = 1.0
    z = np.clip((filled - mean) / sd, -3.0, 3.0)
    male = (sex == 1).astype(np.float64)
    female = (sex == 2).astype(np.float64)
    conditions = np.column_stack((z, male, female)).astype(np.float32)
    ref = conditions[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    info = {"train_n": int(len(train_indices)), "age_valid_train_n": int(len(train_valid)),
            "age_missing_or_invalid_train_n": int(len(train_indices)-len(train_valid)),
            "age_median": median, "age_mean": mean, "age_std_population": sd,
            "unknown_sex_train_n": int(np.sum(~np.isin(sex[train_indices], [1, 2]))),
            "age_z_clip": [-3.0, 3.0], "reference": ref.tolist()}
    return conditions, ref, info


def private_seed(seed: int, fold: int, purpose: str) -> int:
    raw = hashlib.sha256(f"{seed}/{fold}/IAFiLM/{purpose}".encode()).hexdigest()
    return int(raw[:15], 16) % (2**32)


def derange_within_splits(n: int, splits: dict[str, np.ndarray], sites,
                          seed: int, fold: int):
    sites = np.asarray(sites).astype(str)
    permutation = np.arange(n, dtype=np.int64)
    records = {}
    seen = set()
    for split_name in ("train", "val", "test"):
        indices = np.asarray(splits[split_name], dtype=np.int64)
        assert not seen.intersection(indices.tolist())
        seen.update(indices.tolist())
        for site in np.unique(sites[indices]):
            members = indices[sites[indices] == site]
            rng = np.random.default_rng(private_seed(seed, fold, f"F4/{split_name}/{site}"))
            if len(members) >= 2:
                order = rng.permutation(members)
                permutation[order] = np.roll(order, 1)
                assert np.all(permutation[members] != members)
            key = f"{split_name}/{site}"
            records[key] = {"n": int(len(members)), "changed": int(np.sum(permutation[members] != members)),
                            "singleton_n": int(len(members) == 1)}
    assert len(seen) == n and np.array_equal(np.sort(permutation), np.arange(n))
    assert all(np.array_equal(np.sort(permutation[idx]), np.sort(idx))
               for idx in splits.values())
    return permutation, records


class FilmHook:
    """One removable pre-hook at either MM attention or the classifier."""
    def __init__(self, core, film: IdentityAnchoredFiLM, position: str):
        if position not in ("pre_attention_image_only", "post_attention_joint_pre_classifier"):
            raise ValueError(position)
        self.core, self.film, self.position = core, film, position
        self.enabled = False
        self.epoch = 0
        self.conditions = None
        self.reference = None
        self.last = None
        module = core.mm_attention if position == "pre_attention_image_only" else core.clf
        self.handle = module.register_forward_pre_hook(self._pre_hook)
        self.call_count = 0

    def configure(self, enabled: bool, epoch: int, conditions: torch.Tensor | None,
                  reference: torch.Tensor | None):
        self.enabled, self.epoch = bool(enabled), int(epoch)
        self.conditions, self.reference = conditions, reference
        self.last = None
        self.call_count = 0

    def _pre_hook(self, module, args):
        self.call_count += 1
        if self.call_count != 1:
            raise AssertionError("IA-FiLM hook fired more than once in a core forward")
        if not self.enabled:
            return None
        if self.conditions is None or self.reference is None:
            raise AssertionError("FiLM hook enabled without frozen conditions/reference")
        if self.position == "pre_attention_image_only":
            h = args[0]
            transformed, stats = self.film.modulate(h, self.conditions, self.reference,
                                                     self.epoch, True)
            self.last = {"raw": h, "modulated": transformed, **stats}
            return (transformed, *args[1:])
        h = args[0]
        transformed, stats = self.film.modulate(h, self.conditions, self.reference,
                                                 self.epoch, True)
        self.last = {"raw": h, "modulated": transformed, **stats}
        return (transformed, *args[1:])

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

