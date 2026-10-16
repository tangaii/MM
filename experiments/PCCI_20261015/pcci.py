"""Fold-specific phenotype encoding and the shared bounded residual module."""

import hashlib
import numpy as np
import torch
from torch import nn


def private_seed(seed, fold, purpose):
    return int(hashlib.sha256(f"{seed}/{fold}/{purpose}".encode()).hexdigest()[:15], 16)


def encode_conditions(metadata, train, seed, fold, cfg):
    age = np.asarray(metadata["age"], dtype=np.float64)
    sex = np.asarray(metadata["sex"])
    site = np.asarray(metadata["site"])
    mean, sd = float(age[train].mean()), float(age[train].std())
    if not np.isfinite(age).all() or sd <= 0:
        raise ValueError("Invalid age data or training-only standard deviation")
    categories = cfg["sex_categories"]
    if set(sex) - set(categories):
        raise ValueError("Unexpected original SEX category")
    vocabulary = sorted(set(site[train]))
    if len(vocabulary) >= cfg["site_slots"]:
        raise ValueError("Training site vocabulary exceeds registered capacity")
    values = np.zeros((len(age), cfg["condition_dim"]), dtype=np.float32)
    values[:, 0] = (age - mean) / sd
    values[:, 1:3] = np.stack([sex == category for category in categories], 1)
    lookup = {name: i for i, name in enumerate(vocabulary)}
    slots = np.array([lookup.get(name, cfg["unknown_site_slot"]) for name in site])
    values[np.arange(len(age)), 3 + slots] = 1
    reference = values[train].astype(np.float64).mean(0).astype(np.float32)
    reference[0] = 0
    rng = np.random.default_rng(private_seed(seed, fold, "PCCI-site-pair-shuffle"))
    permutation = np.arange(len(age))
    for name in sorted(set(site)):
        members = rng.permutation(np.where(site == name)[0])
        if len(members) > 1:
            permutation[members] = np.roll(members, 1)
    changed = np.any(np.stack((age, sex.astype(float)), 1) !=
                     np.stack((age, sex.astype(float)), 1)[permutation], 1)
    if not np.array_equal(site, site[permutation]):
        raise ValueError("Condition shuffle crossed site boundaries")
    info = {"age_train_mean": mean, "age_train_std_ddof0": sd,
            "age_statistics_indices": np.asarray(train).tolist(),
            "sex_categories_original": categories, "sex_semantics": {"1": "male", "2": "female"},
            "site_vocabulary_train": vocabulary, "unknown_site_slot": cfg["unknown_site_slot"],
            "condition_columns": ["age_z", "sex_1", "sex_2"] + [f"site_slot_{i}" for i in range(cfg["site_slots"])],
            "reference": reference.tolist(), "permutation": permutation.tolist(),
            "shuffle_moved_fraction": float(np.mean(permutation != np.arange(len(age)))),
            "shuffle_changed_age_sex_fraction": float(changed.mean()),
            "shuffle_preserves_joint_pairs_and_site": True,
            "shuffle_by_site": {name: {"n": int(np.sum(site == name)),
                                      "moved_fraction": float(np.mean(permutation[site == name] != np.where(site == name)[0])),
                                      "changed_age_sex_fraction": float(changed[site == name].mean())}
                                for name in sorted(set(site))}}
    return torch.from_numpy(values), torch.from_numpy(reference), info


class PCCI(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, c = cfg["embedding_dim"], cfg["hidden"], cfg["condition_dim"]
        self.cfg = cfg
        self.norm = nn.LayerNorm(d)
        self.image = nn.Linear(d, h)
        self.gamma = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, h))
        self.beta = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, h))
        self.output = nn.Linear(h, d)
        self.gate = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.activation = nn.GELU()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @classmethod
    def isolated(cls, cfg, seed, device):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            module = cls(cfg)
        return module.to(device)

    def components(self, imaging, conditions, reference, method, intervention="true", permutation=None):
        spec = self.cfg["methods"][method]
        condition = conditions
        if spec["strategy"] == "shuffled" or intervention == "shuffled":
            condition = conditions[permutation]
        if spec["strategy"] == "constant" or intervention == "reference":
            condition = reference.expand(len(imaging), -1)
        if not spec["site"]:
            condition = torch.cat((condition[:, :3], torch.zeros_like(condition[:, 3:])), 1)
        ref = reference
        if not spec["site"]:
            ref = torch.cat((reference[:3], torch.zeros_like(reference[3:])))
        gamma_condition = ref.expand(len(imaging), -1) if spec["strategy"] == "additive" else condition
        normalized = self.norm(imaging)
        gamma = 1 + self.gamma(gamma_condition).tanh()
        beta = self.beta(condition)
        gate = self.gate(normalized).sigmoid().squeeze(1)
        raw = self.output(self.activation(gamma * self.image(normalized) + beta))
        delta = self.cfg["residual_scale"] * gate[:, None] * raw.tanh()
        if intervention == "off" or spec["strategy"] == "off":
            delta = torch.zeros_like(imaging)
        return {"delta": delta, "gate": gate, "gamma": gamma, "beta": beta,
                "condition": condition, "raw": raw}


class ResidualBridge:
    """Capture one imaging output and modify the single existing clf call."""
    def __init__(self, core, module, conditions, reference, permutation, method):
        self.core, self.module = core, module
        self.conditions, self.reference, self.permutation = conditions, reference, permutation
        self.method, self.enabled = method, True
        self.imaging = self.phenotype = self.base = self.last = None
        self.calls = 0
        self.handles = [core.img_unet.register_forward_hook(self._image),
                        core.ph_unet.register_forward_hook(self._phenotype),
                        core.clf.register_forward_pre_hook(self._inject)]

    def _image(self, module, inputs, output):
        self.imaging = output

    def _phenotype(self, module, inputs, output):
        self.phenotype = output

    def _inject(self, module, inputs):
        self.calls += 1
        self.base = inputs[0]
        if self.enabled:
            self.last = self.module.components(self.imaging, self.conditions, self.reference,
                                               self.method, permutation=self.permutation)
            return (self.base + self.last["delta"],)
        self.last = None
        return inputs

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.imaging = self.phenotype = self.base = self.last = None
