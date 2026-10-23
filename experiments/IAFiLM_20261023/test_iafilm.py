"""Synthetic boundary and Jacobian checks for the frozen IA-FiLM module."""
from __future__ import annotations

import copy
import json
import numpy as np
import torch
from torch import nn

from film import (FilmHook, IdentityAnchoredFiLM, derange_within_splits,
                  fit_encode_conditions)


def run_tests():
    torch.manual_seed(2501)
    checks = {}
    film = IdentityAnchoredFiLM()
    assert sum(p.numel() for p in film.parameters()) == 576
    d = torch.tensor([[-1.1, 1., 0.], [-.3, 0., 1.], [.2, 1., 0.], [1.4, 0., 1.]])
    ref = d.mean(0)
    h = torch.linspace(-1.3, 1.1, 64).reshape(4, 16).requires_grad_()
    out0, s0 = film.modulate(h, d, ref, epoch=9)
    ref_out0, _ = film.modulate(h[:1], ref[None], ref, epoch=9)
    checks["parameter_count_576"] = sum(p.numel() for p in film.parameters()) == 576
    checks["zero_head_strict_identity"] = torch.equal(out0, h)
    checks["zero_head_reference_identity"] = torch.equal(ref_out0, h[:1])
    checks["zero_head_gamma_beta_identity"] = torch.equal(s0["gamma"], torch.ones_like(s0["gamma"])) and torch.equal(s0["beta"], torch.zeros_like(s0["beta"]))
    checks["rho_epoch0_zero"] = IdentityAnchoredFiLM.rho(0) == 0
    checks["rho_epoch9_cap"] = IdentityAnchoredFiLM.rho(9) == .1 and IdentityAnchoredFiLM.rho(50) == .1

    with torch.no_grad():
        film.net[-1].weight.normal_(mean=0.0, std=0.15)
    out, stats = film.modulate(h, d, ref, epoch=9)
    ref_out, ref_stats = film.modulate(h[:1], ref[None], ref, epoch=9)
    checks["reference_condition_strict_identity"] = torch.equal(ref_out, h[:1]) and torch.equal(ref_stats["delta"], torch.zeros_like(ref_stats["delta"]))
    checks["gamma_bound"] = bool((stats["gamma"] >= .9).all() and (stats["gamma"] <= 1.1).all())
    checks["beta_bound"] = bool((stats["beta"].abs() <= .1).all())
    checks["real_conditions_change_output"] = not torch.equal(out, h)
    checks["age_condition_changes_output"] = not torch.allclose(out[0], out[1])
    checks["sex_condition_changes_output"] = not torch.allclose(out[0], out[2])
    grad = torch.autograd.grad(out.square().sum(), h, retain_graph=True)[0]
    film_grad = torch.autograd.grad(out.square().sum(), tuple(film.parameters()), allow_unused=True)
    checks["input_gradient_nonzero"] = float(grad.norm()) > 0
    checks["film_parameters_receive_gradient"] = all(g is not None and torch.isfinite(g).all() for g in film_grad) and float(film_grad[-1].norm()) > 0

    rng = np.random.default_rng(2511)
    age = rng.normal(17, 5, 80); sex = rng.choice([1., 2., 9.], 80); age[[2, 35, 36]] = [np.nan, 200, 0]
    ids = np.arange(50)
    encoded, reference, meta = fit_encode_conditions(age, sex, ids)
    train_age = age[ids]
    valid_train_age = train_age[np.isfinite(train_age) & (train_age >= 0) & (train_age <= 120)]
    checks["age_statistics_train_only"] = meta["train_n"] == len(ids) and meta["age_median"] == float(np.median(valid_train_age))
    checks["zero_age_is_valid"] = np.isclose(encoded[36, 0], np.clip((0-meta["age_mean"])/meta["age_std_population"],-3,3))
    checks["age_clip"] = float(encoded[:, 0].min()) >= -3 and float(encoded[:, 0].max()) <= 3
    checks["unknown_sex_zero"] = np.all(encoded[sex == 9, 1:] == 0)
    checks["sex_male_female_order"] = np.array_equal(encoded[sex == 1, 1:], np.tile([1., 0.], (np.sum(sex == 1), 1))) and np.array_equal(encoded[sex == 2, 1:], np.tile([0., 1.], (np.sum(sex == 2), 1)))
    heldout_changed = age.copy(); heldout_changed[50:] = 99
    encoded2, reference2, meta2 = fit_encode_conditions(heldout_changed, sex, ids)
    checks["heldout_age_does_not_change_fit"] = np.array_equal(encoded[:50], encoded2[:50]) and np.array_equal(reference, reference2) and meta["age_mean"] == meta2["age_mean"]
    checks["reference_is_train_mean"] = np.array_equal(reference, encoded[ids].mean(0, dtype=np.float64).astype(np.float32))

    splits = {"train": np.arange(0, 40), "val": np.arange(40, 60), "test": np.arange(60, 80)}
    sites = np.asarray([f"s{i % 4}" for i in range(80)])
    sites[39] = "singleton"; sites[59] = "singleton2"; sites[79] = "singleton3"
    perm, records = derange_within_splits(80, splits, sites, 911, 0)
    checks["F4_permutation_bijection"] = np.array_equal(np.sort(perm), np.arange(80))
    checks["F4_permutation_split_and_site_bounded"] = all(np.array_equal(sites[idx], sites[perm[idx]]) and np.array_equal(np.sort(perm[idx]), np.sort(idx)) for idx in splits.values())
    checks["F4_non_singleton_derangement"] = all(
        item["changed"] == item["n"] if item["n"] >= 2 else item["changed"] == 0
        for item in records.values())
    checks["F4_singletons_retained"] = int(np.sum(perm == np.arange(80))) == 3
    perm2, _ = derange_within_splits(80, splits, sites, 911, 0)
    checks["F4_fixed_sha_seed"] = np.array_equal(perm, perm2)

    class Pair(nn.Module):
        def forward(self, a, b):
            return a + b
    class IdentityInput(nn.Module):
        def forward(self, x):
            return x[0]
    class FakeCore(nn.Module):
        def __init__(self):
            super().__init__(); self.mm_attention = Pair(); self.clf = IdentityInput()
    fake = FakeCore()
    hook = FilmHook(fake, film, "pre_attention_image_only")
    cond = torch.as_tensor(encoded)
    ref_t = torch.as_tensor(reference)
    hook.configure(True, 0, cond, ref_t)
    before = torch.randn(80, 16)
    result = fake.mm_attention(before, torch.zeros_like(before))
    checks["F2_hook_called_once"] = hook.call_count == 1
    checks["F2_raw_input_unchanged"] = hook.last is not None and torch.equal(hook.last["raw"], before)
    checks["F2_epoch0_identity"] = torch.equal(result, before)
    hook.remove()
    checks["hook_removal"] = len(fake.mm_attention._forward_pre_hooks) == 0

    # An asymmetric target order catches logits/label concatenation mistakes.
    targets = torch.tensor([0, 0, 1, 1, 0, 1])
    order = torch.tensor([3, 0, 5, 2, 1, 4])
    logits = torch.tensor([[3., 0.], [2., 0.], [0., 2.], [0., 3.], [1., 0.], [0., 1.]])[order]
    correct = nn.functional.cross_entropy(logits, targets[order])
    wrong = nn.functional.cross_entropy(logits, targets)
    checks["asymmetric_label_order_detects_permutation"] = not torch.isclose(correct, wrong)
    checks = {key: bool(value) for key, value in checks.items()}
    assert all(checks.values()), {k: v for k,v in checks.items() if not v}
    return {"passed": True, "passed_count": len(checks), "checks": checks,
            "formal_metrics_computed": 0, "model_forwards": 0}


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
