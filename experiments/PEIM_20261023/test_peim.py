"""Frozen synthetic mathematics checks for PEIM's evidence and interaction path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evidence import (_site_design, diagnosis_target, interaction_projector,
                      main_effect_projector, projector_properties,
                      predict_reference_delta, shuffle_conditions)
from interaction import LowRankInteraction
from runtime import encode_conditions


BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text())


def run_tests():
    rng = np.random.default_rng(26023)
    n = 240
    sites = np.repeat(np.asarray([f"S{i}" for i in range(6)]), n // 6)
    age = rng.normal(size=n)
    sex = rng.choice([-1.0, 1.0], size=n)
    d = np.column_stack((age, sex))
    z = rng.normal(size=(n, 16))
    latent = 0.8 * z[:, 0] * age - 0.6 * z[:, 1] * sex + 0.2 * z[:, 2]
    labels = np.where(latent >= np.median(latent), 0, 1)
    ref = np.arange(180)
    query = np.arange(180, n)

    interaction = interaction_projector(z[ref], d[ref], labels[ref], sites[ref], CFG)
    main = main_effect_projector(z[ref], d[ref], labels[ref], sites[ref], CFG)
    assert interaction["rank"] <= 2 and main["rank"] <= 2
    for projector in (interaction["P"], main["P"]):
        props = projector_properties(projector)
        assert props["symmetry_max_abs"] < 1e-10
        assert props["idempotence_frobenius"] < 1e-8
        assert props["minimum_eigenvalue"] > -1e-8
        assert props["rank"] <= 2

    # The residualized cross-product fit must not change when query labels or z change.
    mutated_z = z.copy()
    mutated_y = labels.copy()
    mutated_z[query] = rng.normal(size=(len(query), 16)) * 1e5
    mutated_y[query] = 1 - mutated_y[query]
    untouched = interaction_projector(mutated_z[ref], d[ref], mutated_y[ref], sites[ref], CFG)
    assert np.allclose(interaction["P"], untouched["P"], atol=1e-10)
    assert np.allclose(interaction["B_int"], untouched["B_int"], atol=1e-10)

    prediction = predict_reference_delta(z, ref, query, labels, d, sites, CFG)
    assert np.isfinite(prediction["delta_mse"])
    assert prediction["rank"] <= 2

    # A non-symmetric query order catches labels accidentally left in source order.
    y_order = np.asarray([0, 0, 1, 0, 1, 1, 0, 1, 0])
    query_order = np.asarray([7, 1, 5, 0, 8, 3])
    assert np.array_equal(diagnosis_target(y_order, query_order), [-1, 1, -1, 1, 1, 1])
    assert not np.array_equal(y_order[query_order], y_order[query_order[::-1]])

    # Outer-train-only age statistics and the registered sex contrast survive invalid inputs.
    meta = {"age": np.asarray([10., 20., np.nan, 40., 100.]),
            "sex": np.asarray([1., 2., 7., 1., 2.]),
            "site": np.asarray(["A", "B", "A", "B", "C"])}
    encoded, stats = encode_conditions(meta, np.asarray([0, 1, 2, 3]))
    assert np.isfinite(encoded).all() and stats["age_fill_train_median"] == 20.0
    assert np.allclose(encoded[:4, 1] + stats["d_ref"][1], [-1, 1, 0, -1])
    changed_heldout = dict(meta, age=np.asarray([10., 20., np.nan, 40., -999.]))
    _, changed_stats = encode_conditions(changed_heldout, np.asarray([0, 1, 2, 3]))
    assert stats == changed_stats
    site_ref, site_query, vocab = _site_design(np.asarray(["B", "A", "B"]), np.asarray(["C", "B"]))
    assert vocab == ["A", "B"] and np.array_equal(site_query[0], [0, 0])
    assert np.array_equal(site_query[1], [0, 1])

    # The fixed shuffle is a no-label derangement within private-fold x site buckets.
    private = np.arange(n) % 3
    perm, shuffle_stats = shuffle_conditions(np.arange(n), sites, private, age, sex, 9109)
    assert np.array_equal(sites[perm], sites)
    assert np.array_equal(private[perm], private)
    assert shuffle_stats["index_change_fraction"] == 1.0

    # The original graph consults diagnosis only on train-train edges.
    from types import SimpleNamespace
    from model.rp_graph import create_reward_penalty_graph
    graph_args = SimpleNamespace(scores=["SEX"], ages="AGE_AT_SCAN")
    ph_dict = {"SEX": np.asarray([1, 1, 2, 2, 1])}
    graph_labels_a = np.asarray([0, 1, 0, 1, 0])
    graph_labels_b = np.asarray([0, 1, 1, 0, 1])
    graph_a = create_reward_penalty_graph(ph_dict, graph_labels_a, [0, 1], [2], [3, 4], graph_args)
    graph_b = create_reward_penalty_graph(ph_dict, graph_labels_b, [0, 1], [2], [3, 4], graph_args)
    assert np.array_equal(graph_a, graph_b)

    module = LowRankInteraction().double()
    assert torch.equal(module.W_d, torch.zeros_like(module.W_d))
    zt = torch.tensor(rng.normal(size=(n, 16)), dtype=torch.float64, requires_grad=True)
    dt = torch.as_tensor(d, dtype=torch.float64)
    part = {"indices": np.arange(n), "mean": zt.detach().mean(0).numpy(),
            "std": np.maximum(zt.detach().std(0, unbiased=False).numpy(), 0.05),
            "P": interaction["P"], "prototypes": np.stack((
                ((zt.detach().numpy() - zt.detach().numpy().mean(0)) /
                 np.maximum(zt.detach().numpy().std(0), 0.05))[labels == 0].mean(0),
                ((zt.detach().numpy() - zt.detach().numpy().mean(0)) /
                 np.maximum(zt.detach().numpy().std(0), 0.05))[labels == 1].mean(0))),
            "tau": 1.0}
    original, _ = module.apply(zt, dt, [part], 9, CFG, selective=False)
    assert torch.equal(original, zt)
    with torch.no_grad():
        module.W_d.copy_(torch.as_tensor(rng.normal(size=(16, 2)) * 0.02, dtype=torch.float64))
    modulated, stats = module.apply(zt, dt, [part], 9, CFG, selective=False)
    assert torch.all(stats["relative_l2"] <= 0.05 + 1e-10)
    assert torch.any((modulated - zt).abs() > 0)
    loss = modulated.square().mean()
    gradient = torch.autograd.grad(loss, module.W_d)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0

    zero_d = torch.zeros_like(dt)
    zero_condition, _ = module.apply(zt, zero_d, [part], 9, CFG, selective=False)
    assert torch.equal(zero_condition, zt)
    zero_z = torch.zeros_like(zt, requires_grad=True)
    zero_image, _ = module.apply(zero_z, dt, [part], 9, CFG, selective=False)
    assert torch.equal(zero_image, zero_z)
    epoch_zero, _ = module.apply(zt, dt, [part], 0, CFG, selective=False)
    assert torch.equal(epoch_zero, zt)

    selective, selective_stats = module.apply(zt, dt, [part], 9, CFG, selective=True)
    assert torch.all(selective_stats["gate"] > 0) and torch.all(selective_stats["gate"] <= 1)
    assert torch.all(selective_stats["relative_l2"] <= 0.05 + 1e-10)
    assert torch.isfinite(selective).all()

    # Independent central difference checks the custom low-rank interaction Jacobian.
    index = (4, 1)
    analytic = gradient[index].item()
    original_value = module.W_d[index].item()
    eps = 1e-6
    with torch.no_grad():
        module.W_d[index] = original_value + eps
    plus, _ = module.apply(zt, dt, [part], 9, CFG, selective=False)
    plus_loss = plus.square().mean().item()
    with torch.no_grad():
        module.W_d[index] = original_value - eps
    minus, _ = module.apply(zt, dt, [part], 9, CFG, selective=False)
    minus_loss = minus.square().mean().item()
    with torch.no_grad():
        module.W_d[index] = original_value
    numeric = (plus_loss - minus_loss) / (2 * eps)
    assert abs(analytic - numeric) < 2e-5

    # Sign flips of SVD basis vectors leave the subspace projector unchanged.
    basis = interaction["u"]
    p_a = basis @ basis.T
    p_b = (basis * np.asarray([-1.0, 1.0])[:basis.shape[1]]) @ (basis * np.asarray([-1.0, 1.0])[:basis.shape[1]]).T
    assert np.allclose(p_a, p_b, atol=1e-12)
    return {"passed": True, "checks": 30, "synthetic_n": n,
            "interaction_rank": interaction["rank"], "main_rank": main["rank"],
            "delta_mse": prediction["delta_mse"], "gradient_norm": float(gradient.norm()),
            "finite_difference_abs_error": abs(analytic - numeric),
            "null_index_change_fraction": shuffle_stats["index_change_fraction"],
            "null_joint_value_change_fraction": shuffle_stats["joint_value_change_fraction"]}


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
