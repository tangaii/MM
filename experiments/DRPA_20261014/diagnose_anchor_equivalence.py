"""Locate an actual-anchor equivalence failure before any DRPA variant runs."""

import argparse
import json
import os
import torch
import torch.nn.functional as F

from drpa import DRPA, EmbeddingHooks
from run import (BASE, anchor_dir, atomic_json, capture_rng, config, context,
                 core_hash, export_core, import_core, load_torch, make_options,
                 private_seed, require_ppu, restore_rng, rng_hash, utc)
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic-all", action="store_true")
    args = parser.parse_args()
    if args.deterministic_all:
        torch.use_deterministic_algorithms(True)
    device, cfg, ctx = require_ppu(), config(), context()
    seed, fold = 911, 0
    adir = anchor_dir(seed, fold)
    start = load_torch(adir / "train_start.pt")
    split = load_torch(adir / "splits.pt")
    splits = [split[k] for k in ("train", "val", "test")]
    x = load_torch(adir.parent / "selected_features.pt")
    ph = torch.from_numpy(ctx["ph_data"]).float()
    labels = torch.as_tensor(ctx["labels"], device=device, dtype=torch.long)
    opt = make_options(seed, adir.parent / "vae")
    affinity = torch.from_numpy(create_reward_penalty_graph(ctx["ph_dict"], ctx["labels"], *splits, opt)).float()
    snapshots, records, core = [], [], None
    for kind in ("native_first", "native_reused", "native_fresh", "unused_heads_fresh"):
        if kind != "native_reused":
            core = MM_GTUNets(opt, fold).to(device)
        import_core(core, start["core"])
        initial_hash = core_hash(export_core(core))
        core.train()
        restore_rng(start["rng"])
        before = rng_hash(capture_rng())
        auxiliary = DRPA.isolated(cfg, private_seed(seed, fold, "heads"), device) if kind == "unused_heads_fresh" else None
        isolated_rng_equal = before == rng_hash(capture_rng())
        observer = EmbeddingHooks(core) if auxiliary is not None else None
        observed = {}
        handle = core.vae.register_forward_hook(lambda module, args, output: observed.update({"latent": output[0]}))
        params = list(core.parameters()) + (list(auxiliary.parameters()) if auxiliary is not None else [])
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.wd)
        optimizer.zero_grad()
        logits, graph = core(x, ph, affinity)
        forward_end_rng = rng_hash(capture_rng())
        tensors = {"logits": logits.detach().cpu().clone(), "graph_loss": graph.detach().cpu().clone(),
                   "latent": observed["latent"].detach().cpu().clone(),
                   "rp_graph": core.rp_graph.detach().cpu().clone(),
                   "fused_sim": core.fused_sim.detach().cpu().clone(),
                   "fused_graph": core.fused_graph.detach().cpu().clone(),
                   "joint_embed": core.joint_embed.detach().cpu().clone()}
        permutations = {"img": [p.detach().cpu().clone() for p in core.img_unet.perms],
                        "ph": [p.detach().cpu().clone() for p in core.ph_unet.perms]}
        loss = F.cross_entropy(logits[splits[0]], labels[splits[0]]) + graph
        loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in core.named_parameters() if p.grad is not None}
        optimizer.step()
        after = export_core(core)
        snapshot = {"tensors": tensors, "permutations": permutations,
                    "gradients": gradients, "after": after}
        report = {"kind": kind, "initial_core_matches_anchor": initial_hash == core_hash(start["core"]),
                  "isolated_rng_equal": isolated_rng_equal, "before_rng": before,
                  "after_forward_rng": forward_end_rng, "end_rng": rng_hash(capture_rng()),
                  "backend": capture_rng()["backend"], "official_loss": float(loss.detach())}
        if snapshots:
            reference = snapshots[0]
            report["max_abs_errors"] = {key: float((value - reference["tensors"][key]).abs().max())
                                        for key, value in tensors.items()}
            report["topk_permutation_equal"] = {name: [torch.equal(a, b) for a, b in
                 zip(permutations[name], reference["permutations"][name])] for name in permutations}
            report["gradient_max_abs_by_parameter"] = {name: float((value - reference["gradients"][name]).abs().max())
                                                       for name, value in gradients.items()}
            report["step_max_abs_by_tensor"] = {name: float((value - reference["after"]["state"][name]).abs().max())
                                               for name, value in after["state"].items() if value.is_floating_point()}
        snapshots.append(snapshot); records.append(report)
        print("ACTUAL ANCHOR DIAGNOSTIC", json.dumps(report), flush=True)
        handle.remove()
        if observer is not None:
            observer.close()
        del logits, graph, optimizer, auxiliary
    result = {"time_utc": utc(), "seed": seed, "fold": fold, "physical_ppu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "records": records, "no_research_hyperparameter_changes": True,
              "deterministic_algorithms": args.deterministic_all,
              "note": "One-step numerical diagnostics only; not formal candidates or threshold relaxation"}
    suffix = "_deterministic" if args.deterministic_all else ""
    atomic_json(result, BASE / f"artifacts/anchor_equivalence_diagnosis{suffix}.json")


if __name__ == "__main__":
    main()
