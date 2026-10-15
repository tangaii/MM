"""Read-only PPU diagnostics. JSON-lines on stdout; never writes artifacts."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "experiments/DRPA_20261014"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax
import run as historical
from drpa import DRPA, EmbeddingHooks

EXPECTED = "9186fe8b76142e6ef114d850409cb7d8e9ac5f2691d8e132647c389ee60e5572"


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def vector_statistics(grads, n_parameters):
    norm = sum(float(g.detach().cpu().double().square().sum()) for g in grads if g is not None) ** 0.5
    return {"norm": norm, "rms": norm / max(n_parameters, 1) ** 0.5,
            "dimensions": n_parameters,
            "connected_dimensions": sum(g.numel() for g in grads if g is not None)}


def gradient_cosine(a, b):
    aa = sum(float(x.detach().cpu().double().square().sum()) for x in a if x is not None)
    bb = sum(float(x.detach().cpu().double().square().sum()) for x in b if x is not None)
    dot = sum(float((x.detach().cpu().double() * y.detach().cpu().double()).sum())
              for x, y in zip(a, b) if x is not None and y is not None)
    return dot / (aa * bb) ** 0.5 if aa > 0 and bb > 0 else None


def drift(core, saved):
    params = dict(core.named_parameters())
    pmax = max(float((params[k].detach().cpu() - v).abs().max())
               for k, v in saved["state"].items() if k in params)
    return {"max_parameter_abs_change": pmax,
            "max_unregistered_rp_abs_change": float((core.rp_attention.weights.detach().cpu()
                                                      - saved["rp_weights"]).abs().max())}


class Loaded:
    def __init__(self, task, ctx, input_cache):
        self.task, self.ctx = task, ctx
        method, seed, fold = task["method"], task["seed"], task["fold"]
        out = historical.fold_dir(method, seed, fold)
        self.path = out / "best.pt"
        self.sha = historical.sha_file(self.path)
        if self.sha != task["sha256"]:
            raise RuntimeError("Checkpoint changed after input audit")
        self.saved = historical.load_torch(self.path)
        if self.saved["training_code_sha256"] != EXPECTED:
            raise RuntimeError("Not a formal checkpoint")
        adir = historical.anchor_dir(seed, fold)
        if (seed, fold) not in input_cache:
            split = historical.load_torch(adir / "splits.pt")
            historical.check_splits([split[k] for k in ("train", "val", "test")])
            x = historical.load_torch(adir.parent / "selected_features.pt")
            rng = historical.load_torch(adir / "train_start.pt")["rng"]
            opt = historical.make_options(seed, out)
            # Restore static input tables only; do not refit learned AMRS or RFE.
            aff = torch.from_numpy(historical.create_reward_penalty_graph(
                ctx["ph_dict"], ctx["labels"], split["train"], split["val"], split["test"], opt)).float()
            pairing = self.saved["pairing"]
            if historical.array_hash(x) != pairing["features_fingerprint"]:
                raise RuntimeError("Selected features no longer match training")
            if historical.split_fingerprint([split[k] for k in ("train", "val", "test")]) != pairing["split_fingerprints"]:
                raise RuntimeError("Saved split fingerprint mismatch")
            vae_path = adir.parent / "vae" / f"fold{fold}_pretrain.pth"
            if historical.sha_file(vae_path) != pairing["vae_sha256"]:
                raise RuntimeError("Saved VAE fingerprint mismatch")
            input_cache[(seed, fold)] = (split, x, aff, rng, opt, vae_path)
        self.split, x, aff, self.rng, opt, vae_path = input_cache[(seed, fold)]
        self.core = historical.MM_GTUNets(opt, fold).to(opt.device)
        self.core.vae_save_path = str(vae_path)
        for parameter in self.core.vae.parameters():
            parameter.requires_grad_(False)
        self.auxiliary = None
        if self.saved["auxiliary"] is not None:
            self.auxiliary = DRPA.isolated(historical.config(), historical.private_seed(seed, fold, "heads"), opt.device)
            self.auxiliary.load_state_dict(self.saved["auxiliary"], strict=True)
        self.x, self.ph, self.aff = x.to(opt.device), torch.from_numpy(ctx["ph_data"]).to(opt.device), aff.to(opt.device)
        self.labels = torch.as_tensor(ctx["labels"], dtype=torch.long, device=opt.device)
        self.sites = torch.as_tensor(ctx["site_codes"], dtype=torch.long, device=opt.device)
        self.train = torch.as_tensor(self.split["train"], dtype=torch.long, device=opt.device)
        self.hooks = EmbeddingHooks(self.core)
        self.restore()

    def restore(self):
        historical.import_core(self.core, self.saved["core"])

    def close(self):
        self.restore()
        if historical.core_hash(historical.export_core(self.core)) != historical.core_hash(self.saved["core"]):
            raise RuntimeError("In-memory checkpoint was not restored exactly")
        if historical.sha_file(self.path) != self.sha:
            raise RuntimeError("Historical checkpoint changed")
        self.hooks.close()


def frozen_gradients(loaded):
    c, a, t = loaded.core, loaded.auxiliary, loaded.task
    loaded.restore()
    c.train()
    a.train()
    historical.restore_rng(loaded.rng)
    before = historical.rng_hash(historical.capture_rng())
    logits, graph = c(loaded.x, loaded.ph, loaded.aff)
    aux, align, _, _ = a.objective(loaded.hooks.img, loaded.hooks.ph, loaded.labels,
                                  loaded.sites, loaded.train, t["method"], apply_shuffle=False)
    losses = {"official": F.cross_entropy(logits[loaded.train], loaded.labels[loaded.train]) + graph,
              "weighted_aux": 0.05 * aux,
              "weighted_align": 0.1 * align if align is not None else None}
    groups = {"img_encoder": list(c.img_unet.parameters()),
              "ph_encoder": list(c.ph_unet.parameters()),
              "project_img": list(a.project_img.parameters()),
              "project_ph": list(a.project_ph.parameters()),
              "predict_img": list(a.predict_img.parameters()),
              "predict_ph": list(a.predict_ph.parameters()),
              "aux_heads": list(a.head_img.parameters()) + list(a.head_ph.parameters()),
              "raw_outputs_all": [loaded.hooks.img, loaded.hooks.ph]}
    targets = [v for values in groups.values() for v in values]
    offsets, pos = {}, 0
    for name, values in groups.items():
        offsets[name] = slice(pos, pos + len(values)); pos += len(values)
    gradients = {}
    for name, loss in losses.items():
        values = (torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
                  if loss is not None and loss.requires_grad else [None] * len(targets))
        for g in values:
            if g is not None and not torch.isfinite(g).all():
                raise FloatingPointError("Non-finite frozen diagnostic gradient")
        gradients[name] = values
    result = {"losses": {k: float(v.detach()) if v is not None else 0.0 for k, v in losses.items()},
              "groups": {}, "rng_before": before,
              "rng_after": historical.rng_hash(historical.capture_rng()),
              "forward_parameter_drift": drift(c, loaded.saved["core"]),
              "mode": "train; original stochastic edges/VAE/dropout and graph regularizer; no updates; buffers restored",
              "official_label_scope": "CE uses training rows only; unchanged transductive graph term spans 871 nodes"}
    for name, values in groups.items():
        gs = {k: v[offsets[name]] for k, v in gradients.items()}
        record = {k: vector_statistics(v, sum(x.numel() for x in values)) for k, v in gs.items()}
        record["official_align_cosine"] = gradient_cosine(gs["official"], gs["weighted_align"])
        record["official_aux_cosine"] = gradient_cosine(gs["official"], gs["weighted_aux"])
        record["align_official_ratio"] = (record["weighted_align"]["norm"] / record["official"]["norm"]
                                            if record["official"]["norm"] > 0 else None)
        result["groups"][name] = record
    raw = {k: v[offsets["raw_outputs_all"]] for k, v in gradients.items()}
    for name, loc in (("raw_img_train", 0), ("raw_ph_train", 1)):
        gs = {k: [v[loc][loaded.train] if v[loc] is not None else None] for k, v in raw.items()}
        record = {k: vector_statistics(v, len(loaded.train) * 16) for k, v in gs.items()}
        record["official_align_cosine"] = gradient_cosine(gs["official"], gs["weighted_align"])
        result["groups"][name] = record
    loaded.restore()
    return result


def pool_choices(core):
    result = {}
    for name, encoder in (("img", core.img_unet), ("ph", core.ph_unet)):
        parent = np.arange(871)
        values = []
        for perm in encoder.perms:
            parent = parent[perm.detach().cpu().numpy()]
            values.append(parent.copy())
        result[name] = values
    return result


def compare_forward(first, other, test):
    result = {f"max_abs_{key}": float(np.max(np.abs(first[key] - other[key])))
              for key in ("logits", "probs", "img", "ph", "vae")}
    p, q = first["logits"].argmax(1), other["logits"].argmax(1)
    result.update({"label_flips_all": int(np.sum(p != q)), "label_flips_test": int(np.sum(p[test] != q[test])),
                   "topk": {}})
    for name in ("img", "ph"):
        result["topk"][name] = [{"ordered_positions_changed": int(np.sum(a != b)),
                                  "selected_set_symmetric_difference": len(set(a) ^ set(b))}
                                 for a, b in zip(first["topk"][name], other["topk"][name])]
    return result


def repeat_condition(loaded, condition, repeats=3):
    c = loaded.core
    c.eval()
    original = c.vae.reparameterize
    fixed_noise = None
    latent = []
    handle = c.vae.register_forward_hook(lambda module, inputs, outputs: latent.append(outputs[0].detach().cpu().numpy()))
    historical.restore_rng(loaded.rng)
    noise_generator = torch.Generator(device=c.device)
    noise_generator.set_state(torch.cuda.get_rng_state(0))

    def intervention(self, mu, logvar):
        nonlocal fixed_noise
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if condition == "fixed_vae_vary_dropout":
            if fixed_noise is None:
                fixed_noise = eps.detach().clone()
            eps = fixed_noise
        elif condition == "fixed_dropout_vary_vae":
            eps = torch.randn(std.shape, dtype=std.dtype, device=std.device, generator=noise_generator)
        return mu + eps * std

    if condition in ("fixed_vae_vary_dropout", "fixed_dropout_vary_vae"):
        c.vae.reparameterize = types.MethodType(intervention, c.vae)
    snapshots, records = [], []
    try:
        for repeat in range(repeats):
            loaded.restore()
            if condition in ("same_rng", "deterministic_same_rng", "fixed_dropout_vary_vae"):
                historical.restore_rng(loaded.rng)
            before = historical.rng_hash(historical.capture_rng())
            latent.clear()
            with torch.no_grad():
                logits, _ = c(loaded.x, loaded.ph, loaded.aff)
            torch.cuda.synchronize()
            values = logits.detach().cpu().numpy()
            snapshot = {"logits": values, "probs": softmax(values, axis=1),
                        "img": loaded.hooks.img.detach().cpu().numpy(),
                        "ph": loaded.hooks.ph.detach().cpu().numpy(), "vae": latent[0],
                        "topk": pool_choices(c)}
            test = loaded.split["test"]
            metric = historical.binary_metrics(loaded.ctx["labels"][test], values[test].argmax(1), snapshot["probs"][test, 0])
            records.append({"repeat": repeat, "rng_before": before,
                            "rng_after": historical.rng_hash(historical.capture_rng()),
                            "private_vae_rng_after": historical.array_hash(noise_generator.get_state()) if condition == "fixed_dropout_vary_vae" else None,
                            "logits_sha256": historical.array_hash(values), "diagnostic_metrics_not_formal_scores": metric,
                            "topk": {k: [{"n": len(v), "sha256": historical.array_hash(v)} for v in vv]
                                     for k, vv in snapshot["topk"].items()},
                            "forward_parameter_drift": drift(c, loaded.saved["core"])})
            snapshots.append(snapshot)
        comparisons = [{"pair": [i, j], **compare_forward(snapshots[i], snapshots[j], test)}
                       for i in range(repeats) for j in range(i + 1, repeats)]
        return {"repeats": records, "pairwise": comparisons,
                "acc_span": max(r["diagnostic_metrics_not_formal_scores"]["ACC"] for r in records)
                            - min(r["diagnostic_metrics_not_formal_scores"]["ACC"] for r in records)}
    finally:
        handle.remove()
        c.vae.reparameterize = original
        loaded.restore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gradients", "native", "deterministic"), required=True)
    args = parser.parse_args()
    requests = json.loads(sys.stdin.read())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    historical.source_guard()
    if historical.training_code_hash() != EXPECTED:
        raise RuntimeError("Historical training sources changed")
    historical.require_ppu()
    deterministic = args.stage != "native"
    if deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic environment must precede CUDA initialization")
    torch.use_deterministic_algorithms(deterministic)
    ctx, cache = historical.context(), {}
    for task in requests:
        loaded = None
        started = time.monotonic()
        try:
            with redirect_stdout(sys.stderr):
                loaded = Loaded(task, ctx, cache)
                if args.stage == "gradients":
                    value = frozen_gradients(loaded)
                else:
                    conditions = (["deterministic_same_rng"] if deterministic else
                                  ["original_free_rng", "same_rng", "fixed_vae_vary_dropout", "fixed_dropout_vary_vae"])
                    value = {name: repeat_condition(loaded, name) for name in conditions}
                loaded.close()
            emit({"state": "complete", "task": task, "stage": args.stage, "value": value,
                  "device": torch.cuda.get_device_name(0), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  "seconds": time.monotonic() - started})
        except Exception as exc:
            emit({"state": "error", "task": task, "stage": args.stage,
                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)})
        finally:
            del loaded
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
