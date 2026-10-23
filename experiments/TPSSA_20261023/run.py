"""Experiment 24 stage-gated runner. Stage A contains no training path."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.dont_write_bytecode = True
from common import (BASE, ROOT, HISTORY, D4, CFG, SOURCE, STAGE_A_CODE,
                    sha_file, digest, array_hash, core_hash, export_core, import_core,
                    load_torch, atomic_json, private_seed, historical_module,
                    read_metadata, capture_rng, rng_hash, log, utc)


def audit_inputs():
    import numpy as np
    import torch
    before = time.monotonic()
    hist = historical_module("tpssa_history_runner", D4 / "run.py")
    ctx_path = HISTORY / "artifacts/context.pt"
    ctx = load_torch(ctx_path)
    raw_meta = read_metadata(ctx)
    assert len(ctx["ids"]) == len(set(ctx["ids"])) == 871
    historical_audit = json.loads((D4 / "audit.json").read_text())
    anchors = {}
    protected = {}
    # Read and hash the requested reports and implementation, without importing
    # any historical launch command or executing its audit/training functions.
    review_dirs = [ROOT / "experiments" / name for name in (
        "DRPA_20261014", "PCDCA_20261019", "PCDPAX_20261021", "PCDPAX2_20261021",
        "PCDPACM_20261021", "PCCI_20261015", "CMCV_20261019", "LCRA_20261018")]
    registered_r3 = ROOT / "experiments/PCRPA_20261022"
    actual_r3 = registered_r3 if registered_r3.is_dir() else Path("/tmp/PCRPA_20261022")
    if not actual_r3.is_dir():
        raise FileNotFoundError("Experiment 23 history could not be located")
    review_dirs.append(actual_r3)
    for directory in review_dirs:
        for p in directory.iterdir():
            if p.is_file() and p.suffix in (".py", ".json", ".md"):
                protected[str(p)] = sha_file(p)
    for name in SOURCE:
        protected[str(ROOT / name)] = sha_file(ROOT / name)
    input_dir = ROOT / CFG["data_directory"]
    data_paths = [input_dir / "id.txt", input_dir / "Phenotypic_V1_0b_preprocessed1.csv", ctx_path]
    data_paths += [input_dir / "save_tensor" / n for n in (
        "up_triangles.pt", "labels.pt", "ph_data_SITE_ID_SEX_AGE_AT_SCAN.pt", "ph_dict_SITE_ID_SEX_AGE_AT_SCAN.pt")]
    for p in data_paths:
        protected[str(p)] = sha_file(p)
    old_input_audit = json.loads((HISTORY / "artifacts/audit.json").read_text())
    for p in data_paths:
        if p.is_relative_to(input_dir):
            rel = str(p.relative_to(input_dir))
            assert protected[str(p)] == old_input_audit["data_sha256"][rel], f"Changed original input: {rel}"
    x = load_torch(HISTORY / "artifacts/seed_911/selected_features.pt")
    raw = torch.as_tensor(load_torch(input_dir / "save_tensor/up_triangles.pt"))
    support = np.load(HISTORY / "artifacts/seed_911/rfe_support.npy")
    assert x.shape == (871, 500) and raw.shape == (871, 6670)
    assert support.shape == (500,) and len(np.unique(support)) == 500
    assert torch.equal(raw[:, support].to(x.dtype), x)
    coverage = np.zeros(871, int)
    for fold in range(10):
        directory = hist.anchor(911, fold)  # Actual historical function, not guessed.
        expected_dir = HISTORY / "artifacts/seed_911" / f"fold_{fold}"
        assert directory == expected_dir
        sp = load_torch(directory / "splits.pt")
        joined = np.concatenate([sp[k] for k in ("train", "val", "test")])
        assert len(joined) == 871 and np.array_equal(np.sort(joined), np.arange(871))
        coverage[sp["test"]] += 1
        fp = json.loads((directory / "fingerprints.json").read_text())
        for k in ("train", "val", "test"):
            assert array_hash(sp[k]) == fp["split_fingerprints"][k]
        start = load_torch(directory / "train_start.pt")
        assert core_hash(start["core"]) == fp["train_start_core_fingerprint"]
        assert rng_hash(start["rng"]) == fp["classifier_start_rng"]
        assert array_hash(x) == fp["features_fingerprint"]
        checkpoint = D4 / CFG["historical_d4_checkpoint"].format(fold=fold)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Historical D4 best checkpoint unavailable: {checkpoint}")
        d4 = load_torch(checkpoint)
        assert d4["state"] == "complete" and d4["group"] == "D4" and d4["seed"] == 911 and d4["fold"] == fold
        assert core_hash(d4["core"]) == d4["best_core_sha256"]
        assert d4["pairing"] == fp
        prepared = d4["prototype_preparation"]
        assert np.array_equal(prepared["train_indices"], sp["train"])
        pf = np.asarray(prepared["prototype_fold"], np.int64)
        assert set(pf) == {0, 1, 2} and array_hash(pf) == prepared["prototype_fold_sha256"]
        assert array_hash(np.asarray(ctx["labels"], np.int64)[sp["train"]]) == prepared["train_label_sha256"]
        # Byte-identical historical preparation, trained reference membership only.
        prep2 = hist.prepare(ctx["labels"], raw_meta, sp["train"], 911, fold, hist.CFG)
        assert digest(prep2["info"]) == digest(prepared)
        assert digest(prepared) == digest(historical_audit["anchors"][f"911/{fold}"]["prototype"])
        paths = [directory / n for n in ("splits.pt", "train_start.pt", "fingerprints.json")]
        paths += [directory.parent / "selected_features.pt", directory.parent / "rfe_support.npy",
                  directory.parent / f"vae/fold{fold}_pretrain.pth", checkpoint]
        sha = {str(p): sha_file(p) for p in paths}
        protected.update(sha)
        for path, expected in d4["input_sha256"].items():
            assert sha[path] == expected
        assert sha[str(directory.parent / f"vae/fold{fold}_pretrain.pth")] == fp["vae_sha256"]
        anchors[str(fold)] = {"anchor": str(directory), "d4_checkpoint": str(checkpoint), "sha256": sha,
                              "d4_best_core_sha256": d4["best_core_sha256"],
                              "d4_private_fold_sha256": prepared["prototype_fold_sha256"],
                              "train_start_core_sha256": fp["train_start_core_fingerprint"],
                              "split_sha256": fp["split_fingerprints"], "physical_gpu": fold % 2,
                              "diagnostic_seed": private_seed(fold, "stage-a-native-forward")}
    assert np.all(coverage == 1)
    # Read-only controls: hash every saved checkpoint; never invoke its model.
    controls = [(D4, "D0"), (D4, "D4"), (ROOT / "experiments/PCDPAX_20261021", "M1"),
                (ROOT / "experiments/PCDPAX_20261021", "M3"),
                (ROOT / "experiments/PCDPAX2_20261021", "N3"),
                (ROOT / "experiments/PCDPACM_20261021", "P1"), (actual_r3, "R3")]
    for directory, group in controls:
        for fold in range(10):
            p = directory / "runs" / group / f"s911_f{fold}.pt"
            if p.is_file():
                protected[str(p)] = sha_file(p)
    handoff = ROOT / "handoff.md"
    raw_handoff = handoff.read_bytes()
    result = {"experiment": 24, "name": "TPSSA", "started_utc": utc(),
              "actual_output_directory": str(BASE), "requested_tmp_execution": True,
              "historical_r3_resolved_directory": str(actual_r3),
              "stage_a_original_sources": {n: sha_file(ROOT / n) for n in SOURCE},
              "protected_sha256": protected, "input_inventory_sha256": digest(protected),
              "anchors": anchors, "historical_anchor_function_checked": True,
              "historical_private_folds_exact": True, "rfe_cache_exact": True,
              "data": {"n": 871, "class_counts": [403, 468], "ASD": 0, "TD": 1,
                       "age_range": [float(raw_meta["age"].min()), float(raw_meta["age"].max())],
                       "site_count": len(set(raw_meta["site"])), "csv_sha256": sha_file(data_paths[1])},
              "handoff_at_start": {"bytes": len(raw_handoff), "sha256": hashlib.sha256(raw_handoff).hexdigest(),
                                   "end_marker_count": raw_handoff.count(b"TPSSA_20261023_END")},
              "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                              "host_threads_policy": 1},
              "historical_retrains": {"D0": 0, "D4": 0, "M1": 0, "M3": 0, "others": 0},
              "formal_tasks_started": 0, "stage_a_native_forwards": 0,
              "formal_preflight_forwards": 0, "status": "AUDITED_BEFORE_STAGE_A", "audit_seconds": time.monotonic()-before}
    atomic_json(result, BASE / "audit.json")
    log("input_audit_passed", protected_files=len(protected), anchors=10, seconds=result["audit_seconds"])
    return result


def freeze_stage_a():
    if list((BASE / "stage_a_folds").glob("fold*.json")) if (BASE / "stage_a_folds").exists() else False:
        raise RuntimeError("Stage A already has results; cannot refreeze scientific rules")
    from test_stage_a import run_tests
    tests = run_tests()
    atomic_json(tests, BASE / "math_stage_a.json")
    audit = json.loads((BASE / "audit.json").read_text())
    files = list(STAGE_A_CODE) + ["launch.py"]
    manifest = {"experiment": 24, "name": "TPSSA", "frozen_utc": utc(),
                "stage": "A", "code_sha256": {n: sha_file(BASE / n) for n in files},
                "configuration_sha256": sha_file(BASE / "config.json"),
                "mathematical_tests_sha256": sha_file(BASE / "math_stage_a.json"),
                "input_audit_sha256": sha_file(BASE / "audit.json"),
                "input_inventory_sha256": audit["input_inventory_sha256"],
                "gates": CFG["stage_a"]["gates"], "formal_tasks_authorized": 0,
                "no_stage_a_results_before_freeze": True}
    atomic_json(manifest, BASE / "manifest.json")
    log("stage_a_frozen", manifest_sha256=sha_file(BASE / "manifest.json"), tests=tests["passed_count"])


def guard():
    manifest = json.loads((BASE / "manifest.json").read_text())
    for name, expected in manifest["code_sha256"].items():
        assert sha_file(BASE / name) == expected, f"Frozen Stage A source changed: {name}"
    audit = json.loads((BASE / "audit.json").read_text())
    assert sha_file(BASE / "audit.json") == manifest["input_audit_sha256"]
    return audit, manifest


def stage_a_fold(fold):
    import numpy as np
    import torch
    from stage_a import study, shared_directions
    audit, manifest = guard()
    out = BASE / "stage_a_folds" / f"fold{fold}.json"
    out.parent.mkdir(exist_ok=True)
    with (out.with_suffix(".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            saved = json.loads(out.read_text())
            assert saved["state"] == "complete" and saved["manifest_sha256"] == sha_file(BASE / "manifest.json")
            assert saved["native_forward_count"] == 1
            log("stage_a_resume_skip", fold=fold)
            return saved
        begin_path = out.with_suffix(".begun.json")
        if begin_path.exists():
            raise RuntimeError(f"Stage A fold {fold} previously began; refusing a second stochastic forward")
        ar = audit["anchors"][str(fold)]
        for path, expected in ar["sha256"].items():
            assert sha_file(path) == expected, f"Changed historical input: {path}"
        if not torch.cuda.is_available():
            raise RuntimeError("PPU unavailable")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        name = torch.cuda.get_device_name(0)
        if "PPU" not in name.upper():
            raise RuntimeError(f"Expected PPU, got {name}")
        physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if physical_gpu != str(fold % 2):
            raise RuntimeError(f"Wrong physical GPU for fold{fold}: {physical_gpu}")
        sys.path.insert(0, str(ROOT))
        from opt import OptInit
        from model.mm_gtunets import MM_GTUNets
        from model.rp_graph import create_reward_penalty_graph
        argv = sys.argv
        try:
            sys.argv = ["TPSSA_stage_a_options"]
            opt = OptInit(model="MM_GTUNets", dataset="ABIDE", atlas="aal").args
        finally:
            sys.argv = argv
        directory = Path(ar["anchor"])
        opt.seed, opt.device = 911, torch.device("cuda:0")
        opt.data_folder = str(ROOT / CFG["data_directory"])
        opt.ckpt_path = str(directory.parent / "vae")
        opt.log_save = opt.model_save = False
        ctx = load_torch(HISTORY / "artifacts/context.pt")
        meta = read_metadata(ctx)
        sp = load_torch(directory / "splits.pt")
        checkpoint = load_torch(ar["d4_checkpoint"])
        core = MM_GTUNets(opt, fold).to(opt.device)
        import_core(core, checkpoint["core"])
        assert core_hash(export_core(core)) == ar["d4_best_core_sha256"]
        core.vae_save_path = str(directory.parent / f"vae/fold{fold}_pretrain.pth")
        core.eval()
        x = load_torch(directory.parent / "selected_features.pt").to(opt.device)
        ph = torch.as_tensor(ctx["ph_data"], dtype=torch.float32, device=opt.device)
        aff = torch.as_tensor(create_reward_penalty_graph(ctx["ph_dict"], ctx["labels"],
                              sp["train"], sp["val"], sp["test"], opt), dtype=torch.float32, device=opt.device)
        # Fixed diagnostic stream, isolated from every historical/formal stream.
        start = load_torch(directory / "train_start.pt")
        backend = start["rng"]["backend"]
        torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = backend["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = backend["cudnn_allow_tf32"]
        torch.use_deterministic_algorithms(False)
        seed = ar["diagnostic_seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        pre_rng = rng_hash(capture_rng())
        train = np.asarray(sp["train"], np.int64)
        captured = {}
        count = {"core": 0, "img": 0, "ph": 0, "clf": 0, "fusion": 0}
        def bump(key):
            def hook(module, inp, output):
                count[key] += 1
                if key == "img":
                    assert tuple(output.shape) == (871, 16)
                    captured["z"] = output[torch.as_tensor(train, device=output.device)].detach().cpu().numpy().copy()
            return hook
        handles = [module.register_forward_hook(bump(key)) for key, module in (
            ("core", core), ("img", core.img_unet), ("ph", core.ph_unet), ("clf", core.clf), ("fusion", core.mm_attention))]
        bn0 = int(core.clf[2].num_batches_tracked)
        initial_core = export_core(core)
        atomic_json({"fold": fold, "state": "before_single_forward", "utc": utc(),
                     "manifest_sha256": sha_file(BASE / "manifest.json"), "rng_sha256": pre_rng}, begin_path)
        t0 = time.monotonic()
        try:
            with torch.no_grad():
                logits, graph = core(x, ph, aff)
            torch.cuda.synchronize()
            assert logits.shape == (871, 2) and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(graph))
            # Never index logits by test, compute a metric, or retain any probabilities.
            del logits, graph
        finally:
            for handle in handles:
                handle.remove()
        assert count == {"core": 1, "img": 1, "ph": 1, "clf": 1, "fusion": 1}
        assert int(core.clf[2].num_batches_tracked) == bn0
        after_core = export_core(core)
        drift = {key: float((after_core["state"][key].float() - value.float()).abs().max())
                 for key, value in initial_core["state"].items()
                 if not torch.equal(value, after_core["state"][key])}
        assert all(k.startswith("rp_attention.") or k.startswith("vae.") for k in drift), drift
        assert torch.equal(after_core["rp_weights"], initial_core["rp_weights"])
        post_rng = rng_hash(capture_rng())
        device = {"name": name, "physical_gpu": int(physical_gpu),
                  "total_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        z = captured["z"]
        # Free model memory before small, no-gradient, CPU ridge/SVD calculations.
        del core, x, ph, aff, checkpoint, start, after_core, initial_core
        torch.cuda.empty_cache()
        raw_train = {k: np.asarray(v)[train].copy() for k, v in meta.items()}
        pf = np.asarray(load_torch(ar["d4_checkpoint"])["prototype_preparation"]["prototype_fold"], np.int64)
        probe, encoded = study(z, raw_train, pf, fold)
        # Only the separate protectability diagnostic receives outer-train labels.
        directions, stability = shared_directions(z.astype(np.float64), *encoded, pf,
                                                  np.asarray(ctx["labels"], np.int64)[train])
        assert rng_hash(capture_rng()) == post_rng, "Small statistics disturbed native/global RNG"
        for path, expected in ar["sha256"].items():
            assert sha_file(path) == expected
        result = {"state": "complete", "experiment": 24, "fold": fold, "seed": 911,
                  "completed_utc": utc(), "manifest_sha256": sha_file(BASE / "manifest.json"),
                  "d4_checkpoint_sha256": ar["sha256"][ar["d4_checkpoint"]],
                  "d4_best_core_sha256": ar["d4_best_core_sha256"], "private_fold_sha256": array_hash(pf),
                  "train_indices_sha256": array_hash(train), "train_n": len(train),
                  "z_img_train_sha256": array_hash(z), "z_img_train_shape": list(z.shape),
                  "diagnostic_seed": seed, "rng_before_sha256": pre_rng, "rng_after_sha256": post_rng,
                  "native_forward_count": 1, "module_calls": count, "BN_eval_update_count": 0,
                  "native_expected_inmemory_state_drift": drift, "device": device,
                  "neural_training_count": 0, "test_metric_evaluations": 0, "heldout_labels_used_in_probe": False,
                  "probe": probe, "shared_directions": directions, "shared_direction_stability_absolute_cosine_raw_axes": stability,
                  "elapsed_seconds": time.monotonic()-t0,
                  "scope": "one stochastic best-D4 transductive eval draw, outer-train-only raw phenotype probes"}
        atomic_json(result, out)
        log("stage_a_fold_complete", fold=fold, physical_gpu=int(physical_gpu),
            delta_R2=probe.get("real_delta_R2"), null_p95=probe.get("null_p95"),
            min_protected_norm_squared=min(s["protected_norm_squared"] for s in directions),
            seconds=result["elapsed_seconds"])
        return result


def summarize_stage_a():
    from stage_a import decide
    _, manifest = guard()
    folds = [json.loads((BASE / "stage_a_folds" / f"fold{f}.json").read_text()) for f in range(10)]
    assert all(f["state"] == "complete" and f["manifest_sha256"] == sha_file(BASE / "manifest.json") for f in folds)
    result = {"experiment": 24, "name": "TPSSA", "completed_utc": utc(),
              "manifest_sha256": sha_file(BASE / "manifest.json"), "decision": decide(folds), "folds": folds,
              "total_native_eval_forwards": sum(f["native_forward_count"] for f in folds),
              "formal_tasks_started": 0, "neural_training_tasks": 0, "new_test_metrics_computed": 0,
              "gates_frozen_before_extraction": manifest["no_stage_a_results_before_freeze"]}
    atomic_json(result, BASE / "stage_a.json")
    log("stage_a_decision", **result["decision"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "freeze-a", "stage-a", "summarize-a"])
    parser.add_argument("--folds", nargs="+", type=int)
    args = parser.parse_args()
    if args.mode == "audit":
        if (BASE / "manifest.json").exists():
            raise RuntimeError("Cannot replace frozen input audit")
        audit_inputs()
    elif args.mode == "freeze-a":
        freeze_stage_a()
    elif args.mode == "stage-a":
        for fold in args.folds or []:
            assert 0 <= fold < 10
            stage_a_fold(fold)
    else:
        summarize_stage_a()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("blocked", exception=repr(exc), traceback=traceback.format_exc())
        raise
