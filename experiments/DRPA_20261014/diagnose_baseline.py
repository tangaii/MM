"""A fixed-anchor replay using the original train/evaluate functions verbatim."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run import (BASE, anchor_dir, atomic_json, capture_rng, config, context, core_hash,
                 export_core, import_core, load_torch, make_options, restore_rng, rng_hash,
                 state_hash)
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
import train_mm_gtunets as original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--anchor-folder", type=Path)
    args = parser.parse_args()
    out = BASE / "artifacts" / "baseline_diagnostics" / args.tag
    if out.exists():
        raise RuntimeError("Diagnostic results are immutable; choose a new diagnostic tag")
    out.mkdir(parents=True)
    adir = (args.anchor_folder / f"seed_{args.seed}" / f"fold_{args.fold}") if args.anchor_folder else anchor_dir(args.seed, args.fold)
    start = load_torch(adir / "train_start.pt")
    ctx = context()
    split = load_torch(adir / "splits.pt")
    opts = make_options(args.seed, out, initialize=True)
    core = MM_GTUNets(opts, args.fold).to(opts.device)
    core.vae_save_path = str(adir.parent / "vae" / f"fold{args.fold}_pretrain.pth")
    import_core(core, start["core"])
    if core_hash(export_core(core)) != core_hash(start["core"]):
        raise AssertionError("Diagnostic initialization mismatch")
    original.opt = opts
    original.fold = args.fold
    original.model = core
    original.x = load_torch(adir.parent / "selected_features.pt")
    original.y = ctx["labels"]
    original.ph_features = torch.from_numpy(ctx["ph_data"]).float()
    original.train_ind, original.val_ind, original.test_ind = [split[k] for k in ("train", "val", "test")]
    original.labels = torch.tensor(ctx["labels"], dtype=torch.long).to(opts.device)
    original.affinity_graphs = torch.from_numpy(create_reward_penalty_graph(
        ctx["ph_dict"], ctx["labels"], original.train_ind, original.val_ind, original.test_ind, opts)).float()
    original.optimizer = torch.optim.Adam(core.parameters(), lr=opts.lr, weight_decay=opts.wd)
    original.loss_fn = torch.nn.CrossEntropyLoss()
    original.early_stopping = EarlyStopping(patience=opts.early_stop, verbose=True)
    original.fold_model_path = str(out / f"fold{args.fold}.pth")
    for name in ("accs", "sens", "spes", "f1", "aucs"):
        setattr(original, name, np.zeros(10, dtype=np.float32))
    original.corrects = np.zeros(10, dtype=np.int32)
    restore_rng(start["rng"])
    print("ORIGINAL FUNCTION REPLAY", args.tag, rng_hash(capture_rng()),
          "cudnn_deterministic", torch.backends.cudnn.deterministic,
          "cudnn_benchmark", torch.backends.cudnn.benchmark, flush=True)
    original.train()
    before_test = rng_hash(capture_rng())
    original.evaluate()
    result = {"seed": args.seed, "fold": args.fold, "tag": args.tag,
        "official_functions": True, "model_hooks": False, "extra_backward": False,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "ACC": float(original.accs[args.fold]), "AUC": float(original.aucs[args.fold]),
        "SEN": float(original.sens[args.fold]), "SPE": float(original.spes[args.fold]),
        "F1": float(original.f1[args.fold]), "before_test_rng": before_test,
        "end_rng": rng_hash(capture_rng()),
        "initial_core_fingerprint": core_hash(start["core"]),
        "vae_state_fingerprint": state_hash(load_torch(core.vae_save_path))}
    atomic_json(result, out / "metrics.json")
    print("ORIGINAL REPLAY RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
