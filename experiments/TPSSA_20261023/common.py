"""Small read-only history interface; all new writes are confined to BASE."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/MM-GTUNets")
HISTORY = ROOT / "experiments/DRPA_20261014"
D4 = ROOT / "experiments/PCDCA_20261019"
CFG = json.loads((BASE / "config.json").read_text())
SOURCE = ("model/mm_gtunets.py", "model/gtunet.py", "model/rp_graph.py", "opt.py",
          "train_mm_gtunets.py", "utils/mydataloader.py", "utils/tools.py", "utils/metrics.py")
STAGE_A_CODE = ("config.json", "registration.md", "common.py", "stage_a.py", "run.py", "test_stage_a.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def array_hash(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(array_hash(state[k]).encode())
    return h.hexdigest()


def core_hash(core):
    return state_hash({**core["state"], "unregistered_rp_weights": core["rp_weights"]})


def export_core(model):
    return {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "rp_weights": model.rp_attention.weights.detach().cpu().clone()}


def import_core(model, core):
    import torch
    model.load_state_dict(core["state"], strict=True)
    with torch.no_grad():
        model.rp_attention.weights.copy_(core["rp_weights"].to(model.device))


def payload_hash(value):
    """Exact historical artifact checksum, including explicit type/shape tags."""
    import numpy as np
    import torch
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(b"tensor"); add(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            h.update(b"array"); h.update(array_hash(v).encode())
        elif isinstance(v, np.generic):
            add(v.item())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=repr):
                add(k); add(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(str(type(v)).encode()); h.update(str(len(v)).encode())
            for x in v:
                add(x)
        elif v is None or isinstance(v, (str, int, float, bool)):
            h.update(str(type(v)).encode()); h.update(repr(v).encode())
        else:
            raise TypeError(type(v))
    add(value)
    return h.hexdigest()


def load_torch(path, verify_payload=True):
    import torch
    result = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(result, dict) and set(result) == {"payload", "content_sha256"}:
        if verify_payload and payload_hash(result["payload"]) != result["content_sha256"]:
            raise IOError(f"Historical artifact checksum mismatch: {path}")
        return result["payload"]
    return result


def atomic_json(value, path):
    path = Path(path)
    if BASE not in path.resolve().parents:
        raise ValueError("New writes outside experiment directory are forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("xb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    if path.read_bytes() != data:
        raise IOError(f"Readback failed: {path}")


def private_seed(fold, purpose):
    message = f"911/{fold}/TPSSA/{purpose}"
    return int(hashlib.sha256(message.encode()).hexdigest()[:15], 16) % (2**32)


def historical_module(name, path):
    # The accepted historical runner imports its adjacent `prototype` module.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def read_metadata(ctx):
    import csv
    import numpy as np
    path = ROOT / CFG["data_directory"] / "Phenotypic_V1_0b_preprocessed1.csv"
    with path.open() as f:
        rows = {r["SUB_ID"]: r for r in csv.DictReader(f)}
    ordered = [rows[str(i)] for i in ctx["ids"]]
    def number(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float("nan")
    age = np.asarray([number(r["AGE_AT_SCAN"]) for r in ordered])
    sex = np.asarray([number(r["SEX"]) for r in ordered])
    site = np.asarray([r["SITE_ID"] for r in ordered], str)
    # This verifies input semantics only; probes never receive a diagnosis vector.
    y = np.asarray([int(r["DX_GROUP"]) - 1 for r in ordered])
    assert np.array_equal(y, ctx["labels"])
    assert np.array_equal(np.bincount(y), [403, 468])
    assert [(int(((y == c) & (sex == 2)).sum()), int(((y == c) & (sex == 1)).sum()))
            for c in (0, 1)] == [(54, 349), (90, 378)]
    assert np.array_equal(site, ctx["site_names"])
    assert np.allclose(age, ctx["ph_data"][:, 2], rtol=0, atol=2e-6, equal_nan=True)
    assert np.array_equal(sex - 1, ctx["ph_data"][:, 1])
    return {"age": age, "sex": sex, "site": site}


def rng_hash(state):
    return {"python": hashlib.sha256(repr(state["python"]).encode()).hexdigest(),
            "numpy": hashlib.sha256(repr(state["numpy"][0]).encode() + state["numpy"][1].tobytes()
                                    + repr(state["numpy"][2:]).encode()).hexdigest(),
            "cpu": array_hash(state["cpu"]), "cuda": array_hash(state["cuda"])}


def capture_rng():
    import random
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone()}


def log(event, **values):
    import fcntl
    record = {"utc": utc(), "event": event, **values}
    with (BASE / "run.log").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
