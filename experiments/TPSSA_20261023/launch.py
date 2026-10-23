"""Two PPU read-only Stage A workers. Formal stage is gated separately."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from common import BASE, atomic_json, log, utc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["stage-a", "formal"])
    args = p.parse_args()
    if args.mode == "formal":
        stage = json.loads((BASE / "stage_a.json").read_text())
        if not stage["decision"]["passed"]:
            raise RuntimeError("NO_GO_STAGE_A: formal training and train-start preflight are forbidden (0/40)")
        raise RuntimeError("Stage B requires its mathematical acceptance, single-forward preflight, and frozen implementation before launch")
    t0 = time.monotonic()
    procs = []
    for gpu in (0, 1):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1",
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        folds = [str(f) for f in range(10) if f % 2 == gpu]
        stream = (BASE / f"stage_a_gpu{gpu}.log").open("a")
        proc = subprocess.Popen([sys.executable, "-B", str(BASE / "run.py"), "stage-a", "--folds", *folds],
                                env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=BASE)
        procs.append((gpu, proc, stream))
        log("stage_a_worker_started", gpu=gpu, pid=proc.pid, folds=folds)
    codes = {}
    while procs:
        remaining = []
        for gpu, proc, stream in procs:
            code = proc.poll()
            if code is None:
                remaining.append((gpu, proc, stream))
            else:
                codes[str(gpu)] = code
                stream.close()
                log("stage_a_worker_exit", gpu=gpu, returncode=code)
        procs = remaining
        if procs:
            time.sleep(1)
    atomic_json({"started_utc": utc(), "seconds": time.monotonic()-t0, "worker_exit_codes": codes,
                 "gpu0_folds": [0, 2, 4, 6, 8], "gpu1_folds": [1, 3, 5, 7, 9],
                 "neural_training_tasks": 0}, BASE / "stage_a_execution.json")
    if any(codes.values()):
        raise RuntimeError(f"Stage A worker blocked: {codes}")
    subprocess.run([sys.executable, "-B", str(BASE / "run.py"), "summarize-a"], check=True, cwd=BASE)


if __name__ == "__main__":
    main()
