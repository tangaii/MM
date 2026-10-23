"""Parity-aware, RAM/device-limited, resume-safe two-PPU task launcher."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch

from runtime import BASE, CFG, result_path, utc


def available_ram_mb():
    values = {}
    with open("/proc/meminfo") as stream:
        for line in stream:
            key, value, *_ = line.split()
            if key in ("MemAvailable:", "MemTotal:"):
                values[key] = int(value) // 1024
    return values


def gpu_capacity():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Both registered PPUs must be visible to the launcher")
    info = {}
    for gpu in (0, 1):
        name = torch.cuda.get_device_name(gpu)
        if "PPU" not in name.upper():
            raise RuntimeError(f"GPU {gpu} is not the registered PPU: {name}")
        free_b, total_b = torch.cuda.mem_get_info(gpu)
        requested = int(CFG["workers_per_gpu"])
        device_slots = int(max(free_b - 12 * 1024**3, 0) // (16 * 1024**3))
        if device_slots < 1:
            raise RuntimeError(f"GPU {gpu} has insufficient free memory for one registered worker")
        device_limit = min(requested, device_slots)
        info[str(gpu)] = {"name": name, "free_mib_before": free_b / 1024**2,
                          "total_mib": total_b / 1024**2,
                          "requested_workers": requested,
                          "allowed_workers": device_limit}
    ram = available_ram_mb()
    reserve_mb = 16 * 1024
    estimated_worker_mb = 8 * 1024
    host_capacity = min(4, (ram["MemAvailable:"] - reserve_mb) // estimated_worker_mb)
    if host_capacity < 1:
        raise RuntimeError("Host RAM reserve leaves no capacity for a PEIM worker")
    while sum(item["allowed_workers"] for item in info.values()) > host_capacity:
        gpu = max((0, 1), key=lambda g: info[str(g)]["allowed_workers"])
        if info[str(gpu)]["allowed_workers"] <= 1:
            break
        info[str(gpu)]["allowed_workers"] -= 1
    return info, ram, host_capacity


def run_task(group, fold, gpu, slot, retries=1):
    output = result_path(group, fold)
    log_path = output.with_suffix(".launcher.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = str(CFG["host_threads"])
    env["MKL_NUM_THREADS"] = str(CFG["host_threads"])
    env["OPENBLAS_NUM_THREADS"] = str(CFG["host_threads"])
    env["NUMEXPR_NUM_THREADS"] = str(CFG["host_threads"])
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(BASE), str(BASE.parent.parent), env.get("PYTHONPATH", "")))
    command = [sys.executable, str(BASE / "run.py"), "--group", group,
               "--fold", str(fold), "--gpu", str(gpu)]
    task_started = time.perf_counter()
    errors = []
    for attempt in range(retries + 1):
        started_utc = utc()
        with log_path.open("a") as log:
            log.write(json.dumps({"event": "attempt_start", "utc": started_utc,
                                  "attempt": attempt + 1, "gpu": gpu, "slot": slot}) + "\n")
            log.flush()
            proc = subprocess.run(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  check=False, text=True)
            log.write(json.dumps({"event": "attempt_end", "utc": utc(),
                                  "attempt": attempt + 1, "returncode": proc.returncode}) + "\n")
            log.flush()
        if proc.returncode == 0:
            return {"group": group, "fold": fold, "gpu": gpu, "slot": slot,
                    "returncode": 0, "attempts": attempt + 1,
                    "elapsed_seconds": time.perf_counter() - task_started,
                    "checkpoint": str(output), "log": str(log_path)}
        errors.append({"attempt": attempt + 1, "returncode": proc.returncode})
    return {"group": group, "fold": fold, "gpu": gpu, "slot": slot,
            "returncode": errors[-1]["returncode"], "attempts": len(errors),
            "errors": errors, "elapsed_seconds": time.perf_counter() - task_started,
            "checkpoint": str(output), "log": str(log_path)}


def main():
    resource, ram, host_capacity = gpu_capacity()
    slots = {gpu: threading.Semaphore(resource[str(gpu)]["allowed_workers"]) for gpu in (0, 1)}
    jobs = [(group, fold, fold % 2) for group in CFG["groups"] for fold in range(10)]
    started = time.perf_counter()
    started_utc = utc()
    results = []
    lock = threading.Lock()

    def dispatch(job):
        group, fold, gpu = job
        with slots[gpu]:
            slot = threading.get_ident()
            return run_task(group, fold, gpu, slot)

    max_workers = max(1, min(host_capacity, sum(item["allowed_workers"] for item in resource.values())))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(dispatch, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    completed = sum(row["returncode"] == 0 for row in results)
    record = {"experiment": CFG["experiment"], "started_utc": started_utc,
              "completed_utc": utc(),
              "wall_seconds": time.perf_counter() - started,
              "tasks_total": len(jobs), "tasks_completed": completed,
              "tasks_by_gpu": {str(g): sum(r["gpu"] == g and r["returncode"] == 0 for r in results)
                               for g in (0, 1)},
              "retries": sum(max(0, row["attempts"] - 1) for row in results),
              "failures": [row for row in results if row["returncode"] != 0],
              "gpu_resources_before": resource, "host_memory_before_mb": ram,
              "host_worker_capacity": host_capacity, "worker_limit": max_workers,
              "workers_per_gpu_limit": {key: value["allowed_workers"] for key, value in resource.items()},
              "estimated_worker_memory_mb": 8192,
              "tasks": sorted(results, key=lambda row: (row["group"], row["fold"]))}
    (BASE / "launch.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if completed != 40:
        raise SystemExit(f"Only {completed}/40 tasks completed; see launch.json and task logs")
    print(json.dumps({k: record[k] for k in ("tasks_completed", "tasks_by_gpu", "wall_seconds", "retries")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
