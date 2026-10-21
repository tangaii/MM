"""Resource-aware, resumable two-PPU queue for the registered 40 PCDPA-X2 tasks."""
from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from run import BASE, CFG, ROOT, append_log, atomic_json, guard, result_path, sha_file, valid_result


def host_available_mib() -> float:
    cgroup = Path("/sys/fs/cgroup")
    try:
        if (cgroup / "memory.max").exists():
            limit_text = (cgroup / "memory.max").read_text().strip()
            if limit_text != "max":
                limit = int(limit_text)
                current = int((cgroup / "memory.current").read_text().strip())
                stats = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
                inactive = int(stats.get("inactive_file", 0))
                return max(0.0, (limit - current + inactive) / 1024**2)
        if (cgroup / "memory/memory.limit_in_bytes").exists():
            limit = int((cgroup / "memory/memory.limit_in_bytes").read_text())
            current = int((cgroup / "memory/memory.usage_in_bytes").read_text())
            stats = dict(line.split() for line in (cgroup / "memory/memory.stat").read_text().splitlines())
            inactive = int(stats.get("total_inactive_file", stats.get("inactive_file", 0)))
            return max(0.0, (limit - current + inactive) / 1024**2)
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    current = Path("/sys/fs/cgroup/memory.current")
    maximum = Path("/sys/fs/cgroup/memory.max")
    try:
        limit = maximum.read_text().strip()
        if limit != "max":
            return max(0.0, (int(limit) - int(current.read_text().strip())) / 1024**2)
    except (OSError, ValueError):
        pass
    return float("inf")


def gpu_memory(gpu: int) -> tuple[float, float]:
    free, total = torch.cuda.mem_get_info(gpu)
    return float(free) / 1024**2, float(total) / 1024**2


def task_list():
    return [
        {"group": group, "seed": 911, "fold": fold, "gpu": fold % 2, "attempts": 0}
        for group in CFG["groups"]
        for fold in range(10)
    ]


def queue_state(path: Path, state: dict):
    atomic_json(state, path)


def _run_queue() -> int:
    if len(sys.argv) != 2 or sys.argv[1] != "run":
        raise SystemExit("Usage: python -B launch.py run")
    audit_info = guard(require_manifest=True)
    if not audit_info.get("formal_authorized"):
        raise AssertionError("Formal task queue is not authorized")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Both registered PPUs must be visible to the queue")
    for gpu in (0, 1):
        if "PPU" not in torch.cuda.get_device_name(gpu).upper():
            raise RuntimeError(f"Physical device {gpu} is not the registered PPU")

    queue_path = BASE / "queue.json"
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pending = task_list()
    active = {}
    retries = []
    failures = []
    completed = []
    resource_samples = []
    peak_workers = {"0": 0, "1": 0}
    worker_seconds = 0.0
    max_attempts = CFG["max_attempts"]
    state = {
        "experiment": CFG["experiment"], "state": "running", "started_utc": started_utc,
        "registered_tasks": 40, "planned_gpu_tasks": {"0": 20, "1": 20},
        "workers_per_gpu_cap": CFG["workers_per_gpu"], "completed": [], "pending": [],
        "active": [], "retries": [], "failures": [], "peak_workers": peak_workers,
        "resource_samples": [],
    }
    append_log("queue_started", tasks=40, worker_cap=CFG["workers_per_gpu"])

    while pending or active:
        for process, info in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            info["log_stream"].close()
            active.pop(process)
            task = info["task"]
            task_seconds = time.perf_counter() - info["started"]
            worker_seconds += task_seconds
            if return_code == 0:
                try:
                    result = valid_result(task["group"], task["seed"], task["fold"], audit_info, required=True)
                    completed.append({**task, "attempt": info["attempt"],
                                      "artifact_sha256": sha_file(result_path(task["group"], task["seed"], task["fold"])),
                                      "worker_seconds": result["training_seconds"]})
                except Exception as error:
                    return_code = 86
                    error_text = f"Successful worker left an invalid artifact: {error!r}"
                else:
                    append_log("queue_task_validated", group=task["group"], seed=task["seed"],
                               fold=task["fold"], gpu=task["gpu"], attempt=info["attempt"])
                    continue
            else:
                error_text = f"Worker returned {return_code}"
            failed = {**task, "attempt": info["attempt"], "return_code": return_code,
                      "error": error_text, "log": str(info["log_path"])}
            if info["attempt"] < max_attempts:
                retries.append(failed)
                retry_task = {key: value for key, value in task.items() if key != "attempts"}
                retry_task["attempts"] = info["attempt"]
                pending.insert(0, retry_task)
                append_log("queue_task_retry", group=task["group"], fold=task["fold"],
                           gpu=task["gpu"], attempt=info["attempt"], reason=error_text)
            else:
                failures.append(failed)
                append_log("queue_task_failed", group=task["group"], fold=task["fold"],
                           gpu=task["gpu"], attempt=info["attempt"], reason=error_text)

        for gpu in (0, 1):
            active_gpu = sum(info["task"]["gpu"] == gpu for info in active.values())
            if active_gpu >= CFG["workers_per_gpu"]:
                continue
            try:
                free_mib, total_mib = gpu_memory(gpu)
            except Exception as error:
                failures.append({"gpu": gpu, "error": f"Cannot query PPU memory: {error!r}"})
                continue
            host_mib = host_available_mib()
            host_slots = max(0, int((host_mib - CFG["host_reserve_mib"])
                                    // CFG["host_worker_budget_mib"]))
            gpu_slots = max(0, int((free_mib - CFG["gpu_reserve_mib"])
                                   // CFG["gpu_worker_budget_mib"]))
            allowed = min(CFG["workers_per_gpu"], gpu_slots, host_slots)
            sample = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu,
                "free_mib": free_mib, "total_mib": total_mib, "host_available_mib": host_mib,
                "active_workers_gpu": active_gpu, "allowed_workers_gpu": allowed,
            }
            resource_samples.append(sample)
            if active_gpu >= allowed or not pending:
                continue
            task_index = next((i for i, item in enumerate(pending) if item["gpu"] == gpu), None)
            if task_index is None:
                continue
            task = pending.pop(task_index)
            attempt = task.pop("attempts") + 1
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(CFG["host_threads"]),
                "MKL_NUM_THREADS": str(CFG["host_threads"]), "OPENBLAS_NUM_THREADS": str(CFG["host_threads"]),
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            })
            log_path = result_path(task["group"], task["seed"], task["fold"]).with_suffix(".launcher.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("ab", buffering=0)
            command = [sys.executable, "-B", str(BASE / "run.py"), "train", "--group", task["group"],
                       "--seed", str(task["seed"]), "--fold", str(task["fold"])]
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[proc] = {"task": task, "attempt": attempt, "started": time.perf_counter(),
                            "log_stream": stream, "log_path": log_path}
            active_count = sum(item["task"]["gpu"] == gpu for item in active.values())
            peak_workers[str(gpu)] = max(peak_workers[str(gpu)], active_count)
            append_log("queue_task_started", group=task["group"], seed=task["seed"], fold=task["fold"],
                       gpu=gpu, attempt=attempt, pid=proc.pid, allowed_workers=allowed,
                       free_mib=free_mib, host_available_mib=host_mib)

        state.update({
            "completed": completed, "pending": pending,
            "active": [{"pid": proc.pid, **info["task"], "attempt": info["attempt"]}
                       for proc, info in active.items()],
            "retries": retries, "failures": failures, "peak_workers": peak_workers,
            "resource_samples": resource_samples[-200:],
            "elapsed_seconds": time.perf_counter() - started,
        })
        queue_state(queue_path, state)
        if active or pending:
            time.sleep(5)

    elapsed = time.perf_counter() - started
    state.update({"state": "complete" if len(completed) == 40 and not failures else "partial",
                  "completed": completed, "pending": pending, "active": [], "retries": retries,
                  "failures": failures, "peak_workers": peak_workers,
                  "resource_samples": resource_samples, "elapsed_seconds": elapsed,
                  "worker_seconds": worker_seconds, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    queue_state(queue_path, state)

    audit_info = json.loads((BASE / "audit.json").read_text())
    audit_info["execution"] = {
        "tasks": 40, "gpu0": sum(task["gpu"] == 0 for task in completed),
        "gpu1": sum(task["gpu"] == 1 for task in completed), "completed": len(completed),
        "failures": failures, "retries": len(retries), "queue_wall_seconds": elapsed,
        "worker_seconds": worker_seconds, "peak_workers": peak_workers,
        "resource_samples": resource_samples,
    }
    atomic_json(audit_info, BASE / "audit.json")
    append_log("queue_finished", state=state["state"], completed=len(completed),
               retries=len(retries), failures=len(failures), wall_seconds=elapsed,
               worker_seconds=worker_seconds)
    return 0 if state["state"] == "complete" else 1


def main() -> int:
    lock_path = BASE / "launcher.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A PCDPA-X2 task queue is already active") from error
        try:
            return _run_queue()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
