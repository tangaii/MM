"""Single-owner, resource-aware two-PPU queue with verified resume."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import traceback
import run as r
import analyze


CHILDREN = []
METHODS = ("C1", "C2", "C3", "C4", "C5", "C6")
WORKER_BOOTSTRAP = """
import json, sys, time
from pathlib import Path
base = Path(sys.argv.pop(1))
sys.path.insert(0, str(base))
import run as worker
original_guard = worker.source_guard
def metadata_guard():
    deadline = time.monotonic() + 30
    while True:
        try:
            return original_guard()
        except FileNotFoundError as exc:
            if exc.filename != str(base / 'summary.json') or time.monotonic() >= deadline:
                raise
        except json.JSONDecodeError:
            if time.monotonic() >= deadline:
                raise
        time.sleep(.2)
worker.source_guard = metadata_guard
sys.exit(worker.worker())
"""


def log(event, **values):
    line = json.dumps({"time_utc": r.utc(), "event": event, **values}, ensure_ascii=False, allow_nan=False)
    with (r.BASE / "run.log").open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def jobs():
    return [{"seed": seed, "fold": fold} for seed in r.config()["seeds"] for fold in range(10)]


def key(phase, method, job):
    return f"{phase}/{method}/{job['seed']}/{job['fold']}"


def finished(phase, method, job, state):
    seed, fold = job["seed"], job["fold"]
    if phase == "checks":
        return state.get("checks", {}).get(f"{seed}/{fold}", {}).get("passed", False)
    if phase == "evaluate" and method == "C0":
        value = state.get("baseline_frozen", {}).get(f"{seed}/{fold}")
        if not value:
            return False
        baseline = state["audit"]["baseline"][f"{seed}/{fold}"]
        for i, metrics in enumerate(value["draw_metrics"]):
            calculated = r.binary_metrics(baseline["labels"], value["test_pred"][i], value["test_prob_asd"][i])
            if any(abs(metrics[k] - calculated[k]) > 1e-12 for k in r.KEYS):
                raise RuntimeError("C0 saved frozen metrics failed recalculation")
        return True
    return r.verify_result(method, seed, fold, state, require_aux=phase == "evaluate")


def dispatch(phase, method, selected, state):
    selector = selectors.DefaultSelector()
    processes = []
    with (r.BASE / "run.log").open("a") as errors:
        for gpu in (0, 1):
            assigned = [job for job in selected if job["fold"] % 2 == gpu]
            if not assigned:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            if phase == "checks":
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            else:
                env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            process = subprocess.Popen([sys.executable, "-B", str(r.BASE / "run.py"), "--phase", phase, "--method", method],
                                       cwd=r.ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=errors, text=True, bufsize=1)
            CHILDREN.append(process); processes.append(process)
            process.stdin.write(json.dumps(assigned)); process.stdin.close()
            selector.register(process.stdout, selectors.EVENT_READ, {"gpu": gpu, "process": process, "jobs": assigned})
            log("worker_started", phase=phase, method=method, gpu=gpu, pid=process.pid, tasks=len(assigned))
        last_heartbeat = time.monotonic()
        while selector.get_map():
            events = selector.select(timeout=10)
            for entry, _ in events:
                line = entry.fileobj.readline()
                if not line:
                    selector.unregister(entry.fileobj); entry.fileobj.close()
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict) or "kind" not in value:
                    continue
                kind = value["kind"]
                if kind == "progress":
                    log("progress", method=method, seed=value["seed"], fold=value["fold"],
                        epoch=value["epoch"], val_loss=value["val_loss"])
                    continue
                identity = key(phase, method, value)
                if kind == "start":
                    state["tasks"][identity] = {"state": "running", "gpu": value["physical_gpu"],
                                                "started_utc": value["time_utc"], "attempt": state["attempts"].get(identity, 0)}
                    log("task_started", task=identity, gpu=value["physical_gpu"])
                elif kind == "done":
                    if phase == "checks":
                        state.setdefault("checks", {})[f"{value['seed']}/{value['fold']}"] = value["result"]
                    elif phase == "evaluate" and method == "C0":
                        state.setdefault("baseline_frozen", {})[f"{value['seed']}/{value['fold']}"] = value["result"]
                    state["tasks"][identity] = {"state": "complete", "gpu": value["physical_gpu"],
                                                "completed_utc": value["time_utc"], "attempt": state["attempts"].get(identity, 0)}
                    log("task_complete", task=identity, gpu=value["physical_gpu"],
                        result=value["result"] if phase == "train" else {"verified": True})
                elif kind == "error":
                    state["errors"].append(value)
                    state["tasks"][identity] = {"state": "failed", "error": value["error"], "gpu": value["physical_gpu"]}
                    log("task_failed", task=identity, error=value["error"], traceback=value["traceback"])
                r.atomic_json(state)
            if time.monotonic() - last_heartbeat >= 60:
                log("heartbeat", phase=phase, method=method,
                    complete=sum(v.get("state") == "complete" for k, v in state["tasks"].items() if k.startswith(f"{phase}/{method}/")))
                last_heartbeat = time.monotonic()
        selector.close()
        for process in processes:
            status = process.wait()
            CHILDREN.remove(process)
            log("worker_exit", pid=process.pid, code=status)


def stage(phase, method, state):
    state["phase"] = f"{phase}/{method}"
    r.atomic_json(state)
    pending = []
    for job in jobs():
        if not finished(phase, method, job, state):
            pending.append(job)
        else:
            identity = key(phase, method, job)
            previous = state["tasks"].get(identity, {})
            state["tasks"][identity] = {**previous, "state": "complete", "gpu": job["fold"] % 2,
                                        "verified_on_resume_utc": r.utc()}
    r.atomic_json(state)
    while pending:
        for job in pending:
            identity = key(phase, method, job)
            attempt = state["attempts"].get(identity, 0) + 1
            if attempt > r.config()["max_attempts"]:
                raise RuntimeError(f"Task exhausted fixed retry budget: {identity}; inspect recorded errors before resume")
            state["attempts"][identity] = attempt
        r.atomic_json(state)
        dispatch(phase, method, pending, state)
        pending = [job for job in pending if not finished(phase, method, job, state)]
    log("stage_complete", phase=phase, method=method, tasks=30)


def initialize():
    path = r.BASE / "summary.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["training_code_sha256"] != r.code_hash():
            trained = list((r.BASE / "runs").glob("C*/*.pt"))
            if trained:
                raise RuntimeError("Training code changed after results exist; refuse mixed provenance")
            log("pretraining_implementation_revalidated", previous=state["training_code_sha256"], current=r.code_hash())
            state = None
        else:
            for name, value in state["audit"]["manifest"].items():
                if r.sha_file(r.ROOT / name) != value["sha256"]:
                    raise RuntimeError(f"Read-only historical input changed: {name}")
            return state
    log("audit_started", source_commit=r.config()["source_commit"])
    audited = r.audit()
    state = {"experiment": "PCCI_20261015", "created_utc": r.utc(), "phase": "audited", "config": r.config(),
             "training_code_sha256": r.code_hash(), "script_sha256": {name: r.sha_file(r.BASE / name) for name in ("run.py", "pcci.py", "launch.py", "analyze.py", "config.json")},
             "audit": audited, "checks": {}, "tasks": {}, "attempts": {}, "errors": [], "groups": {},
             "preregistration": {"primary": r.config()["primary_comparisons"], "comparisons": r.config()["comparisons"],
                                 "selection": "original validation CE+graph only; never test labels/metrics", "draws": 5,
                                 "strong_continuation_rule": "Frozen ACC/AUC both positive in all three seeds for C3-C0, C3-C2, C3-C6, and positive within-model real-vs-shuffled CE benefit; descriptive conservative rule, not a significance test"}}
    r.atomic_json(state)
    log("audit_complete", baseline=30, protected_files=len(audited["manifest"]), phenotype_columns=audited["phenotype_columns_original"])
    return state


def take_over(pid):
    """Retire only the coordinator, retaining readers for every healthy worker."""
    root = Path(f"/proc/{pid}")
    if not root.exists():
        return []
    command = (root / "cmdline").read_bytes().split(b"\0")
    if not any(Path(arg.decode()).name == "launch.py" for arg in command if arg):
        raise RuntimeError("Handover target is not this experiment's launcher")
    if Path(os.readlink(root / "cwd")).resolve() != r.ROOT:
        raise RuntimeError("Handover target belongs to another workspace")
    os.kill(pid, signal.SIGSTOP)
    readers = []
    try:
        for _ in range(100):
            if (root / "stat").read_text().split(") ", 1)[1].startswith("T"):
                break
            time.sleep(.01)
        else:
            raise RuntimeError("Old coordinator did not enter a stable handover state")
        children = (root / "task" / str(pid) / "children").read_text().split()
        state = json.loads((r.BASE / "summary.json").read_text())
        for child in children:
            proc = Path(f"/proc/{child}")
            if (proc / "stat").read_text().split(") ", 1)[1].startswith("Z"):
                continue
            args = [arg.decode() for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
            if not any(str(r.BASE) in arg for arg in args):
                raise RuntimeError(f"Refuse to abandon an unrecognized child: {child}")
            phase, method = args[args.index("--phase") + 1], args[args.index("--method") + 1]
            environment = dict(item.split("=", 1) for item in (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
            gpu = int(environment["CUDA_VISIBLE_DEVICES"])
            pipe = os.readlink(proc / "fd/1")
            matches = [fd for fd in (root / "fd").iterdir() if os.readlink(fd) == pipe]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot retain worker {child}'s stdout before coordinator retirement")
            fd = os.open(matches[0], os.O_RDONLY | os.O_NONBLOCK)
            assigned = [{"phase": phase, "method": method, **job} for job in jobs()
                        if job["fold"] % 2 == gpu and state["tasks"].get(key(phase, method, job), {}).get("state") != "complete"]
            readers.append({"pid": int(child), "gpu": gpu, "fd": fd, "buffer": b"", "jobs": assigned,
                            "process": None, "adopted": True})
        # Each child now has a replacement pipe reader, so SIGTERM cannot cause SIGPIPE.
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        for _ in range(300):
            if not root.exists() or (root / "stat").read_text().split(") ", 1)[1].startswith("Z"):
                break
            time.sleep(.01)
        else:
            raise RuntimeError("Old coordinator did not retire")
        log("coordinator_handover", retired_pid=pid, retained_workers=[{"pid": w["pid"], "gpu": w["gpu"],
            "reserved_tasks": len(w["jobs"])} for w in readers], training_workers_interrupted=False)
        return readers
    except BaseException:
        for worker in readers:
            os.close(worker["fd"])
        if root.exists():
            os.kill(pid, signal.SIGCONT)
        raise


class Resources:
    def __init__(self, maximum):
        self.maximum = maximum
        self.previous_cpu = None
        self.budget_mib = {0: 12288, 1: 12288}
        self.cooldown = {0: 0., 1: 0.}
        self.penalty = {0: 0, 1: 0}

    def sample(self, workers):
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                                          "--format=csv,noheader,nounits"], text=True, timeout=10)
        devices = {}
        for row in output.splitlines():
            index, total, used, util = [int(value.strip()) for value in row.split(",")]
            devices[index] = {"total_mib": total, "used_mib": used, "utilization_percent": util}
        if not all(gpu in devices for gpu in (0, 1)):
            raise RuntimeError("Both physical PPU devices must be observable")
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                          "--format=csv,noheader,nounits"], text=True, timeout=10)
        process_memory = {}
        for row in output.splitlines():
            if row.strip():
                pid, used = row.split(",")
                process_memory[int(pid)] = int(used.strip())
        memory = {line.split(":", 1)[0]: int(line.split()[1]) // 1024
                  for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line and len(line.split()) >= 2}
        cpu = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        busy = 0.
        if self.previous_cpu is not None:
            delta = [a - b for a, b in zip(cpu, self.previous_cpu)]
            busy = 1. - (delta[3] + delta[4]) / max(sum(delta), 1)
        self.previous_cpu = cpu
        cpus = len(os.sched_getaffinity(0))
        snapshot = {"time_utc": r.utc(), "devices": devices, "ram_available_mib": memory["MemAvailable"],
                    "cpu_busy_fraction": busy, "cpu_count": cpus, "load1": os.getloadavg()[0]}
        active = {gpu: [w for w in workers if w["gpu"] == gpu] for gpu in (0, 1)}
        capacities = {}
        for gpu in (0, 1):
            if time.monotonic() >= self.cooldown[gpu]:
                self.penalty[gpu] = 0
            for worker in active[gpu]:
                used = process_memory.get(worker["pid"], 0)
                self.budget_mib[gpu] = max(self.budget_mib[gpu], int(used * 1.3))
            budget = self.budget_mib[gpu]
            projected = devices[gpu]["used_mib"] + sum(max(0, budget - process_memory.get(w["pid"], 0)) for w in active[gpu])
            free = devices[gpu]["total_mib"] - 10240 - projected
            slots = max(0, free // budget)
            capacities[gpu] = min(self.maximum - self.penalty[gpu], len(active[gpu]) + slots)
            devices[gpu].update(projected_mib=projected, worker_budget_mib=budget,
                                active_workers=len(active[gpu]), worker_limit=max(0, capacities[gpu]))
        new_slots = max(0, (memory["MemAvailable"] - 16384) // 8192)
        total_limit = min(self.maximum * 2, cpus, len(workers) + new_slots)
        if busy > .95 or snapshot["load1"] > cpus * 1.5:
            total_limit = len(workers)
        return snapshot, capacities, total_limit

    def failed(self, gpu, error):
        if any(word in error.lower() for word in ("out of memory", "oom", "allocation failed")):
            self.penalty[gpu] = min(self.maximum - 1, self.penalty[gpu] + 1)
            self.cooldown[gpu] = time.monotonic() + 90
            self.budget_mib[gpu] = int(self.budget_mib[gpu] * 1.25)


def reconcile_legacy_reservations(state):
    """Old batches debited every reserved fold, including jobs never started."""
    starts, inherited = {}, set()
    worker_groups = {}
    for line in (r.BASE / "run.log").read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if value.get("event") == "worker_started":
            worker_groups[value["pid"]] = value
        elif value.get("event") == "coordinator_handover":
            for worker in value["retained_workers"]:
                original = worker_groups.get(worker["pid"], {})
                if original.get("phase") == "train" and original.get("tasks", 0) > 1:
                    inherited.update(key("train", original["method"], job) for job in jobs() if job["fold"] % 2 == worker["gpu"])
        elif value.get("event") == "task_started":
            starts.setdefault(value["task"], set()).add(value.get("time_utc"))
    adjustments = state.setdefault("scheduling", {}).setdefault("legacy_reservation_reconciliation", {})
    for identity in sorted(inherited):
        phase, method, seed, fold = identity.split("/")
        job = {"seed": int(seed), "fold": int(fold)}
        if identity in adjustments or finished(phase, method, job, state):
            continue
        old, actual = state["attempts"].get(identity, 0), len(starts.get(identity, set()))
        if old > actual:
            adjustments[identity] = {"previous_batch_debits": old, "actual_started_executions": actual,
                "time_utc": r.utc(), "reason": "Release unstarted legacy batch reservations; fixed two-execution retry budget retained"}
            state["attempts"][identity] = actual
            log("legacy_unstarted_reservations_released", task=identity, previous=old, actual_starts=actual)
    r.atomic_json(state)


def unified_stage(phase, methods, state, maximum=6, adopted=None):
    state["phase"] = f"{phase}/unified"
    policy = state.setdefault("scheduling", {})
    policy.update(mode="unified_resource_aware", maximum_workers_per_gpu=maximum,
                  configuration_override="Scheduling only; frozen config.json workers_per_gpu is not edited",
                  queue_groups=list(METHODS), gpu_reserve_mib=10240, ram_reserve_mib=16384,
                  initial_worker_budget_mib=12288, healthy_workers_preempted=False,
                  metadata_startup_retry="Up to 30s for summary.json ENOENT/JSON visibility during fuse.ossfs replacement; no RNG/model/training changes")
    policy.setdefault("started_utc", r.utc())
    policy.setdefault("affinity_exceptions", {})
    policy.setdefault("peak_workers", {"0": 0, "1": 0})
    selector, active = selectors.DefaultSelector(), list(adopted or [])
    complete, pending, blocked = set(), [], []
    all_jobs = [{"phase": phase, "method": method, **job} for job in jobs() for method in methods]
    for task in all_jobs:
        identity = key(phase, task["method"], task)
        if finished(phase, task["method"], task, state):
            complete.add(identity)
            previous = state["tasks"].get(identity, {})
            state["tasks"][identity] = {**previous, "state": "complete", "verified_on_resume_utc": r.utc()}
        else:
            pending.append(task)
    if phase == "train":
        policy.setdefault("validated_fits_skipped", len(complete))
        manifest = policy.setdefault("preexisting_artifacts", {})
        for path in sorted((r.BASE / "runs").glob("C*/*.pt")):
            name, fingerprint = str(path.relative_to(r.BASE)), r.sha_file(path)
            if name in manifest and manifest[name] != fingerprint:
                raise RuntimeError(f"Preexisting validated formal artifact changed: {name}")
            manifest[name] = fingerprint
    reserved = {key(task["phase"], task["method"], task) for worker in active for task in worker["jobs"]}
    pending = [task for task in pending if key(phase, task["method"], task) not in reserved]
    for worker in active:
        selector.register(worker["fd"], selectors.EVENT_READ, worker)
    announced = {method for method in methods if phase == "train" and method in state.get("groups", {})
                 or phase == "evaluate" and "frozen_probability_mean" in state.get("groups", {}).get(method, {})}
    resources = Resources(maximum)
    last_sample = last_heartbeat = 0.
    snapshot, capacities, total_limit = {}, {0: 0, 1: 0}, 0
    fatal, telemetry_error = None, None

    def announce():
        for method in methods:
            if method not in announced and all(key(phase, method, job) in complete for job in jobs()):
                analyze.update(state, [method], auxiliary=phase == "evaluate")
                announced.add(method)
                log("group_training_complete" if phase == "train" else "group_frozen_complete", method=method,
                    formal=state["groups"][method]["formal"]["mean"], handoff_updated=True)

    def accepted(task, gpu, when):
        identity = key(phase, task["method"], task)
        if not finished(phase, task["method"], task, state):
            raise RuntimeError(f"Worker reported completion without a validated artifact: {identity}")
        complete.add(identity)
        previous = state["tasks"].get(identity, {})
        state["tasks"][identity] = {**previous, "state": "complete", "gpu": gpu, "completed_utc": when}

    def consume(worker, value):
        if not isinstance(value, dict) or "kind" not in value:
            return
        kind, method = value["kind"], value["method"]
        identity = key(phase, method, value)
        if kind == "progress":
            log("progress", method=method, seed=value["seed"], fold=value["fold"], epoch=value["epoch"], val_loss=value["val_loss"])
            return
        if kind == "start":
            state["tasks"][identity] = {"state": "running", "gpu": worker["gpu"], "pid": worker["pid"],
                "started_utc": value["time_utc"], "attempt": state["attempts"].get(identity, 0)}
            log("task_started", task=identity, gpu=worker["gpu"], pid=worker["pid"])
        elif kind == "done":
            if phase == "checks":
                state.setdefault("checks", {})[f"{value['seed']}/{value['fold']}"] = value["result"]
            elif phase == "evaluate" and method == "C0":
                state.setdefault("baseline_frozen", {})[f"{value['seed']}/{value['fold']}"] = value["result"]
            accepted(value, worker["gpu"], value["time_utc"])
            log("task_complete", task=identity, gpu=worker["gpu"],
                result=value["result"] if phase == "train" else {"verified": True})
            announce()
        elif kind == "error":
            state["errors"].append(value)
            state["tasks"][identity] = {"state": "failed", "gpu": worker["gpu"], "error": value["error"]}
            resources.failed(worker["gpu"], value["error"])
            log("task_failed", task=identity, error=value["error"], traceback=value["traceback"])
        r.atomic_json(state)

    announce()
    r.atomic_json(state)
    log("unified_queue_started", phase=phase, pending=len(pending), validated_skipped=len(complete),
        adopted_workers=len(active), maximum_workers_per_gpu=maximum)
    with (r.BASE / "run.log").open("a") as errors:
        while pending or active:
            now = time.monotonic()
            if now - last_sample >= 5:
                try:
                    snapshot, capacities, total_limit = resources.sample(active)
                    policy["latest_resources"] = snapshot
                    if telemetry_error:
                        log("telemetry_recovered")
                    telemetry_error = None
                except Exception as exc:
                    capacities, total_limit = {0: 0, 1: 0}, 0
                    if str(exc) != telemetry_error:
                        log("telemetry_unavailable_admissions_paused", error=str(exc))
                    telemetry_error = str(exc)
                last_sample = now
            if fatal is None:
                while pending and len(active) < total_limit:
                    counts = {gpu: sum(w["gpu"] == gpu for w in active) for gpu in (0, 1)}
                    available = [gpu for gpu in (0, 1) if counts[gpu] < capacities[gpu]]
                    if not available:
                        break
                    gpu = min(available, key=lambda g: (counts[g], snapshot["devices"][g]["utilization_percent"], snapshot["devices"][g]["projected_mib"]))
                    affine = [task for task in pending if task["fold"] % 2 == gpu]
                    task = (affine or pending)[0]
                    pending.remove(task)
                    method, identity = task["method"], key(phase, task["method"], task)
                    if finished(phase, method, task, state):
                        accepted(task, gpu, r.utc()); announce()
                        continue
                    attempt = state["attempts"].get(identity, 0) + 1
                    if attempt > r.config()["max_attempts"]:
                        blocked.append(identity)
                        log("task_retry_budget_exhausted", task=identity)
                        continue
                    state["attempts"][identity] = attempt
                    if phase == "train" and task["fold"] % 2 != gpu:
                        policy["affinity_exceptions"][identity] = {"preferred_gpu": task["fold"] % 2,
                            "actual_gpu": gpu, "reason": "No affinity-local pending task; use available capacity instead of idling", "time_utc": r.utc()}
                    r.atomic_json(state)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
                    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
                    process = subprocess.Popen([sys.executable, "-B", "-c", WORKER_BOOTSTRAP, str(r.BASE), "--phase", phase, "--method", method],
                        cwd=r.ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, start_new_session=True)
                    CHILDREN.append(process)
                    process.stdin.write(json.dumps([{"seed": task["seed"], "fold": task["fold"]}]).encode())
                    process.stdin.close()
                    os.set_blocking(process.stdout.fileno(), False)
                    worker = {"pid": process.pid, "gpu": gpu, "fd": process.stdout.fileno(), "buffer": b"", "jobs": [task],
                              "process": process, "adopted": False}
                    active.append(worker); selector.register(worker["fd"], selectors.EVENT_READ, worker)
                    policy["peak_workers"][str(gpu)] = max(policy["peak_workers"][str(gpu)], counts[gpu] + 1)
                    log("worker_started", phase=phase, method=method, gpu=gpu, pid=process.pid, tasks=1,
                        attempt=attempt, active_on_gpu=counts[gpu] + 1)
            for entry, _ in selector.select(timeout=1):
                worker = entry.data
                data = os.read(worker["fd"], 65536)
                if data:
                    worker["buffer"] += data
                    while b"\n" in worker["buffer"]:
                        line, worker["buffer"] = worker["buffer"].split(b"\n", 1)
                        try:
                            value = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        try:
                            consume(worker, value)
                        except Exception as exc:
                            fatal = traceback.format_exc()
                            log("scheduler_validation_failed_draining_healthy_workers", error=str(exc), traceback=fatal)
                else:
                    selector.unregister(worker["fd"])
                    if worker["process"] is None:
                        os.close(worker["fd"]); status = None
                    else:
                        worker["process"].stdout.close()
                        status = worker["process"].wait()
                        CHILDREN.remove(worker["process"])
                    active.remove(worker)
                    for task in worker["jobs"]:
                        identity = key(phase, task["method"], task)
                        if identity in complete:
                            continue
                        if finished(phase, task["method"], task, state):
                            accepted(task, worker["gpu"], r.utc())
                        elif not any(key(phase, t["method"], t) == identity for t in pending):
                            pending.append(task)
                            if state["tasks"].get(identity, {}).get("state") != "failed":
                                failure = {"time_utc": r.utc(), "task": identity, "pid": worker["pid"], "physical_gpu": worker["gpu"],
                                           "error": f"Worker exited without a validated result (exit={status})"}
                                state["errors"].append(failure)
                                state["tasks"][identity] = {"state": "failed", **failure}
                                log("worker_result_missing", **failure)
                    announce(); r.atomic_json(state)
                    log("worker_exit", pid=worker["pid"], code=status, adopted=worker["adopted"])
                    last_sample = 0.
            policy["active_workers"] = [{"pid": w["pid"], "gpu": w["gpu"], "tasks": [key(phase, t["method"], t) for t in w["jobs"]
                if key(phase, t["method"], t) not in complete]} for w in active]
            if time.monotonic() - last_heartbeat >= 60:
                r.atomic_json(state)
                log("heartbeat", phase=phase, complete=len(complete), total=len(all_jobs), pending=len(pending),
                    workers={str(g): sum(w["gpu"] == g for w in active) for g in (0, 1)}, resources=snapshot)
                last_heartbeat = time.monotonic()
            if fatal and not active:
                raise RuntimeError(fatal)
        selector.close()
    if blocked:
        raise RuntimeError(f"Tasks exhausted the unchanged fixed retry budget: {blocked}")
    if len(complete) != len(all_jobs):
        raise RuntimeError("Unified queue did not validate every task")
    if phase == "train":
        hashes = {name: r.sha_file(r.BASE / name) for name in policy["preexisting_artifacts"]}
        if hashes != policy["preexisting_artifacts"]:
            raise RuntimeError("A preexisting validated formal artifact was overwritten")
        policy["preexisting_artifacts_unchanged_after_training"] = True
        policy["formal_training_completed_utc"] = r.utc()
    r.atomic_json(state)
    log("unified_stage_complete", phase=phase, tasks=len(complete))


def execute(checks_only=False, maximum=6, adopted=None):
    state = initialize()
    reconcile_legacy_reservations(state)
    if state.get("phase") == "complete":
        log("already_complete_verifying")
        paths = sorted((r.BASE / "runs").glob("C*/*.pt"))
        before = {str(path.relative_to(r.BASE)): r.sha_file(path) for path in paths}
        for method in r.config()["methods"]:
            for job in jobs():
                if not finished("evaluate", method, job, state):
                    raise RuntimeError("Previously complete experiment now fails validation")
        analyze.final_checks(state)
        after = {str(path.relative_to(r.BASE)): r.sha_file(path) for path in paths}
        if before != after or len(paths) != 180:
            raise RuntimeError("Completed-run resume changed formal artifacts or count")
        state["resume_verification"] = {"passed": True, "time_utc": r.utc(), "artifacts": len(paths),
            "artifact_manifest_sha256": r.digest(after), "all_file_sha256_unchanged": True,
            "training_restarted": False, "neural_forward_performed": False}
        analyze.update(state, auxiliary=True)
        return
    stage("checks", "C0", state)
    if len(state["checks"]) != 30 or not all(v["passed"] for v in state["checks"].values()):
        raise RuntimeError("All 30 numerical-equivalence checks are required")
    analyze.update(state, ["C0"])
    if checks_only:
        log("checks_only_complete")
        return
    training_preserved = state.get("scheduling", {}).get("preexisting_artifacts_unchanged_after_training", False)
    if training_preserved and all(finished("train", method, job, state) for method in METHODS for job in jobs()):
        # Frozen inference legitimately appends to .pt files; never compare those
        # augmented files to the pre-inference whole-file hashes or retrain them.
        log("formal_training_already_complete_verified", tasks=180, retrained=0)
    else:
        unified_stage("train", METHODS, state, maximum, adopted)
    unified_stage("evaluate", tuple(r.config()["methods"]), state, maximum)
    state["phase"] = "mechanism_analysis"
    r.atomic_json(state)
    for method in ("C2", "C3", "C4", "C5"):
        state.setdefault("mechanisms", {})[method] = analyze.mechanism(method, state)
        r.atomic_json(state)
        log("mechanism_complete", method=method)
    analyze.final_checks(state)
    for name, value in state["audit"]["manifest"].items():
        if r.sha_file(r.ROOT / name) != value["sha256"]:
            raise RuntimeError(f"Read-only input modified during experiment: {name}")
    state["final_readonly_verification"] = {"passed": True, "files": len(state["audit"]["manifest"]), "time_utc": r.utc()}
    # Identical private initialization is checked after all formal runs, not inferred from parameter counts.
    for identity, fingerprint in state["groups"]["C1"]["initial_pcci_fingerprints"].items():
        if any(state["groups"][method]["initial_pcci_fingerprints"][identity] != fingerprint for method in ("C2", "C3", "C4", "C5", "C6")):
            raise RuntimeError("New-module private initialization was not identical across groups")
    totals = {str(gpu): sum(group["physical_gpu_fits"][str(gpu)] for method, group in state["groups"].items() if method != "C0") for gpu in (0, 1)}
    if sum(totals.values()) != 180:
        raise RuntimeError(f"Formal training assignment does not cover 180 fits: {totals}")
    state["physical_gpu_training_totals"] = totals
    state["phase"], state["completed_utc"] = "complete", r.utc()
    state["scheduling"]["completed_utc"] = state["completed_utc"]
    state["scheduling"]["elapsed_seconds"] = (datetime.fromisoformat(state["completed_utc"]) - datetime.fromisoformat(state["scheduling"]["started_utc"])).total_seconds()
    state["scheduling"]["experiment_elapsed_seconds"] = (datetime.fromisoformat(state["completed_utc"]) - datetime.fromisoformat(state["created_utc"])).total_seconds()
    analyze.update(state, auxiliary=True)
    log("experiment_complete", training_fits=180, frozen_checkpoints=210, device_fits=totals)
    for answer in analyze.research_answers(state):
        print(answer, flush=True)


def drain_workers(adopted):
    selector = selectors.DefaultSelector()
    for process in CHILDREN:
        if process.stdout is not None and not process.stdout.closed:
            selector.register(process.stdout.fileno(), selectors.EVENT_READ)
    for worker in adopted:
        try:
            os.fstat(worker["fd"])
        except OSError:
            continue
        if worker["fd"] not in selector.get_map():
            selector.register(worker["fd"], selectors.EVENT_READ)
    with (r.BASE / "run.log").open("ab", buffering=0) as stream:
        while selector.get_map():
            for entry, _ in selector.select(timeout=10):
                data = os.read(entry.fd, 65536)
                if data:
                    stream.write(data)
                else:
                    selector.unregister(entry.fd)
    selector.close()
    for process in CHILDREN:
        process.wait()
    CHILDREN.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=6)
    parser.add_argument("--adopt-launcher", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers_per_gpu <= 6:
        parser.error("--workers-per-gpu must be between 1 and 6")
    adopted = take_over(args.adopt_launcher) if args.adopt_launcher else []
    with Path(__file__).open("r") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args.checks_only, args.workers_per_gpu, adopted)
        except BaseException as exc:
            log("launcher_interrupted", error=str(exc), traceback=traceback.format_exc())
            log("healthy_workers_draining_without_termination", owned=len(CHILDREN))
            drain_workers(adopted)
            raise


if __name__ == "__main__":
    main()
