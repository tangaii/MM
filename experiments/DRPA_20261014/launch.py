"""Sequential E0 anchors and balanced, memory-guarded multi-process PPU folds."""

import argparse
from collections import Counter, deque
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(BASE))
from run import (atomic_json, completed, config, file_lock, fold_dir,
                 sha_file, source_guard, training_code_hash, utc)

GPUS = (0, 1)
WORKER_MEMORY_MIB = 14 * 1024
MEMORY_RESERVE_MIB = 8 * 1024


def environment(gpu):
    value = os.environ.copy()
    value.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                  "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                  "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(config()["host_threads"]),
                  "MKL_NUM_THREADS": str(config()["host_threads"]),
                  "OPENBLAS_NUM_THREADS": str(config()["host_threads"])})
    return value


def note(message):
    line = f"{utc()} {message}"
    print(line, flush=True)
    with (BASE / "run.log").open("a") as stream:
        stream.write(line + "\n")


def command(arguments, gpu=0, log_path=None):
    path = log_path or BASE / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
                                env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {path}")


def start_task(arguments, gpu, logfile, identity):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    stream = logfile.open("a")
    process = subprocess.Popen([sys.executable, *map(str, arguments)], cwd=ROOT,
                               env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
    note(f"START gpu={gpu} {identity} pid={process.pid}")
    return {"process": process, "stream": stream, "gpu": gpu, "identity": identity, "log": str(logfile)}


def stop_owned(active):
    for task in active.values():
        if task["process"].poll() is None:
            task["process"].terminate()
    for task in active.values():
        try:
            task["process"].wait(timeout=15)
        except subprocess.TimeoutExpired:
            task["process"].kill(); task["process"].wait()
        task["stream"].close()


def reviewed_history():
    path = BASE / "artifacts/history_acceptance.json"
    if not path.exists():
        raise RuntimeError("A documented historical diagnosis is required; no automatic tolerance waiver")
    review = json.loads(path.read_text())
    audit = source_guard()
    expected = {"training_code_sha256": training_code_hash(), "data_digest": audit["data_digest"],
                "history_check_sha256": sha_file(BASE / "artifacts/history_check.json"),
                "diagnostic_verification_sha256": sha_file(BASE / "artifacts/baseline_diagnostics/verification.json")}
    if not review.get("accepted_for_paired_research", False):
        raise RuntimeError("Historical review does not authorize continuation")
    if not review.get("mean_tolerance_passed", False) or any(
            abs(review[key]) > config()["history_mean_tolerance"]
            for key in ("mean_delta_ACC", "mean_delta_AUC")):
        raise RuntimeError("Historical mean tolerance still exceeds the registered threshold")
    for key, value in expected.items():
        if review[key] != value:
            raise RuntimeError(f"Historical diagnosis is stale: {key}")
    for fold, digest in review["seed911_result_sha256"].items():
        if sha_file(fold_dir("E0", 911, int(fold)) / "metrics.json") != digest:
            raise RuntimeError("Reviewed baseline results changed")
    note("Non-exact historical replay reviewed; strict check and all differences remain preserved. "
         "Variants still require all 30 disabled-loss equivalence checks.")
    return True


def baseline_phase(allow_history_review=False):
    cfg = config()
    pending = [seed for seed in cfg["seeds"] if not all(completed("E0", seed, f) for f in range(10))]
    active = {}
    checked = (BASE / "artifacts/history_check.json").exists()
    if checked:
        checked = json.loads((BASE / "artifacts/history_check.json").read_text()).get("passed", False)
    if not checked and allow_history_review:
        checked = reviewed_history()
    try:
        while pending or active:
            for gpu in (0, 1):
                if gpu not in active and pending:
                    seed = pending.pop(0)
                    task = start_task([BASE / "run.py", "anchors", "--seed", seed], gpu,
                        BASE / "outputs/E0" / f"seed_{seed}" / "train.log", f"E0 seed={seed} sequential folds")
                    task["seed"] = seed; active[gpu] = task
            for gpu, task in list(active.items()):
                code = task["process"].poll()
                if code is None:
                    continue
                task["stream"].close(); del active[gpu]
                if code != 0:
                    raise RuntimeError(f"Sequential baseline failed: {task['identity']}; {task['log']}")
                if not all(completed("E0", task["seed"], f) for f in range(10)):
                    raise RuntimeError("Baseline process exited without verified ten-fold completion")
                note(f"COMPLETE {task['identity']}")
            if not checked and all(completed("E0", 911, f, verify=False) for f in range(10)):
                command([BASE / "run.py", "history-check"])
                checked = True
                note("Historical seed911 baseline accepted; detailed precision comparison saved")
            snapshot = {"phase": "E0", "time_utc": utc(), "pending_seeds": pending,
                        "active": [{"gpu": gpu, "seed": t["seed"], "pid": t["process"].pid} for gpu, t in active.items()]}
            atomic_json(snapshot, BASE / "artifacts/queue.json")
            if pending or active:
                time.sleep(5)
        if not checked:
            command([BASE / "run.py", "history-check"])
        command([BASE / "analyze.py", "--group", "E0"])
        note("E0 30/30 complete; handoff updated")
    except BaseException:
        stop_owned(active)
        raise


def summarize_ready(done):
    for method in list(config()["methods"])[1:]:
        if method in done:
            continue
        if all(completed(method, seed, fold, verify=False) for seed in config()["seeds"] for fold in range(10)):
            command([BASE / "analyze.py", "--group", method])
            done.add(method)
            note(f"{method} 30/30 complete; handoff updated")


def balanced_assignments(methods, seeds, finished):
    anchors = [(seed, fold) for seed in seeds for fold in range(10)]
    target = len(anchors) // len(GPUS)
    if target * len(GPUS) != len(anchors):
        raise ValueError("Variant anchor count cannot be divided equally between two cards")
    fixed = {}
    for (method, seed, fold), gpu in finished.items():
        if method not in methods or (seed, fold) not in anchors or gpu not in GPUS:
            raise ValueError("Completed task has an invalid method, anchor, or physical PPU")
        if (seed, fold) in fixed and fixed[seed, fold] != gpu:
            raise ValueError("Completed paired variants already ran on different cards; preserve and review them")
        fixed[seed, fold] = gpu
    counts = Counter(fixed.values())
    if any(counts[gpu] > target for gpu in GPUS):
        raise ValueError("Preserving completed tasks prevents equal per-method card allocation")
    for anchor in anchors:
        if anchor not in fixed:
            gpu = min(GPUS, key=lambda value: counts[value])
            fixed[anchor] = gpu
            counts[gpu] += 1
    return {(method, seed, fold): fixed[seed, fold]
            for seed, fold in anchors for method in methods}


def validate_assignments(assignments, methods, seeds, finished):
    expected = {(method, seed, fold) for method in methods for seed in seeds for fold in range(10)}
    if set(assignments) != expected or any(gpu not in GPUS for gpu in assignments.values()):
        raise ValueError("Saved variant plan has invalid or missing jobs")
    target = len(seeds) * 10 // len(GPUS)
    for method in methods:
        if any(sum(job[0] == method and card == gpu for job, card in assignments.items()) != target
               for gpu in GPUS):
            raise ValueError("Saved variant plan is not balanced within each method")
    for seed in seeds:
        for fold in range(10):
            if len({assignments[method, seed, fold] for method in methods}) != 1:
                raise ValueError("Paired variants are not assigned to the same physical card")
    if any(assignments[job] != gpu for job, gpu in finished.items()):
        raise ValueError("Saved card assignment disagrees with an already completed task")


def round_robin_jobs(assignments, finished, methods, gpu, offset=0):
    groups = {method: deque(sorted(job for job, card in assignments.items()
                                   if card == gpu and job[0] == method and job not in finished))
              for method in methods}
    order = methods[offset % len(methods):] + methods[:offset % len(methods)]
    pending = deque()
    while any(groups.values()):
        for method in order:
            if groups[method]:
                pending.append(groups[method].popleft())
    return pending


def variant_plan(workers_per_gpu):
    cfg = config()
    methods, seeds = list(cfg["methods"])[1:], cfg["seeds"]
    finished = {}
    for method in methods:
        for seed in seeds:
            for fold in range(10):
                if completed(method, seed, fold):
                    result = json.loads((fold_dir(method, seed, fold) / "metrics.json").read_text())
                    finished[method, seed, fold] = int(result["physical_gpu"])
    path = BASE / "artifacts/variant_plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["training_code_sha256"] != training_code_hash():
            raise ValueError("Saved scheduling plan belongs to different training code")
        rows = plan["assignments"]
        assignments = {(r["method"], r["seed"], r["fold"]): r["gpu"] for r in rows}
        if len(assignments) != len(rows):
            raise ValueError("Duplicate task in saved scheduling plan")
        validate_assignments(assignments, methods, seeds, finished)
    else:
        assignments = balanced_assignments(methods, seeds, finished)
        validate_assignments(assignments, methods, seeds, finished)
        plan = {"time_utc": utc(), "training_code_sha256": training_code_hash(),
                "policy": "equal per-method counts; paired variants on the same physical PPU; completed jobs pinned",
                "tasks": len(assignments), "per_gpu_total": {str(g): sum(v == g for v in assignments.values()) for g in GPUS},
                "assignments": [{"method": m, "seed": s, "fold": f, "gpu": g}
                                for (m, s, f), g in assignments.items()]}
        atomic_json(plan, path)
    queues = {gpu: round_robin_jobs(assignments, finished, methods, gpu, gpu * workers_per_gpu)
              for gpu in GPUS}
    policy = {"time_utc": utc(), "workers_per_gpu": workers_per_gpu,
              "worker_memory_budget_MiB": WORKER_MEMORY_MIB, "reserve_MiB_per_gpu": MEMORY_RESERVE_MIB,
              "variant_plan_sha256": sha_file(path), "per_gpu_total": plan["per_gpu_total"],
              "verified_complete": len(finished), "pending_per_gpu": {str(g): len(queues[g]) for g in GPUS},
              "algorithm_and_training_code_unchanged": True, "training_code_sha256": training_code_hash()}
    atomic_json(policy, BASE / "artifacts/scheduling_policy.json")
    return plan, finished, queues


def parse_memory_snapshot(gpu_csv, process_csv):
    cards = {}
    for row in csv.reader(io.StringIO(gpu_csv)):
        if not row:
            continue
        gpu, total, free = map(int, row)
        if gpu in GPUS:
            if not 0 <= free <= total:
                raise ValueError("Invalid accelerator memory measurement")
            cards[gpu] = {"total_MiB": total, "free_MiB": free}
    if set(cards) != set(GPUS):
        raise RuntimeError("Both physical PPUs must be visible to the scheduler")
    processes = {}
    for row in csv.reader(io.StringIO(process_csv)):
        if row:
            pid, used = map(int, row)
            if used < 0:
                raise ValueError("Invalid accelerator process memory measurement")
            processes[pid] = processes.get(pid, 0) + used
    return cards, processes


def memory_snapshot():
    def query(fields, kind):
        return subprocess.run(["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                              check=True, text=True, capture_output=True, timeout=15).stdout
    return parse_memory_snapshot(query("index,memory.total,memory.free", "gpu"),
                                 query("pid,used_gpu_memory", "compute-apps"))


def checked_memory_snapshot():
    try:
        cards, processes = memory_snapshot()
        return cards, processes, None
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
        return None, {}, {"error_type": type(exc).__name__, "error": str(exc)}


def launch_budget(gpu, cards, processes, active):
    # Account for workers whose PPU allocation has not reached its training peak yet.
    growth = sum(max(0, WORKER_MEMORY_MIB - processes.get(task["process"].pid, 0))
                 for task in active.values() if task["gpu"] == gpu)
    return cards[gpu]["free_MiB"] - growth - MEMORY_RESERVE_MIB


def variant_phase(workers_per_gpu=4):
    plan, finished, queues = variant_plan(workers_per_gpu)
    from analyze import update_handoff
    update_handoff()
    active, attempts, done = {}, {}, set()
    cards, memory_time, telemetry_error, telemetry_failures = {}, None, None, 0
    note(f"BALANCED VARIANTS tasks={plan['tasks']} per_gpu={plan['per_gpu_total']} "
         f"already_complete={len(finished)} pending={[len(queues[g]) for g in GPUS]} "
         f"workers_per_gpu={workers_per_gpu}; training code and anchors unchanged")
    summarize_ready(done)
    try:
        while any(queues.values()) or active:
            for key, task in list(active.items()):
                code = task["process"].poll()
                if code is None:
                    continue
                identity = task["job"]
                task["stream"].close(); del active[key]
                if code == 0 and completed(*identity):
                    finished[identity] = task["gpu"]
                    note(f"COMPLETE {task['identity']}")
                else:
                    status_path = fold_dir(*identity) / "status.json"
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                    fatal = status.get("error_type") in ("FloatingPointError", "ValueError", "AssertionError")
                    fatal |= any(word in status.get("error", "").lower() for word in
                                 ("gradient", "mismatch", "illegal", "numerical", "protected", "changed", "pair"))
                    if fatal or attempts[identity] >= 2:
                        raise RuntimeError(f"Failed/fatal task {identity}; inspect {task['log']} (no invalid continuation)")
                    queues[task["gpu"]].appendleft(identity)
                    note(f"RETRY identical anchor {identity}; previous status={status.get('error')}")
            capacity = any(queues[g] and sum(t["gpu"] == g for t in active.values()) < workers_per_gpu
                           for g in GPUS)
            if capacity:
                fresh_cards, processes, error = checked_memory_snapshot()
                if error is not None:
                    telemetry_error = error
                    telemetry_failures += 1
                    note(f"TELEMETRY PAUSE DISPATCH: {error['error_type']}: {error['error']}; "
                         "existing workers continue; no stale-memory launch and no CPU fallback")
                    atomic_json({"time_utc": utc(), "failures": telemetry_failures, **error},
                                BASE / "artifacts/telemetry_last_error.json")
                else:
                    if telemetry_error is not None:
                        note("TELEMETRY RECOVERED; dispatch resumes with a fresh memory measurement")
                    telemetry_error = None
                    cards, memory_time = fresh_cards, utc()
                    for gpu in GPUS:
                        budget = launch_budget(gpu, cards, processes, active)
                        for slot in range(workers_per_gpu):
                            if (gpu, slot) in active or not queues[gpu] or budget < WORKER_MEMORY_MIB:
                                continue
                            method, seed, fold = queues[gpu].popleft()
                            identity = (method, seed, fold)
                            task = start_task([BASE / "run.py", "variant", "--method", method,
                                               "--seed", seed, "--fold", fold], gpu,
                                              fold_dir(method, seed, fold) / "train.log", f"{method} seed={seed} fold={fold}")
                            task["job"], task["slot"] = identity, slot
                            active[gpu, slot] = task
                            attempts[identity] = attempts.get(identity, 0) + 1
                            budget -= WORKER_MEMORY_MIB
            summarize_ready(done)
            atomic_json({"phase": "variants", "time_utc": utc(), "pending": sum(map(len, queues.values())),
                         "workers_per_gpu": workers_per_gpu, "per_gpu_total": plan["per_gpu_total"],
                         "complete_per_gpu": {str(g): sum(v == g for v in finished.values()) for g in GPUS},
                         "pending_per_gpu": {str(g): len(queues[g]) for g in GPUS},
                         "memory": cards, "memory_measured_utc": memory_time,
                         "telemetry_error": telemetry_error, "telemetry_failure_count": telemetry_failures,
                         "launcher_pid": os.getpid(),
                         "training_code_sha256": training_code_hash(),
                         "groups_summarized": sorted(done),
                         "active": [{"gpu": t["gpu"], "slot": t["slot"], "job": list(t["job"]),
                                     "pid": t["process"].pid} for t in active.values()]},
                        BASE / "artifacts/queue.json")
            if any(queues.values()) or active:
                time.sleep(5)
        summarize_ready(done)
        if len(done) != 7:
            raise RuntimeError("Some variant groups did not complete")
    except BaseException:
        stop_owned(active)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnosed-history-review", action="store_true",
                        help="Use an explicit immutable diagnostic review, never silently relax tolerances")
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 5), default=4,
                        help="Concurrent independent variant folds per PPU (default: 4), with a memory guard")
    args = parser.parse_args()
    with file_lock(BASE / "artifacts/launcher.lock"):
        try:
            note("DRPA requested workflow started; protected source and historical data remain read-only")
            audit_path = BASE / "artifacts/audit.json"
            if not audit_path.exists():
                command([BASE / "run.py", "audit"])
            else:
                source_guard()
            test_path = BASE / "artifacts/correctness.json"
            if not test_path.exists() or not json.loads(test_path.read_text()).get("passed", False):
                command(["-m", "unittest", "discover", "-s", "tests", "-p", "test_ppu_custom_conv_equivalence.py"],
                        log_path=BASE / "artifacts/compatibility_test.log")
                command([BASE / "test_drpa.py"], log_path=BASE / "artifacts/correctness_test.log")
            note("Minimal correctness and PPU compatibility tests passed")
            if args.prepare_only:
                return
            baseline_phase(args.diagnosed_history_review)
            paired_path = BASE / "artifacts/paired_disabled_checks.json"
            if not paired_path.exists() or not json.loads(paired_path.read_text()).get("passed", False):
                command([BASE / "test_drpa.py", "--paired"], log_path=BASE / "artifacts/paired_disabled_test.log")
            note("All 30 actual anchor forward/gradient/Adam-step/RNG disabled-loss equivalence checks passed "
                 "under the documented test-only deterministic backend; formal training remains unchanged")
            command([BASE / "test_scheduling.py"], log_path=BASE / "artifacts/scheduling_test.log")
            command([BASE / "test_analysis.py"], log_path=BASE / "artifacts/analysis_test.log")
            atomic_json({"state": "running", "time_utc": utc(), "launcher_pid": os.getpid(),
                         "workers_per_gpu": args.workers_per_gpu, "training_code_sha256": training_code_hash()},
                        BASE / "artifacts/workflow.json")
            variant_phase(args.workers_per_gpu)
            command([BASE / "analyze.py", "--final"])
            atomic_json({"state": "complete", "time_utc": utc(), "tasks": 240, "oof_records": 20904,
                         "training_code_sha256": training_code_hash()}, BASE / "artifacts/workflow.json")
            note("All 240 tasks, statistics, and final handoff complete")
        except BaseException as exc:
            atomic_json({"state": "stopped_error", "time_utc": utc(), "error_type": type(exc).__name__,
                         "error": str(exc)}, BASE / "artifacts/workflow.json")
            note(f"WORKFLOW STOPPED: {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
