"""CPU-only scheduler tests; no neural model is constructed or trained."""

from collections import Counter
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from launch import (GPUS, MEMORY_RESERVE_MIB, WORKER_MEMORY_MIB,
                    balanced_assignments, checked_memory_snapshot, environment, launch_budget,
                    parse_memory_snapshot, round_robin_jobs,
                    validate_assignments, variant_phase)


METHODS = [f"E{i}" for i in range(1, 8)]
SEEDS = [911, 912, 913]
ANCHORS = [(seed, fold) for seed in SEEDS for fold in range(10)]


class SchedulingTests(unittest.TestCase):
    def test_empty_plan_is_balanced_and_paired(self):
        plan = balanced_assignments(METHODS, SEEDS, {})
        validate_assignments(plan, METHODS, SEEDS, {})
        self.assertEqual(len(plan), 210)
        self.assertEqual(Counter(plan.values()), {0: 105, 1: 105})
        for method in METHODS:
            self.assertEqual(Counter(g for job, g in plan.items() if job[0] == method), {0: 15, 1: 15})
        for seed, fold in ANCHORS:
            self.assertEqual(len({plan[method, seed, fold] for method in METHODS}), 1)

    def test_completed_folds_are_pinned_and_not_queued(self):
        finished = {("E1", *anchor): int(i < 9) for i, anchor in enumerate(ANCHORS[:16])}
        plan = balanced_assignments(METHODS, SEEDS, finished)
        validate_assignments(plan, METHODS, SEEDS, finished)
        for job, gpu in finished.items():
            self.assertEqual(plan[job], gpu)
        queues = {g: round_robin_jobs(plan, finished, METHODS, g, g * 4) for g in GPUS}
        self.assertEqual([len(queues[g]) for g in GPUS], [98, 96])
        queued = [job for queue in queues.values() for job in queue]
        self.assertEqual(len(queued), len(set(queued)))
        self.assertEqual(set(queued), set(plan) - set(finished))
        first_methods = {job[0] for queue in queues.values() for job in list(queue)[:4]}
        self.assertEqual(first_methods, set(METHODS))

    def test_impossible_completed_allocations_are_rejected(self):
        conflicts = {("E1", 911, 0): 0, ("E2", 911, 0): 1}
        with self.assertRaisesRegex(ValueError, "different cards"):
            balanced_assignments(METHODS, SEEDS, conflicts)
        overcrowded = {("E1", *anchor): 0 for anchor in ANCHORS[:16]}
        with self.assertRaisesRegex(ValueError, "equal per-method"):
            balanced_assignments(METHODS, SEEDS, overcrowded)

    def test_stale_and_unbalanced_plans_are_rejected(self):
        plan = balanced_assignments(METHODS, SEEDS, {})
        with self.assertRaisesRegex(ValueError, "already completed"):
            validate_assignments(plan, METHODS, SEEDS, {("E1", 911, 0): 1})
        del plan["E7", 913, 9]
        with self.assertRaisesRegex(ValueError, "missing jobs"):
            validate_assignments(plan, METHODS, SEEDS, {})
        plan = balanced_assignments(METHODS, SEEDS, {})
        plan["E7", 911, 0] = 1
        with self.assertRaisesRegex(ValueError, "not balanced"):
            validate_assignments(plan, METHODS, SEEDS, {})

    def test_memory_guard_accounts_for_warming_workers(self):
        cards, processes = parse_memory_snapshot("0, 98304, 70000\n1, 98304, 80000\n",
                                                "123, 10000\n999, 20000\n")
        active = {(0, 0): {"gpu": 0, "process": SimpleNamespace(pid=123)},
                  (0, 1): {"gpu": 0, "process": SimpleNamespace(pid=124)},
                  (1, 0): {"gpu": 1, "process": SimpleNamespace(pid=125)}}
        expected = 70000 - MEMORY_RESERVE_MIB - (WORKER_MEMORY_MIB - 10000) - WORKER_MEMORY_MIB
        self.assertEqual(launch_budget(0, cards, processes, active), expected)
        self.assertEqual(launch_budget(1, cards, processes, active),
                         80000 - MEMORY_RESERVE_MIB - WORKER_MEMORY_MIB)
        with self.assertRaises(RuntimeError):
            parse_memory_snapshot("0, 98304, 70000\n", "")
        with self.assertRaises(ValueError):
            parse_memory_snapshot("0, 98304, 99000\n1, 98304, 80000\n", "")

    def test_workers_bind_cards_without_changing_thread_policy(self):
        for gpu in GPUS:
            env = environment(gpu)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], str(gpu))
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            self.assertEqual(env["MKL_NUM_THREADS"], "1")
            self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_telemetry_timeout_pauses_dispatch_without_raising(self):
        failure = subprocess.TimeoutExpired("nvidia-smi", 15)
        with patch("launch.memory_snapshot", side_effect=failure):
            cards, processes, error = checked_memory_snapshot()
        self.assertIsNone(cards)
        self.assertEqual(processes, {})
        self.assertEqual(error["error_type"], "TimeoutExpired")

    def test_timeout_with_an_active_worker_does_not_stop_training(self):
        from collections import deque
        jobs = {0: deque([("E1", 911, 0), ("E2", 911, 0)]),
                1: deque([("E3", 911, 1)])}
        a, b, c = Mock(pid=110), Mock(pid=210), Mock(pid=120)
        a.poll.side_effect, b.poll.side_effect, c.poll.side_effect = [0], [None, None, 0], [0]
        def task(process, gpu):
            return {"process": process, "gpu": gpu, "stream": Mock(), "identity": "test", "log": "test.log"}
        cards = {g: {"total_MiB": 98304, "free_MiB": 98302} for g in GPUS}
        snapshots = [(cards, {}), subprocess.TimeoutExpired("nvidia-smi", 15), (cards, {})]
        with patch("launch.variant_plan", return_value=({"tasks": 3, "per_gpu_total": {"0": 2, "1": 1}}, {}, jobs)), \
             patch("analyze.update_handoff"), patch("launch.note"), patch("launch.atomic_json") as write, \
             patch("launch.summarize_ready", side_effect=lambda done: done.update(METHODS)), \
             patch("launch.completed", return_value=True), patch("launch.time.sleep"), \
             patch("launch.memory_snapshot", side_effect=snapshots) as query, \
             patch("launch.start_task", side_effect=[task(a, 0), task(b, 1), task(c, 0)]) as start, \
             patch("launch.stop_owned") as stop:
            variant_phase(workers_per_gpu=1)
        self.assertEqual(query.call_count, 3)
        self.assertEqual(start.call_count, 3)
        stop.assert_not_called()
        for process in (a, b, c):
            process.terminate.assert_not_called()
        paused = [args[0][0] for args in write.call_args_list
                  if args[0][1].name == "queue.json" and args[0][0]["telemetry_error"] is not None]
        self.assertEqual(len(paused), 1)
        self.assertEqual(paused[0]["active"][0]["pid"], 210)
        self.assertEqual(paused[0]["pending"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
