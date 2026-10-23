"""Finalize execution and handoff metadata without touching frozen training code."""
from __future__ import annotations

import json
from pathlib import Path

import run


BASE=Path(__file__).resolve().parent
audit=json.loads((BASE/"audit.json").read_text())
queue=json.loads((BASE/"queue.json").read_text())
attempt=json.loads((BASE/"handoff_attempt.json").read_text())
if queue.get("state")!="complete" or queue.get("completed")!=40 or queue.get("failures") or queue.get("retries"):
    raise AssertionError("Queue does not verify the registered 40-task completion")
audit["formal_tasks_started"]=40
audit["formal_tasks_completed"]=40
audit["execution"].update({"tasks":40,"completed":40,"retries":0,"failures":[],
    "queue_state":"complete","wall_seconds":queue["wall_seconds"],
    "worker_elapsed_seconds":queue["worker_elapsed_seconds"],
    "gpu_plan":{"0":20,"1":20}})
audit["handoff"]={"append_attempted":True,"updated":False,
    "error":attempt["tool_error"],"pending_body":"handoff_pending.md",
    "pending_body_sha256_recorded_in":"handoff_attempt.json",
    "original_prefix_bytes":attempt["pre_append_bytes"],
    "original_prefix_sha256":attempt["pre_append_sha256"],
    "prefix_preserved_after_failure":True}
run.atomic_json(audit,BASE/"audit.json")
summary=json.loads((BASE/"summary.json").read_text())
summary["handoff_append"]={"attempted":True,"updated":False,
    "error":attempt["tool_error"],"pending_body":"handoff_pending.md",
    "pending_body_sha256_recorded_in":"handoff_attempt.json",
    "original_prefix_bytes":attempt["pre_append_bytes"],
    "original_prefix_sha256":attempt["pre_append_sha256"],
    "prefix_preserved_after_failure":True}
run.atomic_json(summary,BASE/"summary.json")
print(json.dumps({"formal_tasks_started":audit["formal_tasks_started"],
                  "formal_tasks_completed":audit["formal_tasks_completed"],
                  "audit_sha256":run.sha_file(BASE/"audit.json"),
                  "summary_sha256":run.sha_file(BASE/"summary.json")},indent=2))
