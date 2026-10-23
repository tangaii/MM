"""Resource-aware, resumable launcher for the registered forty PPU tasks."""
from __future__ import annotations
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

from run import BASE, CONFIG, log
import run as IA_RUN


def attach_historical_run():
    module_path=Path("/data/adodas/MM-GTUNets/experiments/PCDPAX_20261021/run.py")
    spec=importlib.util.spec_from_file_location("iafilm_historical_pcdpax_run",module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    IA_RUN.HIST_RUN=module


def host_available_mib():
    try:
        mem=Path("/proc/meminfo").read_text().splitlines()
        return float(next(x for x in mem if x.startswith("MemAvailable:")).split()[1])/1024
    except Exception:return float("inf")


def main():
    if sys.argv[1:]!=["run"]:raise SystemExit("Usage: python -B launch.py run")
    attach_historical_run()
    audit=json.loads((BASE/"audit.json").read_text())
    from run import guard,valid_result,result_path
    audit=guard(True)
    if not audit.get("formal_authorized"):raise AssertionError("Formal run not authorized")
    if torch.cuda.device_count()<2:raise RuntimeError("Both PPUs must be visible")
    for gpu in (0,1):
        if "PPU" not in torch.cuda.get_device_name(gpu).upper():raise RuntimeError(f"GPU{gpu} is not a PPU")
    queue=[{"group":g,"fold":f,"gpu":f%2,"attempt":0} for g in CONFIG["groups"] for f in range(10)]
    pending=[]
    for task in queue:
        if valid_result(task["group"],task["fold"]):
            log("launcher_resume_skip",group=task["group"],fold=task["fold"])
        else:pending.append(task)
    active={};failures=[];retries=[];worker_seconds=0.;start=time.perf_counter();start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    status={"experiment":25,"state":"running","started_utc":start_utc,"completed":40-len(pending),
            "registered_tasks":40,"gpu_plan":{"0":20,"1":20},"retries":retries,"failures":failures}
    atomic_path=BASE/"queue.json"
    def save_state(state):
        temp=atomic_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state,indent=2));os.replace(temp,atomic_path)
    log("launcher_started",pending=len(pending),workers_per_gpu_cap=4)
    max_attempts=2
    while pending or active:
        for proc,info in list(active.items()):
            code=proc.poll()
            if code is None:continue
            info["stream"].close();active.pop(proc)
            elapsed=time.perf_counter()-info["started"];worker_seconds+=elapsed
            task=info["task"]
            if code==0:
                try:valid_result(task["group"],task["fold"],required=True)
                except Exception as exc:code=86;reason=f"artifact_validation:{exc!r}"
                else:
                    status["completed"]+=1
                    log("launcher_task_validated",group=task["group"],fold=task["fold"],gpu=task["gpu"],attempt=task["attempt"])
                    continue
            else:reason=f"worker_exit_{code}"
            failure={**task,"exit_code":code,"reason":reason,"log":str(info["log_path"])}
            if task["attempt"]+1<max_attempts:
                retry={**task,"attempt":task["attempt"]+1};pending.insert(0,retry);retries.append(failure)
                log("launcher_retry",group=task["group"],fold=task["fold"],gpu=task["gpu"],reason=reason,attempt=retry["attempt"])
            else:failures.append(failure);log("launcher_failure",group=task["group"],fold=task["fold"],gpu=task["gpu"],reason=reason)
        for gpu in (0,1):
            active_gpu=sum(info["task"]["gpu"]==gpu for info in active.values())
            if active_gpu>=4:continue
            try:free,total=torch.cuda.mem_get_info(gpu);free_mib=free/1024**2
            except Exception as exc:
                log("launcher_resource_query_error",gpu=gpu,error=repr(exc));continue
            host_mib=host_available_mib()
            gpu_slots=max(0,int((free_mib-8192)//14336))
            host_slots=max(0,int((host_mib-8192)//3072))
            allowed=min(4,gpu_slots,host_slots)
            if active_gpu>=allowed:continue
            index=next((i for i,t in enumerate(pending) if t["gpu"]==gpu),None)
            if index is None:continue
            task=pending.pop(index)
            env=os.environ.copy();env.update({"CUDA_VISIBLE_DEVICES":str(gpu),"OMP_NUM_THREADS":"1",
                "MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONUNBUFFERED":"1",
                "PYTHONDONTWRITEBYTECODE":"1","TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD":"1"})
            log_path=result_path(task["group"],task["fold"]).with_suffix(".launcher.log")
            log_path.parent.mkdir(parents=True,exist_ok=True);stream=log_path.open("ab",buffering=0)
            cmd=[sys.executable,"-B",str(BASE/"run.py"),"train","--group",task["group"],"--fold",str(task["fold"])]
            proc=subprocess.Popen(cmd,cwd=BASE,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[proc]={"task":task,"started":time.perf_counter(),"stream":stream,"log_path":log_path}
            log("launcher_spawn",group=task["group"],fold=task["fold"],gpu=gpu,pid=proc.pid,
                attempt=task["attempt"],allowed_workers=allowed,free_gpu_mib=free_mib,host_available_mib=host_mib)
        status.update({"pending":pending,"active":[{**v["task"],"pid":p.pid} for p,v in active.items()],
                       "retries":retries,"failures":failures,"wall_seconds":time.perf_counter()-start,
                       "worker_elapsed_seconds":worker_seconds})
        save_state(status)
        if active or pending:time.sleep(1)
    status.update({"state":"complete" if not failures else "partial","completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                   "completed":40-len(failures),"wall_seconds":time.perf_counter()-start,"worker_elapsed_seconds":worker_seconds,
                   "retries":retries,"failures":failures})
    save_state(status);log("launcher_finished",state=status["state"],completed=status["completed"],
                          retries=len(retries),failures=len(failures),wall_seconds=status["wall_seconds"],
                          worker_elapsed_seconds=worker_seconds)
    return 0 if not failures else 1


if __name__=="__main__":raise SystemExit(main())
