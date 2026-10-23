"""Frozen IA-FiLM training, one-forward preflight, and artifact validation."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.dont_write_bytecode = True

BASE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/MM-GTUNets")
HISTORY = ROOT / "experiments/DRPA_20261014"
PCDCA = ROOT / "experiments/PCDCA_20261019"
M3DIR = ROOT / "experiments/PCDPAX_20261021"
CONFIG = json.loads((BASE / "config.json").read_text())
TRAIN_FILES = ("config.json", "film.py", "test_iafilm.py", "test_analysis.py", "run.py", "launch.py", "analyze.py")
RESULT_METRICS = ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(M3DIR))
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from opt import OptInit
from utils.tools import EarlyStopping
from film import (FilmHook, IdentityAnchoredFiLM, derange_within_splits,
                  fit_encode_conditions, private_seed)
from alignment import PCDPAXObjective, gradient_budget, prepare
import run as HIST_RUN


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(b: bytes):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def json_digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False).encode())


def array_hash(value):
    a = np.ascontiguousarray(value)
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def atomic_bytes(data: bytes, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_json(value, path):
    atomic_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()+b"\n", path)


def atomic_torch(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def log(event, **fields):
    line = json.dumps({"utc":utc(),"event":event,**fields},ensure_ascii=False,
                      allow_nan=False,separators=(",",":"))+"\n"
    with (BASE/"run.lock").open("a+") as lock, (BASE/"run.log").open("a",encoding="utf-8") as stream:
        fcntl.flock(lock,fcntl.LOCK_EX); stream.write(line); stream.flush(); os.fsync(stream.fileno())
        fcntl.flock(lock,fcntl.LOCK_UN)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def train_code_hashes():
    return {name:sha_file(BASE/name) for name in TRAIN_FILES}


def code_hash():
    return json_digest(train_code_hashes())


def result_path(group, fold):
    return BASE/"runs"/group/f"s911_f{fold}.pt"


def metric_values(y, p, pred):
    y=np.asarray(y,np.int64);p=np.asarray(p,np.float64);pred=np.asarray(pred,np.int64)
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)):
        raise AssertionError("Invalid test labels/probabilities")
    if set(np.unique(y)) != {0,1} or not np.isin(pred,[0,1]).all():
        raise AssertionError("AUC requires both labels; prediction encoding must be binary")
    return {"ACC":float(np.mean(pred==y)),
            "SEN-ASD":float(np.mean(pred[y==0]==0)),
            "SPE-TD":float(np.mean(pred[y==1]==1)),
            "probability_AUC":float(roc_auc_score(y==0,p))}


def source_files():
    return ("opt.py","train_mm_gtunets.py","model/mm_gtunets.py","model/gtunet.py",
            "model/rp_graph.py","utils/mydataloader.py","utils/tools.py","utils/metrics.py")


def audit_inputs():
    started=time.monotonic()
    hist_audit=HIST_RUN.guard(require_manifest=True)
    original={name:sha_file(ROOT/name) for name in source_files()}
    pcdca_audit=json.loads((PCDCA/"audit.json").read_text())
    for name,expected in pcdca_audit["source_sha256"].items():
        if original.get(name)!=expected: raise AssertionError(f"Protected original source changed: {name}")
    pcdax_manifest=json.loads((M3DIR/"manifest.json").read_text())
    pcdax_sources={name:sha_file(M3DIR/name) for name in ("config.json","alignment.py","run.py","launch.py")}
    if json_digest(pcdax_sources)!=pcdax_manifest["training_code_sha256"]:
        raise AssertionError("Historical M3 implementation is not the frozen accepted code")
    context_path=HISTORY/"artifacts/context.pt"
    ctx=load_torch(context_path)
    meta=HIST_RUN.read_metadata(ctx)
    if len(ctx["ids"])!=871 or not np.array_equal(ctx["labels"],np.asarray(ctx["labels"])):
        raise AssertionError("Unexpected historical context")
    if not np.array_equal(np.bincount(np.asarray(ctx["labels"],np.int64)),[403,468]):
        raise AssertionError("ASD=0 / TD=1 cohort encoding mismatch")
    input_files=[context_path, ROOT/"dataset/ABIDE_aal/Phenotypic_V1_0b_preprocessed1.csv",
                 ROOT/"dataset/ABIDE_aal/id.txt", ROOT/"dataset/ABIDE_aal/save_tensor/up_triangles.pt",
                 ROOT/"dataset/ABIDE_aal/save_tensor/labels.pt",
                 ROOT/"dataset/ABIDE_aal/save_tensor/ph_data_SITE_ID_SEX_AGE_AT_SCAN.pt",
                 ROOT/"dataset/ABIDE_aal/save_tensor/ph_dict_SITE_ID_SEX_AGE_AT_SCAN.pt"]
    inputs={str(p):sha_file(p) for p in input_files}
    anchors={};test_coverage=np.zeros(871,np.int64)
    for fold in range(10):
        directory=Path(HIST_RUN.anchor(911,fold))
        expected=HISTORY/"artifacts/seed_911"/f"fold_{fold}"
        if directory!=expected: raise AssertionError(f"Historical anchor() returned {directory}, expected {expected}")
        split=load_torch(directory/"splits.pt")
        start=load_torch(directory/"train_start.pt")
        fp=json.loads((directory/"fingerprints.json").read_text())
        joined=np.concatenate([split[k] for k in ("train","val","test")])
        if len(joined)!=871 or not np.array_equal(np.sort(joined),np.arange(871)):
            raise AssertionError(f"Invalid historical split fold {fold}")
        test_coverage[np.asarray(split["test"],np.int64)]+=1
        split_sha={k:array_hash(np.asarray(split[k])) for k in ("train","val","test")}
        if split_sha!=fp["split_fingerprints"]: raise AssertionError(f"Split fingerprint mismatch fold {fold}")
        if HIST_RUN.core_hash(start["core"])!=fp["train_start_core_fingerprint"]:
            raise AssertionError(f"train_start core mismatch fold {fold}")
        if HIST_RUN.rng_hash(start["rng"])!=fp["classifier_start_rng"]:
            raise AssertionError(f"train_start RNG mismatch fold {fold}")
        features=directory.parent/"selected_features.pt"
        support=directory.parent/"rfe_support.npy"
        vae=directory.parent/"vae"/f"fold{fold}_pretrain.pth"
        fea=load_torch(features); raw=load_torch(ROOT/"dataset/ABIDE_aal/save_tensor/up_triangles.pt")
        selected=torch.as_tensor(np.asarray(raw)[:,np.load(support)],dtype=fea.dtype)
        if fea.shape!=(871,500) or not torch.equal(selected,fea):
            raise AssertionError(f"Historical selected feature cache mismatch fold {fold}")
        prep=prepare(ctx["labels"],meta,split["train"],911,fold,HIST_RUN.CFG["prototype"])
        if json_digest(prep["info"])!=hist_audit["anchors"][str(fold)]["prototype_preparation_sha256"]:
            raise AssertionError(f"Historical M3/D4 prototype mapping differs fold {fold}")
        paths=[directory/"splits.pt",directory/"train_start.pt",directory/"fingerprints.json",
               features,support,vae]
        shas={str(p):sha_file(p) for p in paths}
        for path,digest in shas.items():
            if hist_audit["anchors"][str(fold)]["sha256"].get(path)!=digest:
                raise AssertionError(f"Historical input SHA differs: {path}")
        anchors[str(fold)]={"anchor":str(directory),"sha256":shas,
                            "split_sha256":split_sha,"train_start_core_sha256":HIST_RUN.core_hash(start["core"]),
                            "train_start_rng_sha256":HIST_RUN.rng_hash(start["rng"]),
                            "private_prototype_sha256":json_digest(prep["info"]),"gpu":fold%2}
    if not np.all(test_coverage==1): raise AssertionError("Historical ten-fold test coverage is not exact")
    historical={}
    controls=[("historical_D0",PCDCA,"D0"),("historical_D4",PCDCA,"D4"),
              ("historical_M1",M3DIR,"M1"),("historical_M3",M3DIR,"M3")]
    for key,directory,group in controls:
        rows=[]; paths=[]
        for fold in range(10):
            path=directory/"runs"/group/f"s911_f{fold}.pt"
            saved=load_torch(path)
            if "payload" in saved:
                if HIST_RUN.payload_hash(saved["payload"])!=saved["content_sha256"]:
                    raise AssertionError(f"Historical checkpoint content hash failed: {path}")
                saved=saved["payload"]
            if (saved.get("state"),saved.get("group"),saved.get("seed"),saved.get("fold"))!=("complete",group,911,fold):
                raise AssertionError(f"Unexpected historical control checkpoint: {path}")
            rows.append(saved["metrics"]);paths.append({"path":str(path),"sha256":sha_file(path)})
        historical[key]={"fold_metrics":rows,"checkpoint_sha256":paths}
    pcci_report=ROOT/"experiments/PCCI_20261015/report.md"
    pcci_config=ROOT/"experiments/PCCI_20261015/config.json"
    exp23=Path("/tmp/PCRPA_20261022")
    exp24=Path("/tmp/TPSSA_20261023")
    record={"experiment":25,"name":"IA-FiLM","created_utc":utc(),
            "requested_directory":str(ROOT/"experiments/IAFiLM_20261023"),
            "actual_directory":str(BASE),"requested_directory_creation_error":{"errno":5,"message":"Input/output error"},
            "pending_migration":True,"git_commit_at_audit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
            "historical_anchor_function_checked":"PCDPAX_20261021/run.py:anchor",
            "historical_pcdax_audit_sha256":sha_file(M3DIR/"audit.json"),
            "historical_pcdax_manifest_sha256":sha_file(M3DIR/"manifest.json"),
            "historical_pcdax_m3_files_sha256":pcdax_sources,
            "historical_pcdca_prototype_sha256":sha_file(PCDCA/"prototype.py"),
            "original_source_sha256":original,"inputs_sha256":inputs,"anchors":anchors,
            "historical_readonly_controls":historical,
            "historical_pcci":{"report_sha256":sha_file(pcci_report),"config_sha256":sha_file(pcci_config),
                               "three_seed_C0_C1_C3_C6_summary_from_user_review":True},
            "experiment23_review":{"directory":str(exp23),"residual_py_sha256":sha_file(exp23/"residual.py") if (exp23/"residual.py").is_file() else None,
                                   "residual_path_used_by_M3":False},
            "experiment24_review":{"directory":str(exp24),"report_sha256":sha_file(exp24/"report.md") if (exp24/"report.md").is_file() else None,
                                   "decision":"NO_GO_STAGE_A; not a training source"},
            "dataset":{"subjects":871,"ASD":403,"TD":468,"labels":{"ASD":0,"TD":1},
                       "sites":len(set(np.asarray(meta["site"]).astype(str)))},
            "forbidden_actions":{"historical_retrains":0,"rfe_refits":0,"vae_pretraining":0,
                                  "root_source_modifications":0,"historical_directory_modifications":0},
            "formal_tasks_started":0,"training_code_sha256":None,"manifest_sha256":None,
            "preflight":None,"execution":{"tasks":40,"completed":0,"failures":[],"retries":0}}
    atomic_json(record,BASE/"audit.json")
    log("read_only_audit_passed",historical_folds=10,controls=list(historical),seconds=time.monotonic()-started)
    return record


def guard(require_manifest=True):
    audit=json.loads((BASE/"audit.json").read_text())
    if audit.get("training_code_sha256")!=code_hash(): raise AssertionError("Frozen training source hash changed")
    for name,digest in audit["original_source_sha256"].items():
        if sha_file(ROOT/name)!=digest: raise AssertionError(f"Original model source changed: {name}")
    for fold,record in audit["anchors"].items():
        for path,digest in record["sha256"].items():
            if sha_file(path)!=digest: raise AssertionError(f"Historical input changed: {path}")
    if require_manifest:
        path=BASE/"manifest.json"
        if not path.exists() or sha_file(path)!=audit["manifest_sha256"]: raise AssertionError("Manifest missing or changed")
        manifest=json.loads(path.read_text())
        if manifest["training_code_sha256"]!=code_hash() or not manifest["preflight_passed"]:
            raise AssertionError("Frozen code/preflight mismatch")
    return audit


def check_math():
    from test_iafilm import run_tests as film_tests
    from test_analysis import run_tests as analysis_tests
    own=film_tests()
    patient=analysis_tests()
    # Exercise the original M3/D4/CE/KL and independent output-gradient budget test.
    historical=HIST_RUN.math_tests()
    if historical.get("passed") is not True: raise AssertionError("Accepted historical M3 math tests failed")
    tests={"passed":True,"IAFiLM":own,"patient_transition_direction":patient,
           "historical_M3_reused_math":historical,
           "real_labels":{"ASD":0,"TD":1},"asymmetric_label_permutation":True,
           "formal_metrics":0,"model_forwards":0}
    atomic_json(tests,BASE/"math_tests.json")
    log("math_tests_passed",film_checks=own["passed_count"],historical_checks=len(historical["checks"]))
    return tests


def make_film(fold,device):
    seed=private_seed(911,fold,"film-initialization")
    # Module initialization is isolated from the historical train-start RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module=IdentityAnchoredFiLM()
    module=module.to(device)
    return module,seed


class FoldData:
    def __init__(self,fold,audit):
        if not torch.cuda.is_available(): raise RuntimeError("PPU required; CPU fallback is forbidden")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (str(fold%2),None):
            raise AssertionError("Task assigned to incorrect physical GPU")
        torch.cuda.set_device(0);torch.set_num_threads(1)
        self.fold=fold;self.device=torch.device("cuda:0")
        self.anchor=Path(HIST_RUN.anchor(911,fold));self.anchor_info=audit["anchors"][str(fold)]
        for path,digest in self.anchor_info["sha256"].items():
            if sha_file(path)!=digest: raise AssertionError(f"Input SHA changed: {path}")
        self.ctx=load_torch(HISTORY/"artifacts/context.pt")
        self.meta=HIST_RUN.read_metadata(self.ctx)
        self.split=load_torch(self.anchor/"splits.pt")
        self.start=load_torch(self.anchor/"train_start.pt")
        saved=sys.argv
        try:
            sys.argv=["iafilm_options"]
            self.options=OptInit(model="MM_GTUNets",dataset="ABIDE",atlas="aal").args
        finally: sys.argv=saved
        self.options.seed=911;self.options.device=self.device
        self.options.data_folder=str(ROOT/"dataset/ABIDE_aal")
        self.options.ckpt_path=str(self.anchor.parent/"vae")
        self.options.lr=1e-4;self.options.wd=5e-4;self.options.epoch=500;self.options.early_stop=100
        self.options.log_save=False;self.options.model_save=False
        self.core=MM_GTUNets(self.options,fold).to(self.device)
        self.core.vae_save_path=str(self.anchor.parent/"vae"/f"fold{fold}_pretrain.pth")
        HIST_RUN.import_core(self.core,self.start["core"])
        if HIST_RUN.core_hash(HIST_RUN.export_core(self.core))!=self.anchor_info["train_start_core_sha256"]:
            raise AssertionError("Core is not the historical train_start state")
        self.x_cpu=load_torch(self.anchor.parent/"selected_features.pt").float()
        self.ph_cpu=torch.as_tensor(self.ctx["ph_data"],dtype=torch.float32)
        aff=create_reward_penalty_graph(self.ctx["ph_dict"],self.ctx["labels"],
                 self.split["train"],self.split["val"],self.split["test"],self.options)
        self.aff_cpu=torch.as_tensor(aff,dtype=torch.float32)
        self.x,self.ph,self.aff=[x.to(self.device) for x in (self.x_cpu,self.ph_cpu,self.aff_cpu)]
        self.y=torch.as_tensor(self.ctx["labels"],dtype=torch.long,device=self.device)
        self.indices={k:torch.as_tensor(self.split[k],dtype=torch.long,device=self.device) for k in ("train","val","test")}
        self.prepared=prepare(self.ctx["labels"],self.meta,self.split["train"],911,fold,HIST_RUN.CFG["prototype"])
        if json_digest(self.prepared["info"])!=self.anchor_info["private_prototype_sha256"]:
            raise AssertionError("Historical M3/D4 train-only prototype preparation changed")
        self.conditions,self.d_ref,self.condition_info=fit_encode_conditions(self.meta["age"],self.meta["sex"],self.split["train"])
        self.permutation,self.permutation_records=derange_within_splits(871,self.split,self.meta["site"],911,fold)
        self.conditions_true=torch.as_tensor(self.conditions,dtype=torch.float32,device=self.device)
        self.conditions_wrong=torch.as_tensor(self.conditions[self.permutation],dtype=torch.float32,device=self.device)
        self.d_ref=torch.as_tensor(self.d_ref,dtype=torch.float32,device=self.device)
        self.z_img=None;self.z_ph=None;self.forward_count=0
        self.img_handle=self.core.img_unet.register_forward_hook(self._capture_img)
        self.ph_handle=self.core.ph_unet.register_forward_hook(self._capture_ph)
        self.input_sha={name:digest for name,digest in self.anchor_info["sha256"].items()}
        self.input_digest=HIST_RUN.payload_hash({"x":self.x_cpu,"ph":self.ph_cpu,"affinity":self.aff_cpu,
                                                "labels":self.ctx["labels"],"ids":self.ctx["ids"]})

    def _capture_img(self,module,inputs,output):
        if tuple(output.shape)!=(871,16): raise AssertionError(f"z_img shape {tuple(output.shape)}")
        self.z_img=output

    def _capture_ph(self,module,inputs,output):
        if tuple(output.shape)!=(871,16): raise AssertionError(f"z_ph shape {tuple(output.shape)}")
        self.z_ph=output

    def forward(self):
        self.forward_count+=1
        return self.core(self.x,self.ph,self.aff)

    def reset(self):
        HIST_RUN.import_core(self.core,self.start["core"])
        for p in self.core.parameters():p.grad=None
        HIST_RUN.restore_rng(self.start["rng"])
        torch.use_deterministic_algorithms(False)

    def close(self):
        self.img_handle.remove();self.ph_handle.remove()
        if hasattr(self,"core"): del self.core
        torch.cuda.empty_cache()


def film_state_hash(module):
    return HIST_RUN.state_hash({k:v.detach().cpu() for k,v in module.state_dict().items()})


def finite(value,name):
    if not torch.isfinite(value).all(): raise FloatingPointError(f"NaN/Inf in {name}")


def film_diagnostics(stats, indices=None):
    if stats is None:return None
    g=stats["gamma"].detach();b=stats["beta"].detach()
    h=stats["raw"].detach();mod=stats["modulated"].detach()
    if indices is not None:
        idx=torch.as_tensor(np.asarray(indices),dtype=torch.long,device=g.device)
        g=g.index_select(0,idx);b=b.index_select(0,idx);h=h.index_select(0,idx);mod=mod.index_select(0,idx)
    return {"gamma_mean":float(g.mean()),"gamma_sd":float(g.std(unbiased=False)),
            "beta_mean":float(b.mean()),"beta_sd":float(b.std(unbiased=False)),
            "max_abs_gamma_minus_one":float((g-1).abs().max()),"max_abs_beta":float(b.abs().max()),
            "relative_feature_l2":float((mod-h).norm()/h.norm().clamp_min(1e-12))}


def film_subgroups(stats,data,indices):
    if stats is None:return {}
    indices=np.asarray(indices,dtype=np.int64);age=np.asarray(data.meta["age"],dtype=float);sex=np.asarray(data.meta["sex"],dtype=float)
    valid=np.isfinite(age)&(age>=0)&(age<=120);filled=np.where(valid,age,data.condition_info["age_median"])
    specs={"age_lt12":indices[valid[indices]&(filled[indices]<12)],
           "age_12_18":indices[valid[indices]&(filled[indices]>=12)&(filled[indices]<18)],
           "age_18_30":indices[valid[indices]&(filled[indices]>=18)&(filled[indices]<30)],
           "age_30_plus":indices[valid[indices]&(filled[indices]>=30)],
           "age_invalid_imputed":indices[~valid[indices]],
           "sex_male":indices[sex[indices]==1],"sex_female":indices[sex[indices]==2],
           "sex_unknown":indices[~np.isin(sex[indices],[1,2])]}
    return {name:{"n":int(len(ix)),"film":film_diagnostics(stats,ix) if len(ix) else None}
            for name,ix in specs.items()}


def preflight_one(fold=0):
    audit=json.loads((BASE/"audit.json").read_text())
    if not json.loads((BASE/"math_tests.json").read_text())["passed"]: raise AssertionError("Math tests required")
    if fold!=0:raise ValueError("Registered preflight is seed911/fold0 only")
    data=FoldData(fold,audit);device=data.device
    film1,seed=make_film(fold,device);film2,seed2=make_film(fold,device)
    if seed!=seed2 or film_state_hash(film1)!=film_state_hash(film2):raise AssertionError("F1/F2 film initialization mismatch")
    # Both placement hooks are identity at zero head; one core forward exercises both paths.
    h1=FilmHook(data.core,film1,"post_attention_joint_pre_classifier")
    h2=FilmHook(data.core,film2,"pre_attention_image_only")
    true=data.conditions_true;wrong=data.conditions_wrong;ref=data.d_ref
    h1.configure(True,9,true,ref);h2.configure(True,9,true,ref)
    captures={}
    extra=[]
    for name,module in (("img_lin",data.core.mm_attention.img_lin),
                        ("shared_lin",data.core.mm_attention.shared_lin),
                        ("ph_lin",data.core.mm_attention.ph_lin)):
        extra.append(module.register_forward_hook(lambda m,i,o,n=name:captures.__setitem__(n,o)))
    data.reset()
    data.core.train();film1.train();film2.train();torch.cuda.reset_peak_memory_stats()
    bn_before=int(data.core.clf[2].num_batches_tracked)
    with torch.set_grad_enabled(True): logits,graph=data.forward()
    if data.forward_count!=1 or h1.call_count!=1 or h2.call_count!=1:raise AssertionError("Preflight forward/hook count != 1")
    if logits.shape!=(871,2) or graph.ndim!=0:raise AssertionError("Unexpected native output shape")
    finite(logits,"preflight logits");finite(graph,"preflight graph")
    if int(data.core.clf[2].num_batches_tracked)!=bn_before+1:raise AssertionError("Expected one train BN update")
    if not torch.equal(h1.last["raw"],h1.last["modulated"]) or not torch.equal(h2.last["raw"],h2.last["modulated"]):
        raise AssertionError("Zero-initialized preflight was not strictly identity")
    if not torch.equal(h1.last["raw"],data.core.joint_embed):raise AssertionError("F1 did not receive the original joint representation")
    # Exact historical objectives: D4 image CE, real-condition phenotype CE and M3 KL.
    objective=PCDPAXObjective(data.prepared,device)
    m3=objective(data.z_img,data.z_ph,"M3")
    m1=objective(data.z_img,data.z_ph,"M1")
    ytr=data.y[data.indices["train"]]
    native=F.cross_entropy(logits[data.indices["train"]],ytr)+graph
    d4_loss=m3["image_loss"]
    ph_ce=m3["ph_ce"]
    if not torch.allclose(m3["ph_ce"],m1["ph_ce"],atol=0,rtol=0):raise AssertionError("F1/F2/F3/F4 ph CE mismatch")
    if m3["align_loss"] is None or m1["align_loss"] is not None:raise AssertionError("F1/F2 must have KL; F3/F4 must not compute it")
    if not torch.allclose(m3["image_loss"],m1["image_loss"],atol=0,rtol=0):raise AssertionError("D4 objective differs across groups")
    if not torch.equal(m3["image_logits"],m1["image_logits"]):raise AssertionError("D4 logits changed across groups")
    g_joint=torch.autograd.grad(native,h1.last["modulated"],retain_graph=True,allow_unused=True)[0]
    g_pre_attention=torch.autograd.grad(native,h2.last["modulated"],retain_graph=True,allow_unused=True)[0]
    g_img_lin=torch.autograd.grad(native,captures["img_lin"],retain_graph=True,allow_unused=True)[0]
    g_shared_lin=torch.autograd.grad(native,captures["shared_lin"],retain_graph=True,allow_unused=True)[0]
    g_ph_lin=torch.autograd.grad(native,captures["ph_lin"],retain_graph=True,allow_unused=True)[0]
    path_norms={"film_joint_to_classifier":0. if g_joint is None else float(g_joint.norm()),
                "film_image_to_attention_classifier":0. if g_pre_attention is None else float(g_pre_attention.norm()),
                "native_to_img_attention_linear":0. if g_img_lin is None else float(g_img_lin.norm()),
                "native_to_shared_attention_linear":0. if g_shared_lin is None else float(g_shared_lin.norm()),
                "native_to_ph_attention_linear":0. if g_ph_lin is None else float(g_ph_lin.norm())}
    if not all(path_norms[k]>0 for k in ("film_joint_to_classifier","film_image_to_attention_classifier",
                                        "native_to_img_attention_linear","native_to_shared_attention_linear")):
        raise AssertionError(f"FiLM classifier/attention path gradient missing: {path_norms}")
    gfilm1=torch.autograd.grad(native,tuple(film1.parameters()),retain_graph=True,allow_unused=True)
    gfilm2=torch.autograd.grad(native,tuple(film2.parameters()),retain_graph=True,allow_unused=True)
    gfilm1_norm=float(torch.stack([x.square().sum() for x in gfilm1 if x is not None]).sum().sqrt())
    gfilm2_norm=float(torch.stack([x.square().sum() for x in gfilm2 if x is not None]).sum().sqrt())
    if not gfilm1_norm>0 or not gfilm2_norm>0:raise AssertionError("FiLM native classifier path has no gradient at rho=0.1")
    for handle in extra:handle.remove()
    h1.remove();h2.remove()
    # Compute the real-versus-permuted condition Jacobian on one captured F2 input,
    # holding the native downstream cotangent fixed. This creates no core forward.
    with torch.enable_grad():
        h=data.z_img.detach()
        # Use actual F2-hook output cotangent from the single native graph.
        cot=torch.autograd.grad(native,h2.last["modulated"],retain_graph=True,allow_unused=True)[0]
        if cot is None or float(cot.norm())<=0:raise AssertionError("No gradient from native loss to original image embedding")
        def output_weight_gradient(conditions):
            model, _seed=make_film(fold,device)
            inp=conditions
            out,stats=model.modulate(h,inp,ref,9)
            scalar=(out*cot.detach()).sum()
            grad=torch.autograd.grad(scalar,model.net[-1].weight)[0]
            del model
            return grad
        grad_true=output_weight_gradient(true)
        grad_wrong=output_weight_gradient(wrong)
    difference=float((grad_true-grad_wrong).norm())
    if not difference>1e-8:raise AssertionError("F3/F4 initial zero-head condition Jacobians are indistinguishable")
    del logits,graph,native,m3,m1
    data.close();torch.cuda.empty_cache()
    return {"passed":True,"fold":0,"seed":911,"native_core_forwards":1,
            "hook_calls":{"F1":1,"F2":1},"logits_shape":[871,2],"z_img_shape":[871,16],"z_ph_shape":[871,16],
            "film_parameter_count":sum(p.numel() for p in film1.parameters()),
            "film_init_sha256":film_state_hash(film1),"film_rng_private_seed":seed,
            "max_abs_gamma_minus_one":0.0,"max_abs_beta":0.0,"film_relative_l2":0.0,
            "native_film_parameter_gradient_norm":{"F1_post_attention":gfilm1_norm,"F2_pre_attention":gfilm2_norm},
            "native_jacobian_path_norms":path_norms,
            "classifier_path_has_gradient":True,
            "attention_path":{"img_lin_output_captured":"img_lin" in captures,
                              "shared_lin_output_captured":"shared_lin" in captures,
                              "ph_lin_output_captured":"ph_lin" in captures,
                              "raw_img_raw_ph_preserved":True,
                              "graph_loss_actual_value_recorded":True,
                              "attention_weight_gradient_truncation_preserved":True},
            "D4_CE_equal_across_groups":True,"ph_CE_equal_across_groups":True,
            "M3_KL_computed_only_F1_F2":True,
            "zero_head_preflight_outputs_identity":True,
            "F3_F4_initial_image_logits_equal_by_identity":True,
            "F3_F4_output_layer_condition_gradient_l2_difference":difference,
            "extra_core_forwards":0,"formal_test_metrics_computed":0,
            "historical_RNG_restored_before_any_formal_training":True,
            "note":"Attention scalar weights are created through the historical detached torch.tensor path; the pre-modulation can affect forward weights/Graph value without a gradient path through those weights."}


def freeze_manifest(audit):
    if audit.get("mathematical_tests",{}).get("passed") is not True:raise AssertionError("Math tests must pass")
    if audit.get("preflight",{}).get("passed") is not True:raise AssertionError("Real fold0 preflight must pass")
    files=train_code_hashes()
    manifest={"experiment":25,"name":"IA-FiLM","frozen_utc":utc(),"training_code_sha256":json_digest(files),
              "training_files_sha256":files,"config_sha256":files["config.json"],
              "math_tests_sha256":sha_file(BASE/"math_tests.json"),"preflight_sha256":json_digest(audit["preflight"]),
              "seed":911,"folds":list(range(10)),"groups":list(CONFIG["groups"]),"tasks":40,
              "gpu0_folds":[0,2,4,6,8],"gpu1_folds":[1,3,5,7,9],
              "optimizer":{"name":"Adam","lr":1e-4,"weight_decay":5e-4},
              "historical_m3_alignment_sha256":audit["historical_pcdax_m3_files_sha256"]["alignment.py"],
              "historical_pcdca_prototype_sha256":audit["historical_pcdca_prototype_sha256"],
              "historical_inputs_sha256":{fold:row["sha256"] for fold,row in audit["anchors"].items()},
              "preflight_passed":True,"no_git":True}
    atomic_json(manifest,BASE/"manifest.json")
    audit["training_code_sha256"]=json_digest(files);audit["training_code_files_sha256"]=files
    audit["manifest_sha256"]=sha_file(BASE/"manifest.json");audit["formal_authorized"]=True
    atomic_json(audit,BASE/"audit.json")
    log("formal_code_frozen",manifest_sha256=audit["manifest_sha256"],training_code_sha256=audit["training_code_sha256"])
    return manifest


def valid_result(group,fold,required=False):
    path=result_path(group,fold)
    if not path.is_file():
        if required:raise FileNotFoundError(path)
        return None
    value=load_torch(path);audit=json.loads((BASE/"audit.json").read_text())
    if (value.get("state"),value.get("group"),value.get("seed"),value.get("fold"))!=("complete",group,911,fold):
        raise AssertionError(f"Invalid checkpoint identity: {path}")
    if value.get("manifest_sha256")!=audit["manifest_sha256"]:raise AssertionError("Checkpoint manifest mismatch")
    if value.get("input_sha256")!=audit["anchors"][str(fold)]["sha256"]:raise AssertionError("Checkpoint input mismatch")
    if value.get("best_core_sha256")!=HIST_RUN.core_hash(value["best_core"]):raise AssertionError("Core SHA mismatch")
    if value.get("best_film_sha256")!=film_state_hash_from_state(value["best_film"]):raise AssertionError("FiLM SHA mismatch")
    indices=np.asarray(value["oof"]["indices"],np.int64)
    split=load_torch(HIST_RUN.anchor(911,fold)/"splits.pt")
    if not np.array_equal(indices,np.asarray(split["test"],np.int64)):raise AssertionError("OOF index mismatch")
    y=np.asarray(value["oof"]["labels"],np.int64);p=np.asarray(value["oof"]["probability_asd"],np.float64)
    pred=np.asarray(value["oof"]["prediction"],np.int64);logits=torch.as_tensor(value["oof"]["logits"],dtype=torch.float32)
    if logits.shape!=(len(indices),2) or not np.isfinite(logits.numpy()).all():raise AssertionError("Invalid saved test logits")
    cpu=torch.softmax(logits,dim=1).numpy()[:,0]
    if not np.allclose(cpu,p,rtol=0,atol=2e-7):raise AssertionError("Stored PPU ASD probability differs from saved logits")
    metrics=metric_values(y,p,pred)
    if max(abs(metrics[k]-value["metrics"][k]) for k in RESULT_METRICS)>1e-12:raise AssertionError("Formal metrics do not recompute")
    if value["oof"]["subject_ids"]!=[load_torch(HISTORY/"artifacts/context.pt")["ids"][int(i)] for i in indices]:
        raise AssertionError("Test subject ordering mismatch")
    return value


def film_state_hash_from_state(state):
    return HIST_RUN.state_hash({k:v.detach().cpu() for k,v in state.items()})


def train_one(group,fold):
    audit=guard(True)
    existing=valid_result(group,fold,False)
    if existing is not None:
        log("resume_skip_valid",group=group,fold=fold,artifact_sha256=sha_file(result_path(group,fold)));return
    if not audit.get("formal_authorized"):raise AssertionError("Formal training not authorized")
    start_time=time.perf_counter();start_utc=utc();data=FoldData(fold,audit);device=data.device
    film,film_seed=make_film(fold,device)
    module_sha=film_state_hash(film)
    hook=FilmHook(data.core,film,CONFIG["groups"][group]["film_position"])
    true=data.conditions_true;wrong=data.conditions_wrong if group=="F4" else true
    optimizer=torch.optim.Adam(list(data.core.parameters())+list(film.parameters()),lr=1e-4,weight_decay=5e-4)
    if any(p is data.core.rp_attention.weights for p in optimizer.param_groups[0]["params"]):
        raise AssertionError("Historical unregistered RP weights unexpectedly entered Adam")
    objective=PCDPAXObjective(data.prepared,device)
    objective_group="M3" if CONFIG["groups"][group]["phenotype_kl"] else "M1"
    stopper=EarlyStopping(patience=100,save=False)
    data.reset()
    if HIST_RUN.rng_hash(HIST_RUN.capture_rng())!=audit["anchors"][str(fold)]["train_start_rng_sha256"]:
        raise AssertionError("Full initial RNG differs from historical train_start")
    initial_core_sha=HIST_RUN.core_hash(HIST_RUN.export_core(data.core))
    if initial_core_sha!=audit["anchors"][str(fold)]["train_start_core_sha256"]:raise AssertionError("Bad core init")
    initial_bn=int(data.core.clf[2].num_batches_tracked)
    status=result_path(group,fold).with_suffix(".status.json")
    atomic_json({"state":"running","group":group,"seed":911,"fold":fold,"started_utc":start_utc,"pid":os.getpid()},status)
    history=[];best=None;best_loss=float("inf")
    sample_epochs=set(range(3))|set(range(0,500,25))
    try:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();train_clock=time.perf_counter()
        for epoch in range(500):
            data.core.train();film.train();hook.configure(True,epoch,wrong,data.d_ref)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(True): logits,graph=data.forward()
            if hook.call_count!=1:raise AssertionError("FiLM hook did not fire exactly once in train forward")
            tr=data.indices["train"];native_ce=F.cross_entropy(logits[tr],data.y[tr]);native=native_ce+graph
            values=objective(data.z_img,data.z_ph,objective_group)
            img_scale,img_budget=gradient_budget(native,values["image_loss"],data.z_img,epoch,0.05)
            ph_scale,ph_budget=gradient_budget(native,values["ph_aux"],data.z_ph,epoch,0.05)
            total=native+img_scale*values["image_loss"]+ph_scale*values["ph_aux"]
            for name,value in (("logits",logits),("graph",graph),("native",native),("D4",values["image_loss"]),
                               ("phCE",values["ph_ce"]),("phaux",values["ph_aux"]),("total",total)):
                finite(value,f"{group}/fold{fold}/epoch{epoch}/{name}")
            if values["align_loss"] is not None:finite(values["align_loss"],"M3 KL")
            sample=epoch in sample_epochs
            record={"epoch":epoch,"native_ce":float(native_ce.detach()),"native_graph":float(graph.detach()),
                    "native_total":float(native.detach()),"D4_CE":float(values["image_loss"].detach()),
                    "ph_CE":float(values["ph_ce"].detach()),
                    "KL":None if values["align_loss"] is None else float(values["align_loss"].detach()),
                    "ph_aux":float(values["ph_aux"].detach()),"lambda_image":float(img_scale),"lambda_ph":float(ph_scale),
                    "image_budget":img_budget,"ph_budget":ph_budget,"train_total":float(total.detach()),
                    "train_img_weight":float(data.core.img_weight.detach()),
                    "train_ph_weight":float(data.core.ph_weight.detach()),
                    "film":film_diagnostics(hook.last,data.split["train"])}
            if sample:
                record["film_train_subgroups"]=film_subgroups(hook.last,data,data.split["train"])
                film_norm=torch.autograd.grad(native,tuple(film.parameters()),retain_graph=True,allow_unused=True)
                record["film_native_gradient_norm"]=float(torch.stack([g.square().sum() for g in film_norm if g is not None]).sum().sqrt())
                record["native_raw_embedding_gradient_norms"]={}
                for key,z in (("z_img",data.z_img),("z_ph",data.z_ph)):
                    grad=torch.autograd.grad(native,z,retain_graph=True,allow_unused=True)[0]
                    record["native_raw_embedding_gradient_norms"][key]=0. if grad is None else float(grad.norm())
                if values["align_loss"] is not None:
                    teacher_grad,student_grad=torch.autograd.grad(values["align_loss"],(data.z_img,data.z_ph),retain_graph=True,allow_unused=True)
                    record["KL_raw_gradient_norms"]={"image_teacher":0. if teacher_grad is None else float(teacher_grad.norm()),
                                                       "ph_student":0. if student_grad is None else float(student_grad.norm())}
            total.backward()
            for name,param in list(data.core.named_parameters())+[(f"film.{n}",p) for n,p in film.named_parameters()]:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Nonfinite gradient: {name}")
            if sample:
                record["film_gradient_norm"]=float(torch.stack([p.grad.detach().square().sum() for p in film.parameters() if p.grad is not None]).sum().sqrt())
            optimizer.step()
            if int(data.core.clf[2].num_batches_tracked)!=initial_bn+epoch+1:raise AssertionError("Unexpected train BN count")
            hook.configure(True,epoch,true if group!="F4" else wrong,data.d_ref)
            data.core.eval();film.eval()
            with torch.set_grad_enabled(False): val_logits,val_graph=data.forward()
            if hook.call_count!=1:raise AssertionError("FiLM hook did not fire once in validation")
            va=data.indices["val"];val_ce=F.cross_entropy(val_logits[va],data.y[va]);val_loss=val_ce+val_graph
            finite(val_loss,"native validation CE+Graph")
            if int(data.core.clf[2].num_batches_tracked)!=initial_bn+epoch+1:raise AssertionError("Eval updated BN")
            record.update({"val_ce":float(val_ce),"val_graph":float(val_graph),"val_loss":float(val_loss)})
            record["val_img_weight"]=float(data.core.img_weight.detach())
            record["val_ph_weight"]=float(data.core.ph_weight.detach())
            history.append(record)
            if float(val_loss)<best_loss:
                best_loss=float(val_loss)
                best={"core":HIST_RUN.export_core(data.core),
                      "film":{k:v.detach().cpu().clone() for k,v in film.state_dict().items()},
                      "epoch":epoch,"val_loss":best_loss,"record":record.copy()}
            stopper(val_loss,data.core)
            if epoch%50==0:log("task_progress",group=group,fold=fold,epoch=epoch,val_loss=float(val_loss),
                                lambda_image=float(img_scale),lambda_ph=float(ph_scale),film_grad=record.get("film_gradient_norm"))
            if stopper.early_stop:break
        torch.cuda.synchronize();training_seconds=time.perf_counter()-train_clock
        if best is None:raise AssertionError("No validation checkpoint selected")
        train_end_rng=HIST_RUN.capture_rng()
        HIST_RUN.import_core(data.core,best["core"]);film.load_state_dict(best["film"],strict=True)
        data.core.eval();film.eval();hook.configure(True,best["epoch"],wrong,data.d_ref)
        test_rng_before=HIST_RUN.capture_rng()
        with torch.set_grad_enabled(True):test_logits,test_graph=data.forward()
        test_rng_after=HIST_RUN.capture_rng()
        if data.forward_count!=2*len(history)+1:raise AssertionError("Unexpected train/val/test native forward count")
        if hook.call_count!=1:raise AssertionError("FiLM hook missing from formal test forward")
        test_indices=np.asarray(data.split["test"],np.int64)
        raw_logits=test_logits.detach().cpu().numpy()[test_indices].astype(np.float32)
        p_asd=torch.softmax(test_logits.detach(),dim=1).cpu().numpy()[test_indices,0].astype(np.float64)
        pred=raw_logits.argmax(1).astype(np.int64);labels=np.asarray(data.ctx["labels"],np.int64)[test_indices]
        ids=[str(data.ctx["ids"][int(i)]) for i in test_indices]
        metrics=metric_values(labels,p_asd,pred)
        test_stats=film_diagnostics(hook.last,test_indices)
        if group=="F4":
            pair_true,true_stats=film.modulate(data.z_img,true,data.d_ref,best["epoch"],True)
            pair_wrong,wrong_stats=film.modulate(data.z_img,wrong,data.d_ref,best["epoch"],True)
            tr_idx=data.indices["train"]
            modulation_pair={"feature_relative_l2":float((pair_true-pair_wrong).norm()/data.z_img.norm().clamp_min(1e-12)),
                             "train_feature_relative_l2":float((pair_true[tr_idx]-pair_wrong[tr_idx]).norm()/data.z_img[tr_idx].norm().clamp_min(1e-12)),
                             "actual_film_output_change_fraction":float(torch.any(pair_true!=pair_wrong,dim=1).float().mean()),
                             "train_actual_film_output_change_fraction":float(torch.any(pair_true[tr_idx]!=pair_wrong[tr_idx],dim=1).float().mean()),
                             "gamma_delta_l2":float((true_stats["gamma"]-wrong_stats["gamma"]).norm()),
                             "beta_delta_l2":float((true_stats["beta"]-wrong_stats["beta"]).norm())}
        else:modulation_pair=None
        split_stats={}
        for name,idx in (("train",np.asarray(data.split["train"])),("val",np.asarray(data.split["val"])),("test",test_indices)):
            src=wrong[idx].detach().cpu().numpy();dst=true[idx].detach().cpu().numpy()
            split_stats[name]={"condition_index_change_fraction":float(np.mean(data.permutation[idx]!=idx)) if group=="F4" else 0.,
                               "age_changed_fraction":float(np.mean(src[:,0]!=dst[:,0])) if group=="F4" else 0.,
                               "sex_changed_fraction":float(np.any(src[:,1:]!=dst[:,1:],axis=1).mean()) if group=="F4" else 0.,
                               "joint_condition_changed_fraction":float(np.any(src!=dst,axis=1).mean()) if group=="F4" else 0.}
        film_final={"parameter_count":sum(p.numel() for p in film.parameters()),"initial_sha256":module_sha,
                    "best_sha256":film_state_hash(film),"best_epoch":best["epoch"],
                    "training_reference":data.d_ref.detach().cpu().tolist(),"conditions_fit":data.condition_info,
                    "best_test_split_modulation":test_stats,"F4_split_permutation":split_stats if group=="F4" else None,
                    "F4_permutation_groups":data.permutation_records if group=="F4" else None}
        film_final["F4_true_vs_wrong_modulation_on_same_captured_test_forward"]=modulation_pair
        result={"state":"complete","group":group,"seed":911,"fold":fold,
                "started_utc":start_utc,"completed_utc":utc(),"training_code_sha256":audit["training_code_sha256"],
                "manifest_sha256":audit["manifest_sha256"],"input_sha256":audit["anchors"][str(fold)]["sha256"],
                "input_digest":data.input_digest,"historical_anchor":str(data.anchor),
                "historical_train_start_core_sha256":initial_core_sha,
                "historical_train_start_rng_sha256":HIST_RUN.rng_hash(data.start["rng"]),
                "film_initialization_seed":film_seed,"film_initial_sha256":module_sha,
                "historical_prototype_preparation":data.prepared["info"],"objective_group":"M3" if objective_group=="M3" else "M1-no-KL",
                "best_epoch":best["epoch"],"best_val_loss":best_loss,"best_core":best["core"],
                "best_core_sha256":HIST_RUN.core_hash(best["core"]),"best_film":best["film"],
                "best_film_sha256":film_state_hash(film),"best_record":best["record"],"epochs":history,
                "epochs_run":len(history),"train_end_rng":train_end_rng,
                "train_end_rng_sha256":json_digest(HIST_RUN.rng_hash(train_end_rng)),
                "test_rng_before":HIST_RUN.rng_hash(test_rng_before),"test_rng_after":HIST_RUN.rng_hash(test_rng_after),
                "oof":{"indices":test_indices,"subject_ids":ids,"labels":labels,"logits":raw_logits,
                       "probability_asd":p_asd,"prediction":pred},"metrics":metrics,
                "training_seconds":training_seconds,"task_seconds":time.perf_counter()-start_time,
                "native_core_forwards":data.forward_count,"formal_test_forwards":1,"formal_test_grad_enabled":True,
                "test_graph_loss_actual":float(test_graph.detach()),"test_attention_weights_actual":{
                    "image":float(data.core.img_weight.detach()),"phenotype":float(data.core.ph_weight.detach())},
                "physical_gpu":int(os.environ.get("CUDA_VISIBLE_DEVICES","0")),"device":torch.cuda.get_device_name(0),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(0),"peak_reserved_bytes":torch.cuda.max_memory_reserved(0),
                "film":film_final,"uses_test_labels_for_training_or_selection":False}
        path=result_path(group,fold);atomic_torch(result,path)
        valid_result(group,fold,required=True)
        atomic_json({"state":"complete","group":group,"fold":fold,"artifact_sha256":sha_file(path),
                     "best_epoch":best["epoch"],"training_seconds":training_seconds},status)
        log("task_complete",group=group,fold=fold,best_epoch=best["epoch"],best_val_loss=best_loss,
            training_seconds=training_seconds,artifact_sha256=sha_file(path))
    except Exception as error:
        atomic_json({"state":"failed","group":group,"fold":fold,"failed_utc":utc(),"error":repr(error),
                     "traceback":traceback.format_exc()},status)
        log("task_failed",group=group,fold=fold,error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        hook.remove();data.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("audit","math","preflight","freeze","train"))
    parser.add_argument("--group",choices=list(CONFIG["groups"]))
    parser.add_argument("--fold",type=int)
    args=parser.parse_args()
    if args.action=="audit":audit_inputs();return
    if args.action=="math":
        audit=json.loads((BASE/"audit.json").read_text()) if (BASE/"audit.json").exists() else audit_inputs()
        audit["mathematical_tests"]=check_math();atomic_json(audit,BASE/"audit.json");return
    if args.action=="preflight":
        audit=json.loads((BASE/"audit.json").read_text())
        if audit.get("mathematical_tests",{}).get("passed") is not True:raise AssertionError("Run math tests first")
        value=preflight_one(0);audit["preflight"]=value;atomic_json(audit,BASE/"audit.json");log("real_fold0_preflight_passed",forwards=1);return
    if args.action=="freeze":
        freeze_manifest(json.loads((BASE/"audit.json").read_text()));return
    if args.group is None or args.fold is None or not 0<=args.fold<10:parser.error("train requires --group and --fold")
    lock_path=BASE/"runs"/args.group/f"s911_f{args.fold}.task.lock"
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a+") as task_lock:
        fcntl.flock(task_lock,fcntl.LOCK_EX)
        train_one(args.group,args.fold)
        fcntl.flock(task_lock,fcntl.LOCK_UN)


if __name__=="__main__":
    try:main()
    except Exception as exc:
        log("blocked",error=repr(exc),traceback=traceback.format_exc());raise
