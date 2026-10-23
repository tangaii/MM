"""Read-only verification and analysis of completed IA-FiLM artifacts."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import run
from film import IdentityAnchoredFiLM
from test_analysis import patient_transition,run_tests


METRICS=run.RESULT_METRICS


def attach_historical_run():
    module_path=run.M3DIR/"run.py"
    spec=importlib.util.spec_from_file_location("iafilm_analysis_historical_pcdpax_run",module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run.HIST_RUN=module


def pct(value):return 100.0*float(value)


def mean_sd(values):
    values=np.asarray(values,dtype=np.float64)
    return {"mean":float(values.mean()),"sample_sd":float(values.std(ddof=1)),
            "formatted_percent":f"{pct(values.mean()):.2f} +/- {pct(values.std(ddof=1)):.2f}%"}


def historical_subject_rows(group):
    directory=run.PCDCA if group in ("D0","D4") else run.M3DIR
    audit=json.loads((run.BASE/"audit.json").read_text())
    control_name=f"historical_{group}"
    rows={}
    for fold in range(10):
        path=directory/"runs"/group/f"s911_f{fold}.pt"
        saved_sha=next(x["sha256"] for x in audit["historical_readonly_controls"][control_name]["checkpoint_sha256"] if x["path"]==str(path))
        if run.sha_file(path)!=saved_sha:raise AssertionError(f"Historical {group} checkpoint SHA mismatch: {path}")
        value=run.load_torch(path)
        if "payload" in value:
            value=value["payload"]
        oof=value["oof"]
        ids=oof.get("ids",oof.get("subject_ids"))
        probability=oof.get("probability",oof.get("probability_asd"))
        indices=np.asarray(oof["indices"],dtype=np.int64)
        split_path=run.HISTORY/"artifacts"/"seed_911"/f"fold_{fold}"/"splits.pt"
        expected=np.asarray(run.load_torch(split_path)["test"],dtype=np.int64)
        if not np.array_equal(indices,expected):raise AssertionError(f"Historical {group} fold {fold} test order changed")
        probability=np.asarray(probability,dtype=np.float64);labels=np.asarray(oof["labels"],dtype=np.int64)
        predictions=np.asarray(oof["prediction"],dtype=np.int64)
        if not np.isfinite(probability).all() or not np.all((probability>=0)&(probability<=1)):
            raise AssertionError(f"Invalid historical probabilities: {group}/{fold}")
        if not np.isin(predictions,[0,1]).all() or len(ids)!=len(indices) or len(labels)!=len(indices):
            raise AssertionError(f"Invalid historical OOF structure: {group}/{fold}")
        for index,subject,label,prob,pred in zip(indices,ids,labels,probability,predictions):
            subject=str(subject)
            if subject in rows:raise AssertionError(f"Duplicate historical OOF subject {subject}")
            rows[subject]={"index":int(index),"fold":fold,"label":int(label),"probability_asd":float(prob),"prediction":int(pred)}
    if len(rows)!=871:raise AssertionError(f"Historical {group} OOF has {len(rows)} subjects")
    return rows


def load_new_group(group):
    rows={};fold_metrics=[];checkpoints=[]
    for fold in range(10):
        result=run.valid_result(group,fold,required=True)
        fold_metrics.append({"fold":fold,**result["metrics"]})
        checkpoints.append({"fold":fold,"best_epoch":result["best_epoch"],"best_val_loss":result["best_val_loss"],
                            "training_seconds":result["training_seconds"],"task_seconds":result["task_seconds"],
                            "physical_gpu":result["physical_gpu"],"peak_allocated_bytes":result["peak_allocated_bytes"],
                            "peak_reserved_bytes":result["peak_reserved_bytes"],"sha256":run.sha_file(run.result_path(group,fold)),
                            "core_sha256":result["best_core_sha256"],"film_sha256":result["best_film_sha256"]})
        oof=result["oof"]
        for index,subject,label,prob,pred in zip(oof["indices"],oof["subject_ids"],oof["labels"],oof["probability_asd"],oof["prediction"]):
            subject=str(subject)
            if subject in rows:raise AssertionError(f"Duplicate {group} OOF subject {subject}")
            rows[subject]={"index":int(index),"fold":fold,"label":int(label),"probability_asd":float(prob),"prediction":int(pred)}
    if len(rows)!=871:raise AssertionError(f"{group} OOF has {len(rows)} subjects")
    return rows,fold_metrics,checkpoints


def check_alignment(reference,other,name):
    if set(reference)!=set(other):raise AssertionError(f"Patient ID set differs: {name}")
    ordered=sorted(reference)
    for subject in ordered:
        a,b=reference[subject],other[subject]
        if a["index"]!=b["index"] or a["fold"]!=b["fold"] or a["label"]!=b["label"]:
            raise AssertionError(f"Patient/fold/label mismatch for {name}: {subject}")
    labels=np.asarray([reference[s]["label"] for s in ordered],dtype=np.int64)
    if np.bincount(labels,minlength=2).tolist()!=[403,468]:raise AssertionError(f"Unexpected cohort label counts: {name}")
    return ordered


def transition(left,right,ordered):
    y=np.asarray([left[s]["label"] for s in ordered]);lp=np.asarray([left[s]["prediction"] for s in ordered])
    ry=np.asarray([right[s]["label"] for s in ordered]);rp=np.asarray([right[s]["prediction"] for s in ordered])
    return patient_transition(y,lp,ry,rp)


def compare_fold_metrics(left,right):
    result={}
    for metric in METRICS:
        delta=np.asarray([left[f][metric]-right[f][metric] for f in range(10)],dtype=np.float64)
        loo=[float(np.delete(delta,i).mean()) for i in range(10)]
        result[metric]={"fold_delta":delta.tolist(),"mean_delta":float(delta.mean()),
                        "mean_delta_pp":pct(delta.mean()),"sample_sd_delta_pp":pct(delta.std(ddof=1)),
                        "positive_folds":int(np.sum(delta>0)),"negative_folds":int(np.sum(delta<0)),
                        "leave_one_fold_out_mean_range_pp":[pct(min(loo)),pct(max(loo))]}
    return result


def film_summary(groups):
    summary={}
    for group in groups:
        results=[run.load_torch(run.result_path(group,fold)) for fold in range(10)]
        test=[x["film"]["best_test_split_modulation"] for x in results]
        train_records=[record for result in results for record in result["epochs"] if "film_train_subgroups" in record]
        subgroup_values={}
        for record in train_records:
            for name,item in record["film_train_subgroups"].items():
                if item["film"] is not None:subgroup_values.setdefault(name,[]).append(item["film"])
        subgroup_summary={name:{"sampled_n":len(values),
            "gamma_mean":float(np.mean([x["gamma_mean"] for x in values])),
            "beta_mean":float(np.mean([x["beta_mean"] for x in values])),
            "relative_feature_l2_mean":float(np.mean([x["relative_feature_l2"] for x in values]))}
            for name,values in subgroup_values.items()}
        image_budget=[e["image_budget"] for result in results for e in result["epochs"]]
        ph_budget=[e["ph_budget"] for result in results for e in result["epochs"]]
        sampled=[e for result in results for e in result["epochs"] if "film_gradient_norm" in e]
        film_modules=[]
        for result in results:
            module=IdentityAnchoredFiLM()
            module.load_state_dict(result["best_film"],strict=True);module.eval()
            reference=np.asarray(result["film"]["training_reference"],dtype=np.float32)
            ref=torch.as_tensor(reference).reshape(1,3)
            h=torch.zeros((1,16))
            out,stats=module.modulate(h,ref,ref,result["best_epoch"])
            film_modules.append(bool(torch.equal(out,h) and torch.equal(stats["gamma"],torch.ones_like(stats["gamma"])) and
                torch.equal(stats["beta"],torch.zeros_like(stats["beta"]))))
        summary[group]={"parameter_count_per_fold":sorted({x["film"]["parameter_count"] for x in results}),
            "test_gamma_mean":float(np.mean([x["gamma_mean"] for x in test])),
            "test_gamma_sd_mean":float(np.mean([x["gamma_sd"] for x in test])),
            "test_beta_mean":float(np.mean([x["beta_mean"] for x in test])),
            "test_beta_sd_mean":float(np.mean([x["beta_sd"] for x in test])),
            "test_relative_feature_l2_mean":float(np.mean([x["relative_feature_l2"] for x in test])),
            "test_max_abs_gamma_minus_one_max":float(max(x["max_abs_gamma_minus_one"] for x in test)),
            "test_max_abs_beta_max":float(max(x["max_abs_beta"] for x in test)),
            "best_epoch_mean":float(np.mean([x["best_epoch"] for x in results])),
            "best_epoch_range":[int(min(x["best_epoch"] for x in results)),int(max(x["best_epoch"] for x in results))],
            "best_val_loss_mean":float(np.mean([x["best_val_loss"] for x in results])),
            "checkpoint_hashes_differ_by_fold":len({x["best_core_sha256"] for x in results})>1,
            "sampled_train_subgroup_epochs":len(train_records),
            "trained_weights_reference_is_exact_identity_all_folds":all(film_modules),
            "train_subgroups_sampled":subgroup_summary,
            "gradient_budget":{"image_weighted_ratio_mean":float(np.mean([x["weighted_auxiliary_z_gradient_ratio"] for x in image_budget])),
                "phenotype_weighted_ratio_mean":float(np.mean([x["weighted_auxiliary_z_gradient_ratio"] for x in ph_budget])),
                "image_lambda_cap_fraction":float(np.mean([x["lambda_cap_hit"] for x in image_budget])),
                "phenotype_lambda_cap_fraction":float(np.mean([x["lambda_cap_hit"] for x in ph_budget])),
                "film_gradient_norm_mean_at_sampled_epochs":float(np.mean([x["film_gradient_norm"] for x in sampled])),
                "film_native_gradient_norm_mean_at_sampled_epochs":float(np.mean([x["film_native_gradient_norm"] for x in sampled])),
                "native_raw_img_gradient_norm_mean":float(np.mean([x["native_raw_embedding_gradient_norms"]["z_img"] for x in sampled])),
                "native_raw_ph_gradient_norm_mean":float(np.mean([x["native_raw_embedding_gradient_norms"]["z_ph"] for x in sampled])),
                "kl_mean_if_present":float(np.mean([e["KL"] for r in results for e in r["epochs"] if e["KL"] is not None])) if any(e["KL"] is not None for r in results for e in r["epochs"]) else None,
                "kl_absent_all_epochs":all(e["KL"] is None for r in results for e in r["epochs"])}}
        if group=="F4":
            fields=("condition_index_change_fraction","age_changed_fraction","sex_changed_fraction","joint_condition_changed_fraction")
            summary[group]["permutation_change_fraction_by_split"]={
                split:{field:float(np.mean([x["film"]["F4_split_permutation"][split][field] for x in results])) for field in fields}
                for split in ("train","val","test")}
            pairs=[x["film"]["F4_true_vs_wrong_modulation_on_same_captured_test_forward"] for x in results]
            summary[group]["same_test_forward_true_wrong_relative_feature_l2_mean"]=float(np.mean([x["feature_relative_l2"] for x in pairs]))
            summary[group]["same_test_forward_true_wrong_gamma_delta_l2_mean"]=float(np.mean([x["gamma_delta_l2"] for x in pairs]))
            summary[group]["same_test_forward_true_wrong_beta_delta_l2_mean"]=float(np.mean([x["beta_delta_l2"] for x in pairs]))
    return summary


def position_mechanism(groups):
    values={}
    for group in ("F1","F2","F3","F4"):
        rows=[]
        for fold in range(10):
            result=run.load_torch(run.result_path(group,fold));record=result["best_record"]
            rows.append({"fold":fold,"validation_graph":record["val_graph"],
                         "validation_img_weight":record["val_img_weight"],
                         "validation_ph_weight":record["val_ph_weight"],
                         "test_graph":result["test_graph_loss_actual"],
                         "test_img_weight":result["test_attention_weights_actual"]["image"],
                         "test_ph_weight":result["test_attention_weights_actual"]["phenotype"]})
        values[group]={"mean":{key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key!="fold"},"folds":rows}
    return {"by_group":values,
            "F2_minus_F1_mean_validation_graph":values["F2"]["mean"]["validation_graph"]-values["F1"]["mean"]["validation_graph"],
            "F2_minus_F1_mean_validation_image_weight":values["F2"]["mean"]["validation_img_weight"]-values["F1"]["mean"]["validation_img_weight"],
            "F2_minus_F1_mean_validation_phenotype_weight":values["F2"]["mean"]["validation_ph_weight"]-values["F1"]["mean"]["validation_ph_weight"]}


def checkpoint_comparisons(checkpoints):
    out={}
    pairs=(("F2","F1"),("F3","F2"),("F3","F4"),
           ("F3","historical_D4"),("F3","historical_M1"),("F3","historical_M3"))
    for left,right in pairs:
        a={x["fold"]:x for x in checkpoints[left]};b={x["fold"]:x for x in checkpoints[right]}
        epoch=np.asarray([a[i]["best_epoch"]-b[i]["best_epoch"] for i in range(10)],dtype=np.float64)
        val=np.asarray([a[i]["best_val_loss"]-b[i]["best_val_loss"] for i in range(10)],dtype=np.float64)
        out[f"{left}-{right}"]={"best_epoch_delta_by_fold":epoch.astype(int).tolist(),
            "mean_best_epoch_delta":float(epoch.mean()),"best_epoch_different_fold_fraction":float(np.mean(epoch!=0)),
            "best_val_loss_delta_by_fold":val.tolist(),"mean_best_val_loss_delta":float(val.mean()),
            "core_checkpoint_different_fold_fraction":float(np.mean([a[i]["core_sha256"]!=b[i]["core_sha256"] for i in range(10)])),
            "film_checkpoint_different_fold_fraction":float(np.mean([a[i].get("film_sha256")!=b[i].get("film_sha256") for i in range(10)])) if all("film_sha256" in a[i] and "film_sha256" in b[i] for i in range(10)) else None}
    return out


def render_report(summary):
    lines=["# Experiment 25: IA-FiLM", "",
        f"Status: {summary['status']}. Formal training: {summary['tasks_completed']}/40.", "",
        "Execution was requested in `/tmp/IAFiLM_20261023`; the original `/data` experiment-directory creation returned EIO, so archival migration remains pending.", "",
        "RTGMFF applies direct age/gender FiLM after HWM/CSTE visual fusion and before ASAM, using batch age normalization and unconstrained gamma/beta. This implementation preserves the model's reference point as identity and bounds its modulation. The method uses the FiLM concept, not FiLM itself, as a claimed novelty. See [RTGMFF official model](https://github.com/BeistMedAI/RTGMFF/blob/main/model.py) and [MM-GTUNets paper](https://arxiv.org/html/2406.14455v3).", "",
        "PCCI's C0 was about 81.88% ACC / 88.54% AUC; C1, C3, and C6 were about 77.78% / 85.25%, 77.63% / 85.07%, and 77.44% / 84.98%. C1 had a constant residual, so condition-independent feature disturbance itself could be harmful. Historical M3's phenotype student was near 50% disease accuracy despite KL gradients. F2/F3 isolates KL retention; F3/F4 tests correct versus site/split-bounded patient correspondence.", "",
        "| Group | ACC | SEN-ASD | SPE-TD | probability AUC |", "|---|---:|---:|---:|---:|"]
    for group,row in summary["performance"].items():
        lines.append(f"| {group} | {row['ACC']['formatted_percent']} | {row['SEN-ASD']['formatted_percent']} | {row['SPE-TD']['formatted_percent']} | {row['probability_AUC']['formatted_percent']} |")
    lines.extend(["","Paired fold deltas are left group minus right group, in percentage points. Leave-one-fold-out ranges describe the mean delta after dropping each fold.","",
                  "| Contrast | Metric | Mean delta (pp) | Positive folds | LOO mean range (pp) |","|---|---|---:|---:|---:|"])
    for contrast,metrics in summary["comparisons"].items():
        for metric,item in metrics.items():lines.append(f"| {contrast} | {metric} | {item['mean_delta_pp']:.3f} | {item['positive_folds']}/10 | {item['leave_one_fold_out_mean_range_pp'][0]:.3f} to {item['leave_one_fold_out_mean_range_pp'][1]:.3f} |")
    lines.extend(["","## Mechanism and audit","",f"Patient-level OOF changes: `{json.dumps(summary['patient_transitions'],ensure_ascii=False,sort_keys=True)}`.","",
                  f"FiLM and checkpoint diagnostics: `{json.dumps(summary['film_mechanisms'],ensure_ascii=False,sort_keys=True)}`.","",
                  f"FiLM position diagnostics: `{json.dumps(summary['film_position_mechanism'],ensure_ascii=False,sort_keys=True)}`.","",
                  f"Checkpoint contrasts: `{json.dumps(summary['checkpoint_comparisons'],ensure_ascii=False,sort_keys=True)}`.","",
                  f"Training wall seconds: {summary['execution']['wall_seconds']:.1f}; cumulative worker seconds: {summary['execution']['worker_elapsed_seconds']:.1f}; retries: {summary['execution']['retry_count']}; failures: {summary['execution']['failure_count']}.","",
                  "All probabilities and metrics were independently recomputed from each saved single-forward test output. No analysis step runs a model or accesses test outcomes for selection.","",
                  "Research judgment is limited to a candidate decision under seed 911. Historical RFE cross-fold contamination, transductive 871-node graph, stochastic VAE/eval dropout, and the attention-gradient truncation remain; these results are not leakage-free clinical generalization evidence."])
    return "\n".join(lines)+"\n"


def run_analysis():
    attach_historical_run()
    audit=run.guard(True)
    if not audit.get("formal_authorized"):raise AssertionError("Formal run was not frozen")
    transition_tests=run_tests()
    new={};fold_metrics={};checkpoints={}
    for group in run.CONFIG["groups"]:
        new[group],fold_metrics[group],checkpoints[group]=load_new_group(group)
    controls={name:historical_subject_rows(group) for name,group in (("historical_D4","D4"),("historical_M1","M1"),("historical_M3","M3"),("historical_D0","D0"))}
    reference=new["F1"];ordered=check_alignment(reference,new["F2"],"F2")
    for group in ("F3","F4",*controls):ordered=check_alignment(reference,new.get(group,controls.get(group)),group)
    all_rows={**controls,**new};oof=[]
    for group,rows in all_rows.items():
        for subject in ordered:
            value=rows[subject]
            oof.append({"group":group,"subject_id":subject,"index":value["index"],"fold":value["fold"],
                        "label":value["label"],"probability_asd":value["probability_asd"],"prediction":value["prediction"]})
    if len(oof)!=871*(4+4):raise AssertionError("Expected 8x871 OOF records including read-only controls")
    oof_new=[row for row in oof if row["group"] in run.CONFIG["groups"]]
    # The formal OOF file contains the four new groups only; controls remain read-only references.
    with (run.BASE/"oof.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(oof_new[0]));writer.writeheader();writer.writerows(oof_new)
    rows={**{g:new[g] for g in new},**controls}
    comparisons={"F2-F1":compare_fold_metrics(fold_metrics["F2"],fold_metrics["F1"]),
                 "F3-F2":compare_fold_metrics(fold_metrics["F3"],fold_metrics["F2"]),
                 "F3-F4":compare_fold_metrics(fold_metrics["F3"],fold_metrics["F4"])}
    historic_metrics={name:audit["historical_readonly_controls"][name]["fold_metrics"] for name in controls}
    for name in ("historical_D4","historical_M1","historical_M3"):
        comparisons[f"F3-{name}"]=compare_fold_metrics(fold_metrics["F3"],historic_metrics[name])
    transitions={}
    for other in ("F2","F4","historical_M3","historical_D4"):
        transitions[f"F3-vs-{other}"]=transition(new["F3"],rows[other],ordered)
    performance={}
    for name in ("historical_D0","historical_D4","historical_M1","historical_M3","F1","F2","F3","F4"):
        values=fold_metrics[name] if name in fold_metrics else historic_metrics[name]
        performance[name]={metric:mean_sd([row[metric] for row in values]) for metric in METRICS}
    # Persist exactly the registered four-group, 40-fold table.
    with (run.BASE/"fold_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["group","fold",*METRICS]);writer.writeheader()
        for group in run.CONFIG["groups"]:
            for row in fold_metrics[group]:writer.writerow({"group":group,**row})
    queue=json.loads((run.BASE/"queue.json").read_text())
    total_worker=sum(x["training_seconds"] for group in new for x in [run.load_torch(run.result_path(group,fold)) for fold in range(10)])
    audit["execution"].update({"tasks":40,"completed":40-len(queue.get("failures",[])),"retries":len(queue.get("retries",[])),
                               "failures":queue.get("failures",[]),"queue_state":queue.get("state"),
                               "wall_seconds":queue.get("wall_seconds"),"worker_elapsed_seconds":queue.get("worker_elapsed_seconds"),
                               "sum_training_seconds":total_worker,"gpu_plan":{"0":20,"1":20}})
    film=film_summary(tuple(run.CONFIG["groups"]))
    position_diagnostics=position_mechanism(run.CONFIG["groups"])
    # Conservative descriptive decision: require F3 to lead its patient-mismatch and M3 controls on AUC,
    # with majority-fold direction, non-degenerate modulation and no ASD-sensitivity collapse.
    key=comparisons["F3-F4"]["probability_AUC"];m3=comparisons["F3-historical_M3"]["probability_AUC"]
    sens=performance["F3"]["SEN-ASD"]["mean"]
    modulation=film["F3"]["test_relative_feature_l2_mean"]
    position_auc=comparisons["F2-F1"]["probability_AUC"]
    no_kl=comparisons["F3-F2"]["probability_AUC"]
    baseline_auc=[comparisons[f"F3-{name}"]["probability_AUC"] for name in ("historical_D4","historical_M1","historical_M3")]
    stable_auc=[key,position_auc,no_kl,m3,*baseline_auc]
    stable_direction=all(x["mean_delta"]>0 and x["positive_folds"]>=6 and min(x["leave_one_fold_out_mean_range_pp"])>0 for x in stable_auc)
    acc_not_lower=all(performance["F3"]["ACC"]["mean"]>=performance[name]["ACC"]["mean"]
                      for name in ("historical_D4","historical_M1","historical_M3"))
    sens_spec_not_lower=all(performance["F3"][metric]["mean"]>=performance[name][metric]["mean"]
        for name in ("historical_D4","historical_M1","historical_M3") for metric in ("SEN-ASD","SPE-TD"))
    film_gradient=film["F3"]["gradient_budget"]["film_gradient_norm_mean_at_sampled_epochs"]
    mismatch_effect=film["F4"]["same_test_forward_true_wrong_relative_feature_l2_mean"]
    go=stable_direction and acc_not_lower and sens_spec_not_lower and modulation>0 and film_gradient>0 and mismatch_effect>0 and all(
        film[group]["trained_weights_reference_is_exact_identity_all_folds"] for group in run.CONFIG["groups"])
    judgment="CANDIDATE_GO_FOR_CONFIRMATION" if go else "NO_GO"
    historical_checkpoints={}
    for name,group in (("historical_D4","D4"),("historical_M1","M1"),("historical_M3","M3")):
        directory=run.PCDCA if group=="D4" else run.M3DIR
        historical_checkpoints[name]=[]
        for fold in range(10):
            saved=run.load_torch(directory/"runs"/group/f"s911_f{fold}.pt")
            if "payload" in saved:saved=saved["payload"]
            historical_checkpoints[name].append({"fold":fold,"best_epoch":saved["best_epoch"],
                "best_val_loss":saved["best_val_loss"],"core_sha256":saved["best_core_sha256"]})
    checkpoints_all={**checkpoints,**historical_checkpoints}
    checkpoint_delta=checkpoint_comparisons(checkpoints_all)
    all_results=[run.load_torch(run.result_path(group,fold)) for group in new for fold in range(10)]
    gpu_resources={str(gpu):{"completed_tasks":sum(int(x["physical_gpu"])==gpu for x in all_results),
        "peak_allocated_mib":max(x["peak_allocated_bytes"] for x in all_results if int(x["physical_gpu"])==gpu)/(1024**2),
        "peak_reserved_mib":max(x["peak_reserved_bytes"] for x in all_results if int(x["physical_gpu"])==gpu)/(1024**2),
        "training_seconds":sum(x["training_seconds"] for x in all_results if int(x["physical_gpu"])==gpu)} for gpu in (0,1)}
    summary={"experiment":25,"name":"IA-FiLM","status":"COMPLETE" if not queue.get("failures") and queue.get("completed")==40 else "PARTIAL",
             "tasks_completed":40-len(queue.get("failures",[])),"gpu0_tasks":20-len([x for x in queue.get("failures",[]) if x["gpu"]==0]),
             "gpu1_tasks":20-len([x for x in queue.get("failures",[]) if x["gpu"]==1]),"historical_retrains":0,
             "performance":performance,"comparisons":comparisons,"patient_transitions":transitions,
             "film_mechanisms":film,"film_position_mechanism":position_diagnostics,
             "checkpoints":checkpoints,"checkpoint_comparisons":checkpoint_delta,"fold_metrics":fold_metrics,
             "oof_records_registered_groups":len(oof_new),"oof_records_with_historical_readonly_controls":len(oof),
             "patient_transition_synthetic_test":transition_tests,"research_judgment":judgment,
             "research_judgement_operationalization":run.CONFIG["research_judgement_operationalization"],
             "execution":{"wall_seconds":queue.get("wall_seconds",0.),"worker_elapsed_seconds":queue.get("worker_elapsed_seconds",0.),
                          "training_seconds_sum":total_worker,"retry_count":len(queue.get("retries",[])),"failure_count":len(queue.get("failures",[])),
                          "failures":queue.get("failures",[]),"gpu_plan":{"0":20,"1":20},"gpu_resources":gpu_resources},
             "code_sha256":audit["training_code_files_sha256"],"manifest_sha256":audit["manifest_sha256"],
             "source_sha256":audit["original_source_sha256"],"historical_input_sha256":audit["inputs_sha256"],
             "pending_migration":True,"handoff_updated":False}
    run.atomic_json(summary,run.BASE/"summary.json")
    run.atomic_json(audit,run.BASE/"audit.json")
    (run.BASE/"report.md").write_text(render_report(summary),encoding="utf-8")
    return summary


if __name__=="__main__":
    print(json.dumps(run_analysis(),indent=2,ensure_ascii=False))
