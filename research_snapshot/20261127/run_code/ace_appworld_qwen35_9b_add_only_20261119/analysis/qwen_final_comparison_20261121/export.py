# SPDX-License-Identifier: Apache-2.0
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from math import comb
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.ace.analyze_reflection_accumulation import validate_phase
from scripts.ace.plot_worst_reflection_repeats import RUNS as Q38
from scripts.ace.run_reflection_accumulation import exact_reward

out = Path(__file__).resolve().parent
root = Path.cwd() / "outputs"
q36 = root / "ace_appworld_qwen36_add_only_20261118"
q35 = root / "ace_appworld_qwen35_9b_add_only_20261119"
q38_adv = root / "ace_appworld_advantage_add_20261118"
aud = root / "ace_appworld_aud_accumulation_qwen38_20261117"
models = {"qwen38": Q38, **{name: {m: root / m / "accumulation" for m in Q38}
                            for name, root in (("qwen36", q36), ("qwen35", q35))}}
adv_models = {"qwen38": q38_adv, "qwen36": q36 / "advantage", "qwen35": q35 / "advantage"}
seeds = (42, 43, 44)
phases = ("train_final", "bwt_checkpoint", "val")
arms = ("without_gt", "with_gt")
read = lambda p: json.loads(p.read_text())
provenance, score_rows, scored, baselines, settings = {}, [], [], {}, {}
progress_at = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")

def source(path):
    raw = path.read_bytes()
    provenance[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def metric(rows):
    assert rows and all(type(r["success"]) is bool for r in rows)
    return 100 * mean(float(exact_reward(r)) for r in rows), 100 * mean(r["success"] for r in rows)

def archive_eval(rows, model, mode, variant, phase):
    for r in rows:
        score_rows.append({"model": model, "mode": mode, "variant": variant, "phase": phase, **{
            k:r.get(k) for k in ("task_id","arm","seed","condition_id","source_repeat","evaluation_repeat",
                                  "checkpoint_step","pct","success","passed_tests","total_tests",
                                  "turn_limit_reached","appworld_policy_sha256")
        }, "budget_failure": (r.get("budget_failure") or {}).get("reason", "")})

def record(model, mode, variant, seed, condition, train, bwt, val, selections, initial, candidates):
    tr = {r["task_id"]: r for r in train}
    assert len(train) == len(tr) == 50 == len(val) == len({r["task_id"] for r in val})
    by_task = {c["candidate"]["task_id"]: c["candidate"]["source_repeat"] for c in selections}
    assert len(by_task) == len(selections) == len(bwt) == len({r["task_id"] for r in bwt})
    assert set(by_task) == {r["task_id"] for r in bwt} <= tr.keys() == initial.keys()
    arm = train[0]["arm"]
    selection = condition.removeprefix(f"s{seed}_{arm}_")
    ep, et = map(mean, zip(*(metric(candidates[arm,t,by_task[t]]) if t in by_task else metric(initial[t]) for t in tr)))
    tp, tt = metric(train)
    vp, vt = metric(val)
    bp = 100 * mean(float(exact_reward(tr[r["task_id"]]) - exact_reward(r)) for r in bwt)
    bt = 100 * mean(int(tr[r["task_id"]]["success"]) - int(r["success"]) for r in bwt)
    scored.append(dict(model=model, mode=mode, variant=variant, seed=seed, gt=arm, selection=selection,
                       learned=len(bwt), train_pct=tp, train_tgc=tt, heldout_pct=vp, heldout_tgc=vt,
                       bwt_pct=bp, bwt_tgc=bt, independent_pct=ep, independent_tgc=et))

for model, modes in models.items():
    baselines[model], settings[model] = {}, {}
    for mode, run in modes.items():
        protocol = source(run / "protocol.json")
        settings[model][mode] = {"model":protocol["model"],"model_settings":protocol["model_settings"],
                                "actor_limits":protocol["actor_limits"],"policy_sha256":protocol["appworld_policy_sha256"],
                                "task_orders_sha256": hashlib.sha256(json.dumps(protocol["task_orders"],sort_keys=True).encode()).hexdigest()}
        bank = Path(protocol["source_reflection_run"])
        initial, candidates = defaultdict(list), defaultdict(list)
        for f in sorted((bank / "stage1").glob("*/result.json")):
            row = source(f)
            exact_reward(row)
            initial[row["task_id"]].append(row)
        for f in sorted((bank / "stage3").glob("*/result.json")):
            row = source(f)
            exact_reward(row)
            candidates[row["arm"],row["task_id"],row["source_repeat"]].append(row)
        assert len(initial) == 50 and all({r["source_repeat"] for r in g} == set(range(5)) and len(g)==5 for g in initial.values())
        assert len(candidates) == 500 and all({r["evaluation_repeat"] for r in g} == {0,1,2} and len(g)==3 for g in candidates.values())
        init_rows = [r for rows in initial.values() for r in rows]
        updated = [r for rows in candidates.values() for r in rows]
        archive_eval(init_rows,model,mode,"reflection","initial")
        archive_eval(updated,model,mode,"reflection","independent")
        success_counts = Counter(sum(r["success"] for r in rows) for rows in initial.values())
        b = {"initial": metric(init_rows), **{a:metric([r for r in updated if r["arm"]==a]) for a in arms},
             "success_histogram": dict(success_counts)}
        for k in (3,5):
            b[f"pass@{k}"] = 2 * sum(n * (1 - comb(5-c,k)/comb(5,k)) for c,n in success_counts.items())
            b[f"pass-and-{k}"] = 2 * sum(n * comb(c,k)/comb(5,k) for c,n in success_counts.items())
        baselines[model][mode] = b
        for variant, acc, selected_seeds in [("ADD",run,seeds)] + (
            [("AUD",aud/"thinking",(42,43))] if model=="qwen38" and mode=="thinking" else []
        ):
            p = source(acc/"protocol.json")
            for seed in selected_seeds:
                all_groups = defaultdict(lambda: defaultdict(list))
                for phase in phases:
                    rows = source(acc/f"seed{seed}"/f"{phase}_results.json")
                    validate_phase(acc,p,seed,phase,rows)
                    archive_eval(rows,model,mode,variant,phase)
                    for r in rows: all_groups[r["condition_id"]][phase].append(r)
                for cid,g in all_groups.items():
                    choices = source(acc/"conditions"/cid/"selections.json")
                    record(model,mode,variant,seed,cid,*(g[phase] for phase in phases),choices,initial,candidates)
        a = adv_models[model]
        eval_path = a/"evaluations"
        g = defaultdict(lambda: defaultdict(list))
        for phase in phases:
            rows = [r for r in source(eval_path/f"{phase}_results.json") if r["sampler"]==mode]
            assert len(rows) == (12 * 50 if phase != "bwt_checkpoint" else
                                 sum(job["steps"] for job in read(a/"plan.json")
                                     if Path(job["path"]).parent.name == mode))
            archive_eval(rows,model,mode,"advantage",phase)
            for row in rows:
                exact_reward(row)
                g[row["condition_id"]][phase].append(row)
        for cid,group in g.items():
            sample=group["train_final"][0]
            choices=source(a/mode/sample["arm"]/"conditions"/cid/"selections.json")
            record(model,mode,"advantage",sample["seed"],cid,*(group[p] for p in phases),choices,initial,candidates)

partial = []
for f in sorted((aud/"thinking/seed44/stage3").glob("*/result.json")):
    row = source(f)
    exact_reward(row)
    assert row["phase"] == "train_final"
    partial.append(row)
archive_eval(partial,"qwen38","thinking","AUD_partial_seed44","train_final")
assert len(scored)==192
ids = {(r["model"],r["mode"],r["variant"],r["phase"],r["seed"],r["condition_id"],r["task_id"],r["arm"],
        r["source_repeat"],r["evaluation_repeat"]) for r in score_rows}
assert len(ids)==len(score_rows)

def csv_file(name, rows):
    with (out/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

csv_file("task_scores.csv",score_rows)
csv_file("context_seed_metrics.csv",scored)
groups=defaultdict(list)
for r in scored:
    groups[tuple(r[k] for k in ("model","mode","variant","gt","selection"))].append(r)
fields=("learned","train_pct","train_tgc","heldout_pct","heldout_tgc","bwt_pct","bwt_tgc","independent_pct","independent_tgc")
means=[]
for key, group in groups.items():
    assert len(group)==(2 if key[2]=="AUD" else 3)
    means.append(dict(zip(("model","mode","variant","gt","selection"),key),n_seeds=len(group),
                      **{k:mean(r[k] for r in group) for k in fields}))
csv_file("context_mean_metrics.csv",means)
(out/"baseline_and_reflection.json").write_text(json.dumps(baselines,indent=2)+"\n")
(out/"protocol_settings.json").write_text(json.dumps(settings,indent=2)+"\n")
(out/"source_sha256.json").write_text(json.dumps(provenance,indent=2)+"\n")
(out/"audit.json").write_text(json.dumps({"score_rows":len(score_rows),"contexts_with_finished_evaluation":len(scored),
                                         "aud_seed44_saved_train":len(partial),"snapshot_at":progress_at},indent=2)+"\n")
def fmt(pair):
    return f"{pair[0]:.2f} / {pair[1]:.2f}"

def pair(row, stem):
    return fmt([row[f"{stem}_pct"],row[f"{stem}_tgc"]])

def key_for(r):
    return (r["mode"]!="thinking",r["gt"]!="without_gt",("random","best","worst","best_advantage","worst_advantage").index(r["selection"]))

for model in models:
    lines=["완료된 결과의 점수는 PCT / TGC 순이며 단위는 %입니다. 모든 누적 평가는 고정된 Train 50개와 Held-out 50개를 사용했습니다.",
           "", "## 학습 전과 독립 Reflection 평가", "",
           "| 모드 | 학습 전 | w/o GT 업데이트 | w/ GT 업데이트 | Pass@5 | Pass@3 | Pass-and-3 | Pass-and-5 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode,b in baselines[model].items():
        lines.append(f"| {mode} | {fmt(b['initial'])} | {fmt(b['without_gt'])} | {fmt(b['with_gt'])} | "+
                     " | ".join(f"{b[k]:.2f}%" for k in ("pass@5","pass@3","pass-and-3","pass-and-5"))+" |")
    lines += ["","학습 전 250회. 독립 업데이트는 GT 조건별 750회. 3회 Pass 지표는 5회 중 가능한 모든 3회 조합의 평균입니다.",
              "", "## ADD-only 누적 최종 결과", "",
              "| 모드 | GT | 선택 | 학습 Task | 비축적 기대 | Train | Held-out | BWT (pp) |",
              "| --- | --- | --- | ---: | --- | --- | --- | --- |"]
    for r in sorted((r for r in means if r["model"]==model and r["variant"]!="AUD"), key=key_for):
        lines.append(f"| {r['mode']} | {'O' if r['gt']=='with_gt' else 'X'} | {r['selection']} | {r['learned']:.0f} | "+
                     " | ".join(pair(r,f) for f in ("independent","train","heldout","bwt"))+" |")
    lines += ["", "각 행은 seed 42·43·44 평균입니다. 부호별 누적은 Advantage가 모두 0인 Task를 제외했습니다. "
                 "비축적 기대값은 선택한 후보의 독립 3회 평균이고, 건너뛴 Task는 빈 Context의 5회 평균을 사용했습니다. "
                 "BWT는 실제 학습한 Task에서 최종 Context와 해당 Task 직후 체크포인트의 점수 차이입니다. "
                 "비축적 점수는 후보 선정에도 쓰였으므로 누적 효과 해석에는 선택 편향이 포함됩니다."]
    if model=="qwen38":
        lines += ["","## Thinking AUD 중단 시점과 대응 비교","",
                  f"seed42·43의 학습 및 평가 1,800건을 마쳤습니다. seed44 학습은 완료했고 Train 결과 {len(partial)}/300건을 저장한 상태에서 중단했습니다. Non-thinking AUD는 대기 상태입니다.",
                  "", "| GT | 선택 | Train A → AUD | Held-out A → AUD | BWT A → AUD (pp) |",
                  "| --- | --- | --- | --- | --- |"]
        for r in sorted((r for r in means if r["variant"]=="AUD"),key=key_for):
            match=[x for x in scored if x["model"]==model and x["mode"]=="thinking" and x["variant"]=="ADD"
                   and x["seed"] in (42,43) and x["gt"]==r["gt"] and x["selection"]==r["selection"]]
            a={k:mean(x[k] for x in match) for k in fields}
            lines.append(f"| {r['gt']} | {r['selection']} | "+
                         " | ".join(f"{pair(a,f)} → {pair(r,f)}" for f in ("train","heldout","bwt"))+" |")
        lines += ["", "A와 AUD는 완료된 동일 seed 42·43만 비교했습니다. seed44의 부분 결과는 CSV에 보존했습니다. "
                      "AUD의 비축적 기대값은 공통으로 사용한 ADD-only 후보 평가에서 왔습니다."]
    elif model=="qwen36":
        lines += ["","## 실행 완료","",
                  "두 모드 독립 평가, 전체 누적 Context 36개, Advantage Context 24개의 평가가 완료됐습니다. "
                  "Advantage 평가는 Train 1,200회, BWT 912회, Held-out 1,200회입니다. "
                  "Thinking seed42 Held-out 타임아웃 202건은 서버 재기동 후 재평가를 완료했습니다."]
    else:
        lines += ["", "## 실행 완료", "",
                  "두 모드 독립 평가와 전체 누적 Context 36개, Advantage Context 24개의 평가가 완료됐습니다. "
                  "Advantage 평가는 Train 1,200회, BWT 1,170회, Held-out 1,200회입니다."]
    (out/f"{model}.md").write_text("\n".join(lines)+"\n")

with ZipFile(out/"ace_appworld_qwen_final_20261121.zip","w",ZIP_DEFLATED) as z:
    for name in ("task_scores.csv","context_seed_metrics.csv","context_mean_metrics.csv","baseline_and_reflection.json",
                 "protocol_settings.json","source_sha256.json","audit.json","qwen38.md","qwen36.md","qwen35.md","export.py"):
        z.write(out/name,name)
print((out/"audit.json").read_text())
print("archive size",(out/"ace_appworld_qwen_final_20261121.zip").stat().st_size)
