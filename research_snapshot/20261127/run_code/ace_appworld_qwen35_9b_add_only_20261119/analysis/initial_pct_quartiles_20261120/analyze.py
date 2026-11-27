# SPDX-License-Identifier: Apache-2.0
"""Compute task-level PCT changes by initial-score rank from the consolidated archive."""

import csv
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.ace.plot_worst_reflection_repeats import RUNS as Q38

OUT = Path(__file__).resolve().parent
ARCHIVE = OUT.parent / "notion_consolidation_20261120"
ROOT = Path.cwd() / "outputs"
Q36 = ROOT / "ace_appworld_qwen36_add_only_20261118"
FULL = {"qwen38": Q38, "qwen36": {m: Q36 / m / "accumulation" for m in Q38}}
ADV = {"qwen38": ROOT / "ace_appworld_advantage_add_20261118", "qwen36": Q36 / "advantage"}
AUD = ROOT / "ace_appworld_aud_accumulation_qwen38_20261117" / "thinking"
LABELS = {"qwen38": "Qwen3.8-27B FP8", "qwen36": "Qwen3.6-35B-A3B FP8"}
METHODS = ("random", "best", "worst", "best_advantage", "worst_advantage")
ARMS = ("without_gt", "with_gt")

def read(path):
    return json.loads(path.read_text())

def mean_reward(rows):
    return mean(r for r in rows)

def group_append(data, key, item):
    data[key].append(item)

initial, candidate, accum = (defaultdict(list) for _ in range(3))
with (ARCHIVE / "task_scores.csv").open() as f:
    for r in csv.DictReader(f):
        model, mode, task = r["model"], r["mode"], r["task_id"]
        if model not in FULL:
            continue
        if r["budget_failure"]:
            assert float(r["pct"]) == 0
            pct = Fraction(0)
        else:
            pct = Fraction(int(r["passed_tests"]), int(r["total_tests"]))
            assert abs(float(pct) - float(r["pct"])) < 1e-12
        if r["phase"] == "initial":
            group_append(initial, (model, mode, task), pct)
        if r["phase"] == "independent":
            group_append(candidate, (model, mode, r["arm"], task, int(r["source_repeat"])), pct)
        if r["phase"] == "train_final" and r["variant"] != "AUD_partial_seed44":
            key = model, mode, r["variant"], int(r["seed"]), r["arm"], r["condition_id"]
            group_append(accum, key, (task, pct))
assert len(initial) == 200 and all(len(r) == 5 for r in initial.values())
assert len(candidate) == 2000 and all(len(r) == 3 for r in candidate.values())

baseline = {key: mean_reward(rows) for key, rows in initial.items()}
reward = {key: mean_reward(rows) for key, rows in candidate.items()}
ranks, membership, quartiles, split_ties = [], {}, [], []
for model in FULL:
    for mode in Q38:
        ordered = sorted((b, task) for (m, v, task), b in baseline.items() if (m, v) == (model, mode))
        assert len(ordered) == 50
        breaks = (0, 13, 26, 38, 50)
        for i, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:], strict=True), 1):
            band = f"Q{i}"
            scores = [b for b, _ in ordered[lo:hi]]
            quartiles.append(dict(model=model, mode=mode, quartile=band, tasks=hi-lo,
                                  initial_pct=100*float(mean(scores)),
                                  min_initial_pct=100*float(scores[0]), max_initial_pct=100*float(scores[-1]),
                                  initial_perfect=sum(s==1 for s in scores)))
            if hi < 50 and ordered[hi-1][0] == ordered[hi][0]:
                split_ties.append(dict(model=model, mode=mode, boundary=f"Q{i}/Q{i+1}", pct=100*float(scores[-1])))
            for n, (b, t) in enumerate(ordered[lo:hi], lo + 1):
                membership[model,mode,t] = band
                ranks.append(dict(model=model, mode=mode, task=t, quartile=band, rank=n, initial_pct=100*float(b)))

task_rows = []
source_sha = {"task_scores.csv": hashlib.sha256((ARCHIVE/"task_scores.csv").read_bytes()).hexdigest()}
methods_by_task = defaultdict(list)
for (model,mode,variant,seed,arm,condition), scores in accum.items():
    assert len(scores)==50 and len({t for t,_ in scores})==50
    cohort = FULL[model][mode] if variant == "ADD" else AUD if variant=="AUD" else ADV[model]/mode/arm
    path = cohort/"conditions"/condition/"selections.json"
    source_sha[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    chosen = {r["candidate"]["task_id"]: r["candidate"]["source_repeat"] for r in read(path)}
    strategy = condition.removeprefix(f"s{seed}_{arm}_")
    for task,pct in scores:
        b = baseline[model,mode,task]
        i = reward[model,mode,arm,task,chosen[task]] if task in chosen else b
        row=dict(model=model,mode=mode,arm=arm,variant=variant,strategy=strategy,seed=seed,task=task,
                 quartile=membership[model,mode,task],learned=int(task in chosen),initial_pct=100*float(b),
                 independent_pct=100*float(i),accumulated_pct=100*float(pct),
                 independent_gain_pp=100*float(i-b),accumulated_gain_pp=100*float(pct-b),
                 added_gain_pp=100*float(pct-i))
        task_rows.append(row)
        methods_by_task[model,mode,arm,variant,strategy,row["quartile"]].append(row)
assert len(task_rows)==6600

combined = []
for key, rows in sorted(methods_by_task.items()):
    seeds = sorted({r["seed"] for r in rows})
    n = len({r["task"] for r in rows})
    assert len(rows) == n * len(seeds)
    combined.append(dict(zip(("model","mode","arm","variant","strategy","quartile"),key),tasks=n,seeds=len(seeds),
                         **{k:mean(r[k] for r in rows) for k in
                            ("learned","initial_pct","independent_pct","accumulated_pct","independent_gain_pp",
                             "accumulated_gain_pp","added_gain_pp")}))

candidate_gains = defaultdict(list)
for (model,mode,arm,t,idx), r in reward.items():
    band = membership[model,mode,t]
    candidate_gains[model,mode,arm,band].append(100*float(r-baseline[model,mode,t]))
candidate_rows = [dict(zip(("model","mode","arm","quartile"),key),tasks=len(rows)//5,candidates=len(rows),
                       independent_gain_pp=mean(rows)) for key,rows in sorted(candidate_gains.items())]

# Matched AUD comparison uses the same task, seeds and candidate selection.
matched = []
a_by_task = {(r["mode"],r["arm"],r["strategy"],r["seed"],r["task"]):r for r in task_rows
             if r["model"]=="qwen38" and r["variant"]=="ADD"}
diffs=defaultdict(list)
for r in task_rows:
    if r["variant"]=="AUD":
        a=a_by_task[r["mode"],r["arm"],r["strategy"],r["seed"],r["task"]]
        assert r["independent_pct"]==a["independent_pct"]
        diffs[r["arm"],r["strategy"],r["quartile"]].append(
            (a["accumulated_gain_pp"],r["accumulated_gain_pp"]))
for (arm,strategy,band), rows in sorted(diffs.items()):
    a,b=map(mean,zip(*rows))
    matched.append(dict(arm=arm,strategy=strategy,quartile=band,tasks=len(rows)//2,seeds=2,
                        add_gain_pp=a,aud_gain_pp=b,aud_minus_add_pp=b-a))

def write_csv(name, rows):
    with (OUT/name).open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

for name,rows in (("task_quartiles.csv",ranks),("quartile_baselines.csv",quartiles),
                  ("all_reflection_candidates.csv",candidate_rows),("task_seed_gains.csv",task_rows),
                  ("quartile_gains.csv",combined),("aud_matched_quartiles.csv",matched)):
    write_csv(name,rows)
(OUT/"source_sha256.json").write_text(json.dumps(source_sha,indent=2)+"\n")
(OUT/"boundary_ties.json").write_text(json.dumps(split_ties,indent=2)+"\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bands = ("Q1","Q2","Q3","Q4")
names={"random":"Random","best":"Best","worst":"Worst",
       "best_advantage":"Best advantage","worst_advantage":"Worst advantage"}
colors=dict(zip(METHODS,("#64748b","#1765a8","#bf5144","#1b8b71","#ac7208")))
for model,label in LABELS.items():
    fig, axes = plt.subplots(4,2,figsize=(13,12),sharex=True,sharey=True,layout="constrained")
    for i,(mode,arm) in enumerate((m,a) for m in Q38 for a in ARMS):
        sub=[r for r in combined if r["model"]==model and r["mode"]==mode and r["arm"]==arm and r["variant"]!="AUD"]
        for j,(key,kind) in enumerate((("independent_gain_pp","One update"),("accumulated_gain_pp","Accumulated"))):
            ax=axes[i,j]
            for method in METHODS:
                scores={r["quartile"]:r[key] for r in sub if r["strategy"]==method}
                ax.plot(bands,[scores[b] for b in bands],marker="o",
                        linestyle="--" if method.endswith("advantage") else "-",color=colors[method],label=names[method])
            if j==0:
                scores={r["quartile"]:r["independent_gain_pp"] for r in candidate_rows
                        if r["model"]==model and r["mode"]==mode and r["arm"]==arm}
                ax.plot(bands,[scores[b] for b in bands],marker=".",color="#161616",linestyle=":",
                        label="All five candidates")
            ax.axhline(0,color="#999999",linewidth=.8)
            ax.grid(axis="y",alpha=.18)
            ax.set_title(f"{mode.title()} / {'GT' if arm=='with_gt' else 'No GT'} — {kind}",fontsize=11)
            if j==0: ax.set_ylabel("PCT gain (pp)")
            if i==3: ax.set_xlabel("Initial task PCT quartile (low → high)")
    axes[0,0].legend(loc="best",fontsize=8,ncol=2)
    axes[0,1].legend(loc="best",fontsize=8,ncol=2)
    fig.suptitle(f"{label} | PCT change from the same task's initial five-run mean",fontsize=14)
    fig.savefig(OUT/f"{model}_quartile_gains.png",dpi=165)
    plt.close(fig)

def get(rows, band, field):
    return next(r[field] for r in rows if r["quartile"]==band)

def format_quartiles(rows, field):
    return " | ".join(f"{get(rows,b,field):+.2f}" for b in bands)

for model,label in LABELS.items():
    lines=[f"# {label}","",
           "| 모드 | Q1 학습 전 | Q2 학습 전 | Q3 학습 전 | Q4 학습 전 |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for mode in Q38:
        p=[r for r in quartiles if (r["model"],r["mode"])==(model,mode)]
        lines.append(f"| {mode} | "+" | ".join(f"{r['initial_pct']:.2f}% ({r['tasks']}개)" for r in p)+" |")
    for mode in Q38:
        for arm in ARMS:
            lines+=["",f"## {mode} · {'w/ GT' if arm=='with_gt' else 'w/o GT'}","",
                    "| 설정 | Q1 단일 / 누적 | Q2 단일 / 누적 | Q3 단일 / 누적 | Q4 단일 / 누적 |",
                    "| --- | ---: | ---: | ---: | ---: |"]
            cand=[r for r in candidate_rows if (r["model"],r["mode"],r["arm"])==(model,mode,arm)]
            lines.append("| 후보 5개 평균 | "+" | ".join(f"{get(cand,b,'independent_gain_pp'):+.2f} / —" for b in bands)+" |")
            for method in METHODS:
                rows=[r for r in combined if (r["model"],r["mode"],r["arm"],r["strategy"])==(model,mode,arm,method)
                      and r["variant"]!="AUD"]
                cells=[f"{get(rows,b,'independent_gain_pp'):+.2f} / {get(rows,b,'accumulated_gain_pp'):+.2f}" for b in bands]
                lines.append(f"| {names[method]} | "+" | ".join(cells)+" |")
    if model=="qwen38":
        lines+=["","## AUD와 같은 seed의 ADD-only","",
                "seed42·43에서 초기 PCT 대비 누적 변화량입니다. 각 셀은 ADD-only / AUD이며 단위는 %p입니다.","",
                "| GT | 선택 | Q1 | Q2 | Q3 | Q4 |","| --- | --- | ---: | ---: | ---: | ---: |"]
        for arm in ARMS:
            for method in METHODS[:3]:
                rows=[r for r in matched if r["arm"]==arm and r["strategy"]==method]
                cells=[f"{get(rows,b,'add_gain_pp'):+.2f} / {get(rows,b,'aud_gain_pp'):+.2f}" for b in bands]
                lines.append(f"| {'O' if arm=='with_gt' else 'X'} | {names[method]} | "+" | ".join(cells)+" |")
    (OUT/f"{model}.md").write_text("\n".join(lines)+"\n")

print("Task×seed×condition rows",len(task_rows),"quartile rows",len(combined))
for m in FULL:
    print(m)
    for q in bands:
        single=[r["independent_gain_pp"] for r in candidate_rows if r["model"]==m and r["quartile"]==q]
        print(q, "all candidates",round(mean(single),2), *(
            (v, round(mean(r["accumulated_gain_pp"] for r in combined if
                           r["model"]==m and r["quartile"]==q and r["variant"]==v),2))
            for v in ("ADD","advantage")))
print("split ties",split_ties)
