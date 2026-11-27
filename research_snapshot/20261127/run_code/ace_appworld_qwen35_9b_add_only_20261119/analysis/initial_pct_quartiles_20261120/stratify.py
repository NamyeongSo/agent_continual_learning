# SPDX-License-Identifier: Apache-2.0
"""Compare Reflection arms and accumulation selectors within initial PCT bands."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

P = Path(__file__).resolve().parent
MODELS = {"qwen38": "Qwen3.8", "qwen36": "Qwen3.6"}
ARMS = {"without_gt": "w/o GT", "with_gt": "w/ GT"}
METHODS = ("random", "best", "worst", "best_advantage", "worst_advantage")
BANDS = ("Q1", "Q2", "Q3", "Q4")
MODES = ("thinking", "nonthinking")


def read(name):
    with (P / name).open() as f:
        return list(csv.DictReader(f))


def save(name, rows):
    with (P / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


summary = {
    tuple(r[k] for k in ("model", "mode", "arm", "strategy", "quartile")): r
    for r in read("quartile_gains.csv") if r["variant"] != "AUD"
}
candidates = {
    tuple(r[k] for k in ("model", "mode", "arm", "quartile")): float(r["independent_gain_pp"])
    for r in read("all_reflection_candidates.csv")
}
assert len(summary) == 160 and len(candidates) == 32

# A best or worst tie can change the selected R index, but its independent PCT is identical.
by_task = defaultdict(list)
for r in read("task_seed_gains.csv"):
    if r["variant"] == "ADD" and r["strategy"] in ("best", "worst"):
        key = tuple(r[k] for k in ("model", "mode", "arm", "quartile", "task", "strategy"))
        by_task[key].append(float(r["independent_gain_pp"]))
task_counts = defaultdict(Counter)
for key, points in by_task.items():
    assert len(points) == 3 and max(points) - min(points) < 1e-10
    model, mode, arm, q, task, strategy = key
    sign = "zero" if abs(points[0]) <= 1e-10 else "positive" if points[0] > 0 else "negative"
    task_counts[model, mode, arm, q, strategy][sign] += 1

def avg(m, a, s, q, key, modes=MODES):
    return mean(float(summary[m,mode,a,s,q][key]) for mode in modes)


pool, gt_effect, best_worst = [], [], []
for m in MODELS:
    for a in ARMS:
        for q in BANDS:
            b = avg(m,a,"best",q,"independent_gain_pp")
            w = avg(m,a,"worst",q,"independent_gain_pp")
            p = avg(m,a,"best",q,"accumulated_gain_pp")
            z = avg(m,a,"worst",q,"accumulated_gain_pp")
            counts = {f"{method}_{sign}": sum(task_counts[m,mode,a,q,method][sign] for mode in MODES)
                      for method in ("best","worst") for sign in ("positive","zero","negative")}
            best_worst.append(dict(model=m, arm=a, quartile=q, model_mode_task_count=sum(counts[k] for k in
                              ("best_positive","best_zero","best_negative")), best_independent_gain_pp=b,
                              worst_independent_gain_pp=w, independent_best_minus_worst_pp=b-w,
                              accumulated_best_minus_worst_pp=p-z, **counts))
            for s in METHODS:
                i = avg(m,a,s,q,"independent_gain_pp")
                n = avg(m,a,s,q,"accumulated_gain_pp")
                pool.append(dict(model=m, arm=a, strategy=s, quartile=q, independent_gain_pp=i,
                                 accumulated_gain_pp=n, accumulated_minus_independent_pp=n-i,
                                 learned_fraction=avg(m,a,s,q,"learned")))
    for q in BANDS:
        for mode_set in (MODES, ("thinking",), ("nonthinking",)):
            mode_label = "equal_mode_mean" if len(mode_set)==2 else mode_set[0]
            no = mean(candidates[m,mode,"without_gt",q] for mode in mode_set)
            yes = mean(candidates[m,mode,"with_gt",q] for mode in mode_set)
            gt_effect.append(dict(model=m,mode=mode_label,strategy="all_candidates",quartile=q,
                                  independent_gt_minus_no_gt_pp=yes-no,accumulated_gt_minus_no_gt_pp=None))
            for s in METHODS:
                d = [avg(m,"with_gt",s,q,k,mode_set)-avg(m,"without_gt",s,q,k,mode_set)
                     for k in ("independent_gain_pp","accumulated_gain_pp")]
                gt_effect.append(dict(model=m,mode=mode_label,strategy=s,quartile=q,
                                      independent_gt_minus_no_gt_pp=d[0],accumulated_gt_minus_no_gt_pp=d[1]))

save("strategy_by_gt.csv",pool)
save("gt_paired_differences.csv",gt_effect)
save("independent_best_worst.csv",best_worst)

def cells(rows, key):
    return " | ".join(f"{next(r[key] for r in rows if r['quartile']==q):+.2f}" for q in BANDS)

lines=[
    "초기 PCT 사분위의 Task 배정은 앞선 분석을 사용했습니다. 아래 표는 같은 모델 안에서 Thinking과 Non-thinking을 동일하게 가중했습니다. "
    "모드별 수치는 기존 상세 표와 CSV에 있습니다.",
    "", "### Reflection 타입의 차이", "",
    "같은 모델·모드·Task 구간에서 w/ GT − w/o GT를 계산했습니다. 누적은 세 seed 평균입니다. 단위는 %p입니다.", "",
    "| 모델 | 방식 | Q1 | Q2 | Q3 | Q4 |", "| --- | --- | ---: | ---: | ---: | ---: |",
]
for m,label in MODELS.items():
    for s in ("all_candidates",*METHODS):
        rs=[r for r in gt_effect if r["model"]==m and r["mode"]=="equal_mode_mean" and r["strategy"]==s]
        if s=="all_candidates":
            lines.append(f"| {label} | 단일 후보 5개 평균 | {cells(rs,'independent_gt_minus_no_gt_pp')} |")
        else:
            lines.append(f"| {label} | 누적 {s} | {cells(rs,'accumulated_gt_minus_no_gt_pp')} |")

lines += ["", "### 누적 전략별 향상", "",
          "ADD 계열 결과입니다. 각 셀은 같은 Task의 학습 전 PCT 대비 최종 Context의 변화량(%p)입니다. "
          "Advantage 실험에서 학습을 건너뛴 Task도 최종 평가의 구간 평균에 포함합니다.", "",
          "| 모델 | GT | 누적 전략 | Q1 | Q2 | Q3 | Q4 |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
for m,label in MODELS.items():
    for a,title in ARMS.items():
        for s in METHODS:
            r=[x for x in pool if (x["model"],x["arm"],x["strategy"])==(m,a,s)]
            lines.append(f"| {label} | {title} | {s} | {cells(r,'accumulated_gain_pp')} |")

lines += ["", "### 비축적 best와 worst", "",
          "각 Task의 후보 다섯 개 중 독립 3회 평균 PCT가 최고·최저였던 후보의 변화량입니다. "
          "각 셀은 best / worst이며 단위는 %p입니다. 같은 Task·모드의 동점 후보는 동일한 PCT를 갖습니다.", "",
          "| 모델 | GT | Q1 | Q2 | Q3 | Q4 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
for m,label in MODELS.items():
    for a,title in ARMS.items():
        g={r["quartile"]:r for r in best_worst if (r["model"],r["arm"])==(m,a)}
        c=" | ".join(f"{g[q]['best_independent_gain_pp']:+.2f} / {g[q]['worst_independent_gain_pp']:+.2f}" for q in BANDS)
        lines.append(f"| {label} | {title} | {c} |")

lines += ["", "Best·worst는 보고한 독립 평가로 선정했습니다. 별도 재평가에서 기대할 수 있는 격차를 추정하려면 새 반복 평가가 필요합니다. "
          "누적 쪽은 별도 rollout이므로 격차의 변화에는 누적 자체와 선정에 따른 평균 회귀가 함께 들어갑니다. "
          "Q4의 학습 전 평균은 100%이므로 단일 적용도 개선할 수 없습니다."]

with (P/"aud_matched_quartiles.csv").open(newline="") as f:
    aud_deltas = [{**r, "aud_minus_add_pp": float(r["aud_minus_add_pp"])} for r in csv.DictReader(f)]
lines += ["", "### AUD와 ADD", "",
          "Qwen3.8 Thinking의 Train 평가가 완료된 seed 42·43을 맞췄습니다. 각 셀은 누적 후 AUD − ADD PCT(%p)입니다.", "",
          "| GT | 후보 선택 | Q1 | Q2 | Q3 | Q4 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
for a,title in ARMS.items():
    for s in ("random","best","worst"):
        r=[x for x in aud_deltas if (x["arm"],x["strategy"])==(a,s)]
        lines.append(f"| {title} | {s} | {cells(r,'aud_minus_add_pp')} |")
(P/"stratified_followup.md").write_text("\n".join(lines)+"\n")
with ZipFile(P/"ace_quartile_stratified_followup.zip","w",compression=ZIP_DEFLATED) as z:
    for name in ("strategy_by_gt.csv","gt_paired_differences.csv","independent_best_worst.csv",
                 "aud_matched_quartiles.csv","stratified_followup.md","stratify.py"):
        z.write(P/name,name)

for r in best_worst:
    if r["quartile"] == "Q1":
        print(r)
