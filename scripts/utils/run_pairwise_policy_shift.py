# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Legacy ΔCP study: independent temperature-zero draws and all-checkpoint common histories."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import itertools
import json
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.utils import run_checkpoint_policy_shift as common  # noqa: E402
from scripts.utils.policy_shift_metrics import canonical, digest, load_json, score_actions, write_json  # noqa: E402

METRICS = ("loose_accuracy", "strict_accuracy", "loose_f1", "strict_f1", "accuracy")


def now():
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).isoformat()


def build_jobs(db, plan):
    settings = canonical({k: plan[k] for k in ("model", "temperature", "input_tokens", "output_tokens")})

    def job(anchor, prompt, sample):
        key = digest(canonical([anchor, prompt, sample, settings]))
        db.execute("INSERT OR IGNORE INTO jobs(id,anchor,prompt,sample) VALUES(?,?,?,?)",
                   (key, anchor, prompt, sample))
        return key

    for policy in db.execute("SELECT * FROM policies").fetchall():
        root = db.execute("SELECT prompt FROM policies WHERE stream=? AND point=0", (policy["stream"],)).fetchone()[0]
        anchors = db.execute("SELECT id FROM anchors WHERE benchmark=? AND seed=?",
                             (policy["benchmark"], policy["seed"])).fetchall()
        for anchor in anchors:
            reference = job(anchor[0], root, f"cp:{policy['stream']}:0")
            if policy["point"] == 0:
                prediction = job(anchor[0], root, f"repeat:{policy['stream']}:0")
            elif policy["prompt"] == root:
                prediction = reference
            else:
                prediction = job(anchor[0], policy["prompt"], f"policy:{policy['stream']}")
            db.execute("INSERT OR REPLACE INTO bindings VALUES(?,?,?,?,?)",
                       (policy["stream"], policy["point"], anchor[0], prediction, reference))
        db.commit()


def deduplicate_jobs(db):
    """Reuse one response for identical policy prompts within a stream/history."""
    policies = db.execute("SELECT * FROM policies ORDER BY stream,point").fetchall()
    by_stream = defaultdict(list)
    for policy in policies:
        by_stream[policy["stream"]].append(policy)

    rewired_bindings = 0
    root_alias_bindings = 0
    shared_policy_bindings = 0
    for stream, rows in by_stream.items():
        root = next(row for row in rows if row["point"] == 0)
        prompt_groups = defaultdict(list)
        for row in rows:
            if row["point"]:
                prompt_groups[row["prompt"]].append(row["point"])
        anchors = db.execute(
            "SELECT id FROM anchors WHERE benchmark=? AND seed=?",
            (root["benchmark"], root["seed"]),
        ).fetchall()
        for anchor_row in anchors:
            anchor = anchor_row[0]
            root_binding = db.execute(
                "SELECT reference FROM bindings WHERE stream=? AND point=0 AND anchor=?",
                (stream, anchor),
            ).fetchone()
            if root_binding is None:
                raise ValueError(f"Missing CP0 binding for {stream}/{anchor}")
            root_reference = root_binding[0]
            for prompt, points in prompt_groups.items():
                if prompt == root["prompt"]:
                    target = root_reference
                    root_alias_bindings += len(points)
                    shared_policy_bindings += len(points)
                else:
                    if len(points) > 1:
                        shared_policy_bindings += len(points)
                    placeholders = ",".join("?" for _ in points)
                    candidates = db.execute(
                        f"""SELECT DISTINCT j.id,j.status FROM bindings b JOIN jobs j ON j.id=b.job
                        WHERE b.stream=? AND b.anchor=? AND b.point IN ({placeholders})
                        ORDER BY CASE j.status WHEN 'completed' THEN 0 WHEN 'pending' THEN 1
                                 WHEN 'running' THEN 2 WHEN 'error' THEN 3 ELSE 4 END,j.id""",
                        (stream, anchor, *points),
                    ).fetchall()
                    if not candidates:
                        raise ValueError(f"Missing prediction job for {stream}/{anchor}/{points}")
                    target = candidates[0]["id"]
                    if candidates[0]["status"] == "running":
                        db.execute("UPDATE jobs SET status='pending' WHERE id=?", (target,))
                for point in points:
                    old = db.execute(
                        "SELECT job FROM bindings WHERE stream=? AND point=? AND anchor=?",
                        (stream, point, anchor),
                    ).fetchone()[0]
                    if old != target:
                        db.execute(
                            "UPDATE bindings SET job=? WHERE stream=? AND point=? AND anchor=?",
                            (target, stream, point, anchor),
                        )
                        rewired_bindings += 1
        db.commit()

    db.execute("CREATE INDEX IF NOT EXISTS bindings_job_idx ON bindings(job)")
    db.execute("CREATE INDEX IF NOT EXISTS bindings_reference_idx ON bindings(reference)")
    orphaned_rows = db.execute("""SELECT j.id,j.status FROM jobs j
        LEFT JOIN bindings bj ON bj.job=j.id LEFT JOIN bindings br ON br.reference=j.id
        WHERE bj.job IS NULL AND br.reference IS NULL AND j.status!='deduplicated'""").fetchall()
    orphaned = [row[0] for row in orphaned_rows]
    orphaned_completed = sum(row[1] == "completed" for row in orphaned_rows)
    for start in range(0, len(orphaned), 500):
        batch = orphaned[start:start + 500]
        placeholders = ",".join("?" for _ in batch)
        db.execute(f"UPDATE jobs SET status='deduplicated' WHERE id IN ({placeholders})", batch)
    if orphaned:
        db.commit()
    counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
    return {
        "rewired_bindings": rewired_bindings,
        "root_alias_bindings": root_alias_bindings,
        "shared_policy_bindings": shared_policy_bindings,
        "deduplicated_jobs": len(orphaned),
        "completed_duplicate_responses_preserved": orphaned_completed,
        "effective_requests": sum(value for key, value in counts.items() if key != "deduplicated"),
        "counts_after_deduplication": counts,
    }


def prepare_fork(output, source):
    """Reuse immutable inputs/policies, never responses produced at another temperature."""
    from scripts.utils.schedule_checkpoint_policy_shift import api

    output.mkdir(parents=True, exist_ok=False)
    previous = load_json(source / "plan.json")
    prepared = load_json(source / "prepared.json")
    for path, sha in prepared["source_hashes"].items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError(f"Original policy/history source changed: {path}")
    source_db = common.own_db(source)
    db = common.own_db(output)
    for row in source_db.execute("SELECT * FROM anchors"):
        data = Path(row["file"]).read_bytes()
        if digest(canonical(json.loads(data))) != row["id"]:
            raise ValueError("Original fixed history changed")
        path = output / "histories" / f"{row['id']}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        db.execute("INSERT INTO anchors VALUES(?,?,?,?,?)",
                   (row["id"], row["benchmark"], row["seed"], row["task"], str(path)))
    for row in source_db.execute("SELECT * FROM prompts"):
        if digest(row["content"]) != row["sha"]:
            raise ValueError("Original frozen prompt changed")
        db.execute("INSERT INTO prompts VALUES(?,?)", tuple(row))
    for row in source_db.execute("SELECT * FROM policies"):
        db.execute("INSERT INTO policies VALUES(?,?,?,?,?,?)", tuple(row))
    db.commit()
    source_db.close()
    parser = output / "frozen/terminus_json_plain_parser.py"
    parser.parent.mkdir()
    shutil.copy2(previous["terminal_parser_path"], parser)
    if digest(parser.read_bytes()) != previous["terminal_parser_sha256"]:
        raise ValueError("Official native action parser changed")
    plan = {k:v for k,v in previous.items() if k not in {"journal_id","command","analysis_corrections"}}
    plan.update(output=str(output), created_at=now(), temperature=0.0, workers=128,
                comparison="all_within_seed_checkpoint_pairs_by_distance", checkpoint_mode="all",
                request_sampling="One draw per distinct stream/prompt/history; exact duplicate prompts share a response; CP0 repeat remains independent",
                common_history_scope="All available checkpoints within each method/benchmark/seed",
                aggregation="Mean histories within task, equal tasks per checkpoint pair; then mean and sample SD "
                            "across within-seed checkpoint pairs from all seeds at each ΔCP",
                sampling_limitations="Temperature0; independent CP0 repeat diagnoses serving nondeterminism",
                reference="All CPi–CPj pairs on identical histories, grouped by ΔCP=j−i",
                source_history_bank=str(source), terminal_parser_path=str(parser),
                metrics_file="distance_metrics.csv",
                analysis_caption="Temperature0 · all-checkpoint common histories · mean 1−accuracy/F1 versus ΔCP",
                legacy_differences="Current 36-stream cohort and saved shared histories; unchanged prompts share responses; global input100k/output2048 caps retained")
    build_jobs(db, plan)
    prepared.update(source_history_bank=str(source), temperature=0.0,
                    distinct_requests=db.execute("SELECT count(*) FROM jobs").fetchone()[0],
                    effective_requests=db.execute("SELECT count(*) FROM jobs").fetchone()[0],
                    deduplicated_request_count=0,
                    policy_instances=db.execute("SELECT count(*) FROM policies").fetchone()[0],
                    comparison=plan["comparison"], request_sampling=plan["request_sampling"],
                    reused_response_count=0, created_at=now())
    prepared["protocol_sha256"] = digest(canonical({
        key: plan[key]
        for key in ("model", "temperature", "input_tokens", "output_tokens", "checkpoint_mode",
                    "selected_stream_ids", "terminal_parser_sha256")
    }))
    db.close()
    snapshot = output / "code_snapshot"
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("policy_shift_metrics.py","run_checkpoint_policy_shift.py",
                 "schedule_checkpoint_policy_shift.py","run_pairwise_policy_shift.py"):
        target = snapshot / "scripts/utils" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts/utils" / name,target)
    write_json(output / "code_hashes.json", {str(p.relative_to(snapshot)):digest(p.read_bytes())
                                              for p in snapshot.rglob("*") if p.is_file()})
    plan["command"] = [sys.executable,"-u",str(snapshot / "scripts/utils/run_pairwise_policy_shift.py"),
                       "--output",str(output)]
    record = api("/api/experiments", {
        "title":"Qwen3.6 · ACE/GEPA/SkillOpt/PEACE · ΔCP policy shift · temperature0 · queue128",
        "purpose":"이전 방식으로 모든 checkpoint쌍의 행동 차이를 동일history에서 측정하고 ΔCP별 평균/SD 집계",
        "status":"planned","method":"ACE AUD / GEPA / SkillOpt / PEACE AUD · pairwise ΔCP",
        "benchmark":"BFCL, AppWorld, Terminal-Bench","model":plan["model"],"seeds":"42,43,44",
        "run_path":str(output),"git_commit":plan.get("git_commit", ""),
        "split":plan["history_bank"],"budget":"temperature0, thinkingoff, global128, input100k/output2048; "
                "independent per-CP inference and CP0 repeat control",
        "command":" ".join(plan["command"]),"summary":f"기존0.7실험을 보존하고 새실험 준비. "
                   f"history{prepared['histories']}, policy{prepared['policy_instances']}, fresh requests{prepared['distinct_requests']}.",
        "limitations":plan["legacy_differences"]+"; missing final checkpoints remain missing",
        "tags":["policy-shift","pairwise","temperature0","qwen36","queue128"]})
    plan["journal_id"] = record["id"]
    write_json(output / "plan.json",plan)
    write_json(output / "prepared.json",prepared)
    shutil.copy2(source / "source_scope.json",output / "source_scope.json")
    shutil.copy2(source / "history_audit.json",output / "history_audit.json")
    return plan,prepared


def analyze(output, plan):
    db=common.own_db(output)
    pair_rows,task_rows,repeat_rows,coverage=[],[],[],[]
    streams=db.execute("SELECT DISTINCT stream,benchmark,method,seed FROM policies ORDER BY benchmark,method,seed").fetchall()
    for stream in streams:
        points=[r[0] for r in db.execute("SELECT point FROM policies WHERE stream=? ORDER BY point",(stream["stream"],))]
        rows=db.execute("""SELECT b.anchor,a.task,b.point,j.status,j.action,
            repeat.status AS repeat_status,repeat.action AS repeat_action
            FROM bindings b JOIN anchors a ON a.id=b.anchor
            JOIN jobs j ON j.id=CASE WHEN b.point=0 THEN b.reference ELSE b.job END
            LEFT JOIN jobs repeat ON repeat.id=b.job AND b.point=0
            WHERE b.stream=? ORDER BY b.anchor,b.point""",(stream["stream"],))
        totals=defaultdict(lambda:[0,defaultdict(float)])
        repeats=defaultdict(lambda:[0,defaultdict(float)])
        available=common_histories=0
        common_tasks=set()
        for _,anchor_rows in itertools.groupby(rows,key=lambda r:r["anchor"]):
            anchor_rows=list(anchor_rows);available+=1
            if {r["point"] for r in anchor_rows}!=set(points) or any(r["status"]!="completed" for r in anchor_rows):
                continue
            actions={r["point"]:json.loads(r["action"]) for r in anchor_rows}
            common_histories+=1;task=anchor_rows[0]["task"];common_tasks.add(task)
            for left,right in itertools.combinations(points,2):
                scores=score_actions(actions[left],actions[right],stream["benchmark"])
                item=totals[(left,right,task)];item[0]+=1
                for metric,value in scores.items():item[1][metric]+=value
            root=next(r for r in anchor_rows if r["point"]==0)
            if root["repeat_status"]=="completed":
                item=repeats[task];item[0]+=1
                for metric,value in score_actions(json.loads(root["repeat_action"]),actions[0],stream["benchmark"]).items():
                    item[1][metric]+=value
        coverage.append({**dict(stream),"checkpoints":points,"available_histories":available,
                         "common_histories":common_histories,"common_tasks":len(common_tasks)})
        by_pair=defaultdict(lambda:defaultdict(list))
        for (left,right,task),(count,scores) in sorted(totals.items()):
            for metric,total in scores.items():
                agreement=total/count
                task_rows.append({**dict(stream),"left":left,"right":right,"distance":right-left,
                                  "task_id":task,"metric":metric,"agreement":agreement,"histories":count})
                by_pair[(left,right)][metric].append(agreement)
        for (left,right),metrics in sorted(by_pair.items()):
            pair_rows.append({**dict(stream),"left":left,"right":right,"distance":right-left,
                              "common_histories":common_histories,"common_tasks":len(common_tasks),
                              **{metric:statistics.mean(metrics[metric]) if metric in metrics else None for metric in METRICS}})
        repeated=defaultdict(list)
        for count,scores in repeats.values():
            for metric,total in scores.items():repeated[metric].append(total/count)
        for metric,values in repeated.items():
            repeat_rows.append({**dict(stream),"metric":metric,"agreement":statistics.mean(values),
                                "policy_shift":1-statistics.mean(values),"tasks":len(values)})
    groups=defaultdict(list)
    for row in pair_rows:
        for metric in METRICS:
            if row[metric] is not None:groups[(row["benchmark"],row["method"],row["distance"],metric)].append(
                (row["seed"],1-row[metric]))
    distances=[]
    for (benchmark,method,distance,metric),values in sorted(groups.items()):
        shifts=[v[1] for v in values]
        distances.append({"benchmark":benchmark,"method":method,"checkpoint_distance":distance,"metric":metric,
                          "mean_shift":statistics.mean(shifts),"pair_sample_sd":statistics.stdev(shifts) if len(shifts)>1 else 0,
                          "checkpoint_pairs":len(values),"seeds":len({v[0] for v in values})})
    common.write_csv(output/'analysis/pairwise_task_metrics.csv',task_rows)
    common.write_csv(output/'analysis/pairwise_metrics.csv',pair_rows)
    common.write_csv(output/'analysis/distance_metrics.csv',distances)
    common.write_csv(output/'analysis/repeatability_metrics.csv',repeat_rows)
    write_json(output/'analysis/coverage.json',coverage)
    counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status').fetchall())
    summary={"status":"completed_with_errors" if counts.get('error') else "completed","counts":counts,
             "workers":plan['workers'],"temperature":plan['temperature'],"comparison":plan['comparison'],
             "common_histories":sum(r['common_histories'] for r in coverage),
             "checkpoint_pair_rows":len(pair_rows),"scores":distances}
    if not distances:raise RuntimeError('No all-checkpoint common histories; cannot measure ΔCP shift')
    write_json(output/'analysis/summary.json',summary);db.close()
    return summary


def plot(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=summary['scores']
    colors={'ace':'#7753A6','gepa':'#D58A21','skillopt':'#2B9472','premise':'#2877B6'}
    for benchmark in ['bfcl','appworld','terminalbench2']:
        metrics=['accuracy'] if benchmark=='terminalbench2' else ['loose_accuracy','strict_accuracy','loose_f1','strict_f1']
        fig,axes=plt.subplots(1,len(metrics),figsize=(5*len(metrics),4.5),squeeze=False)
        for ax,metric in zip(axes[0],metrics,strict=True):
            for method in common.METHODS:
                group=sorted([r for r in rows if (r['benchmark'],r['method'],r['metric'])==(benchmark,method,metric)],
                             key=lambda r:r['checkpoint_distance'])
                if not group:continue
                x=[r['checkpoint_distance'] for r in group];y=[r['mean_shift'] for r in group];sd=[r['pair_sample_sd'] for r in group]
                ax.plot(x,y,marker='o',markersize=3,color=colors[method],label=common.DISPLAY[method])
                ax.fill_between(x,[max(0,a-b) for a,b in zip(y,sd,strict=True)],
                                [min(1,a+b) for a,b in zip(y,sd,strict=True)],color=colors[method],alpha=.12)
            ax.set(xlabel='Checkpoint distance (ΔCP)',ylabel='Policy shift',ylim=(-.02,1.02),title='1 − '+metric)
            ax.grid(alpha=.2);ax.legend()
        fig.suptitle(benchmark+' · temperature 0 · all-checkpoint common histories')
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(output/f'analysis/{benchmark}_delta_cp_policy_shift.{ext}',dpi=180)
        plt.close(fig)


def run(output):
    plan=load_json(output/'plan.json')
    if not common.checkpoint_gate(plan)['ready']:raise RuntimeError('Checkpoint evaluation gate not satisfied')
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared=load_json(output/'prepared.json')
        for path,sha in prepared['source_hashes'].items():
            if digest(Path(path).read_bytes())!=sha:raise ValueError('A frozen source changed: '+path)
        if digest(Path(plan['terminal_parser_path']).read_bytes())!=plan['terminal_parser_sha256']:
            raise ValueError('Official native parser changed')
        db=common.own_db(output)
        deduplication=deduplicate_jobs(db)
        counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status').fetchall())
        prepared.update(effective_requests=sum(value for key,value in counts.items() if key!='deduplicated'),
                        deduplicated_request_count=counts.get('deduplicated',0),
                        reused_response_count=deduplication['shared_policy_bindings'])
        write_json(output/'prepared.json',prepared)
        audit_path=output/'analysis/deduplication.json'
        if not audit_path.exists() or deduplication['deduplicated_jobs']:
            write_json(audit_path,deduplication)
        saved_audit=load_json(audit_path)
        db.close()
        plan['request_sampling']='One draw per distinct stream/prompt/history; exact duplicate prompts share a response; CP0 repeat remains independent'
        plan['deduplication']=saved_audit
        write_json(output/'plan.json',plan)
        asyncio.run(common.probe(output,plan))
        summary=analyze(output,plan);plot(output,summary)
        write_json(output/'status.json',{'phase':summary['status'],'pid':__import__('os').getpid(),
                                       'counts':summary['counts'],'workers':plan['workers']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-source',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    if args.prepare_source:prepare_fork(output,args.prepare_source.resolve())
    if not args.prepare_only:run(output)


if __name__=='__main__':main()
