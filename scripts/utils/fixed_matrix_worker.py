#!/usr/bin/env python3
"""One isolated train→frozen-test job; only a valid full test releases its lane."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def training_complete(job):
    folder=Path(job['train_dir'])
    if job['method']=='ace':
        p=folder/'training_complete.json'
        return p.exists() and read(p).get('completed_task_count')==job['train_count']
    if job['method']=='premise':
        p=folder/'state.json'
        return p.exists() and read(p).get('cursor')==job['train_count']
    p=folder/'training_result.json'
    return p.exists() and len(list((folder/'online').glob('*.json')))==job['train_count']


def validate_final(job):
    root=Path(job['output'])
    marker=root/'job_complete.json'
    if not marker.exists() or not training_complete(job):return False
    value=read(marker)
    return (value.get('complete') is True and value.get('test_ids')==job['test_ids']
            and value.get('tasks')==len(job['test_ids']) and value.get('infrastructure_errors')==0
            and value.get('job_id')==job['id'] and isinstance(value.get('score'),(float,int)))


def legacy_test(job):
    from scripts.utils.run_qwen_isolated_when_free import frozen_evaluation_jobs,evaluate_one,evaluation_worker_init
    tasks=frozen_evaluation_jobs(job)
    directory=Path(job['output'])/'evaluation';records=directory/'records';records.mkdir(parents=True,exist_ok=True)
    complete={}
    for task in tasks:
        p=records/f"{task['index']:03d}.json"
        if p.exists():
            row=read(p)
            if row.get('status')=='completed' and row.get('frozen_sha256')==task['frozen_sha256']:
                complete[task['index']]=row
    pending=[task for task in tasks if task['index'] not in complete]
    for attempt in range(3):
        if not pending:break
        retry=[]
        with ProcessPoolExecutor(max_workers=job['test_workers'],mp_context=multiprocessing.get_context('spawn'),initializer=evaluation_worker_init,initargs=(job.get('base_url'),)) as pool:
            futures={pool.submit(evaluate_one,task):task for task in pending}
            for future in as_completed(futures):
                task=futures[future];row=future.result()
                if row.get('status')=='completed':
                    write(records/f"{task['index']:03d}.json",row);complete[task['index']]=row
                else:
                    write(directory/'attempts'/f"{task['index']:03d}_{attempt}.json",row);retry.append(task)
                write(Path(job['output'])/'job_status.json',{'phase':'test','at':now(),'completed':len(complete),'total':len(tasks)})
        pending=retry
    if pending:raise RuntimeError(f'Final test has {len(pending)} infrastructure failures; lane remains blocked')
    rows=[complete[i] for i in range(len(tasks))]
    if [r['task_id'] for r in rows]!=job['test_ids']:raise ValueError('Test ID order differs')
    summary={'complete':True,'tasks':len(rows),'score':sum(r['score'] for r in rows)/len(rows),
        'successes':sum(r['score']==1 for r in rows),'test_ids':job['test_ids'],'infrastructure_errors':0,
        'metric':'TGC' if job['benchmark']=='appworld' else 'accuracy','at':now()}
    write(directory/'summary.json',summary)
    return summary


def run(plan,job):
    root=Path(job['output']);root.mkdir(parents=True,exist_ok=True)
    if validate_final(job):return
    write(root/'job_status.json',{'phase':'training','at':now(),'pid':os.getpid()})
    env=os.environ.copy()
    env.update(plan.get('environment',{}))
    env.update(OPENAI_API_BASE=plan['base_url'],OPENAI_BASE_URL=plan['base_url'],OPENAI_API_KEY='EMPTY',
        LITELLM_LOCAL_MODEL_COST_MAP='True',PYTHONUNBUFFERED='1',EXGENTIC_PORT_BLOCK_START=str(job['port_block_start']))
    env['PATH']=str(Path(plan['snapshot'])/'.venv/bin')+os.pathsep+env.get('PATH','')
    if job['method'] in ('ace','premise'):
        if not training_complete(job):
            process=subprocess.run(job['train_command'],cwd=plan['snapshot'],env=env)
            if process.returncode:raise RuntimeError(f'Training exited {process.returncode}')
        if not training_complete(job):raise RuntimeError('Training did not consume the complete frozen stream')
        write(root/'job_status.json',{'phase':'test','at':now(),'completed':0,'total':len(job['test_ids'])})
        summary=legacy_test(job)
    else:
        # Official GEPA/SkillOpt entrypoints freeze their selected policy and
        # run full final_test before returning. No next job starts on train-only completion.
        process=subprocess.run(job['train_command'],cwd=plan['snapshot'],env=env)
        if process.returncode:raise RuntimeError(f'Optimizer/train-test entrypoint exited {process.returncode}')
        folder=Path(job['train_dir']);summary=read(folder/'summary.json');rows=read(folder/'test_results.json')
        if not training_complete(job) or not summary.get('complete') or summary.get('tasks')!=len(job['test_ids']):
            raise RuntimeError('Incomplete baseline train/test result')
        if [row['id'] for row in rows]!=job['test_ids']:raise RuntimeError('Final test IDs differ')
        summary.update(test_ids=job['test_ids'],infrastructure_errors=0,
                       metric='TGC' if job['benchmark']=='appworld' else 'accuracy')
    write(root/'job_complete.json',{**summary,'complete':True,'job_id':job['id'],'at':now(),
        'split_manifest_sha256':plan['manifest_sha256'],'validation_source':job.get('validation_source','val')})
    if not validate_final(job):raise RuntimeError('Final completion evidence failed validation')
    write(root/'job_status.json',{'phase':'completed','at':now(),'score':summary['score'],'test_tasks':len(job['test_ids'])})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--job',required=True)
    args=parser.parse_args();plan=read(args.plan);job=next(j for j in plan['jobs'] if j['id']==args.job)
    try:run(plan,job)
    except Exception as exc:
        write(Path(job['output'])/'job_status.json',{'phase':'error','at':now(),'error':repr(exc)});raise


if __name__=='__main__':main()
