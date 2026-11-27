#!/usr/bin/env python3
"""Freeze six baseline lanes; after CKA restart serving and run train→test jobs."""
import argparse
from collections import Counter
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import httpx
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
VARIANTS=(('ace_aud','ace','val'),('gepa_val','gepa','val'),('skillopt_val','skillopt','val'),
          ('peace_aud','premise','val'),('gepa_train_as_val','gepa','train'),('skillopt_train_as_val','skillopt','train'))
BENCHMARKS=('bfcl','appworld','terminalbench2')


def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def identity(pid):
    try:
        p=Path('/proc')/str(pid);stat=p.joinpath('stat').read_text().rpartition(')')[2].split()
        if stat[0] in ('Z','X'):return None
        return {'pid':pid,'start_ticks':stat[19],'argv':p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')}
    except (OSError,IndexError):return None


class ExistingProcess:
    def __init__(self,saved):self.saved=saved;self.pid=saved['pid']
    def poll(self):return None if identity(self.pid)==self.saved else 0


def patch_snapshot(code):
    changes=[]
    def replace(relative,old,new,count=1):
        path=code/relative;text=path.read_text()
        if text.count(old)!=count:raise ValueError(f'Patch source differs: {relative}: {old[:70]!r}')
        before=sha(path);path.write_text(text.replace(old,new));changes.append({'file':relative,'before':before,'after':sha(path)})
    for relative in ['scripts/ace/run_experiment.py','src/exgentic/agents/premise/pipeline.py']:
        replace(relative,'        random.Random(order_seed).shuffle(train)',
                '        if manifest.get("seed_scope") != "within_split_order_only":\n            random.Random(order_seed).shuffle(train)')
    replace('src/exgentic/agents/premise/pipeline.py','        self.rng = random.Random(seed)',
            '        self.algorithm_seed = int(json.loads(self.manifest.read_text()).get("algorithm_seed", seed))\n        self.rng = random.Random(self.algorithm_seed)')
    replace('src/exgentic/agents/premise/pipeline.py','            "count": self.count, "past_count": self.past_count, "model": self.model,',
            '            "count": self.count, "past_count": self.past_count, "model": self.model,\n            "algorithm_seed": self.algorithm_seed,')
    replace('src/exgentic/agents/premise/pipeline.py','    score = session_result.score\n    if score is None:',
        '    partial_score = session_result.score\n    score = partial_score\n'
        '    if benchmark == "appworld":\n'
        '        if not isinstance(session_result.success, bool):\n'
        '            raise ValueError("AppWorld TGC requires official success flag")\n'
        '        score = float(session_result.success)\n    if score is None:')
    replace('src/exgentic/agents/premise/pipeline.py','        "actions": actions, "score": float(score), "details": session_result.details,',
        '        "actions": actions, "score": float(score), "details": session_result.details,\n'
        '        **({"partial_score": partial_score, "score_metric": "appworld_tgc_v1"} if benchmark == "appworld" else {}),')
    replace('scripts/baselines/run_experiment.py','    if args.pilot:\n        train, validation, test = train[:5], validation[:2], test[:2]',
        '    if args.pilot:\n        train, validation, test = train[:5], validation[:2], test[:2]\n'
        '    if args.validation_source == "train":\n'
        '        validation = [{"task_id": item["task_id"], "split": "val", "source_split": "train"} for item in train]')
    replace('scripts/baselines/run_experiment.py','        "mode": "isolated", "model": args.model, "base_url": args.base_url, "workers": args.workers,',
        '        "mode": "isolated", "model": args.model, "base_url": args.base_url, "workers": args.workers,\n'
        '        "optimizer_seed": args.optimizer_seed if args.optimizer_seed is not None else args.seed,\n'
        '        "validation_source": args.validation_source,')
    replace('scripts/baselines/run_experiment.py','    parser.add_argument("--pilot", action="store_true",',
        '    parser.add_argument("--validation-source", choices=("val", "train"), default="val")\n'
        '    parser.add_argument("--optimizer-seed", type=int, default=None)\n'
        '    parser.add_argument("--pilot", action="store_true",')
    for relative in ['scripts/baselines/gepa_runtime.py','scripts/baselines/skillopt_runtime.py']:
        replace(relative,'seed=config["seed"]','seed=config.get("optimizer_seed", config["seed"])')
    replace('scripts/baselines/runtime.py','"selection": "validation_only"',
            '"selection": runtime.config.get("validation_source", "val") + "_only"')
    replace('scripts/baselines/runtime.py','    results = runtime.batch(runtime.config["test"], learned, "final_test")',
        '    write(runtime.output.parent / "job_status.json", {"phase": "test", "total": len(runtime.config["test"])})\n'
        '    results = runtime.batch(runtime.config["test"], learned, "final_test")')
    # Displayed playbook IDs are bracketed. Accept the displayed form as well as
    # the canonical ID; do not silently reject valid UPDATE/DELETE operations.
    replace('src/exgentic/agents/ace/playbook_utils.py','            target = op.get("entry_id", op.get("bullet_id"))\n',
        '            target = op.get("entry_id", op.get("bullet_id"))\n'
        '            if isinstance(target, str):\n'
        '                target = target.strip().removeprefix("[").removesuffix("]")\n')
    replace('src/exgentic/agents/ace/playbook_utils.py','            if "bullet_id" in op and op["bullet_id"] != target:',
        '            if "bullet_id" in op and str(op["bullet_id"]).strip().removeprefix("[").removesuffix("]") != target:')
    replace('scripts/utils/run_qwen_isolated_when_free.py','        bm = {"subset": SUBSETS[job["benchmark"]], "use_cache": False}',
        '        bm = {"subset": SUBSETS[job["benchmark"]], "use_cache": False}\n'
        '        if job["benchmark"] == "appworld":\n            bm["interaction_mode"] = "code_generation"')
    replace('scripts/utils/run_qwen_isolated_when_free.py','            benchmark_kwargs=bm, agent_kwargs=kwargs,',
        '            benchmark_kwargs=bm, agent_kwargs=kwargs, max_steps=30 if job["benchmark"] == "bfcl" else 40,')
    return changes


def prepare(output,after_cka,manifest):
    if (output/'launch_manifest.json').exists():raise ValueError('Already frozen; use run')
    output.mkdir(parents=True,exist_ok=True);code=output/'code_snapshot';code.mkdir(exist_ok=False)
    raw=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','--cached','--others','--exclude-standard','--','src','scripts'])
    for name in [Path(os.fsdecode(p)) for p in raw.split(b'\0') if p]+[Path('pyproject.toml'),Path('uv.lock')]:
        source=ROOT/name;destination=code/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    for name in ['.venv','.exgentic']:
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    # BFCL checker vendoring is ignored source data; reuse the existing install.
    installation=Path('src/exgentic/benchmarks/bfcl/installation')
    if (ROOT/installation).exists() and not (code/installation).exists():(code/installation).symlink_to(ROOT/installation,target_is_directory=True)
    changes=patch_snapshot(code)
    data=read(manifest);data['algorithm_seed']=42
    frozen_manifest=output/'task_splits.json';write(frozen_manifest,data)
    policy_source=ROOT/'outputs/appworld_code_generation_mixed5_teammate_20261102_151041/appworld_teammate_policy.txt'
    policy=output/'appworld_teammate_policy.txt';shutil.copy2(policy_source,policy)
    jobs=[];python=str(code/'.venv/bin/python');base='http://127.0.0.1:18006/v1'
    for lane,(variant,method,validation_source) in enumerate(VARIANTS):
        previous=None
        for seed in [42,43,44]:
            for benchmark in BENCHMARKS:
                section=data[f'{benchmark}_{seed}'];root=output/'runs'/variant/benchmark/f's{seed}';train=root/'train'
                job={'id':f'{variant}_{benchmark}_s{seed}','variant':variant,'method':method,
                    'benchmark':benchmark,'seed':seed,'data_order_seed':seed,'optimizer_seed':42,
                    'validation_source':validation_source,'train_ids':section['train'],'val_ids':section['val'],'test_ids':section['test'],
                    'train_count':len(section['train']),'train_dir':str(train),'output':str(root),
                    'lane':lane,'predecessor':previous,'test_workers':50,'port_block_start':27000+lane*1000,
                    'playbook_edit_mode':'add_update_delete' if method in ('ace','premise') else None}
                common=['--seed',str(seed),'--model','openai/Qwen/Qwen3.6-35B-A3B']
                if method=='ace':
                    command=[python,'-u',str(code/'scripts/ace/run_experiment.py'),'--mode','isolated',*common,
                        '--benchmarks',benchmark,'--task-split-manifest',str(frozen_manifest),'--all-train-tasks',
                        '--resume-sessions','--temperature','0.7','--playbook-edit-mode','add_update_delete',
                        '--max-tokens','8192','--disable-thinking','--training-time','--bfcl-max-turns','30',
                        '--appworld-max-turns','40','--terminalbench2-max-turns','40','--terminalbench2-actor','terminus2','--output-dir',str(train)]
                elif method=='premise':
                    command=[python,'-u',str(code/'scripts/premise/run_experiment.py'),'--mode','isolated',*common,
                        '--benchmarks',benchmark,'--task-split-manifest',str(frozen_manifest),'--all-train-tasks',
                        '--base-url',base,'--temperature','0.7','--playbook-edit-mode','add_update_delete',
                        '--past','3','--parent-workers','2','--reflection-workers','4','--mutation-workers','4',
                        '--replay-workers','28','--llm-calls','32','--unlimited-benchmark-sessions',
                        '--bfcl-max-steps','30','--appworld-max-steps','40','--terminalbench2-max-steps','40','--output-dir',str(train)]
                else:
                    command=[python,'-u',str(code/'scripts/baselines/run_experiment.py'),'--method',method,
                        '--benchmark',benchmark,*common,'--optimizer-seed','42','--validation-source',validation_source,
                        '--manifest',str(frozen_manifest),'--base-url',base,'--workers','50','--output',str(train)]
                if benchmark=='appworld':
                    command+=['--appworld-policy',str(policy)]
                    job.update(appworld_policy=str(policy),appworld_policy_sha256=sha(policy),appworld_interaction_mode='code_generation')
                job['train_command']=command
                job['command']=[python,'-u',str(code/'scripts/utils/fixed_matrix_worker.py'),'--plan',str(output/'launch_manifest.json'),'--job',job['id']]
                jobs.append(job);previous=job['id']
    cka=read(after_cka/'plan.json')
    server=['/home/thskadud/.vllm/bin/vllm','serve','Qwen/Qwen3.6-35B-A3B','--tensor-parallel-size','1',
        '--gpu-memory-utilization','0.95','--chat-template','/home/thskadud/chat_templates/qwen3_6-27B.jinja',
        '--max-model-len',str(cka['max_model_len']),'--max-num-batched-tokens','16384','--reasoning-parser','qwen3',
        '--kv-cache-dtype','fp8','--language-model-only','--enable-prefix-caching','--enable-chunked-prefill',
        '--max-num-seqs','256','--port','8006','--revision',cka['model_revision'],'--tokenizer-revision',cka['model_revision']]
    plan={'created_at':now(),'after_cka':str(after_cka),'snapshot':str(code),'manifest':str(frozen_manifest),
        'manifest_sha256':sha(frozen_manifest),'source_manifest':str(manifest),'source_manifest_sha256':sha(manifest),
        'max_parallel':6,'schedule':'six independent variant lanes; each lane seed42 BFCL→AppWorld→TB, then43, then44; train+full test before release',
        'jobs':jobs,'base_url':base,'model':cka['model'],'temperature_all_roles':0.7,'optimizer_seed':42,
        'actor_input_tokens':100000,'actor_output_tokens':2048,'turns':{'bfcl':30,'appworld':40,'terminalbench2':40},
        'server_command':server,'gpu':'1','prior_server_identity':identity(1163043),
        'prior_engine_identity':identity(1163766),'server_max_num_seqs':256,
        'appworld_policy':str(policy),'appworld_policy_sha256':sha(policy),'total_training_limit':None,
        'training_jobs':54,'training_tasks':sum(j['train_count'] for j in jobs),'test_tasks':sum(len(j['test_ids']) for j in jobs)}
    write(output/'snapshot_patches.json',changes)
    write(output/'code_hashes.json',{str(f.relative_to(code)):sha(f) for folder in ['src','scripts'] for f in (code/folder).rglob('*') if f.is_file() and 'installation' not in f.parts})
    write(output/'launch_manifest.json',plan)
    write(output/'progress.json',{'at':now(),'jobs':{j['id']:{'phase':'pending','launches':0} for j in jobs}})
    write(output/'status.json',{'phase':'prepared','at':now(),'after_cka':str(after_cka),'max_parallel':6,'jobs':54})
    print(json.dumps({'jobs':54,'first_jobs':[j['id'] for j in jobs if j['predecessor'] is None],
        'max_parallel':6,'snapshot':str(code)},ensure_ascii=False),flush=True)


def cka_complete(plan):
    directory=Path(plan['after_cka'])
    try:
        status=read(directory/'status.json');summary=read(directory/'analysis/final_cka_plot_summary.json')
        return (status['phase']=='completed' and status['counts']=={'completed':status['total']}
                and summary['status']=='completed' and summary['errors']==0 and not identity(status['pid']))
    except (OSError,ValueError,KeyError):return False


def port_free(port):
    with socket.socket() as sock:
        try:sock.bind(('127.0.0.1',port));return True
        except OSError:return False


def services(output,plan):
    existing_path=output/'server_process.json'
    if existing_path.exists():
        saved=read(existing_path)
        if saved.get('command')==plan['server_command'] and identity(saved['identity']['pid'])==saved['identity']:
            return ExistingProcess(saved['identity'])
    previous=plan['prior_server_identity'];current=identity(previous['pid']) if previous else None
    if current:
        if current!=previous:raise RuntimeError('Recorded CKA server identity changed; refusing to stop unknown process')
        os.killpg(os.getpgid(current['pid']),signal.SIGTERM)
        deadline=time.monotonic()+60
        engine=plan.get('prior_engine_identity')
        while (identity(current['pid']) or (engine and identity(engine['pid']))) and time.monotonic()<deadline:time.sleep(1)
        if identity(current['pid']) or (engine and identity(engine['pid'])):raise RuntimeError('CKA server did not shut down normally')
    if not port_free(8006):raise RuntimeError('Port8006 is occupied by another process')
    import shlex
    (output/'serve_training.sh').write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=1\nexport VLLM_ALLOW_LONG_MAX_MODEL_LEN=1\nexec '+shlex.join(plan['server_command'])+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','VLLM_ALLOW_LONG_MAX_MODEL_LEN':'1'}
    env.pop('PYTHONPATH',None)
    with (output/'vllm_training.log').open('ab') as log:
        server=subprocess.Popen(plan['server_command'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(output/'server_process.json',{'at':now(),'identity':identity(server.pid),'command':plan['server_command']})
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if server.poll() is not None:raise RuntimeError('Training vLLM exited; see vllm_training.log')
        try:
            response=httpx.get('http://127.0.0.1:8006/v1/models',timeout=5,trust_env=False);response.raise_for_status()
            if response.json()['data'][0]['max_model_len']==131072:break
        except (httpx.HTTPError,KeyError):pass
        time.sleep(5)
    else:raise TimeoutError('Training server startup exceeded30minutes')
    if port_free(18006):
        with (output/'parser_bridge.log').open('ab') as log:
            bridge=subprocess.Popen(['/home/thskadud/.vllm/bin/python','-u',str(Path(plan['snapshot'])/'scripts/utils/serve_vllm_tool_parser_bridge.py'),'--port','18006'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(output/'bridge_process.json',{'at':now(),'identity':identity(bridge.pid)})
    for attempt in range(30):
        try:
            response=httpx.get('http://127.0.0.1:18006/healthz',timeout=3,trust_env=False)
            if response.status_code==200 and response.json().get('parser')=='qwen3_coder':break
        except httpx.HTTPError:pass
        time.sleep(1)
    else:raise RuntimeError('Qwen parser bridge not healthy')
    sys.path.insert(0,plan['snapshot']);sys.path.insert(0,str(Path(plan['snapshot'])/'src'))
    from scripts.utils.run_qwen_isolated_when_free import preflight_model
    preflight_model(output)
    return server


def journal(output,phase,message=None):
    plan=read(output/'launch_manifest.json')
    if not plan.get('journal_id'):return
    try:
        with httpx.Client(base_url='http://127.0.0.1:8878',timeout=20,trust_env=False) as client:
            headers={'X-Journal-Token':client.get('/api/info').json()['csrf_token']};endpoint='/api/experiments/'+plan['journal_id']
            record=client.get(endpoint).json();state=read(output/'status.json')
            r=client.put(endpoint,json={'revision':record['revision'],'status':phase,'summary':json.dumps(state,ensure_ascii=False)},headers=headers);r.raise_for_status()
            if message:
                r=client.post(endpoint+'/notes',json={'text':message},headers=headers);r.raise_for_status()
    except Exception as exc:write(output/'journal_error.json',{'at':now(),'error':repr(exc)})


def run(output):
    plan=read(output/'launch_manifest.json')
    if not read(output/'preflight_validation.json').get('passed'):raise RuntimeError('Preparation validation required before arming')
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(output/'controller_process.json',{'at':now(),'identity':identity(os.getpid())})
        while not cka_complete(plan):
            write(output/'status.json',{'phase':'waiting_cka','at':now(),'after_cka':plan['after_cka'],'max_parallel':6,'jobs':54})
            time.sleep(20)
        write(output/'status.json',{'phase':'restarting_vllm','at':now(),'server_max_num_seqs':256})
        journal(output,'running','1000-prefix CKA 완료 확인. 학습용 vLLM 동시성256 재기동 후6개baseline의BFCL seed42부터학습+test시작.')
        server=services(output,plan)
        sys.path.insert(0,plan['snapshot'])
        from scripts.utils.fixed_matrix_worker import validate_final
        progress=read(output/'progress.json')['jobs'];active={}
        for job in plan['jobs']:
            if validate_final(job):progress[job['id']]['phase']='completed'
            elif progress[job['id']]['phase']=='running':
                row=progress[job['id']];saved=identity(row.get('pid'))
                if saved and saved['argv']==job['command']:active[job['id']]=ExistingProcess(saved)
                elif saved:raise RuntimeError('Worker PID reused by another command')
                else:row['phase']='pending'
        last_journal=0
        while True:
            if server.poll() is not None:raise RuntimeError('Training vLLM stopped; preserve queue')
            for name,process in list(active.items()):
                if process.poll() is None:continue
                job=next(j for j in plan['jobs'] if j['id']==name);row=progress[name]
                if validate_final(job):
                    row.update(phase='completed',result=read(Path(job['output'])/'job_complete.json'))
                else:
                    row['phase']='pending' if row['launches']<3 else 'error'
                row['finished_at']=now();del active[name]
            for job in plan['jobs']:
                row=progress[job['id']]
                if row['phase']!='pending' or len(active)>=6:continue
                if job['predecessor'] and progress[job['predecessor']]['phase']!='completed':continue
                folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
                env={**os.environ,'PYTHONUNBUFFERED':'1','LITELLM_LOCAL_MODEL_COST_MAP':'True'}
                env.update(plan.get('environment',{}))
                with (folder/'worker.log').open('ab') as log:
                    process=subprocess.Popen(job['command'],cwd=plan['snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row.update(phase='running',pid=process.pid,started_at=now(),launches=row['launches']+1)
                active[job['id']]=process
            counts=Counter(v['phase'] for v in progress.values())
            write(output/'progress.json',{'at':now(),'jobs':progress})
            phase='running' if active else ('completed' if counts['completed']==54 else 'blocked')
            write(output/'status.json',{'phase':phase,'at':now(),'max_parallel':6,'running':len(active),'counts':dict(counts),
                'completed_jobs':counts['completed'],'total_jobs':54,'server_max_num_seqs':256})
            if time.monotonic()-last_journal>120:
                journal(output,'running' if phase=='running' else phase);last_journal=time.monotonic()
            if not active:break
            time.sleep(15)
        journal(output,phase,f'6baseline×3benchmark×3seed 학습+test 종료. 완료{counts["completed"]}/54. 미완료lane은다음실험으로진행하지않음.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--after-cka',type=Path);parser.add_argument('--manifest',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if args.command=='prepare':prepare(output,args.after_cka.resolve(),args.manifest.resolve())
    else:
        try:run(output)
        except Exception as exc:
            write(output/'status.json',{'phase':'blocked','at':now(),'error':repr(exc)})
            journal(output,'blocked','학습 큐 중단: '+repr(exc));raise


if __name__=='__main__':main()
