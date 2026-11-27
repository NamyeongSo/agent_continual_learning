#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Adopt existing jobs and schedule fresh train→test jobs across two local model servers."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.fixed_matrix_worker import validate_final


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rpartition(')')[2].split()
        if fields[0] in {'Z', 'X'}:
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'argv': (path / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except (OSError, IndexError):
        return None


def same_process(saved):
    """Check the stable PID/start-time token; argv can change during exec."""
    try:
        fields = (Path('/proc') / str(saved['pid']) / 'stat').read_text().rpartition(')')[2].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] not in {'Z', 'X'} and fields[19] == saved['start_ticks']


def worker_catalog():
    """Find existing workers by their complete command, including the unique plan path."""
    catalog = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        saved = identity(int(path.name))
        if saved and any(arg.endswith('/fixed_matrix_worker.py') for arg in saved['argv'][1:]):
            catalog.setdefault(tuple(saved['argv']), []).append(saved)
    return catalog


def existing_worker(job, row, catalog):
    expected = row.get('dispatch_command', job['command'])
    matches = catalog.get(tuple(expected), [])
    if len(matches) > 1:
        raise RuntimeError('Multiple live workers for the same job: ' + job['id'])
    if matches:
        return matches[0]
    saved = row.get('process_identity')
    if saved and saved.get('argv') == expected and same_process(saved):
        return saved
    return None


def metrics(endpoint):
    try:
        with urllib.request.urlopen(endpoint['server'] + '/metrics', timeout=2) as response:
            lines = response.read().decode().splitlines()
        def value(name):
            return sum(float(line.rsplit(' ', 1)[1]) for line in lines
                       if line.startswith('vllm:' + name + '{'))
        return {'healthy': True, 'running': value('num_requests_running'),
                'waiting': value('num_requests_waiting'), 'kv_cache': value('kv_cache_usage_perc'),
                'generation_tokens': value('generation_tokens_total'),
                'completed_requests': value('request_success_total')}
    except (OSError, ValueError):
        return {'healthy': False}


def check_generation_progress(loads, history, timestamp, stall_timeout=300):
    """A responsive HTTP server can still have an engine that produces no output."""
    for gpu, load in loads.items():
        if not load.get('healthy'):
            continue
        counters = (load['generation_tokens'], load['completed_requests'])
        previous = history.get(gpu)
        busy = load.get('running', 0) + load.get('waiting', 0) > 0
        if previous is None or previous['counters'] != counters or not busy:
            previous = history[gpu] = {'counters': counters, 'at': timestamp}
        elapsed = max(0, timestamp - previous['at'])
        load['seconds_without_progress'] = round(elapsed, 1)
        load['stalled'] = busy and elapsed >= stall_timeout
        if load['stalled']:
            load['healthy'] = False


def select_gpu(job, row, counts, loads, endpoints, capacity=7):
    available = [gpu for gpu in endpoints if counts[gpu] < capacity and loads.get(gpu, {}).get('healthy')]
    # Partial jobs keep their original endpoint so optimizer caches/configurations remain compatible.
    if row.get('gpu') is not None:
        gpu = int(row['gpu'])
        return gpu if gpu in available else None
    if job['seed'] == 44 and job['benchmark'] == 'bfcl' and 0 in available:
        return 0
    if not available:
        return None
    def cost(gpu):
        load = loads[gpu]
        return (counts[gpu] / capacity + (load.get('running', 0) + 4 * load.get('waiting', 0)) / 256
                + .25 * load.get('kv_cache', 0), counts[gpu], gpu)
    return min(available, key=cost)


def ready_jobs(plan, rows):
    return [job for job in plan["jobs"] if rows[job["id"]]["phase"] == "pending" and
            (not job["predecessor"] or rows[job["predecessor"]]["phase"] == "completed")]


def route_prepared_config(job, endpoint):
    """Move only an unused dry-run configuration; keep learned caches pinned."""
    if job['method'] not in {'gepa', 'skillopt'}:
        return
    config_path = Path(job['train_dir']) / 'config.json'
    if not config_path.exists():
        return
    config = read(config_path)
    if config.get('base_url') == endpoint:
        return
    folder = config_path.parent
    if any((folder / name).exists() for name in [
        'rollouts', 'online', 'analysis', 'analysis_calls.jsonl', 'optimizer.lock',
        'training_result.json', 'test_results.json',
    ]):
        raise RuntimeError('Cannot migrate a configuration with learning artifacts: ' + job['id'])
    archive = Path(job['output']) / 'routing_history' / ('config_' + uuid.uuid4().hex + '.json')
    write(archive, config)
    write(config_path, {**config, 'base_url': endpoint})
    write(archive.with_suffix('.migration.json'), {'at': now(), 'from': config.get('base_url'),
          'to': endpoint, 'changed_fields': ['base_url'], 'learning_started': False})


def dispatch_plan(plan, job, endpoint, gpu):
    assigned = dict(job, base_url=endpoint['client'], gpu=gpu)
    command = job['train_command'].copy()
    if '--base-url' in command:
        command[command.index('--base-url') + 1] = endpoint['client']
    assigned['train_command'] = command
    path = Path(job['output']) / 'dispatch_plan.json'
    assigned['command'] = [str(Path(plan['snapshot']) / '.venv/bin/python'), '-u',
                           str(Path(plan['snapshot']) / 'scripts/utils/fixed_matrix_worker.py'),
                           '--plan', str(path), '--job', job['id']]
    return path, {**plan, 'base_url': endpoint['client'], 'gpu': gpu, 'jobs': [assigned]}, assigned


def journal(root, plan, status):
    if not plan.get('journal_id'):
        return
    try:
        url = 'http://127.0.0.1:8878'
        def request(path, body=None, method=None, token=None):
            headers = {'Content-Type': 'application/json'}
            if token:
                headers['X-Journal-Token'] = token
            query = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                           method=method, headers=headers)
            with urllib.request.urlopen(query, timeout=10) as response:
                return json.load(response)
        token = request('/api/info')['csrf_token']
        path = '/api/experiments/' + plan['journal_id']
        record = request(path)
        phase = status['phase'] if status['phase'] in {'completed', 'blocked'} else 'running'
        request(path, {'revision': record['revision'], 'status': phase,
                       'summary': json.dumps(status, ensure_ascii=False)}, 'PUT', token)
    except Exception as exc:
        write(root / 'journal_error.json', {'at': now(), 'error': repr(exc)})


def run(output):
    config = read(output / 'dispatch_manifest.json')
    endpoints = {int(key): value for key, value in config['endpoints'].items()}
    maximum = config.get('max_parallel', 14)
    capacity = config.get('per_gpu_capacity', 7)
    with (output / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = {'at': now(), 'identity': identity(os.getpid()), 'output': str(output)}
        write(output / 'controller_process.json', owner)
        sources = []
        source_locks = []
        active = {}
        catalog = worker_catalog()
        for name in config['sources']:
            root = Path(name)
            source_lock = (root / 'controller.lock').open('a')
            fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            source_locks.append(source_lock)
            plan = read(root / 'launch_manifest.json')
            if plan.get('coordinator_output') != str(output):
                raise RuntimeError('Source was not handed over to this coordinator')
            rows = read(root / 'progress.json')['jobs']
            sources.append((root, plan, rows))
            write(root / 'controller_process.json', owner)
            for job in plan['jobs']:
                row = rows[job['id']]
                if validate_final(job):
                    row['phase'] = 'completed'
                else:
                    saved = existing_worker(job, row, catalog)
                    if saved:
                        if row['phase'] != 'running' or row.get('pid') != saved['pid']:
                            print(json.dumps({'at': now(), 'event': 'adopted_existing_worker',
                                              'job': job['id'], 'previous_phase': row['phase'],
                                              'previous_pid': row.get('pid'), 'pid': saved['pid']}), flush=True)
                        row.update(phase='running', pid=saved['pid'])
                        row.pop('finished_at', None)
                        row.setdefault('gpu', 1)
                        row['process_identity'] = saved
                        active[job['id']] = (root, job, row, saved)
                    elif row['phase'] == 'running':
                        row['phase'] = 'pending' if row['launches'] < 3 else 'error'
        last_journal = 0
        engine_progress = {}
        while True:
            for name, (root, job, row, saved) in list(active.items()):
                if same_process(saved):
                    continue
                found = existing_worker(job, row, worker_catalog())
                if found:
                    row.update(pid=found['pid'], process_identity=found)
                    active[name] = (root, job, row, found)
                    print(json.dumps({'at': now(), 'event': 'reconciled_existing_worker',
                                      'job': name, 'pid': found['pid']}), flush=True)
                    continue
                row['phase'] = 'completed' if validate_final(job) else ('pending' if row['launches'] < 3 else 'error')
                row['finished_at'] = now()
                del active[name]
                print(json.dumps({'at': now(), 'job': name, 'event': row['phase'], 'gpu': row.get('gpu')}), flush=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                sampled = list(pool.map(metrics, endpoints.values()))
            loads = dict(zip(endpoints, sampled))
            check_generation_progress(loads, engine_progress, time.monotonic())
            counts = Counter(int(item[2]['gpu']) for item in active.values())
            ready = [(root, plan, rows, job) for root, plan, rows in sources for job in ready_jobs(plan, rows)]
            ready.sort(key=lambda item: (not (item[3]['seed'] == 44 and item[3]['benchmark'] == 'bfcl'),
                                         item[3]['seed'], item[3]['id']))
            for root, plan, rows, job in ready:
                if len(active) >= maximum:
                    break
                row = rows[job['id']]
                gpu = select_gpu(job, row, counts, loads, endpoints, capacity)
                if gpu is None:
                    continue
                route_prepared_config(job, endpoints[gpu]['client'])
                path, dispatch, assigned = dispatch_plan(plan, job, endpoints[gpu], gpu)
                write(path, dispatch)
                env = {**os.environ, **plan.get('environment', {}), 'PYTHONUNBUFFERED': '1',
                       'LITELLM_LOCAL_MODEL_COST_MAP': 'True'}
                with (Path(job['output']) / 'worker.log').open('ab') as log:
                    process = subprocess.Popen(assigned['command'], cwd=plan['snapshot'], env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(assigned)
                row.update(phase='running', gpu=gpu, base_url=endpoints[gpu]['client'], pid=process.pid,
                           started_at=now(), launches=row['launches'] + 1, dispatch_command=assigned['command'],
                           process_identity=identity(process.pid))
                if row['process_identity'] is None:
                    row['phase'] = 'pending' if row['launches'] < 3 else 'error'
                    continue
                active[job['id']] = (root, job, row, row['process_identity'])
                counts[gpu] += 1
                print(json.dumps({'at': now(), 'event': 'started', 'job': job['id'], 'gpu': gpu}), flush=True)
            combined = Counter()
            for root, plan, rows in sources:
                state_counts = Counter(row['phase'] for row in rows.values())
                combined.update(state_counts)
                own_active = [item for item in active.values() if item[0] == root]
                phase = 'completed' if state_counts['completed'] == len(rows) else 'running'
                if not own_active and not state_counts['pending'] and state_counts['error']:
                    phase = 'blocked'
                status = {'phase': phase, 'at': now(), 'max_parallel': plan['max_parallel'],
                          'running': len(own_active), 'counts': dict(state_counts),
                          'completed_jobs': state_counts['completed'], 'total_jobs': len(rows),
                          'gpu_counts': dict(Counter(int(item[2]['gpu']) for item in own_active)),
                          'server_max_num_seqs': 256, 'coordinator_output': str(output)}
                write(root / 'launch_manifest.json', plan)
                write(root / 'progress.json', {'at': now(), 'jobs': rows})
                write(root / 'status.json', status)
                if time.monotonic() - last_journal > 120:
                    journal(root, plan, status)
            last_journal = time.monotonic() if time.monotonic() - last_journal > 120 else last_journal
            all_done = sum(combined.values()) == combined['completed']
            blocked = not active and not any(ready_jobs(plan, rows) for _, plan, rows in sources) and combined['error']
            write(output / 'status.json', {'at': now(), 'phase': 'completed' if all_done else 'blocked' if blocked else 'running',
                  'max_parallel': maximum, 'running': len(active), 'counts': dict(combined),
                  'gpu_counts': dict(counts), 'gpu_metrics': loads, 'total_jobs': sum(combined.values())})
            if all_done or blocked:
                return
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve())
    except Exception as exc:
        write(args.output / 'coordinator_error.json', {'at': now(), 'error': repr(exc)})
        raise
