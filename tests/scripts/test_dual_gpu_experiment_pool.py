# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from scripts.utils.run_dual_gpu_experiment_pool import dispatch_plan, ready_jobs, select_gpu

ENDPOINTS = {0: {'server': 'http://127.0.0.1:8007', 'client': 'http://127.0.0.1:18007/v1'},
             1: {'server': 'http://127.0.0.1:8006', 'client': 'http://127.0.0.1:18006/v1'}}


def test_seed44_bfcl_first_wave_fills_gpu0_without_exceeding_seven_slots():
    loads = {0: {'healthy': True}, 1: {'healthy': True, 'running': 100}}
    counts = {0: 0, 1: 7}
    for _ in range(7):
        gpu = select_gpu({'seed': 44, 'benchmark': 'bfcl'}, {}, counts, loads, ENDPOINTS)
        assert gpu == 0
        counts[gpu] += 1
    assert select_gpu({'seed': 44, 'benchmark': 'bfcl'}, {}, counts, loads, ENDPOINTS) is None


def test_new_jobs_use_available_load_but_partial_jobs_keep_their_endpoint():
    job = {'seed': 43, 'benchmark': 'appworld'}
    loads = {0: {'healthy': True, 'running': 100, 'waiting': 20}, 1: {'healthy': True, 'running': 0}}
    assert select_gpu(job, {}, {0: 4, 1: 4}, loads, ENDPOINTS) == 1
    assert select_gpu(job, {'gpu': 0}, {0: 4, 1: 4}, loads, ENDPOINTS) == 0
    assert select_gpu(job, {'gpu': 0}, {0: 7, 1: 0}, loads, ENDPOINTS) is None
    loads[0]['healthy'] = False
    assert select_gpu(job, {'gpu': 0}, {0: 4, 1: 0}, loads, ENDPOINTS) is None


def test_seed44_is_independent_while_each_benchmark_waits_for_full_previous_job():
    jobs = [{'id': 'tb43', 'predecessor': 'app43'}, {'id': 'bfcl44', 'predecessor': None},
            {'id': 'app44', 'predecessor': 'bfcl44'}, {'id': 'tb44', 'predecessor': 'app44'}]
    rows = {name: {'phase': 'pending'} for name in ['app43', 'tb43', 'bfcl44', 'app44', 'tb44']}
    rows['app43']['phase'] = 'running'
    assert [j['id'] for j in ready_jobs({'jobs': jobs}, rows)] == ['bfcl44']
    rows['bfcl44']['phase'] = 'running'
    assert ready_jobs({'jobs': jobs}, rows) == []
    rows['bfcl44']['phase'] = 'completed'
    assert [j['id'] for j in ready_jobs({'jobs': jobs}, rows)] == ['app44']


def test_dispatch_routes_actor_analysis_and_frozen_test_to_the_assigned_endpoint(tmp_path):
    job = {'id': 'example', 'train_command': ['python', 'run', '--base-url', ENDPOINTS[1]['client'], '--temperature', '.7'],
           'output': str(tmp_path / 'job'), 'method': 'premise', 'seed': 44}
    original = job['train_command'].copy()
    path, plan, assigned = dispatch_plan({'snapshot': str(tmp_path / 'snapshot')}, job, ENDPOINTS[0], 0)
    assert plan['base_url'] == assigned['base_url'] == ENDPOINTS[0]['client']
    assert assigned['train_command'][assigned['train_command'].index('--base-url') + 1] == ENDPOINTS[0]['client']
    assert assigned['command'][-3:] == [str(path), '--job', 'example']
    assert job['train_command'] == original and assigned['gpu'] == 0


def test_prepared_optimizer_configuration_moves_endpoint_without_changing_scientific_settings(tmp_path):
    import json
    from scripts.utils.run_dual_gpu_experiment_pool import route_prepared_config
    folder = tmp_path / 'job/train'
    folder.mkdir(parents=True)
    config = {'base_url': ENDPOINTS[1]['client'], 'temperature': .7, 'seed': 44,
              'actor_output_tokens': 2048, 'train': [{'task_id': 'one'}]}
    (folder / 'config.json').write_text(json.dumps(config))
    job = {'id': 'job', 'method': 'gepa', 'train_dir': str(folder), 'output': str(folder.parent)}
    route_prepared_config(job, ENDPOINTS[0]['client'])
    updated = json.loads((folder / 'config.json').read_text())
    assert updated == {**config, 'base_url': ENDPOINTS[0]['client']}
    assert len(list((folder.parent / 'routing_history').glob('*.migration.json'))) == 1


def test_endpoint_migration_refuses_any_existing_learning_artifacts(tmp_path):
    import json
    import pytest
    from scripts.utils.run_dual_gpu_experiment_pool import route_prepared_config
    folder = tmp_path / 'job/train'
    (folder / 'online').mkdir(parents=True)
    config = {'base_url': ENDPOINTS[1]['client'], 'seed': 44}
    (folder / 'config.json').write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match='learning artifacts'):
        route_prepared_config({'id': 'job', 'method': 'skillopt', 'train_dir': str(folder),
                               'output': str(folder.parent)}, ENDPOINTS[0]['client'])
    assert json.loads((folder / 'config.json').read_text()) == config


def test_process_tracking_uses_start_time_and_rejects_exited_or_reused_pids():
    import subprocess
    import sys
    from scripts.utils.run_dual_gpu_experiment_pool import identity, same_process

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        saved = identity(process.pid)
        assert saved is not None
        assert same_process({**saved, "argv": ["argv-may-change-during-exec"]})
        assert not same_process({**saved, "start_ticks": str(int(saved["start_ticks"]) + 1)})
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert not same_process(saved)


def test_error_row_adopts_the_existing_worker_without_launching_a_duplicate():
    from scripts.utils.run_dual_gpu_experiment_pool import existing_worker

    command = ["python", "/snapshot/fixed_matrix_worker.py", "--plan", "/run/a/dispatch_plan.json", "--job", "a"]
    job = {"id": "a", "command": command}
    saved = {"pid": 100, "start_ticks": "1000", "argv": command}
    row = {"phase": "error", "pid": 200, "dispatch_command": command}
    assert existing_worker(job, row, {tuple(command): [saved]}) == saved
    other_command = [*command[:3], "/different/run/dispatch_plan.json", *command[4:]]
    assert existing_worker(job, row, {tuple(other_command): [saved]}) is None


def test_recovery_refuses_ambiguous_duplicate_workers():
    import pytest
    from scripts.utils.run_dual_gpu_experiment_pool import existing_worker

    job = {"id": "a", "command": ["python", "/snapshot/fixed_matrix_worker.py", "--plan", "/unique/a.json"]}
    with pytest.raises(RuntimeError, match="Multiple live workers"):
        existing_worker(job, {}, {tuple(job["command"]): [{"pid": 1}, {"pid": 2}]})


def test_responsive_http_with_frozen_busy_engine_is_unhealthy_and_cannot_receive_new_jobs():
    from scripts.utils.run_dual_gpu_experiment_pool import check_generation_progress

    history = {}
    sample = {"healthy": True, "running": 55, "waiting": 0, "generation_tokens": 100, "completed_requests": 10}
    loads = {0: dict(sample), 1: {"healthy": False}}
    check_generation_progress(loads, history, timestamp=0)
    assert loads[0]["healthy"] and not loads[0]["stalled"]
    loads[0] = dict(sample)
    check_generation_progress(loads, history, timestamp=301)
    assert loads[0]["stalled"] and not loads[0]["healthy"]
    assert select_gpu({"seed": 44, "benchmark": "bfcl"}, {}, {0: 0, 1: 0}, loads, ENDPOINTS) is None


def test_idle_engine_is_healthy_and_token_progress_or_server_restart_clears_a_stall():
    from scripts.utils.run_dual_gpu_experiment_pool import check_generation_progress

    history = {}
    load = {"healthy": True, "running": 0, "waiting": 0, "generation_tokens": 100, "completed_requests": 10}
    loads = {0: dict(load)}
    check_generation_progress(loads, history, timestamp=0)
    check_generation_progress(loads, history, timestamp=1000)
    assert loads[0]["healthy"] and not loads[0]["stalled"]
    loads[0] = {**load, "running": 2}
    check_generation_progress(loads, history, timestamp=1301)
    assert loads[0]["stalled"]
    loads[0] = {**load, "running": 2, "generation_tokens": 101}
    check_generation_progress(loads, history, timestamp=1302)
    assert loads[0]["healthy"] and loads[0]["seconds_without_progress"] == 0
    loads[0] = {**load, "running": 1, "generation_tokens": 0, "completed_requests": 0}
    check_generation_progress(loads, history, timestamp=1800)
    assert loads[0]["healthy"] and not loads[0]["stalled"]


def test_http_failure_is_not_overridden_by_generation_health_check():
    from scripts.utils.run_dual_gpu_experiment_pool import check_generation_progress

    loads = {0: {"healthy": False}}
    check_generation_progress(loads, {}, timestamp=0)
    assert not loads[0]["healthy"]
