# Experiment monitoring

## Standalone evaluation queues

The journal's `http://127.0.0.1:8878/#live` page also reads standalone evaluation
queues registered by `outputs/<run>/evaluation_monitor.json`. It refreshes every
five seconds and shows queue order, dependency, controller state, split coverage,
successes, measured failures, infrastructure errors and failed attempts.
Infrastructure errors are excluded from score denominators; incomplete coverage
is labeled as an interim score. The reader never launches or retries a task.
Recovery cards also appear on their source experiment's detail page, and native
cards show the actual active task worker count and task IDs independently of
completed results. Refresh an already open page to load updated display code.

The November 13 Qwen3.8 cohort includes ReAct Terminal-Bench 2.1 (89 tasks),
ReAct BFCL/AppWorld (150 each), then native Terminus-2 Terminal-Bench 2.1 (89).
`scripts/litellm/evaluate_terminus2_after_queue.py` waits for terminal records for
all 300 predecessor tasks and a final controller marker before launching.
It freezes the default Harbor JSON prompt/parser/tmux protocol, empty playbook,
the requested xhigh sampler, retained reasoning history, and unlimited output.
The existing 100k input and 40 episode policy remains recorded separately from
Harbor's defaults. The queue has 100 worker slots and at most 89 active tasks.
Historical ReAct measurements and their audit limitations remain separate.

The latest November 13 follow-up uses **10 workers** to retry only the **48 native
Terminus-2 task-timeout failures** after all BFCL/AppWorld tasks finish. Selection
requires both original failure and `agent_timed_out=true`; successful timed-out
trajectories, other verifier failures and policy zeros are excluded. The earlier
100-slot native cohort and the interrupted full c10 rerun remain separate
historical artifacts. A five-task ReAct recovery queue follows the 48-task native
recovery and selects only the five original missing
outcomes. Its `disable_request_timeout=true` setting sends `httpx.Timeout(None)`
explicitly; omitting `timeout` would silently restore LiteLLM's 600-second default.
Recovery keeps the original ReAct sampler and budgets, uses a separate output
directory, and appears in the same live monitor.

The current replacement cohorts additionally set `exclude_model_call_time=true`:
the task's agent deadline charges wall time minus model-call intervals, including
internal retries. The prior six timeout outcomes remain in their interrupted
wall-clock cohort. New 48-task native and five-task ReAct outputs, journal records
and monitor labels identify the changed time policy explicitly.

`scripts/utils/monitor_experiments.py` audits learning and frozen checkpoint evaluation separately.
It uses the saved launch manifests, the seed-specific split JSON, result files, and actual Linux worker processes.
The current configuration and generated reports live under the ignored `outputs/experiment_monitor_20260929/`.

## Initial check and twenty-minute cycle

Run from the Exgentic directory:

```bash
.venv/bin/python scripts/utils/monitor_experiments.py \
  --config outputs/experiment_monitor_20260929/config.json \
  --output outputs/experiment_monitor_20260929 \
  --repair-known-infrastructure
```

Review `latest.md`, investigate its alerts, and finish any needed debugging before starting the timer:

```bash
sleep 1200
```

Wait for this process to finish. During the timer, perform no further experiment inspection or debugging.
Then repeat the audit, report progress and changes, address new failures, and start the next timer.
Do not launch another training matrix when existing workers are already alive.

For unattended report generation, adding `--watch` starts a real `sleep 1200` subprocess after each audit.
The monitor does no polling, file inspection, or model requests during that wait.
Stopping this monitor terminates its timer; it does not terminate training, evaluation, or vLLM.

## Report fields

- Learning: method, setting, seed, completed tasks, benchmark counts, next task, worker PIDs,
  skipped iterations, recent exception lines, and changes since the preceding report.
  Worker identity includes Linux start ticks so a restart is detectable even when a PID is reused.
- Progress within the current task: parent and replay completion, acting events, LLM response log growth,
  internal Terminus-2 turns and recorded episodes. A retry supersedes its failed attempt in the replay count.
  These signals reveal work in progress even when the training cursor has not advanced.
- Evaluation: checkpoints 50, 100, and 130 on all three benchmarks for each experiment.
  Native results, early checkpoint evaluations, and corrected copies count once.
- `processed`: tasks with a recorded result, including execution errors.
- `valid_results`: recorded tasks without execution errors. An officially graded zero is valid.
- `clean_complete`: every planned test ID is present, the frozen evaluation protocol matches,
  and no execution errors remain. Process exit code zero alone does not establish this.
- Frozen protocol checks: split hash, checkpoint hash, test IDs, evaluation without learning,
  and rank 1 for PREMiSE variants.
- SWE-Milestone: job status, milestone progress, worker survival, and validity for statistics.
- Alerts: stopped or duplicate workers, missing evaluation owners, stale activity, execution errors,
  protocol changes, and an unreachable model endpoint. Slow tasks are reported without automatic termination.

`latest.json` and `latest.md` contain the newest snapshot. `snapshots/` keeps timestamped JSON and Markdown;
`history.jsonl` stores aggregate changes. Reports use Korean Standard Time.

## Recovery of known infrastructure errors

`--repair-known-infrastructure` reuses frozen checkpoint workers to retry Docker address pool or subnet
failures, provider timeouts, and runner RPC timeouts in completed early PREMiSE evaluations.
Only affected records are reopened.
Original results are archived under the early evaluation batch's `repairs/` directory before replacement.
Valid results and context skips are retained. Live learning state and checkpoints are not modified.
An already active worker is allowed to finish before recovery starts.

ACE Terminal-Bench recovery waits for each original checkpoint summary and retries its failed tasks,
preserving deliberate context skips. Corrected summaries live in the ACE output's
`test_evaluation_terminalbench2_retry/`; original summaries remain intact.
Remaining errors continue to appear as alerts for investigation.

Recovery processes use separate service port blocks and file locks. Attempts are recorded in
`repair_workers.json` and bounded by `max_repair_attempts`; logs live in the monitor's `repairs/` directory.
Private endpoint and Docker settings are loaded from `worker_environment_file` into subprocess environments.
Reports do not record environment values or full process command lines. Keep configuration and results ignored.

Terminal-Bench's venv transport timeout covers the longest official agent, verifier, or environment setup
deadline in the task release plus ten minutes for communication and cleanup. This removes the generic
thirty-minute RPC limit without changing the task's official deadline or trajectory/context policy.
Existing Python workers load changes only when restarted; recovery launches use the updated code.

Model accessibility probes allow two minutes for a response under concurrent inference load.
A timed-out probe cancels its request, and a retained probe timeout is eligible for targeted provider recovery.
Authentication failures remain execution errors requiring diagnosis.

### SWE-Milestone SSH failures

The backend retries SSH identification or key-exchange failures up to three times within the original call deadline.
These failures precede remote command execution. Other disconnects are reported without repeating a mutating command.

The monitor can queue repository recovery with `--repair-known-infrastructure`.
Recovery waits on the original launcher's lock until its sequential queue finishes and also checks for orphaned workers.
It then reruns affected repository streams with a fresh ACE and namespace, retaining the original task order and settings.
This preserves the one-repository concurrency in the launch manifest.
Artifacts remain under the original output's `repairs/`; `recovery_index.json` references hash-checked replacement results.
Official unsuccessful results and blocked prerequisites alone do not trigger recovery.

Submission capture uses the same official source/test partition resolver as the pinned evaluator.
This avoids outdated dataset metadata omitting production directories declared by a migrated repository config.
Streams already affected by a capture-config mismatch are also eligible for a fresh repository recovery.

### SWE-Milestone context recovery

Only a real context overflow triggers Terminus-2 summarization. If the summary request also overflows,
retry the native summary with fewer history messages, at most three attempts. The full saved trajectory remains intact.
Failed fallback recovery raises an execution error immediately; it cannot consume the remaining turn budget
with Harbor's synthetic "Technical difficulties" replies.

Retained trajectories containing repeated synthetic replies invalidate the affected repository result for statistics.
The result reader records this diagnosis without rewriting the original score files, and the repository joins
the bounded recovery queue. A single transient synthetic reply in an older trajectory does not invalidate it by itself.

### Docker SSH connection reuse

One shared OpenSSH master can exhaust the server's session-channel limit under parallel Docker calls.
The local experiment environment installs `scripts/utils/docker_ssh_pool.py` as an ignored
`.venv/bin/ssh` symlink. Docker `system dial-stdio` requests share 16 masters with at most eight
channels each. Authentication is serialized to avoid the server's connection-startup limit;
other SSH commands pass through to the system client. Private sockets and lock files stay in
the user's cache. The wrapper applies to subsequent Docker calls in existing workers and leaves
experiment/actor concurrency unchanged. Validate transport with simultaneous read-only Docker probes.

Terminal-Bench artifact copies require an actual official reward file, even if directory copying
reports success. Empty copies are retried and reward files can be copied separately; transport failures
remain infrastructure errors. Native GEPA/SkillOpt actors derive binary task success from Harbor's
official reward, since successful verifier execution alone does not mean that the task passed.
Training histories produced using the wrong reward must be preserved and excluded, then replaced
by fresh runs with matched settings; do not relabel their optimizer decisions as valid.
