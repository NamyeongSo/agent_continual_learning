# Deferred isolated Qwen experiments

`scripts/utils/run_qwen_isolated_when_free.py` executes a prepared
`launch_manifest.json`. The scheduled comparison contains ACE and PREMiSE on
BFCL, AppWorld and Terminal-Bench 2.0, with seeds 42, 43 and 44: 18 independent
learning streams, 780 training iterations and 774 final test tasks.

All Qwen roles use temperature 0.7, including acting, shortlisting, reflection,
curation, mutation, alignment judging and native Terminus-2. The optional
`--temperature` flags in ACE/PREMiSE apply an explicit override; omitted flags
preserve previous role-specific sampling. Actor limits follow
[the shared experiment policy](experiment_limits.md). Analysis retains its
8,192-token output budget. The frozen split, ADD-only updates, past=3, beam size,
prompts, task ordering and benchmark deadlines retain their isolated baseline settings.

The controller requires GPU 1 to report no compute processes, at most 512 MiB
memory and 0% utilization for three consecutive checks 30 seconds apart. Both
ports 8006 and 18006 must be available. It never signals existing GPU owners.
It holds a per-GPU lease against duplicate Exgentic controllers, checks the supplied
server script's hash, and invokes `/home/thskadud/vllm_serve_qwen.sh` unchanged.
Model readiness must be served by a process in the controller's server process group.

The script starts the existing Qwen parser bridge on port 18006. Before any
benchmark runs, it verifies identical auto/none tool-choice prompt token IDs and
performs a small parsed-tool-call smoke check on its owned server. A failed check
records `blocked` rather than starting incompatible experiments.

Six learning streams run concurrently, filling available slots with the first
pending stream in manifest order across methods and benchmarks. PREMiSE
keeps parent/reflection/mutation/replay limits 2/4/4/28 and 32 shared LLM slots per
training process. Benchmark session semaphores remain disabled. Final rank-one
playbooks are frozen independently as training finishes and evaluated through a
shared 100-process pool. ACE evaluation disables learning. Budget failures are
measured zero-score failures; infrastructure errors retain separate status and
are retried at most twice, then reported as unresolved rather than valid zeros.
Evaluation does not prevent the scheduler from filling available training slots.
An optional `runtime_schedule.json` in the output directory, containing
`{"train_workers": 6}`, overrides training capacity on each scheduling cycle.
This changes only scheduling; per-stream learning settings and evaluation capacity
remain unchanged. A capacity reduction lets existing workers finish.

Each prepared output directory retains the source snapshot and file hashes,
task split, launch commands, server-script hash, controller/server/bridge process
identities, `status.json`, `progress.json`, training logs and per-task evaluation
attempts. The local research journal records preparation, actual startup and the
final state. `running` is set only after model readiness and preflight pass.

The detached controller survives the originating terminal. A machine reboot or
controller crash requires inspecting the recorded processes and state before
recovery. After stopping only the previous controller and its evaluation workers,
`--resume-existing` adopts the recorded server, bridge and live training processes,
validates their identities and commands, and preserves completed evaluation records.
The previous controller and evaluation pool must be stopped before resuming to
avoid concurrent evaluation writes. The per-GPU lease prevents duplicate controllers.
Record a scheduling change and controller source hash alongside the original
manifest and source hashes; preserve historical training/evaluation artifacts.

## Queue a curator variant after the current comparison

`scripts/utils/run_qwen_followup.py --output-root NEW --after-output BASELINE`
waits until every predecessor learning stream is terminal, its final evaluation
queue is empty, and all completed/evaluation-error streams have complete task
records and summaries. It acquires the shared GPU lease only after this check, so
the waiting controller does not interfere with the baseline. A controller crash
with unfinished jobs does not count as completion. Terminal infrastructure errors
are retained in `predecessor_completion.json` and the journal rather than hidden.

The follow-up uses its own prepared `launch_manifest.json`, code snapshot, source
hashes and output directories. Set `playbook_edit_mode=add_update_delete` when
building the plan to add the flag to all 18 ACE/PREMiSE training commands and
verify it when freezing final checkpoints. Global training capacity can be 12
with `train_workers_per_method={"ace":6,"premise":6}` to fill six slots per method;
the shared final evaluation pool remains 100. The original baseline artifacts are
read only. The follow-up reuses the recorded Qwen server and parser bridge after
checking their process/port ownership and repeating the model preflight.

While queued, `status.json` reports `waiting_predecessor` and the journal stays
`planned`. The detached waiting process survives terminal disconnection. The
model is not relaunched, and its original log paths are recorded in
`services_reused.json` after handoff. If the recorded services have stopped, the
follow-up records a blocker for recovery rather than using an unrelated server.

## Scheduled overnight suspension

`scripts/utils/schedule_qwen_pause.py --plan PLAN.json` runs as a detached controller
with explicit timezone-aware `pause_at` and `resume_at` timestamps. The current
reservation uses Asia/Seoul: stop on 2026-10-21 at 18:00 KST and start service recovery
on 2026-10-22 at 10:00 KST. The schedule folder is recorded in
`/tmp/exgentic_qwen_pause_schedule_path`; its `state.json`, `heartbeat.json` and
`scheduler.log` show execution and the next action. The live journal at `#live`
shows paused streams separately from errors and displays the restart time in KST.

Suspension first freezes the three queue controllers, then stops their retained
process identities, the owned vLLM/parser bridge, and exact matching benchmark
containers. Completed results, playbook checkpoints and cached optimizer decisions
remain in their original directories. Interrupted cancellation artifacts are archived
under the schedule output rather than recorded as new official task failures.
An interrupted trajectory or analysis call can run again after resumption; this is
checkpoint recovery rather than in-memory continuation. Scheduled suspension does
not consume a bounded infrastructure failure retry.

At the reserved restart time, the scheduler waits for the GPU lease, an idle GPU 1
and free service ports, starts the unchanged supplied vLLM script and parser bridge,
then repeats model preflight. Old vLLM logs are renamed before the script opens its
new log. It restarts the original frozen queues and commands with retained results,
ACE/PREMiSE capacity six each and GEPA/SkillOpt capacity three each; the curator
variant still follows its original predecessor. GPU use by another owner is never
interrupted. The scheduler survives terminal disconnection; machine/container
shutdown still requires inspecting the saved state and restarting the controller.

The Qwen launchers construct their worker environment explicitly, including both
`OPENAI_API_BASE` and `OPENAI_BASE_URL` for the parser bridge, the project `.venv/bin`
at the front of `PATH`, and the inherited private Docker transport. Model preflight
checks that the Docker CLI is available before training starts. These settings apply
to new training processes, resumed queues, follow-up queues, and evaluation workers;
resumption must not depend on a user's interactive shell configuration.

The 2026-10-22 GPU handoff adopted the already running, identically configured GPU 0
Qwen server. Its queue wrappers and PID/start-time socket checks are retained under
`outputs/qwen_gpu0_migration_20261022`, with a GPU 0 lease and an
`active_service_override.json` in each original queue directory. The historical
launch manifests and actor source snapshots remain unchanged. Thirty-five ACE
attempts affected by omitted endpoint/PATH settings during that handoff were
archived together with their infrastructure-only checkpoints and metrics, then
requeued from the prior completed prefix. The handoff's `error_audit.json` and
`environment_recovery/repair_summary.json` record the affected tasks and recovery.
# 2026-10-23 concurrent curator launch

The ADD+UPDATE+DELETE variant now shares the existing GPU 0 Qwen server while
the ADD-only and native optimizer runs continue. The user requested ACE's nine
benchmark/seed streams immediately and PREMiSE's six BFCL/AppWorld streams first.
Each completed PREMiSE training stream frees a slot for the next Terminal-Bench
seed, keeping PREMiSE at six concurrent learners. Final testing does not occupy
a training slot. The total initial learner concurrency is fifteen.

The pending variant's `runtime_schedule.json` and launch manifest record
`train_workers=15`, `train_workers_per_method={"ace":9,"premise":6}` and
`port_block_base=50000` (ports 50000–64999). Its controller uses the explicit
`--shared-followup` entry in the recorded GPU 0 operational launcher. Model,
temperature, seeds, task splits, actor limits and internal method parallelism
retain their frozen values. Five-task checkpoints and final evaluations use
the existing shared 100-worker inference queue. Launch evidence is in
`outputs/curator_launch_20261023/`.
