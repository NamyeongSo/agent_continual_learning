# Terminal-Bench 2.0 and 2.1

## Restore native Terminus-2 execution

The standalone native evaluator supports `benchmark_kwargs.native_terminus2=true`.
It constructs Harbor's original `Terminus2` class directly, preserving the JSON
parser, prompt, summarization, truncation recovery, and recording defaults. It
does not use the shared `BudgetedTerminus2` wrapper. Set `max_interactions=null`
and omit `max_tokens` from model settings to avoid the extra 40-turn and output
caps. The native model metadata must match the serving endpoint's actual context
limit; `native_model_info` supplies this to Harbor's model registry.

An explicit native `max_interactions` value is forwarded as Harbor's `max_turns`;
for example, `max_interactions=100` permits 100 main agent episodes. It does not
restore the shared input/output budgets. Summary requests and provider retries
are not additional main agent turns.

The November 14 Qwen3.8 all89 run uses two workers, native summarization, no
shared 100k input or 40-turn limit, and no explicit output cap. The earlier user
choice to exclude model-call time from the task deadline remains an explicit
exception to official wall-clock timing. Frozen configs and per-task
`native_agent_config.json` record these conditions separately from older runs.

### Repair context overflow without changing the actor prompt

For the separate context-repair cohort, set `native_context_recovery=true` with
`native_terminus2=true`. `RecoveringTerminus2` retains Harbor's prompt, parser,
agent loop and three-stage summary prompts. It counts inputs through the serving
endpoint's `/tokenize`, including preserved reasoning. The counter mirrors
vLLM's `reasoning_content` to `reasoning` normalization: its tokenization endpoint
does not apply the chat-completion request's legacy-field normalization itself.
Real request preflight verifies counts against `usage.prompt_tokens`.

When necessary, recovery removes complete old exchanges while preserving the
task and latest exchange, and retries the native summary at most three times.
Failed recovery raises an infrastructure error instead of allowing upstream's
synthetic `Technical difficulties` response to drive repeated agent episodes.
Per-task `context_recovery.jsonl` records reductions and summary outcomes.

`model_settings.disable_request_timeout=true` passes `httpx.Timeout(None)` and
disables hidden provider retries, avoiding LiteLLM's default 600-second request
cutoff during healthy long generation. Transport errors still have finite Harbor
and worker retries. This does not remove command/verifier deadlines or add an
output cap.

The November 14 repair run uses a dedicated GPU0 endpoint at port 8008 with
1,000,000-token context and the model card's YaRN factor 4 configuration. The
existing port 8007 service remains separate. A real 270,076-token generation
passed preflight. All 89 outcomes are generated afresh at concurrency 2; old
262k results remain in their original cohort. This is a documented protocol
variant, not a claim of exact reproduction of the HF 73.0% result.

## Resize a running standalone queue

The controller can adopt existing workers through `adopt_workers_file`. Manifest
entries record task IDs, PIDs, Linux process start ticks and exact commands.
Adopted tasks occupy pool slots before pending tasks launch, preventing duplicate
trajectories during a concurrency increase. A worker that exits without a
committed outcome is an infrastructure error. Preserve earlier config/source
snapshots and record the concurrency history. The output directory can retain
its original name; the config and live monitor show the current concurrency.

The November 14 run changed from 2 to 10 workers at the user's request. Server
`max_num_seqs` changed to 10 as well. Its existing two workers were paused during
model calls, after the task clock suspended the deadline; responses were drained
before restarting the dedicated server. The replacement controller adopts those
trajectories and fills the remaining slots with pending tasks.

## Apply a turn limit to existing trajectories

The user's subsequent GPU restriction moves this experiment to physical GPU 1
only. The GPU1 endpoint at port 8007 is upgraded from 262k to 1M with the same
YaRN factor 4 parameters. The GPU0 port 8008 service stays stopped. Completed
outcomes are preserved; canceled session directories are archived before the
remaining tasks restart at concurrency 10 and a native 100-turn limit.

The November 14 run later received a user-requested 100-turn cap. New workers
use native `max_turns=100`. Already running workers retain their chat and task
files: a per-container tmux guard counts the native loop's `has-session` checks
and returns false before episode 101, allowing the original loop to exit and
the official verifier to evaluate the preserved state. Other tmux commands pass
through to the original binary. Each guard is seeded at an unambiguous in-flight
episode and recorded in `agent/live_turn_limit.json`.

For these live guards, Harbor may report `n_episodes=101` for the blocked
liveness check; count the main trajectory's `source=agent` steps as executed
turns. The cap excludes summary subagents. Retroactive review keeps original
records before setting any outcome with more than 100 actual agent turns to
zero under the explicit user policy; these overrides are separate from official
verifier rewards and infrastructure errors.

## Excluding model-call time from the task deadline

Set `benchmark_kwargs.exclude_model_call_time=true` to charge only non-model
agent time against each task's agent deadline. This is an opt-in user protocol;
the default retains the official wall-clock deadline. Native Terminus-2 pauses
the deadline around the full provider conversation call, including its internal
retries, and resumes it before running commands. Interactive ReAct shares the
same accounting through a per-session request ledger.

Environment command time and verifier deadlines remain enforced. Individual
provider request timeouts are independent of the task deadline. The native
runner transport ceiling is removed in this mode so it cannot reintroduce an
overall wall-clock cutoff. Results record `task_time_accounting`, including
`wall_seconds`, `model_call_seconds`, `charged_seconds`, and policy version
`agent-active-time-v1`. Keep these results separate from wall-clock cohorts.

## Run the corrected 2.1 release

[Terminal-Bench 2.1](https://www.tbench.ai/news/terminal-bench-2-1) repairs issues
in 28 of the 89 tasks: external dependency drift, insufficient resources, and
instructions that did not match their tests. The adapter uses the
[official 2.1 repository](https://github.com/harbor-framework/terminal-bench-2-1),
pinned to `12b7565679ce5006230b820666353673a133db93`. Its 89 task IDs match
the installed 2.0 release, so existing frozen split IDs can be reused.

New installations install both releases:

```bash
uv run exgentic install --benchmark terminalbench2
```

If Terminal-Bench is already installed, add only the 2.1 data without reinstalling
the shared Harbor environment. Run from this Exgentic project directory:

```bash
EXGENTIC_TERMINALBENCH21_TASK_ROOT="$HOME/.exgentic/benchmarks/terminalbench2/terminal-bench-2-1" \
  bash src/exgentic/benchmarks/terminalbench2/setup.sh 2.1

.venv/bin/exgentic list tasks --benchmark terminalbench2 --subset 2.1 --limit 5
docker info

bash scripts/run_terminalbench21.sh \
  --agent tool_calling --model <provider/model> --num-tasks 1
```

The wrapper calls `exgentic evaluate --benchmark terminalbench2 --subset 2.1`
and saves to `outputs/terminalbench21`. Pass `--output-dir` or set
`TERMINALBENCH21_OUTPUT_DIR` to separate model runs. `DOCKER_HOST=ssh://...`
works as with 2.0. The standard CLI can also select `--subset 2.1` directly;
Python callers use `TerminalBench2Benchmark(subset="2.1")`.

2.1 tasks live under
`~/.exgentic/benchmarks/terminalbench2/terminal-bench-2-1/tasks`.
`EXGENTIC_TERMINALBENCH21_TASK_ROOT` can point to the repository or its `tasks/`
directory. This is independent of the older `EXGENTIC_TERMINALBENCH2_TASK_ROOT`.
2.1 requires a clean Git checkout at the pinned commit, so a 2.0 directory or
locally modified verifier cannot silently be used as 2.1. Each session and
aggregate records `dataset_version` and `dataset_commit`. Aggregation rejects
results from another release. Shared actor limits still apply: 40 turns,
100,000 input tokens and 2,048 output tokens per turn. Official agent/verifier
timeouts and per-task resources come from the selected release's `task.toml`.

ACE accepts `--terminalbench2-version 2.1`; its Qwen shell launcher accepts
`ACE_TERMINALBENCH2_VERSION=2.1`. PREMiSE accepts the same CLI option, or
`"subset": "2.1"` in the manifest's `benchmark_kwargs`. For example:

```bash
.venv/bin/python scripts/ace/run_experiment.py \
  --benchmarks terminalbench2 --terminalbench2-version 2.1 \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json \
  --mode isolated --seed 42 --all-train-tasks \
  --model <provider/model> --training-time \
  --output-dir outputs/ace_terminalbench21_s42 --dry-run

.venv/bin/python scripts/premise/run_experiment.py \
  --benchmarks terminalbench2 --terminalbench2-version 2.1 \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json \
  --mode isolated --seed 42 --all-train-tasks \
  --output-dir outputs/premise_terminalbench21_s42 --dry-run
```

Remove `--dry-run` to train with the chosen model and endpoint. Both methods
save the selected subset in `benchmark_configs`, carry it into replay/frozen
evaluation, and reject resuming an output directory with another release.
The existing default remains 2.0 for historical experiments; select 2.1
explicitly and use fresh output directories.

GEPA and SkillOpt use the same version option for training, validation gates
and final frozen-policy testing:

```bash
.venv/bin/python scripts/baselines/run_experiment.py \
  --method gepa --benchmark terminalbench2 --terminalbench2-version 2.1 \
  --seed 42 --workers 50 --output outputs/gepa_terminalbench21_s42 --dry-run

.venv/bin/python scripts/baselines/run_experiment.py \
  --method skillopt --benchmark terminalbench2 --terminalbench2-version 2.1 \
  --seed 42 --workers 50 --output outputs/skillopt_terminalbench21_s42 --dry-run
```

Remove `--dry-run` to execute. Their immutable `config.json` records
`benchmark_kwargs.subset`, which reaches every actor request and checkpoint
evaluation. Reusing a 2.0 output directory for 2.1 is rejected. To create a new
18-run GEPA/SkillOpt queue, pass `--terminalbench2-version 2.1` to
`scripts/baselines/run_queue.py`; the six Terminal-Bench jobs receive that
selection. An existing frozen queue retains its original release.

## Existing 2.0 workflow

This adapter uses the [Terminal-Bench 2.0 task release](https://github.com/harbor-framework/terminal-bench-2)
and [Harbor's Docker environment and verifier](https://www.harborframework.com/docs/run-jobs/run-evals).
ACE and PREMiSE use [Harbor Terminus-2](https://www.harborframework.com/docs/agents/terminus-2)
as their default Terminal-Bench actor. Both pass their current playbook to Terminus-2
and use the official verifier reward. PREMiSE reflects on Terminus-2's internal
episode trajectory and replays an action prefix before candidate continuation.
The interactive `bash` and `finish` actions remain available as an alternate actor.

## Install

Use Python 3.12 or newer, `git`, a Docker CLI with Compose, and a reachable
Docker daemon. `DOCKER_HOST=ssh://...` is supported; verifier artifacts are
copied with Docker instead of host bind mounts. The benchmark is installed in
its own Exgentic venv and the 89 task definitions are pinned to commit
`2fd12b88aafdd04a52c298e3940bcb189f9766d6`.

```bash
uv run exgentic install --benchmark terminalbench2
docker info
uv run exgentic list benchmarks
```

The installed task root is
`~/.exgentic/benchmarks/terminalbench2/terminal-bench-2`. To use an existing
copy of that exact release, set `EXGENTIC_TERMINALBENCH2_TASK_ROOT` before
running Exgentic. The default benchmark runner is `venv`, so Harbor stays out
of the host project's Python environment.

## Images and parallel Docker networks

All pinned 2.0 tasks use prebuilt, explicitly tagged Docker images. Sessions
start with `force_build=False`, reuse images cached on the selected Docker
daemon, and preserve those images during container teardown. A missing image
is pulled on first use; there is no startup check that preloads all 89 images.
Replay creates a fresh writable container layer from the same cached image.

Each session retains its own isolated Compose network. Its subnet is a `/28`
drawn from `10.240.0.0/16`, avoiding Docker's small default network address
pools. Set `EXGENTIC_TERMINALBENCH2_NETWORK_POOL` to an unused private IPv4
range on another host. Docker subnet collisions are retried with another
subnet; teardown removes the session network and temporary Compose override.
This does not impose a session concurrency limit or disk quota.

## Frozen train/validation/test split

The 2.0 release contains 89 tasks. The current split assigns shuffled positions
1–30 to training, 31–60 to validation, and 61–89 to test. BFCL, AppWorld, and
SWE-bench use 50/50/50. Task IDs are sorted before one deterministic shuffle
per benchmark and split seed. Only training IDs are passed to the learning
methods. See [the split policy](../scripts/utils/TASK_SPLITS.md) for seed derivation,
the nine benchmark/seed partitions, and evaluation rules.

`scripts/utils/task_splits/seed42_train_val_test.json` contains the seed-42
runtime manifest. To verify or recreate it after installing all four datasets:

```bash
.venv/bin/python scripts/utils/create_task_splits.py \
  --seed 42 \
  --output scripts/utils/task_splits/seed42_train_val_test.json
```

## Run ACE and PREMiSE

```bash
DOCKER_HOST=ssh://your-docker-host \
  bash scripts/ace/run_terminalbench2_qwen_training_time.sh

DOCKER_HOST=ssh://your-docker-host \
  bash scripts/premise/run_terminalbench2.sh
```

The ACE wrapper runs five selected Terminal-Bench training tasks by default,
drawn from the frozen 30-task training split. The pilot manifest lists `fix-git`,
`filter-js-from-html`, `sqlite-db-truncate`, `openssl-selfsigned-cert`, and
`regex-log`, with at most 40 Terminus-2 turns per task under the
[shared actor budget](experiment_limits.md). Larger launcher values cannot raise that cap. For the complete training split, set `NUM_TASKS=30` and
`ACE_TASK_SPLIT_MANIFEST` to the seed-42 runtime manifest. To use the
older step-by-step ACE actor, set `ACE_TERMINALBENCH2_ACTOR=interactive`. For
PREMiSE, the wrapper uses all 50 BFCL, 50 AppWorld, and 30 Terminal-Bench train
tasks by default. Pass `--num-tasks 1 --dry-run` for a wiring check.

Concurrent parent and candidate runs can have the same deterministic ExGentic
session ID for a task. Harbor's Docker Compose project gets a fresh execution
suffix so these runs use separate containers and networks. The ExGentic session
ID and per-run artifact paths remain stable for resuming and analysis.

Terminus-2 receives the task instruction plus ACE's current playbook and operates
the task container with its tmux terminal. The full ATIF trajectory is saved
under each session's `benchmark/harbor/agent/` directory. Tests are uploaded
only during `score()` after the agent stops acting. Harbor's verifier writes the
official `reward`; this is the Exgentic session score. All agent messages,
commands, and terminal observations from the trajectory are supplied to ACE's
reflector with the reward, followed by its curator update. No per-turn or
whole-trajectory clipping is applied to this reflection input. Terminal-Bench
uses the same reflection prompt as the other benchmarks, without a separate
100-word field limit. Terminus-2 context summarization is disabled. If the
generator, reflector, or curator raises a context length error, ACE skips the
current task's remaining learning steps and preserves the previous playbook.
The actor uses the shared 100,000-input/2,048-output token budget per episode.
A budget violation or truncated actor output is a completed zero-score task failure.
Reflection and curation have independent token budgets. Skips are recorded in each session's
`agent/task_skip.json`. The task's interaction limit and model output limit are
configured separately from the reflection input.
Any verifier failure is an experiment error, not a zero reward.

Verifier artifact downloads retry directory transfers up to three times. If an
auxiliary log still prevents the transfer, the adapter downloads the official
reward files separately and lets Harbor parse them. Transfer attempts and their
underlying exceptions are saved in `benchmark/harbor/verifier_transfer.json`;
phase failures retain their tracebacks in `phase_timings.json`. After a verifier
failure, the adapter spends at most 15 seconds preserving diagnostics before
deleting the container. This does not extend the verifier's official deadline
or accept a reward produced after that deadline.

`scripts/utils/retry_interleaved_queue_errors.py` retries execution errors from
interleaved seed-44 test queues with the original frozen checkpoints and model
settings. It preserves original results, discovers errors from sources that are
still running, and writes combined scores to the recovery output's `scores.json`.
Each failed tuple receives one additional full session; completed original
tasks are retained and the full benchmark denominator stays unchanged.

For evaluations without elapsed-time cutoffs, set the benchmark option
`enforce_session_timeouts=False`. This disables the task's agent and verifier
deadlines and the venv runner's RPC response deadline. The interaction limit,
individual command timeout, environment setup checks, and model request behavior
still apply. Result metadata records this option so these evaluations can be
distinguished from runs using the official task deadlines.

PREMiSE stores each Terminus-2 response and ATIF episode without extra trajectory
clipping. For candidate replay, it passes the recorded episode file path and the
selected prefix length to a fresh Harbor environment. Terminus-2 executes the
prefix through its parser and tmux executor without model calls, then continues
under the candidate playbook using fresh terminal observations. The same 40-turn
budget includes prefix and continuation. See [the PREMiSE protocol](../scripts/premise/README.md)
for the six-run task order, official-score selection, and checkpoint tests.

The benchmark's own agent can be run separately for a baseline:

```bash
~/.exgentic/benchmarks/terminalbench2/venv/bin/harbor run \
  --dataset terminal-bench@2.0 --agent terminus-2 --model <model>
```

See [transport recovery and automatic container cleanup](terminalbench_operations.md)
for ownership checks, bounded teardown and the background janitor.
