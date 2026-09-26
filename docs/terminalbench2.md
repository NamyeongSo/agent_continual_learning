# Terminal-Bench 2.0 as a training benchmark

This adapter uses the [Terminal-Bench 2.0 task release](https://github.com/harbor-framework/terminal-bench-2)
and [Harbor's Docker environment and verifier](https://www.harborframework.com/docs/run-jobs/run-evals).
ACE uses [Harbor Terminus-2](https://www.harborframework.com/docs/agents/terminus-2)
as its default Terminal-Bench actor. ACE passes its current playbook to Terminus-2,
then uses the agent trajectory and official verifier reward for post-task reflection
and playbook updates. PREMiSE retains the interactive `bash` and `finish` actions.

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

## Frozen training split

The 2.0 release contains 89 tasks. The included manifest assigns 50 to
training and 39 to validation. BFCL and AppWorld keep their existing seed-42
training IDs and take the first 39 validation IDs from the previous manifest.
The Terminal-Bench IDs are sorted, shuffled with the benchmark-specific seed,
and divided 50/39. Only training IDs are passed to the learning methods.

`scripts/utils/task_splits/seed42_train50_val39_terminalbench2.json` is ready
to use. To verify or recreate it after installing the benchmark and the other
two datasets:

```bash
.venv/bin/python scripts/utils/create_task_splits.py \
  --benchmarks bfcl,appworld,terminalbench2 \
  --seed 42 --train-count 50 --val-count 39 \
  --output scripts/utils/task_splits/seed42_train50_val39_terminalbench2.json
```

## Run ACE and PREMiSE

```bash
DOCKER_HOST=ssh://your-docker-host \
  bash scripts/ace/run_terminalbench2_qwen_training_time.sh

DOCKER_HOST=ssh://your-docker-host \
  bash scripts/premise/run_terminalbench2.sh
```

The ACE wrapper runs five selected Terminal-Bench training tasks by default,
drawn from the frozen 50-task training split. The pilot manifest lists `fix-git`,
`filter-js-from-html`, `sqlite-db-truncate`, `openssl-selfsigned-cert`, and
`regex-log`, with 15 Terminus-2 turns per task. Set `ACE_TERMINALBENCH2_MAX_TURNS`
for a longer run. For more tasks, set `NUM_TASKS` and `ACE_TASK_SPLIT_MANIFEST`
to the full 50/39 manifest. To use the
older step-by-step ACE actor, set `ACE_TERMINALBENCH2_ACTOR=interactive`. For
PREMiSE, pass `--num-tasks 1 --dry-run` to the wrapper for a wiring check.

Terminus-2 receives the task instruction plus ACE's current playbook and operates
the task container with its tmux terminal. The full ATIF trajectory is saved
under each session's `benchmark/harbor/agent/` directory. Tests are uploaded
only during `score()` after the agent stops acting. Harbor's verifier writes the
official `reward`; this is the Exgentic session score. The trajectory excerpt
and reward are supplied to ACE's reflector, followed by its curator update.
Any verifier failure is an experiment error, not a zero reward.

The benchmark's own agent can be run separately for a baseline:

```bash
~/.exgentic/benchmarks/terminalbench2/venv/bin/harbor run \
  --dataset terminal-bench@2.0 --agent terminus-2 --model <model>
```
