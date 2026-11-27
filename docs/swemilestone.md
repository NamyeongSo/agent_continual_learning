# SWE-Milestone: a development stream of milestone sessions

This adapter treats the complete development stream as one benchmark run and
each milestone as one Exgentic session. A repository's Docker container, Git
history, files, and dependency caches continue across sessions. The agent starts
a fresh conversation at each milestone; ACE can update its shared playbook after
each session.

The adapter uses the [official SWE-Milestone harness](https://github.com/DeepCommit-ai/SWE-Milestone)
at `17a8f1593e172e26b36cea15e2b30fb9536c93f5` and
[data v1.0.2](https://huggingface.co/datasets/DeepCommit-ai/SWE-Milestone-data/tree/v1.0.2)
at `3500478e740d42583277c2658aa7729eb36e10f4`. All Docker images are resolved
from that harness's digest manifest. Installations and experiment artifacts stay
outside tracked source files. The upstream harness is MIT licensed; it is
downloaded with its license rather than copied into this package.

## Install on an SSH Docker server

The official evaluator bind-mounts host files into Docker. Consequently both
the harness and its data run on the Docker server, while Exgentic and the agent
run on the client. No Docker CLI is needed on the client. The server needs
OpenSSH, Git, Python 3.10 or later, Docker access, outbound HTTPS for installation,
and enough disk for the benchmark images and container snapshots.
The Terminus-2 client runtime requires Python 3.12 or later. Harbor 0.23.0 is
installed only in the local `swemilestone` benchmark venv, outside the core
project environment; the official evaluator has a separate server venv.

The default Exgentic bridge uses `runner="direct"`; upstream dependencies stay
in the server's private environment. The default evaluation deadline is three
hours (`evaluation_timeout_sec=10800`) to accommodate long official suites.
Explicitly choosing the local `venv` runner also imposes its existing 30-minute
RPC deadline, so use the default bridge for longer evaluations.

Configure a working SSH alias using the team's ignored private handoff document,
then export it as the Docker endpoint. The example alias below is a placeholder;
the team's actual access instructions belong in `TEAM_PRIVATE_DOCKER_ACCESS.md`.
Never copy keys or private server addresses into benchmark configs or this guide.

```bash
uv sync --frozen
export DOCKER_HOST=ssh://YOUR_DOCKER_SSH_ALIAS
uv run exgentic install --benchmark swemilestone
uv run python -m exgentic.benchmarks.swemilestone.install --repo ripgrep --pull-images
uv run exgentic list benchmarks
```

Installation creates a private Python environment (including `ast-grep-cli==0.45.0`
for the official Rust test filter), pinned source/data checkouts,
and workers under `~/.exgentic/benchmarks/swemilestone` **on the server**. It does
not install system packages. Missing digest-pinned Docker images are otherwise
pulled when needed. Pre-pull all seven repositories with `--repo all --pull-images`.
With `DOCKER_HOST` unset, the same worker runs beside a local Docker engine.
When upgrading an earlier installation, use
`uv run exgentic install --benchmark swemilestone --force` to add Harbor.

`docker_host` and `backend_root` benchmark options override the endpoint and
server installation directory. For a custom installation directory, also pass
`--backend-root` to the installer. Keep the SSH endpoint exported when using
automatic environment installation.

## Run ACE one milestone at a time

```bash
export DOCKER_HOST=ssh://YOUR_DOCKER_SSH_ALIAS
uv run python examples/run_swemilestone.py \
  --repo ripgrep --model YOUR_LITELLM_MODEL --actor terminus2 \
  --training-feedback --run-id ripgrep-ace-train
```

`--num-tasks 3` runs the first three milestones in dependency order. It does
not sample or shuffle tasks. Available subsets are `ripgrep`, `navidrome`,
`dubbo`, `go-zero`, `nushell`, `element-web`, `scikit-learn`, and `all`.
The default is `ripgrep`. The complete dataset has 101 sessions: 98 graded
milestones and three ungraded milestones which still contribute code to later
milestones. In ripgrep, `maintenance_style_1` and
`milestone_seed_119407d_1_sub-01` are ungraded.

The example sets `curator_frequency=1`, `shuffle_mode="sequential"`, and
`agent.runner="direct"`, so ACE's in-process playbook survives between sessions.
`max_workers=1` is required and validated before execution. Each repository gets
its own development stream when using `all`; the example shares ACE learning
across those streams. Model credentials remain with the client agent.
The example defaults to `--actor terminus2`. `--max-turns` defaults to 200,
and `--agent-timeout` defaults to 10,800 seconds. Use `--actor interactive` for
the direct Bash tool loop. The lightweight benchmark config defaults to
`actor="interactive"`; set `actor="terminus2"` when constructing it directly.

The equivalent CLI configuration is:

```bash
uv run exgentic evaluate --benchmark swemilestone --subset ripgrep --agent ace \
  --model YOUR_LITELLM_MODEL --max-workers 1 --num-tasks 3 \
  --set agent.runner=direct --set agent.shuffle_mode=sequential \
  --set agent.benchmark_id=swemilestone --set agent.curator_frequency=1 \
  --set benchmark.actor=terminus2 \
  --set agent.training_time=true
```

## Repeat isolated repository experiments

Run all seven repositories sequentially and repeat the whole experiment three
times with the local Qwen server:

```bash
export DOCKER_HOST=ssh://YOUR_DOCKER_SSH_ALIAS
uv run python -u scripts/ace/run_swemilestone_isolated.py \
  --repeats 3 \
  --output-root outputs/swemilestone_ace_terminus2_isolated_3runs
```

The order is `ripgrep`, `navidrome`, `dubbo`, `go-zero`, `nushell`,
`element-web`, and `scikit-learn`, followed by the same order for the next
repeat. Only one repository runs at a time. Each of the 21 repository runs has
its own worker process, empty ACE playbook, output directory, and development
stream; learning and code persist between that repository's milestones.
The frozen plan contains all **303 sessions**, including nine ungraded sessions.
Use `--dry-run` to inspect it without starting agents.

Defaults match the local experiment server: `openai/Qwen/Qwen3.6-35B-A3B`
at `http://127.0.0.1:8006/v1`, no thinking, 131,072 context tokens, 8,192 output
tokens, 200 acting turns per milestone, and official training feedback enabled.
Terminus retains backend sampling defaults; ACE uses its existing temperature
of 1. The task-selection seed stays at 42 and the dependency order is fixed.
Each repeat makes fresh model calls; these are repeated trials with the same
settings, rather than different task shuffles. The zero model cost fields
describe this self-hosted server and do not estimate GPU costs. Override model
and token limits for a different server; `--no-training-feedback` retains silent
grading. Context recovery still occurs only after an overflow error.

`launch_manifest.json` freezes the plan. `status.json` tracks the active worker,
and `<repeat_repo>/progress.json` tracks its milestone phase. Per-session agent
checkpoints and a final repository playbook are retained. `summary.json` and
`summary.md` update after each repository run: they report each repository's
mean and **sample standard deviation (`ddof=1`)**, as well as the mean and
standard deviation of the three overall scores. An overall score is the
unweighted mean of the seven official repository scores within one repeat.

An unsubmitted graded milestone contributes zero to the official repository
denominator, including descendants blocked by an unsubmitted prerequisite.
Infrastructure errors are recorded separately and excluded from score
statistics. Pending or excluded repeats are never filled with zero; a final
three-repeat estimate requires `n=3`. Use a fresh output directory for each
launch to preserve existing results.

## Terminus-2 recovery after context overflow

The Terminus actor uses `enable_summarize=True` and
`proactive_summarization_threshold=0`. It also disables the proactive check
explicitly: Harbor 0.23.0 otherwise compares negative free-token estimates
against zero, which can trigger a summary before an overflow error. No token
threshold triggers compaction in this adapter. Only `ContextLengthExceededError`
enters Harbor's native summary and recovery path. That path attempts the
three-stage summary, questions, and answers handoff, then its shorter-summary
and terminal-state fallbacks if necessary. Recovery can fail if the task
instruction itself exceeds the model context or the provider remains unavailable.

Terminus operates the existing official development container through the SSH
worker; it does not create a Harbor task container or run Harbor's verifier.
For a new development stream, the official initialization hook prepares `tmux`
before network quarantine. Each milestone receives a fresh tmux session and
conversation in `/testbed`. Its terminal is stopped before submission, while
repository files and caches persist. An older stream without tmux requires a
fresh `run_id` or `stream_id`; the adapter does not reopen its quarantined network.

ACE supplies the current playbook to `run_terminus2`, receives the complete
terminal trajectory, and calls `finish`. Only Terminus's confirmed completion
(two completion signals, as required by its native loop) creates the official
submission. Exhausting turns or failing the runtime does not submit partial
work. Official scoring and ACE reflection occur after submission. Terminus's
context recovery applies to its acting conversation; ACE reflector and curator
context errors still skip learning and preserve the previous playbook.

Local artifacts under `benchmark/terminus/` include `trajectory.json`, any
summary/question/answer trajectories, and `runtime.log`. Session metadata records
the Harbor version, summary count, usage, and `context_recovery="on_overflow"`.
For custom models, `model_info` supplies Harbor's context and output token limits.
`terminus_python` overrides the isolated interpreter when needed.

## Submission, grading, and state

The interactive actor has `bash` and `finish` tools. Bash runs as the official development
user in `/testbed`; commands start at the repository root. `finish` commits the
working tree and tags it `agent-impl-{milestone_id}`. For graded milestones, the
official orchestrator captures the immutable source archive and integrity
sidecar, excluding hidden tests and disallowed paths. Its official evaluator
then runs the milestone image with frozen repository configuration and quarantine
policy. The agent cannot act again after submission.

Dependencies follow the official strong and additional dependency DAG. A valid
submitted snapshot unlocks its dependents even when its tests fail, matching
the official default early-unblock protocol. If the session ends without
submission, closing it schedules restoration of the milestone's starting image;
its dependents stay blocked. A session lease and server file lock reject
concurrent changes to the same stream.

Session scores use upstream's precision/recall harmonic mean, including official
invalid-test filters and build-failure handling. When compilation produces no
test report, the adapter retains upstream's error summary and collector score
zero, with `build_failure=true` and compiler diagnostics in session metadata.
Infrastructure-invalid results
raise an execution error rather than become an agent score of zero. `success`
means execution and grading completed; `session_metrics.resolved` records whether
all targets were fixed without regressions. Ungraded submissions report score
zero with `graded=false`; zero is a placeholder, not ground-truth failure feedback.
Those sessions omit training score feedback, so ACE uses its observation-only
reflection. Ungraded milestones still allow playbook updates.

Benchmark aggregation averages graded milestones within each repository, then
averages repositories. Missing graded milestones count as zero in the full
selected repository's denominator. A short prefix reports `partial=true` and
the number of missing milestones; it must not be presented as a completed
official benchmark result. Session rewards remain usable for online learning.

Local session artifacts include `benchmark/start_snapshot.json`, the scored
`benchmark/evaluation_result.json`, and `benchmark/results.json`. Server trials
retain original/filtered evaluation files, logs, source snapshots, and stream
state under `trials/{repository}/{namespace}`. The namespace derives from the
local run directory and optional `stream_id`; use a fresh run ID for independent
experiments. Containers and checkpoint images are retained for continuation and
replay. Cleanup is explicit on the Docker host; the adapter does not prune shared
Docker resources. Remote code persistence does not itself restore ACE's playbook
when restarting the client process; restore its saved checkpoint separately.

## Replay branches for PREMiSE and ps_st_pareto

Every milestone starts with a full container snapshot, preserving ignored build
files, Git state, and previous submissions. A candidate can replay that milestone
from the same state on the same Docker server:

```python
import json
from pathlib import Path
from exgentic import SWEMilestoneBenchmark

snapshot = json.loads(Path("outputs/RUN/sessions/SESSION/benchmark/start_snapshot.json").read_text())
candidate = SWEMilestoneBenchmark(
    subset="ripgrep",
    stream_id="candidate-001",
    start_snapshot=snapshot,
)
# Execute only the target milestone using its task ID and score that session.
# Use another fresh stream_id for each candidate or past-task replay.
```

Snapshot identities include the repository, harness/data commits, baseline
commit, and already-submitted milestones. A branch cannot overwrite a populated
primary stream or silently switch its original snapshot. Selecting a candidate's
code into the primary stream and adapting the existing independent-task PREMiSE
training scripts require a separate development-stream selection policy. This
adapter supplies isolated replay and official per-session rewards; it does not
automatically run a Pareto search or promote candidate code.

For a past milestone, replay **its own start snapshot**, not the current codebase.
Otherwise past-score comparisons measure changes in repository state as well as
changes in the playbook. Candidate-only aggregate scores use the full repository
denominator; use `SessionScore.score` for each current/past Pareto objective.

## Protocol difference from the original benchmark

The [official protocol](https://github.com/DeepCommit-ai/SWE-Milestone/blob/17a8f1593e172e26b36cea15e2b30fb9536c93f5/README.md)
uses one continuing agent conversation and silent evaluation. This adapter
deliberately adds fresh milestone conversations. Enabling `--training-feedback`
also passes the official session reward to ACE before its reflection/curation
step; omitting it retains silent grading toward the agent. Report the setting
alongside results: milestone-session adaptation and feedback-assisted training
are research protocols, and are not directly comparable to the original silent
continuous-conversation leaderboard.

## Validation (2026-09-28)

The SSH Docker server installation and all 101 task definitions were checked.
The 12 required ripgrep images were pulled by manifest digest. A no-model smoke
run through the local venv bridge submitted a milestone, recorded its official
compilation-failure score of zero, continued to its dependent milestone with the
same filesystem, restored unsubmitted edits, and opened an isolated branch from
the original start snapshot. The default direct bridge was also checked against
the recorded official outcome. These are infrastructure checks, not agent
performance results.

Focused adapter/orchestrator/ACE tests: **39 passed**. Runner tests: **133 passed,
11 skipped**. The broader suite, excluding live installers and integrations:
**380 passed, 18 skipped, 3 failed**; the failures are existing `test_cost.py`
expectations that disagree with LiteLLM's model cost map. Both repository-wide
pre-commit and explicit checks of the new files passed. Full Ruff still reports
errors in existing source files; the new adapter, tests, and example pass Ruff.
Full evaluations of the other six repositories and model-based training were
not run.

The Terminus-2 integration additionally passed **55 focused tests, 3 skipped**
and **10 tests in the isolated Harbor environment** (including existing episode
replay and Terminal-Bench adapter regressions). The host skips require the
isolated Harbor dependency and are covered by the isolated run. A local
OpenAI-compatible mock API injected a context
overflow while the real Terminus runtime operated the SSH development container.
The first milestone recovered through the native three-stage handoff exactly
once; the next milestone started a fresh conversation, reused the persistent
marker, and performed no summary. Both used the official submission and grading
path. These scripted no-op implementations scored zero and do not measure model
performance. No paid model calls were made.
