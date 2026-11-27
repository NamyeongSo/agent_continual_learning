# PREMiSE request budgets and timing records

`scripts/premise/run_experiment.py` accepts `--llm-timeout` (default 600 seconds per request)
and `--llm-retries` (default 1 retry). These apply to PREMiSE's tool selection, acting,
reflection, mutation, and alignment calls. They do not configure the native Terminus-2 actor.
Terminal-Bench continues to use its native sampling settings and official agent/verifier deadlines;
PREMiSE's tool-calling sampling settings are not forwarded to it.

All future BFCL, AppWorld, and Terminal-Bench actor trajectories use the
[shared experiment limits](experiment_limits.md): 100,000 input tokens and 2,048 output
tokens per environment turn, at most 30 BFCL turns or 40 AppWorld/Terminal-Bench turns.
Tool selection and acting share the turn budget; replay prefixes count toward the
trajectory limit. The same limits cover training, candidate replay, and evaluation.
Analysis calls retain their own output budget. Larger old launcher settings cannot
raise these hard caps.

Acting connection/timeout retries use one explicit retry loop with LiteLLM retries
disabled. Analysis requests use the OpenAI client's bounded retry policy. Exhausted
service requests retain a `CompletionRetriesExhaustedError` marker through serialized
session errors; training skips that update under its service-recovery policy.
Actor budget violations and output truncation instead terminate the trajectory with
score 0 and `success=false`, with no output-truncation retry, whole-session retry,
or service-error skip. A failed candidate remains a measured failed candidate.
Recoverable environment failures still use `--replay-retries`.

Each analysis call writes a private JSON file under `<output>/llm_calls/` containing its stage,
prompt, response, error, request timeout/retry settings, queue duration, and request duration.
Invalid JSON responses are retained. A timeout with no received response records the error;
it cannot reveal text that remained on the server. These files can contain benchmark data and
must stay out of version control. SDK retries are included in the recorded request duration.

Terminal-Bench writes these diagnostic files in each session's `benchmark/harbor/` directory:

- `phase_timings.json`: environment startup, agent setup/run, and verifier boundaries, including
  failures and timeouts.
- `agent/execution_metrics.json`: Harbor's agent metadata, including available API request durations.
- `agent/query_events.jsonl`: PREMiSE's native query start/end/error records. They exclude command
  execution. Replay calls are marked separately. Recursive recovery calls have a `depth` field;
  nested durations overlap and must not be summed as independent time.

The budgeted Terminal-Bench actor bypasses Harbor's output-truncation recovery loop.
Only responses accepted within the shared budget become replay episodes and execute commands.
The legacy replay helper retains recovery behavior for other benchmark integrations.

These changes do not shorten official test scripts, remove downloads or dependency installations,
or change Harbor's connection-retry policy. A long verifier phase still needs its stdout/stderr and, if
necessary, a separate timed reproduction to identify the command responsible. The timing record
alone does not prove that a specific download caused the whole delay.

Existing running Python processes do not reload these source changes. Fresh processes use the
new behavior; historical artifacts and already loaded workers retain their original behavior.

## Restart sequential training after BFCL

`scripts/premise/restart_sequential.py` copies checkpoint 50 and iterations 1–50 into a new
output root. The copied state retains both beam entries, replay memory, task order, and RNG state.
The original experiment is read only. Copied files and the source checkpoint have SHA-256
provenance in each new seed directory's `restart_provenance.json`.

```bash
.venv/bin/python scripts/premise/restart_sequential.py \
  --source-root outputs/original_experiment \
  --output-root outputs/new_experiment \
  --seeds 43 44 --evaluation-workers 50 --port-block-base 30000
```

Each seed's training process runs AppWorld tasks 51–100 and immediately continues with
Terminal-Bench tasks 101–130. The controller watches immutable checkpoint files 100 and 130
and submits every `(seed, checkpoint, benchmark, task_id)` tuple to one 50-worker evaluation
queue. AppWorld, Terminal-Bench, and BFCL test tasks are mixed in the queue. There are no
per-benchmark evaluation limits or waits for previous checkpoint evaluations to finish.
Training uses separate processes and its existing worker settings.

For seeds 43 and 44 with the frozen 50/50/29 test splits, this schedules 516 test executions.
Results keep the existing `test_evaluation/cp100_<benchmark>.json` and `cp130_<benchmark>.json`
format, with task-level persistence and checkpoint/manifest identity checks. Execution errors
are explicitly recorded and contribute zero under the existing evaluation convention.

`--prepare-only` copies and validates the inputs without starting tasks. `--run-prepared`
starts such a prepared directory; it refuses one already marked as started, to avoid duplicate
training processes. The controller holds a file lock. Progress is in `restart_status.json` and
`logs/s<seed>_training.log`; `restart_complete.json` appears after training and the evaluation
queue have both finished. Completed test rows can be reused by the evaluation queue, but the
launcher does not automatically adopt training processes after a controller crash.

`--execution-tag` separates new session and AppWorld environment IDs when resuming from the
latest saved training state after a runtime policy change. It leaves checkpoint configuration,
beam, memory, and RNG state intact. Prior session directories remain available.

Add `--interleaved-seeds 42` to prepare a fresh interleaved run alongside sequential seeds 43/44.
It reuses the original frozen configuration and task order, initializes the beam and memory empty,
and resets the RNG from seed 42. Its checkpoints 50/100/130 join the same evaluation queue as
the sequential checkpoints 100/130. Training remains independent of evaluation. The combined
50/50/29 test splits schedule 903 evaluations through one pool of 50 workers.

## Ground reflection and mutation in the exposed API

PREMiSE traces retain the public action names and exact schemas/descriptions for recorded tools
and completion actions. For older memory traces, reflection and mutation read these definitions
from the original session's public `session.json`; they do not read private evaluator files.
Hidden action definitions are excluded. Native Terminal-Bench episode tools are not constrained
by the outer adapter's gateway schema.

Terminal-Bench reflection uses a native-episode prompt and a JSON schema with bounded integer
span indices. It removes the duplicate `agent_trace` and request/usage metadata from feedback.
The full episode text is retained when it fits. The serving model's `/tokenize` endpoint checks
the complete system/user messages against its context size, reserving 8,192 output tokens and
a 1,024-token margin. Only overflowing fields are shortened, with explicit middle-omission markers;
every episode keeps its original index for replay. If tokenization is unavailable, the current Qwen
configuration uses a conservative UTF-8 byte bound. This does not summarize the acting conversation.

`scripts/premise/rerun_isolated_terminalbench2_reflection.py` creates fresh isolated seed 42/43/44
training runs and cp030 tests in a separate output root. It verifies the original task order and
stores the source hashes and task split. By default it disables session timeouts, following the
experiment's timeout-removal policy; `--enforce-session-timeouts` retains the official time budget.
Disabling timeouts makes the scores diagnostic rather than standard Terminal-Bench scores.
Progress is recorded in `status.json`; each seed automatically evaluates its 29 test tasks after
all 30 training iterations finish.

Both prompts prioritize the task and public API documentation over existing playbook claims.
This priority is also provided as a system instruction so imperative playbook entries are treated
as data to check. AppWorld grading retains failed requirements, labels and metrics in the prompts;
private assertion traces with expected-answer values are excluded, while original records remain intact.
They distinguish allowed and required parameters, backend aliases and advertised agent actions,
successful task termination and official semantic grading, and recovered errors and final failures.
Mutation also receives the full original trajectory to check later recovery beyond the selected span.
Concrete API rules must name their benchmark, and unsupported reflection proposals should produce
no addition. The default preserves the existing ADD-only mutation protocol.
Both ACE and PREMiSE also support explicit [playbook edit modes](playbook_edit_modes.md)
for ADD + UPDATE and ADD + UPDATE + DELETE in new runs; changing a checkpoint's
edit policy on resume is rejected.

For AppWorld, the exposed `finish` argument description says to pass `answer` iff the task
requests an answer; state-change tasks retain its default `None` by omitting the argument.
This is public API guidance, not a rule inferred from hidden expected answers. The regression
example demonstrates why changing an invalid `output` argument to an unnecessary answer summary
can still fail semantic grading after every requested state change succeeds.
An optional default of `None` does not by itself permit explicitly passing JSON null. The prompts
recommend omitting the argument when null is excluded from the input schema.

## AppWorld environment setup and replay overhead

AppWorld session metadata is now constructed by the lightweight benchmark class with the same
helper used by its evaluator. Task execution no longer starts a separate evaluator process only
to assemble constructor arguments. Evaluators remain available for discovery and aggregation;
the real task environment is still created in its own isolated runner for every attempt.

The benchmark declares its action space static within a session. Prefix replay then fetches the
468 public action definitions once and reuses them for all prefix actions and continuation setup.
It still executes every recorded prefix action in a fresh environment. Benchmarks with dynamic
action spaces, including BFCL, retain per-action definition lookups by default. Definitions and
mutable task state are never shared across attempts.

Direct setup QA on two tasks, two repetitions each, with four concurrent workers measured:

| Phase | Before, median seconds | After, median seconds |
| --- | ---: | ---: |
| Initial setup, including fetching the acting inputs | 8.72 | 5.92 |
| Replay setup with 10–11 prefix actions | 17.32 | 6.63 |
| Prefix start/restoration alone | 9.15 | 1.51 |

These measurements include no LLM calls. All eight before/after pairs had identical task text,
context, public action schemas, and byte-identical snapshots of all 12 task databases. They are
setup measurements on those tasks, not an estimate of end-to-end training speed. The remaining
roughly five-second environment process startup includes CLI/service imports, AppWorld imports,
fresh world creation/database loading, API documentation loading, and action model construction.

`benchmark/appworld_startup_timings.json` records constructor phases. Its manifest phase includes
the nested tool documentation and action-model phases; do not sum nested durations. The clock
retains the real monotonic callable before AppWorld freezes time, so task dates do not corrupt
duration measurements. `benchmark/prefix_replay_timings.json` records the schema transfer and
prefix action execution durations separately.

The timestamp gap between framework `config.json` and `session.json` is not a prefix replay
timer: the session manifest is written inside environment construction, before replay starts.

The restart launcher prepends the workspace `src` directory to `PYTHONPATH` for its controller,
training workers, and their isolated benchmark subprocesses. Benchmark venvs otherwise import
the framework copy installed when the environment was created. Dependencies continue to come
from the selected isolated venv. This ensures the running benchmark adapter matches the source
used in setup QA without rebuilding or reinstalling benchmark environments.

## Tool parser recovery (2026-10-04)

The server error requiring `--enable-auto-tool-choice` and `--tool-call-parser` is a
configuration failure. PREMiSE now stops without advancing the task or writing a skip
when this error crosses the session boundary. Provider timeouts retain their service-recovery policy. Actor output truncation now follows
the shared zero-score failure policy described above.

`scripts/utils/serve_vllm_tool_parser_bridge.py`, run with the vLLM environment's Python,
provides the installed `Qwen3CoderToolParser` to selected experiment clients. For the
current Qwen template, upstream `tool_choice=none` retains the same tools and generation
prompt as `auto`; `/tokenize` must confirm identical token IDs before using this bridge.
It parses nonstreaming tool responses and preserves truncated finish reasons. Native
text calls and streams pass through with their sampling and token settings unchanged.

The recovery archives the previous state, tail iterations, checkpoints and evaluations.
Sequential seeds 43/44 restore the saved cursor 101; interleaved seed 42 restores cursor 56,
including the original beam, memory and RNG state. A new execution tag keeps session and
AppWorld environment paths separate. All checkpoint evaluations are repeated in the shared
queue with 50 workers, while training proceeds independently.

This recovery does not change reflection, mutation or judge prompts, their token limits,
or handling of context overflow. Those failures remain separately visible in iteration records.

## Verification of the runtime changes (2026-10-03)

- PREMiSE and matrix launcher tests: 78 passed.
- Terminal-Bench adapter and native replay tests in the installed Harbor environment: 15 passed.
  These use fake environments/LLMs and Harbor's real parser/recovery loop; they do not run Docker
  tasks or measure end-to-end speed. The Harbor test environment reports an existing unknown
  `asyncio_mode` pytest option warning.
- Ruff passes for the new regression test and the changed model settings, PREMiSE instance,
  Terminal-Bench adapter/replay files, and their tests. The other touched files retain 43 existing
  lint findings (long lines, import placement, and existing pipeline/script warnings).
- `git diff --check` passes.

Restored-history QA before these changes did not reproduce the original 600-second acting
timeouts. The historical `sam-cell-seg` verifier log ends in a MobileSAM Git dependency failure;
its 38.8 MB weights download took 17 seconds. There are no per-command timestamps to assign
the entire roughly 93-minute verifier interval to that Git operation. These runtime fixes do not
establish or resolve the cause of that historical infrastructure failure.
