# Checkpoint policy shift after frozen evaluation

`scripts/utils/schedule_checkpoint_policy_shift.py` schedules the current isolated
original-policy ACE AUD, GEPA, SkillOpt and PEACE AUD streams for seeds 42–44 across
BFCL, AppWorld and Terminal-Bench. It reads the checkpoint queue without changing it.
The completion gate requires the selected scope to match, all its tasks to be completed,
no pending/running/error tasks or invalidated selected streams, an explicitly completed
controller dashboard and zero active workers. An empty pending count alone is insufficient.

```bash
uv run python scripts/utils/schedule_checkpoint_policy_shift.py \
  --prepare --check-only --queue-output outputs/qwen_five_task_cp_queue_20261022 \
  --output outputs/policy_shift
uv run python scripts/utils/schedule_checkpoint_policy_shift.py --output outputs/policy_shift
```

Preparation freezes the runtime, scope, official Harbor JSON parser and configuration.
The runner then freezes the available checkpoint policies and their source hashes.
CP0 is the actual initial policy: ACE's initial playbook, PEACE's original parent,
GEPA candidate zero, or SkillOpt skill zero. Later policies use the checkpoint queue's
existing selections, including GEPA's highest full-validation mean available at each boundary.
Unavailable final checkpoints remain missing; earlier policies are not substituted for them.

## Fixed histories and inference

For each benchmark/seed, all four methods share the acting inputs saved by ACE's earliest
available checkpoint on the frozen training task set. All valid acting turns are retained.
Tool-selection calls, malformed/truncated source responses and duplicate native logging
callbacks are excluded with a task-level coverage audit. Failed trajectories may still supply
valid earlier acting inputs. The measured comparison uses new CP0 predictions on these fixed
inputs, rather than treating the observed source actions as correct answers.

Only the actor's policy changes. BFCL/AppWorld replace the original system instruction;
native Terminal-Bench replaces the exact learned-policy span in the initial user instruction.
Task descriptions, all other messages, previous tool-call IDs, observations, images and actual
available tools remain identical. AppWorld therefore retains the API interface and environment
policy of the recorded experiment; this study does not migrate histories to Python code generation.
No environments, Docker containers, learning updates or predicted actions are executed.

One shared queue serves all methods, seeds, checkpoints and benchmarks with at most **128**
concurrent requests. It uses Qwen3.6, temperature **0.7**, thinking disabled, at most **100,000**
input and **2,048** output tokens. Input size is checked through the serving tokenizer before
generation. Budget violations and malformed measured outputs are not resampled. Transport
retries are finite; an unavailable model service pauses dispatch and preserves pending work.
Successful immutable requests and identical policies share cached predictions. CP0 has a separate
independent repeat, providing a sampling-variance control without pinning a provider seed.
When a later checkpoint has exactly the initial policy, it shares the independent root
repeat used for evaluation, rather than comparing the cached reference response with itself.
This preserves the same stochastic comparison at CP0 and unchanged checkpoints.

## Metrics

| Benchmark | Action agreement |
|---|---|
| BFCL, AppWorld | Loose and strict accuracy and F1 |
| Terminal-Bench | Accuracy of the native ordered commands, keystroke strings, durations and task-completion flag |

Loose function-call agreement compares names. Strict agreement includes parsed JSON arguments,
canonicalizing object-key order. Accuracy requires the full ordered sequence to match; F1 uses
multiset overlap and retains duplicate calls. Native Terminal-Bench uses the frozen official
Harbor parser and excludes free-form analysis and planning text from the action.

Primary curves use the common successful-history intersection across all methods and available
checkpoints within a benchmark/seed. Histories are averaged within task, tasks equally within seed,
then seeds equally; bands show sample seed standard deviation. Explicit errors and missing data
are coverage losses, never zero-filled agreement. These are conditional action-agreement metrics,
separate from benchmark task success. `policy_shift = 1 - agreement`. The CP0 curve point uses
its independent repeat, making sampling noise visible.

Results include `policy_shift.sqlite3`, immutable histories/policies/responses, `history_audit.json`,
`analysis/{task,seed,checkpoint}_metrics.csv`, coverage and summary JSON, and per-benchmark
PNG/PDF/SVG plots. The scheduler publishes final artifacts to the local research journal.
