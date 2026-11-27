# PEACE candidate evaluation with saved-action strict F1

Enable the variant with `--candidate-evaluation span_policy_shift`. Parent task execution,
reflection, AUD mutation, replay-memory sampling, two-parent beam size, and current/past
Pareto selection remain the PEACE procedure. Candidate evaluation makes model requests only:
it creates no benchmark session, restores no environment, executes no actions, and calls no
alignment judge. Final validation/test still execute the frozen selected playbook normally.

## Scoring

For a source trajectory's reflection span `[start, end)`, retain the original actor request
before every action in that interval. Replace only the learned actor instruction with the
candidate instruction. All other messages, observations, tool-call IDs, environment policy,
images and the exact available tools remain as recorded. Generate one action per saved prefix.
Never append a generated action or obtain its observation before evaluating the next prefix.
Four positions therefore require four candidate actor requests, regardless of the remaining
trajectory length.

The reference is the **stored training action**, including its original sampled arguments.
The source policy is not queried again. Its history, action and original official outcome stay
fixed when the trajectory is sampled from memory in later iterations.

For each position, compute strict F1 using exact function-name/parsed-argument multiset overlap,
with canonical JSON object-key order and duplicate counts. Average position F1 equally within
the span. This is the strict F1 definition in the CP10 → ΔCP policy-shift experiment, not string
or semantic similarity between code fragments. In AppWorld code-generation mode, an action's
entire `execute_python` code argument participates in exact matching.

- Originally successful trajectory: evaluation score = mean strict F1.
- Originally failed trajectory: evaluation score = 1 − mean strict F1.

The direction uses the original **official success boolean**, not the reflection's span type,
the candidate's output, a partial test score, or an inferred threshold. Missing outcomes cannot
be used as failures. New AppWorld parent trajectories retain partial scores separately and use
official TGC for their reported score.

Terminal-Bench uses its recorded official native JSON parser without importing Harbor or an
environment in the evaluator. For this variant, strict F1 compares command atoms containing
exact keystrokes and duration, plus a completion atom when `task_complete=true`. Analysis and
plan prose are excluded. This extends the old Terminal-Bench accuracy metric explicitly;
BFCL/AppWorld strict F1 is unchanged. Ordinary assistant-message actions compare their content.

Each candidate receives `(current evaluation score, mean past evaluation score)`. Past scores
are averaged per trajectory, so long spans do not get extra weight. With no past memory,
selection uses only the current score. Pareto fronts, descending score sum, descending minimum
score, and ascending numeric candidate ID are the existing no-alignment tie breakers. The top
two candidates become parents. Evaluation scores are stored as `current_evaluation_score` and
`past_evaluation_score`, never as official benchmark scores.

## Running

```bash
uv run python scripts/premise/run_experiment.py \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json \
  --output-dir outputs/peace_aud_span_policy_shift_s42 \
  --benchmarks appworld --mode isolated --seed 42 --all-train-tasks \
  --model openai/Qwen/Qwen3.6-35B-A3B --base-url http://127.0.0.1:8006/v1 \
  --temperature 0.7 --playbook-edit-mode add_update_delete --past 3 \
  --candidate-evaluation span_policy_shift --policy-shift-temperature 0
```

Use the same split manifest and `--appworld-policy` file as the intended comparison cohort.
Keep the variant in its own output directory. Selecting this evaluator disables alignment
automatically. It requires `--method premise` and `--candidate-validation current_past`.

`--temperature` still controls parent acting, reflection and mutation. The separate
`--policy-shift-temperature` defaults to zero, matching the cited shift experiment's probes;
set it explicitly to another value when desired. This does not regenerate reference actions.
All candidates' current and past span positions enter one shared action queue per iteration.
For four candidates, four trajectories each and four positions per span, the queue contains
64 independent action jobs. `--replay-workers` bounds simultaneous action workers (default 28),
and `--llm-calls` bounds actual shared model calls (default 32). Effective request concurrency
is at most the smaller limit. There are no nested per-span worker pools. Completion order may
vary; results are restored to action-position order before averaging F1.

## Provenance and limits

Opt-in parent recordings save exact acting prefixes, tools and original response actions in
`policy_shift_actions.jsonl`; tool-selection requests are excluded. Native recordings also
pin the parser path and SHA-256. Memory points at original parent recordings, not candidate
predictions. Prefixes from a previous replay continuation are rejected.

`state.json` pins `candidate_evaluation`, `span-policy-shift-strict-f1-v1`, reference type,
temperature, aggregation, `shared_action_queue_v1` scheduling and outcome source.
A different evaluator or probe temperature cannot
silently resume the same checkpoint. Existing runs continue to default to environment replay.

Candidate request/response/usage records are saved under `policy_shift_calls/`, with per-position
F1 and history hashes in iteration artifacts. Credentials are excluded. Replay session counts
are zero; span evaluations and action generations have separate counters. Queue metrics also
record submitted positions, peak action workers and cancelled pending jobs. Official parent
results remain separate from surrogate selection scores.

The shared `actor-turn-v1` policy applies to every generated action: at most 100,000 input and
2,048 output tokens per position, including measured retries, and original position limits of
30 BFCL or 40 AppWorld/Terminal-Bench turns. Truncated or over-budget output ends that candidate's
trajectory evaluation with score zero, never novelty credit, and is not retried. Pending action
jobs for that span are cancelled. Already running requests finish and their usage is retained;
other spans continue. Transport
retries are finite; malformed outputs and missing source records cannot earn a shift reward.

## Matched fresh BFCL comparison

`scripts/utils/launch_peace_span_policy_shift.py` adds the new evaluator to a frozen fresh
two-epoch temperature-zero BFCL cohort. It copies the baseline snapshot and overlays only the
new evaluation/recording code. Baseline actor, reflector/curator prompts, mutation application,
budget policy, task order and algorithm seed remain pinned. No learned playbook, checkpoint,
trajectory, or evaluation cache is copied.

The launcher creates three lanes, one per seed 42–44. Each lane runs `past=9`, then `past=3`,
then `past=6`, using a fresh initial playbook for each run. A completed seed starts its own next
setting without waiting for other seeds. The gate is the existing complete training plus
matching full final test with zero infrastructure errors. Frozen final train/val evaluation
uses the same separate watcher as the comparison cohort.

Each setting uses current=1, action workers=28 and LLM calls=28. Past memory sampling uses up
to the configured count, without replacement and excluding the current task. Output variant
names include the past count, keeping all nine runs distinct.

```bash
uv run python scripts/utils/launch_peace_span_policy_shift.py prepare \
  --source outputs/qwen_bfcl_temp0_fresh_two_epochs_12runs_20261109 \
  --output outputs/qwen_bfcl_span_policy_shift_temp0_fresh_two_epochs_p9_p3_p6_20261109
```

The generated `span_runner.py` provides `check`, `run`, `worker` and `train` commands.
Run `check --output OUT --job JOB_ID` for each planned job before starting the controller.
The existing baseline controller and its frozen code are not changed.
