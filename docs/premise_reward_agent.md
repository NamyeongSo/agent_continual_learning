# PREMiSE reward judge: one-decision bandit

For the separate batch rubric discovery/selection method from PReMISE (arXiv:2605.30803v1),
see [the rubric judge implementation](premise_rubric_judge.md).

The reward judge selects A or B for a pair of trajectories attempting the same task.
Its binary training reward is agreement with the higher official task score. Equal-score pairs are omitted.
It learns a shared playbook through three parents, three children per parent, reflection, mutation, and full decision replay.
Candidates are selected from current/past accuracy Pareto fronts. No environment or action span is replayed;
the original trajectory comparison is an immutable input to the single judge decision.

## Frozen data and validation

`scripts/premise/run_reward_agent.py` builds a snapshot from the counted pair metadata.
Use `--source-seed 42` to include only original PREMiSE seed 42 data from both source settings.
The split targets 20% of each benchmark's pairs for validation, keeping every pair of the same benchmark/task
in one split. Whole-task subset selection matches the rounded pair target where possible.
A/B winner positions are balanced separately within each benchmark and split. The two training settings
use identical examples, labels, holdouts, initial empty playbooks, and model configuration.

The judge sees public task/context/tool contracts and anonymized trajectories, without scores, labels,
benchmark/task identifiers, source settings, or original/replay designations. Recorded continuation prefixes
are restored using the actual prefix feedback shown to the continuation agent; native Terminal trajectories already contain it.
Identical prefixes are represented once and apply to both attempts.
Private verifier output, native final metrics, and source model/playbook metadata are excluded.
Public tool parameters are retained from the session's `arguments_schema` field, including required arguments
and optional completion parameters. Earlier frozen datasets that omitted this field remain immutable;
use a new dataset/output directory to train with the corrected contracts.
Historical composite action logs can also omit the parameters of nested tool calls because they were
serialized through the base `SingleAction` schema. New trajectory recordings retain concrete arguments.
Dataset preparation recovers missing composite parameters by exact call ID and tool name from public
assistant call history (or BFCL's executed `turn_step_actions`). It never uses checker answers for recovery.
Valid composite calls with unrecoverable required parameters reject dataset preparation. Recorded prefix
checks account for the same recoverable serialization loss while still rejecting changed nonempty arguments.
Actual prefix actions retain the parameters replay really received, including historical empty parameters;
their actual feedback can differ from the parent's feedback. Parent validation reports are removed from
these prefix entries because replay rebuilds validation against its own execution parameters.
Previously frozen judge inputs and reported scores require a fresh dataset and evaluation to measure this fix.

Each turn's **combined environment feedback is limited to 500 Qwen model tokens**.
The explicit `[환경 feedback 생략됨: 턴당 500토큰 제한]` marker is included in this limit.
Agent actions and their arguments are preserved. Feedback from multiple tool calls in one turn shares this budget.
The same frozen representation feeds the judge, reflector, mutator, current/past replays, and validation.

## Learning and reports

Sequential training processes BFCL, AppWorld, then Terminal-Bench. Interleaved training randomly selects a
nonempty benchmark pool and visits every training pair once. At most ten previous training pairs are
sampled with benchmark-balanced quotas; validation never enters this memory or learning feedback.
Reflection and mutation use temperature 0.7 for candidate diversity; judgment uses temperature 0.
Rules may be added or updated. Deletion is disabled, and there is no hard rule-count or total playbook text limit.
Mutations containing
benchmark names, observed concrete identifiers and their API aliases, function-like identifiers, or case examples
are rejected. The reflector's evidence may cite concrete actions, while learned criteria use generic task roles.

Baseline and checkpoints after the sequential benchmark boundaries are evaluated on the same held-out pairs.
Both settings use the same checkpoint step counts. Validation uses frozen rank-1 rules and a shared tuple queue
with concurrency 50. A global limit of 50 model requests includes training calls; training does not wait for
validation to finish. Per-benchmark `accuracy` in `validation/cp*.json` is the mean binary judge reward.
Execution errors are reported separately; an official validation accuracy is published only after a clean completion.

All calls use output limit 2048, request timeout 600 seconds, one additional transient request retry, and
up to five additional identical requests after output truncation. Exhaustion invalidates that decision/candidate;
Fewer valid candidates than the configured parent count skip the training update. Saved state, checkpoints, and completed validation
records are retained when resuming.

Judge reasons are limited by instruction to 60 words; reflection fields to 35 words each / 250 words total;
mutation rules to two sentences / 60 words each. This avoids retelling long trajectories and repeated output-limit calls.
Mutation rejection records include the validation message. LLM traces retain each attempt's duration and raw content,
including truncated replies. These instructions reduce overruns but are not hard tokenizer limits on generation.
To intentionally revise prompts during a run, use `--resume --revise-prompts`. The old state/configuration is archived
in `prompt_revisions/`; the cursor, beam, memory, data split and RNG are preserved. Other configuration changes remain
rejected. Existing checkpoints keep their original configuration; later checkpoints record the revised prompt hash.

```bash
.venv/bin/python -u scripts/premise/run_reward_agent.py \
  --output outputs/premise_reward_agent_seed42_RUN \
  --source-seed 42 --seed 42 --prepare-only

.venv/bin/python -u scripts/premise/run_reward_agent.py \
  --output outputs/premise_reward_agent_seed42_RUN \
  --source-seed 42 --seed 42 --resume
```

`dataset/manifest.json` freezes split membership, tokenization, labels, input hashes, and provenance.
`launch_manifest.json` records the exact prompts. `status.json` reports both training cursors and all validation groups.
Prompts/responses and benchmark traces can contain private benchmark data; experiment outputs remain ignored.

## Comparing 3x3 and 2x2 beams

`--compare-beams` runs both `(parents=3, children_per_parent=3)` and `(parents=2, children_per_parent=2)` in
sequential and interleaved order. All four training runs and their validation requests share one global LLM
semaphore (`--workers 50`). Both variants use the same frozen dataset, holdouts, seed, prompts and past-count 10.
The score is solely whether the A/B judge preference agrees with the higher official environment score:
1 for agreement, 0 for disagreement. Current and sampled past accuracy remain the Pareto objectives;
there is no alignment score or alignment call. Outputs are in `p3_c3/` and `p2_c2/`, with an aggregate `status.json`.

```bash
.venv/bin/python -u scripts/premise/run_reward_agent.py \
  --output outputs/premise_reward_agent_seed42_beam_comparison_RUN \
  --source-seed 42 --seed 42 --past-count 10 --workers 50 --compare-beams --resume
```

For a single beam variant, use `--parents 2 --children-per-parent 2 --past-count 10` (or the defaults 3/3/10).
Existing experiments with different structural settings cannot be resumed under the new defaults;
keep their outputs and start a separate experiment with the same frozen dataset.

## Diagnosing overly abstract rules

`scripts/premise/analyze_judge_specificity.py` runs a diagnostic on frozen seed-42 data.
It compares the existing 20-rubric selection with a 20-rubric selection that includes criteria rejected by
the genericity filter. Each selection is tested with and without additional public-contract applicability
guidance; an empty-rubric baseline is included. Both trajectory orientations are evaluated.

The diagnostic also restores previously rejected concrete rules and compares strict versus relaxed
reflection/mutation on selected training cases. Each parent and proposal is replayed on its source pair
and ten pairs from different training task groups. Validation labels never enter generation or selection.
Rejected proposals retain the parent in the diagnostic's intention-to-treat summary. This reporting policy
does not change the production learner's selection algorithm.

```bash
.venv/bin/python -u scripts/premise/analyze_judge_specificity.py \
  --rubric-source outputs/COMPLETED_RUBRIC_EXPERIMENT \
  --reward-source outputs/COMPLETED_BEAM_COMPARISON \
  --output outputs/judge_specificity_diagnostic_RUN \
  --cases-per-benchmark 3 --generation-cases-per-benchmark 2 --workers 50
```

Use `--prepare-only` to inspect available rejected cases and rubric selections without model requests.
Repeated execution reuses completed requests when the frozen configuration and code hashes match;
`--retry-errors` retries failed requests while preserving their previous records. The output includes
`status.json`, `summary.json`, `report.md`, prompt variants, source snapshots and request traces.
Intervals resample task groups, keeping both orientations together. These development comparisons and
small, deliberately affected case samples cannot substitute for a new independent final test.
Production judge, reflector, mutator and mutation validation remain unchanged by this script.

`scripts/premise/analyze_judge_contracts.py` measures the impact of restoring previously omitted
public API argument schemas. It changes only `public_tools` on the same frozen validation pairs.
Rubrics, labels, trajectory order, feedback truncation and splits remain identical. It waits for the
specificity queue to complete, then evaluates the empty-rubric and expanded/scoped conditions at
concurrency 50. Terminal pairs without public tool schemas reuse their unchanged baseline results.

```bash
.venv/bin/python -u scripts/premise/analyze_judge_contracts.py \
  --specificity-source outputs/judge_specificity_diagnostic_RUN \
  --output outputs/judge_specificity_diagnostic_RUN/contract_schema_control --workers 50
```

## Measuring accuracy after input repairs

`scripts/premise/evaluate_repaired_judge.py` compares the original frozen dataset with a newly prepared
dataset containing public API schemas, recovered composite-call arguments and actual replay-prefix
feedback. It rejects changes to pair IDs, validation splits, labels or official-score provenance.
The empty-rubric and existing discovered-rubric conditions use both A/B orientations and the same
strict `{"choice": "A" | "B"}` response schema. Labels and scores are excluded from model messages.
This measures input repairs with fixed criteria; it does not retrain or select criteria on validation.

```bash
.venv/bin/python -u scripts/premise/evaluate_repaired_judge.py \
  --original-dataset outputs/ORIGINAL_REWARD_EXPERIMENT/dataset \
  --repaired-dataset outputs/REPAIRED_DATASET_RUN/dataset \
  --rubric-source outputs/COMPLETED_RUBRIC_EXPERIMENT \
  --output outputs/judge_repaired_input_evaluation_RUN --workers 50 --timeout 120
```

All conditions share one queue. Identical requests, including unchanged Terminal inputs, are reused
while retaining every evaluation row. `status.json` records progress; `calls/` contains request and
response traces. Final `report_ko.md` and `summary.json` contain per-benchmark accuracy, order sensitivity
and paired task-group intervals. Errors are reported separately and prevent an official accuracy for
the affected condition. Repeating the command reuses completed calls; `--retry-errors` retries failed
calls and archives their previous records. Historical results using `json_object` are not directly
comparable, and this development validation is not an independent final test.
