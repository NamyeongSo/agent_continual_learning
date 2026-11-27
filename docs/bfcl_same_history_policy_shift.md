# BFCL same-history policy shift, CP every five tasks

This study reproduces the AppWorld analysis in
`/home/thskadud/agent_context/reports/figures/appworld_policy_shift` with BFCL native function calls.
It measures changes in the next action under a fixed history as the learned context changes.
The reference action is an observed initial-policy action, rather than a benchmark answer.
These agreement metrics therefore measure behavior changes, rather than BFCL task reward.

## Scheduled study

The frozen plan is
`outputs/bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/plan.json`.
The scheduler waits for Qwen, Gemma, and GPT-OSS's current proposal–evaluation alignment runs
to finish, including final analysis summaries and all four correlation/gain PDF figures.
An interim figure or an exhausted erroring process does not satisfy the completion gate.
Completed runs with recorded evaluation errors are accepted; their error counts stay in the schedule status.

Each model uses its own PREMiSE learning, seeds 42, 43, and 44, and beam rank 1 at
CP 5, 10, 15, 20, 25, 30, 35, 40, 45, and 50. CP 0 is an additional initial-context control.
The training method is the existing two-parent/four-candidate PREMiSE, with three past-task replays.
This policy probe does not generate new context candidates.

Qwen already completed all 50 training tasks. Gemma resumes its BFCL-only sequential prefixes
at 42, 49, and 37 tasks; GPT-OSS resumes its isolated 21-task prefixes for all three seeds.
The continuation forks preserve the original beam, memory, RNG state, and frozen task order.
Gemma's state is converted to a BFCL-only isolated branch so AppWorld/TerminalBench do not start.
Source experiments are preserved. Full state reconstruction verifies all parent contexts, selected
updates, replay memory, and sampler RNG before checkpoint extraction.

New continuation rollouts and reference rollouts have a **20-step** environment budget.
The inherited Qwen/Gemma learning prefixes used a 50-step budget; this difference is recorded
in `prepared.json` and `analysis/summary.json`. GPT-OSS's inherited learning used 20 steps.
CP numbers count attempted training tasks. Skipped updates are explicitly counted in the analysis
and retain the context reached before that skipped task.

Endpoints are Qwen 18006, Gemma 18008, and GPT-OSS 18010. Qwen/Gemma requests set
`enable_thinking=false`; GPT-OSS uses `reasoning_effort=low`, since disabling its reasoning is unsupported.
The limit is **150 concurrent model calls per model**. The three models can run on their own endpoints
after all current alignment analyses finish. Construct all seed pipelines before setting the shared
training semaphore, so construction cannot replace a semaphore while another seed is already running.

## Fixed history bank and requests

For each model and seed, rollout the initial context on the same frozen 50 training tasks,
at temperature 0 with 8,192 output tokens and the 20-step budget.
Save the actual acting requests and their next tool-call sequences. Exclude tool-shortlisting
requests and steps without a valid reference tool call, recording every exclusion and task coverage.
Valid earlier actions in a rollout that later fails can still supply histories; the rollout status
and source trace are preserved for each history.

Each reference history is queried at all eleven contexts, including an independent CP 0 repeat.
Only the generator's first system context is replaced. All other messages, observations, previous
tool-call IDs, and the actual available tools remain identical. This holds tool shortlisting fixed
and measures the acting policy conditional on that tool set. Probed actions are saved without execution
or appending to another probe's history. Each model has its own initial-policy history bank;
comparisons within a model share a bank, while comparisons across models involve different banks.

The exact probe count is `11 × H`, where `H` is the valid history count across the three seeds.
Reference generation adds 150 rollouts per model. These counts are separate from the remaining
own-model training work. `history_bank.json` freezes the final `H` and request hashes before probing.
Queue order mixes seeds and checkpoints, rather than querying one checkpoint exclusively at a time.

## Metrics and missing results

- **Loose**: function name only.
- **Strict**: function name and parsed JSON arguments; object keys are canonicalized.
- **Accuracy**: the full ordered call sequence must match, including duplicate calls.
- **Precision/recall/F1**: multiset overlap, including duplicates, ignoring call order.
- Tool-call IDs, assistant prose, and reasoning text do not define a function-call action.
- A valid text-only response contains zero calls and scores zero against a nonempty reference.
- Transport failures, empty protocol responses, invalid JSON arguments, and truncated responses
  are explicit errors. Up to five retries resend the identical request. Errors are never filled with zero accuracy.

Primary curves use the intersection of successfully evaluated histories across **all eleven CPs**,
so checkpoint coverage is constant. Scores are averaged within task, equally across tasks within seed,
then equally across seeds. The band is sample standard deviation across available seeds.
Task/anchor counts and the number of usable seeds are saved. A secondary step-weighted mean and
available-case means support sensitivity checks; exclusions can still introduce selection bias.
CP 0 repeatability provides a control for nondeterminism even at temperature 0.

The same saved predictions also provide all 55 within-seed checkpoint pairs. Pairwise action
disagreement is `1 - strict accuracy` and `1 - strict F1`, grouped by checkpoint distance in tasks.
Pairs share the same complete history bank; the band is sample SD across pairs.
Per-CP Pearson/Spearman correlations use CP 5–50 and are descriptive: dependent checkpoints
do not justify treating them as independent samples for p-values.

## Outputs and operation

Each model directory contains frozen contexts, source recovery manifests, continued training,
reference rollouts, history hashes, per-anchor probe requests/responses, and status updates.
The analysis saves:

- `checkpoint_agreement.pdf/png`: all eight loose/strict accuracy/precision/recall/F1 curves.
- `pairwise_policy_shift.pdf/png`: strict ACC/F1 disagreement versus checkpoint distance.
- `checkpoint_metrics.csv`, `seed_summary.csv`, `task_metrics.csv`, and `pairwise_metrics.csv`.
- `checkpoint_correlations.csv`, `coverage.json`, and `summary.json` with provenance and skipped-training audit.

The detached watcher logs to `scheduler.log` and updates `schedule_status.json` every 30 seconds.
It writes per-model logs once the dependencies finish. Inspect its gate without starting any work:

```bash
uv run python scripts/premise/schedule_bfcl_policy_shift.py \
  --plan outputs/bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/plan.json --check-only
```

`run_bfcl_policy_shift.py --prepare-only` verifies source states and continuation configuration
without model requests. A normal run continues incomplete branches, reuses immutable histories
and successful probes, and retries previously failed probes. File locks prevent duplicate controllers.
All experiment outputs remain under the ignored `outputs/` directory.

An individual model can also start immediately on a requested endpoint:

```bash
uv run python scripts/premise/run_bfcl_policy_shift.py \
  --plan outputs/bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/plan.json \
  --model-key gemma --base-url http://127.0.0.1:18009/v1
```

The endpoint override changes runtime routing while preserving `prepared.json` and all source states.
`gemma/runtime_manifest.json` records the active endpoint, controller identity, and experiment settings;
restarts retain that routing unless another override is supplied. The scheduler observes separately started
controllers and adopts their results instead of launching duplicate work. Other scheduled models still
wait for the original completion gate. Training status records each seed cursor every ten seconds.
