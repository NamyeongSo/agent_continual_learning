# Group-relative prompt optimization with versioned rollout groups

`scripts/premise/run_group_relative.py` implements a separate optimizer using PEACE's
actor, reflector and mutator. Existing PEACE pipelines and scheduled experiments
keep their original procedure.

## Procedure

1. Run the initial PEACE actor prompt **five times per training task** before any
   update. Save original trajectories, official rewards, exact acting requests,
   tools and reference actions. Initial actor sampling defaults to temperature 1.
   BFCL uses the official scored-user-turn pass fraction, AppWorld uses the official
   individual-test pass fraction (PCT), and Terminal-Bench uses passed / all collected
   verifier tests, including skipped tests in the denominator. Missing partial-score
   evidence is an error, never a fallback to binary success. Official reporting scores
   and success flags remain separate from these optimization rewards.
2. Compute `A_i = (R_i - mean(R)) / (population_std(R) + 1e-8)` within each task.
   Equal-reward groups have exactly zero advantages. Individual zero-advantage
   rollouts contribute zero and incur no prefix-probe requests.
3. Visit training tasks in the frozen manifest order. For each update propose
   **C=5** children of the current prompt. Each proposal independently reflects on
   the saved best and worst trajectories using the existing PEACE reflector.
   The existing PEACE mutator receives the worst trajectory/reflection plus both
   recorded trajectories and their reflections through its evidence input.
   The CLI uses the live Qwen3.8 model-call configuration for reflection and
   mutation as well, including temperature 1. Best/worst ties use the
   first original rollout. ADD-only remains the PEACE default; AUD is optional.
4. Evaluate every child and the incumbent on every saved acting prefix of the
   group's nonzero-advantage trajectories. Replace only the learned instruction.
   Generate one action at each independent prefix; execute no candidate action
   and never append candidate actions to later prefixes. Probe temperature is 1.
   Rollout and probe temperatures must match. Saved actor requests retain the actual
   model, sampling parameters, reasoning settings and template options; probes replay
   these parameters, changing only the learned context. An incompatible active
   inference profile is rejected. Old recordings without generation settings cannot
   silently enter this version's matched evaluation.
   Submit all independent prefixes across the five children and current context
   to one shared queue. Defaults are **128 action workers and 128 model-call
   slots**, matching the current GPU1 server's `max_num_seqs=128` setting. Completed
   requests immediately free workers for queued requests. No per-candidate or
   per-trajectory barrier serializes the prefix calls. Propose the five children
   concurrently before submitting this scoring batch. Identical policy texts may
   share probes as described below.
5. Average the **lower canonical action F1** within each trajectory and maximize
   `J(c) = sum_i A_i * mean_prefix_canonical_F1_lower(c, stored_action_i) / 5`.
   Positive advantages favor agreement; negative advantages favor change. Zero
   advantages retain their zero contribution in the original group denominator.
   Take the agreement's lower value before multiplying either sign of advantage.
   Keep the incumbent (the currently used prompt/context) unless a valid child's
   selection score exceeds its score by more than `1e-12`. Rank children by that
   scalar score; ties use ascending numeric candidate ID. Full comparison and
   objective intervals remain available as diagnostics.
6. Repeat in the fixed task order. If **all** advantages of a task's current group
   are zero, skip that update and mark the task for refresh. After finishing the
   full epoch, collect a new group of five rollouts for each marked task using the
   epoch-end learned context at temperature 1. Compute new partial rewards and
   advantages for this new generation, and use it on the next pass. This refresh
   also occurs after the final configured epoch, so that last refreshed generation
   is saved but not optimized unless a later pass is configured. `--epochs N` is a
   finite number of passes, not an unbounded retry loop. Nonzero-advantage groups
   keep their original trajectories and rewards.

The default evaluation group is the current task's five rollouts (`--past 0`).
`--past N` additionally samples up to N distinct previously visited task groups,
excluding the current task and groups with no usable signal. Groups receive equal
weight. Isolated runs sample only the same benchmark. Sampling is seeded and its
RNG state is checkpointed. Initial collection and epoch-boundary refresh use
environments; candidate selection uses model inference alone. A trajectory with no recorded action has
no prefix evidence and contributes zero. A task without contrasting usable best
and worst trajectories is skipped without proposing an update.

## Agreement and trust controls

New runs default to **`--metric canonical_f1`**. The comparator uses the original
same-prefix request to describe actions without executing generated Python or
shell, starting a benchmark server, or asking the actor/judge model:

- AppWorld: Python AST removes comments, spacing and quote-style differences.
  A bounded static interpreter resolves literal values, direct aliases and small
  literal loops into ordered API requests. Historical variable versions and
  persistent exports are retained. Different unresolved variables, overwritten
  bindings, dynamic code, unsupported control flow and uncertain API defaults
  remain unknown; variable names are never globally discarded.
- BFCL: compare function names and typed arguments. IDs, selectors, paths and
  non-text values retain their exact identity. Free-text fields `content`,
  `message`, `body`, `description` and `subject` can use local bidirectional NLI.
  AppWorld text arguments and printed text use the same text comparison.
- Terminal-Bench: parse Bash with tree-sitter only when the saved terminal output
  confirms a shell prompt. Normalize literal quoting and comments while retaining
  expansion, globbing, operators, redirections, command order and action timing.
  Different complex requests or interactive/unknown terminal state remain unknown.

The static description is a behavioral abstraction, **not general program
equivalence**. In particular different literal shell requests can differ in this
metric even if their effects happen to be equivalent. Free-text comparison uses
`cross-encoder/nli-deberta-v3-xsmall`, pinned at commit
`a150876415327c80daeff35ca6f68f5ed8cf5c24`, on **CPU**. Both entailment directions
must exceed 0.95 to estimate equality; both contradiction directions must exceed
0.95 to estimate difference. Other outcomes, changed protected numbers/identifiers/
negations and input exceeding the comparator's 512-token bound remain unknown.
There is no text truncation and no actor-output cap introduced by this bound.
NLI results carry `approximate: true`; these estimates can be wrong and are not
equivalence proofs. Model assets and runtime versions are pinned in the run.

Each action pair produces an agreement interval `[lo, hi]`. F1 uses an
order-preserving weighted LCS, retaining duplicate counts and prefix boundaries.
`--metric canonical_accuracy` instead requires complete ordered agreement, with
unknown equality represented by `[0, 1]`. Known agreement is a point interval;
unresolved comparisons remain `[0, 1]`. Bounds describe comparator uncertainty,
not statistical confidence intervals or guarantees about runtime semantics.

The default **`--canonical-score-mode lower`** uses `r_lo` as the scalar sampling
proxy for both children and the current prompt. For prefix-mean agreement
`[r_lo, r_hi]`, the contribution to `J` is **`A * r_lo` for both signs of A**.
Unknown agreement contributes zero. For a negative advantage this also removes
that sample's negative contribution, so unknown agreement can improve a child's
score relative to a known match. This is the requested lower-agreement convention;
it is different from the worst-case lower bound of the signed objective.

Average over the **original G=5 denominator**, then equally over evaluated task
groups. Unknown prefixes are never dropped. The selected scalar is recorded in
`surrogate` and `selection_score`, with `agreement` and prefix
`sampling_ratio_value` showing the lower agreement actually used.

The full signed interval is still recorded for diagnostics:

```text
A >= 0: [A * r_lo, A * r_hi]
A <  0: [A * r_hi, A * r_lo]
```

Thus `surrogate_lower/upper` and `selection_lower/upper` are diagnostic bounds,
and their lower endpoint can differ from the selected scalar when A is negative.
`--canonical-score-mode interval` restores the previous conservative procedure:
maximize the signed objective's lower bound and accept a child only if its lower
selection bound exceeds the current prompt's upper bound. Its original v1
configuration is retained for explicit resumption. Lower-agreement runs use
`group-relative-canonical-action-lower-v1` and a separate output directory.

This is the
sampling-ratio **behavioral proxy**, not a likelihood ratio, importance-sampling
correction or new measurement of candidate task success. Generation settings and
temperature 1 remain matched between rollout and prefix probes.

`--metric strict_f1` retains the legacy exact function-name/parsed-argument multiset
overlap, including duplicate counts. `--metric strict_accuracy` requires exact
ordered equality of the normalized action sequence at each prefix. Native
Terminal-Bench uses the saved official parser; assistant messages compare exact
content. AppWorld code-generation arguments compare the entire code string, so
equivalent rewritten code may have zero agreement. These options preserve the
historical method/configuration for explicit resumption of old runs. Use a new
output directory for canonical metrics; old cohorts are never silently rescored.

Optional `--shift-penalty L` selects by `J(c) - L * Delta(c, incumbent)`.
`Delta` averages `1 - agreement` between child and incumbent predictions on the
same active saved prefixes, first within trajectory, then within task, then across
tasks. Its default coefficient is 0. `--max-policy-shift D` additionally rejects
children whose upper shift bound exceeds D. For canonical metrics, agreement
`[lo, hi]` gives shift `[1-hi, 1-lo]`. Lower mode uses `1-lo` for the shift
penalty and hard bound, selecting by `J_lower_agreement - L*shift_hi` for both
the current prompt and each child. Ties retain the current prompt. Diagnostic
selection bounds are `[J_lo - L*shift_hi, J_hi - L*shift_lo]`; interval mode uses
the additional child-lower versus current-upper gate described above.
The incumbent has shift zero.
Identical policy texts share probe
requests, including no-op children; they still count as proposed candidates.
Sharing a probe result does not establish stochastic repeatability. At temperature
1, independent draws under the same context may differ, so leave the penalty at its
default zero until the independent repeatability audit supports its use.

## Run

From the Exgentic repository:

```bash
uv sync --frozen --extra dev --extra action-agreement

.venv/bin/python scripts/premise/run_group_relative.py \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json \
  --output-dir outputs/group_relative_bfcl_s42 \
  --benchmarks bfcl --seed 42 \
  --model openai/Qwen/Qwen3.8-27B-FP8 \
  --base-url http://127.0.0.1:8007/v1 \
  --rollouts 5 --candidates 5 --epochs 1 \
  --metric canonical_f1 \
  --canonical-score-mode lower \
  --proposal-workers 5 --action-workers 128 --llm-calls 128 \
  --playbook-edit-mode add_update_delete
```

Use a separate output directory and pin the desired `--appworld-policy` and
`--terminalbench2-version` for those benchmarks. `--dry-run` validates and writes
configuration without model/environment calls. `--collect-only` completes just the
initial bank. `--stop-after N` limits total optimization steps, after full bank
collection; it does not reduce the task corpus. Reuse the same command/output to
resume. Increasing epochs or changing method parameters requires a new output
directory; endpoint routing and worker counts may change on resume.
The 128-request default is the observed capacity of the current endpoint, not
an unlimited concurrency guarantee. `--action-workers` and `--llm-calls` can be
set explicitly for a different serving capacity. The queue records its total
prefix jobs and peak active workers in each iteration.

The comparator lazily downloads its pinned public model snapshot before collecting
the bank, failing early on missing dependencies/model assets. Subsequent runs can
reuse the local Hugging Face cache. `--agreement-text-model` and
`--agreement-text-revision` select another compatible pinned model;
`--agreement-text-threshold`, `--agreement-text-max-tokens` and
`--agreement-cpu-threads` are also checkpointed. The current adapter requires the
documented contradiction/entailment/neutral label mapping. Defaults use four CPU
threads. The CLI pins `CUDA_VISIBLE_DEVICES=1` for any local GPU work; the text
comparator explicitly stays on CPU. A remote serving endpoint must independently
use physical GPU 1 per the experiment's device restriction.

`--model-call-config` defaults to the active monitor's
`outputs/qwen38_terminalbench21_terminus2_timeout48_c10_20261113/config.json`.
The CLI copies its `model_settings` and pins the source SHA-256. The current
profile is Qwen3.8-27B-FP8, temperature 1, top-p 0.95, top-k 20, min-p 0,
presence penalty 0, repetition penalty 1, reasoning effort `xhigh`, thinking and
preserve-thinking enabled, and **no output-token cap** (`max_tokens: null`).
The CLI sets `EXGENTIC_ACTOR_OUTPUT_UNLIMITED=1`; the actor and probes omit both
`max_tokens` and `max_completion_tokens` on the wire. Model settings are also
forwarded explicitly to the native Terminal-Bench actor. Input 100,000-token and
BFCL 30 / AppWorld 40 / Terminal-Bench 40 turn limits remain in force. For a Python
caller, pass `model_call_settings` and opt into the same output-unlimited environment
policy explicitly. Other PEACE variants retain their defaults.

## Artifacts and failures

- `bank/rollouts/`: immutable trajectories, partial rewards, collection-context hashes
  and history-file SHA-256 hashes, named by task and generation.
- `bank/groups/`: five rollout references, fixed rewards and group advantages per generation.
- `bank/refresh/`: restart-safe epoch-end refresh plans with pinned collection contexts.
- `iterations/`: proposals, best/worst reflections, original-prefix probes,
  advantage-weighted scores, trust scores, selected policy and committed state.
  Canonical runs include each prefix's lower/upper sampling proxy and trust
  agreement, reasons, approximation flags, resolved coverage, and candidate
  objective/shift/selection bounds. Lower mode records the chosen agreement in
  `sampling_ratio_value` and the selected scalar in `surrogate/selection_score`;
  the diagnostic signed lower bound is a separate quantity.
- `agreement_cache/`, `agreement_runtime.json`: checksummed comparison cache and
  pinned parser/text-model runtime manifest. Keys include comparator source hashes,
  dependency versions, model revision/settings, original request, both action
  sequences and metric. This cache never deduplicates independent actor draws.
- `policy_shift_calls/`, `llm_calls/`: requests, responses and measured usage.
- `state.json`, `checkpoints/`: pinned configuration, optimization cursor and
  selected prompt. `training_complete.json` is written only at full completion.

Collection checkpoints each finished rollout and group. `state.json` keeps the full
group-generation history, pending zero-signal tasks and completed refresh epochs.
Restart reuses completed records and refresh groups; source hashes are verified
before reuse. An iteration written before a
state-checkpoint interruption can be committed without new inference. The CLI
holds a process lock to prevent two writers. Checkpoints pin the method, prompt
sources, environment policy, actor context, reward metric, group/candidate counts,
temperatures, sampling order, edit policy and actor budget version.

Infrastructure failures propagate and never become zero rewards. Candidate
malformed/truncated or over-budget actions invalidate that candidate, preserving
measured usage and giving no novelty credit. A failed incumbent probe prevents an
update. The live CLI profile pins `actor-turn-v3-output-unlimited`: input and
benchmark-turn limits apply to initial acting and all prefix probes, with output
unlimited as requested. Actual success and generalization still need
frozen-policy environment evaluation: this optimizer improves an action-agreement surrogate.

`scripts/premise/audit_canonical_action_agreement.py` rescores the frozen manual100
review corpus with the production comparator on CPU and checks original source
hashes. Its benchmark/label coverage is a **development regression**, not an
independent accuracy estimate. It makes no new actor or environment calls and
stores scores without copying private action text into the report:

```bash
.venv/bin/python scripts/premise/audit_canonical_action_agreement.py \
  --review outputs/group_relative_strict_validity_manual100_qwen38_t0_20261114 \
  --output outputs/group_relative_canonical_action_dev100_20261114
```

## Independent F1 reliability audit

`scripts/premise/audit_group_relative_f1.py` measures each benchmark separately.
It draws one new reference action under the initial context and then independent
repeated actions under the initial and learned contexts at temperature 1, with all
other request fields fixed. The live model-call configuration includes `xhigh`
reasoning, thinking/preserve-thinking, and unlimited output. Historical prefixes
are conditions, not new on-policy
environment rollouts. The initial and learned contexts come from actual PEACE
checkpoints; actions and usage are retained verbatim.

The report includes fixed-reference F1 distributions, per-prefix and aggregate
repeat SD, independent same-context pairwise F1/action equality, cross-context
F1/action equality, function-name equality and concrete changed-action examples.
It also reports precision, recall and F1 in both strict and loose modes from the
same saved actions, with mean, repeat SD, distributions and explicit comparison
direction. Precision divides matched atoms by predicted atoms; recall divides by
reference atoms. Fixed-reference comparisons treat each initial/learned draw as
the prediction. Cross-context pairs treat the learned-context draw as the
prediction and initial-context draw as the reference. Self pairs average both
directions. Each metric is averaged separately; macro F1 is not recomputed from
macro precision/recall. Loose matches only names/types, retaining duplicate counts:
it ignores Python code arguments in AppWorld and command text/duration in native
Terminal-Bench. High loose agreement therefore need not indicate unchanged actions.
A consistently zero F1 can be an uninformative floor, so action repeatability is
reported independently. Comparative means use the common complete-prefix
intersection and equal task weighting. Errors and excluded prefixes are reported
per benchmark and never zero-filled. `--analyze-only` recomputes summaries from
the saved draws without generating new actions.

## Agreement by update distance

`scripts/premise/audit_group_relative_update_distance.py` extends a completed
matched-generation audit with checkpoints at every ten stored update steps:

```bash
.venv/bin/python scripts/premise/audit_group_relative_update_distance.py \
  --baseline outputs/group_relative_f1_reliability_qwen38_live_unlimited_20261113 \
  --output outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113 \
  --workers 12
```

The runner reconstructs only the policy span at each checkpoint on the original
prefixes, verifies source/parser hashes, and copies the exact endpoint generation
settings. Endpoint requests must match byte-for-byte before their completed draws
can be reused. Every intermediate checkpoint gets its own ten independent draws,
even when two checkpoints contain identical policy text. The starting corpus is
the prior audit's complete-prefix intersection; it does not reselect tasks based
on intermediate agreement scores. Cached draws, including errors, are preserved
on resume. Use one runner per output directory.

Distance is the difference between checkpoint step indices, not a count of
accepted prompt edits. At distance 10, for example, CP0–10, CP10–20 and all other
available pairs contribute equally. At distance zero, compare distinct samples
within each checkpoint, in both directions; never compare a draw with itself.
At positive distances, all 100 earlier/later sample combinations contribute, with
the later checkpoint as prediction for precision/recall. Within each checkpoint
pair, average over samples, then prefixes within task, then tasks equally. Finally
average checkpoint pairs equally within a distance. A single common prefix mask
across all checkpoints keeps the corpus and weights fixed along each benchmark's
curve. Invalid actions and infrastructure errors are reported, never zero-filled.

Outputs include strict/loose P/R/F1 curves, distance and checkpoint-pair CSVs,
prefix-level statistics, exact action repetition, context-hash equality, and an
audit of actual generation settings and measured usage. `sampling_adjusted_gap`
is the mean of the two checkpoints' self F1 minus their cross F1: a descriptive
noise control, not a validated importance ratio or a replacement objective.
`f1_pair_sd` averages within-prefix pair-score variability; it is not a standard
error because pairs share draws. Four tasks per benchmark provide a pilot, not
a reliable significance test. Historical contexts/prefixes, checkpoint position,
growing prompt length and endpoint sample reuse remain explicit limitations.

`--prepare-only` validates the corpus and copies endpoint draws without inference.
`--analyze-only` regenerates reports from cached draws; unfinished runs fail unless
`--allow-incomplete` is supplied explicitly for an interim common-prefix report.

## Whole sequences regenerated under both contexts

`audit_group_relative_trajectory.py` recovers every valid acting prefix of the
selected original CP0 trajectories. Under each checkpoint context it independently
generates two complete sets of actions, plus a third independent CP0 reference.
Model, temperature, tools, thinking and sampling settings match on both sides.
Only newly generated action atoms are concatenated, in original prefix order.
Historical actions and shared conversation history never enter either scored
sequence. Generated actions are not executed or appended to later prefixes.

```bash
.venv/bin/python scripts/premise/audit_group_relative_trajectory.py \
  --baseline outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113 \
  --output outputs/group_relative_regenerated_trajectory_qwen38_t1_cp10_20261114 \
  --comparison regenerated_contexts --repeats 2 --workers 24 \
  --reuse-run outputs/group_relative_full_trajectory_qwen38_t1_cp10_20261114
```

`--reuse-run` imports completed draws from an interrupted run only after comparing
the entire reconstructed request and draw identity. Failed cached draws are
preserved in their original run. Resume never silently retries cached failures.
The original-reference comparison is retained behind `--comparison
recorded_original` for reproducing its interrupted protocol; the new default
exclusively compares regenerated contexts.

`analyze_regenerated_trajectories.py` scores complete concatenations once per
sequence pair. It provides all-checkpoint-pair distance curves and a separate
independent regenerated-CP0-reference comparison. Same-context controls use
different repeats in both directions. Positive distances treat the later context
as prediction. Tasks receive equal weights, followed by equal checkpoint-pair
weights within distance. A task with any missing/invalid prefix in any checkpoint,
repeat or reference is excluded across all curves; failures are never zero-filled.

Strict/loose PRF keeps the exact action-multiset definitions. Whole-sequence PRF
can match actions across different original positions; it does not enforce order.
Aligned-prefix PRF is retained on the same new samples to isolate the effect of
changing the scoring unit. BLEU-1, smoothed effective-order BLEU-4 and English
METEOR use NLTK 3.9.2 and WordNet 3.0. Strict/full lexical inputs include action
names, argument keys, and unescaped code/command tokens; loose inputs contain
ordered action names only. English METEOR uses its default lowercase, stemming
and synonym matching, which does not establish code semantic equivalence.

Install optional scoring dependencies in a separate target rather than modifying
the shared environment or project lock. Pass that target and WordNet data folder
explicitly:

```bash
.venv/bin/python scripts/premise/analyze_regenerated_trajectories.py \
  --run outputs/group_relative_regenerated_trajectory_qwen38_t1_cp10_20261114 \
  --dependencies outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113/bleu_meteor/dependencies \
  --nltk-data outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113/bleu_meteor/nltk_data \
  --workers 6
.venv/bin/python scripts/premise/plot_regenerated_trajectories.py \
  --run outputs/group_relative_regenerated_trajectory_qwen38_t1_cp10_20261114
```

Reports and CSV/JSON files live under `analysis/`. Two full reconstructions per
context and four original tasks per benchmark make this an exploratory pilot;
pair counts do not provide independent samples for significance tests. The
optimizer objective and default zero policy-shift penalty remain unchanged.

## Evaluation-only temperature ablation

Use the frozen full-trajectory T1 corpus to test both comparison arms at T0.
The ablation changes only the request temperature; it preserves all histories,
contexts, model, thinking, sampling filters, retry, input and turn settings.
Rollout collection and optimizer defaults keep their existing settings.

```bash
.venv/bin/python scripts/premise/audit_group_relative_trajectory.py \
  --baseline outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113 \
  --paired-run outputs/group_relative_regenerated_trajectory_qwen38_t1_cp10_20261114 \
  --temperature 0 \
  --output outputs/group_relative_regenerated_trajectory_qwen38_t0_cp10_20261114 \
  --repeats 2 --workers 24
```

Each repeat makes its own model call. Cached T1 responses cannot enter T0 banks;
the ablation needs 2,948 fresh draws for the same 12 original trajectories and
248 acting prefixes. Keep T0/T1 output folders and journal records separate.
Analyze temperature effects on intersecting valid tasks so differing coverage
cannot masquerade as better stability. T0 tests conditional greedy-action
agreement; it does not provide importance weights for a T1 stochastic rollout
policy. Score and figure generation read the recorded temperature from the run.

After both runs finish, compare identical valid tasks **and repeat-ID pairs**:

```bash
.venv/bin/python scripts/premise/compare_regenerated_temperatures.py \
  --run-zero outputs/group_relative_regenerated_trajectory_qwen38_t0_cp10_20261114 \
  --run-one outputs/group_relative_regenerated_trajectory_qwen38_t1_cp10_20261114 \
  --output outputs/group_relative_regenerated_trajectory_qwen38_t0_cp10_20261114/comparison
```

The strict view intersects tasks complete at every checkpoint/repeat/reference.
The secondary CP0-bank view intersects complete sequence IDs at both temperatures,
requires independent CP0 self pairs and at least one common candidate at every
checkpoint, and keeps the task cohort constant over the grid. An empty common
cohort is missing evidence, never a zero score. Saved nested pair scores are
reaggregated with the same task/checkpoint weights; every PRF pair is checked
against raw regenerated actions. This analysis makes no additional model calls.

Additional diagnostics preserve action order using unit-cost Levenshtein
similarity on canonical action atoms, report aligned-prefix exact action fractions,
and show the effect of Python AST formatting normalization (AppWorld) or removing
command duration (Terminal). These transformations do not establish semantic
equivalence and are not CodeBLEU. The CP0 distribution plot uses one point per
unordered independent-draw pair; points sharing a generated sequence are dependent.

The November 14 pilot increased CP0 strict self F1 from T1 to T0 on matched
sequence IDs: BFCL 0.5120 to 0.6716 (4 tasks), AppWorld 0.2768 to 0.7269 (3),
and Terminal 0.2472 to 0.5310 (1). T0 therefore improved agreement without making
it deterministic. Terminal had no common task in the stricter all-repeats view;
its secondary single-task comparison does not support a benchmark-wide conclusion.

## Aligned-prefix scoring without trajectory concatenation

`analyze_aligned_prefix_actions.py` reuses the saved T0/T1 regenerations and scores
each pair of responses at the identical original acting prefix. Multiple calls
inside one response stay together; actions from different prefixes never enter
the same PRF/BLEU/METEOR comparison. Average independent draw pairs at each prefix,
then prefixes within task, then equally across tasks and checkpoint pairs.

Two separate coverage views prevent scoring-unit and missing-data effects from
being confused. `matched_sequences` keeps exactly the prior concatenation study's
task/prefix/repeat-ID pairs. `valid_prefixes` uses a fixed intersection of prefixes
valid at every checkpoint/repeat/reference at both temperatures, allowing a task
with an incomplete full reconstruction to contribute its valid prefixes. Neither
view fills failures with zeros. The November 14 intersection has 45 BFCL, 131
AppWorld and 50 Terminal prefixes, four tasks per benchmark; the matched CP0-bank
view has 45/94/21 prefixes from 4/3/1 tasks.

```bash
.venv/bin/python scripts/premise/analyze_aligned_prefix_actions.py \
  --run-zero outputs/group_relative_regenerated_trajectory_qwen38_t0_cp10_20261114 \
  --run-one outputs/group_relative_regenerated_trajectory_qwen38_t1_cp10_20261114 \
  --dependencies outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113/bleu_meteor/dependencies \
  --nltk-data outputs/group_relative_update_distance10_qwen38_live_unlimited_20261113/bleu_meteor/nltk_data \
  --output outputs/group_relative_aligned_prefix_qwen38_t0_t1_cp10_20261114 --workers 6
.venv/bin/python scripts/premise/plot_aligned_prefix_actions.py \
  --run outputs/group_relative_aligned_prefix_qwen38_t0_t1_cp10_20261114 \
  --paired-summary outputs/group_relative_regenerated_trajectory_qwen38_t0_cp10_20261114/comparison/summary.json
```

Figures include strict/loose PRF and lexical metrics, a pure scoring-unit comparison,
normalization/control gaps and task-balanced prefix-score distributions. A gap is
the independent self score minus cross-context score, can be negative and is not
an importance ratio. AST normalization preserves variable names. Removing
cross-prefix matches does not establish semantic equivalence or execution validity.
`repeat_pair_sd_f1` is descriptive dispersion among pair scores; with two draws the
two symmetric within-context F1 pairs have identical scores and zero SD, which is
not evidence of deterministic generation or a sampling standard error.
