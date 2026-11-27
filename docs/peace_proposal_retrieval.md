# PEACE proposal retrieval: action strict F1 and MiniLM reflections

The four variants change only the historical evidence appended to the AUD mutator prompt.
BFCL learning uses a fresh initial playbook, **one epoch of 50 training tasks**, seeds 42–44,
temperature zero for acting/reflection/mutation, current=1 and past=3 **real environment replays**,
28 replay workers and 28 shared LLM slots. Candidate selection remains the no-alignment
current/past official-score Pareto procedure. These runs do not use policy-shift surrogate scoring.

| Mode | Two requested historical examples |
| --- | --- |
| `a` | Two A examples: similar action composition, contrasting reflection |
| `b` | Two B examples: similar reflection, contrasting action composition |
| `ab` | One A, then one B from another task |
| `random` | Uniformly sampled distinct tasks, then one reflection uniformly within each task |

All historical parent reflections with recorded official boolean outcomes are eligible,
including unselected candidates and iterations where candidate selection failed. Only earlier
committed training iterations in that run are read. The current task and other benchmarks are
excluded. Each selected example comes from a distinct task. Initially zero or one example is
used when fewer than two past tasks exist. Proposal history is independent of the accepted-edit
memory used for replay; retrieval never advances the replay RNG.

Trajectory similarity is exact multiset strict F1 over **all** recorded trajectory actions.
Parallel batches are flattened; function names, canonical parsed arguments and duplicate counts
participate. Action order, IDs, observations and validation metadata do not participate. Zero
overlap is recorded as zero, not described as an absolute semantic match.

For each query and task, A retains the reflection with highest action F1 (ties: lower reflection
cosine, then stable ID). It takes the top half of distinct-task representatives by action F1,
including boundary ties, then selects the lowest reflection cosine. B symmetrically shortlists
the highest reflection cosine half, then selects lowest action F1. Shortlists contain at least
the requested example count. `ab` chooses A first and excludes that task from B. All tie-breaking
is deterministic; Random uses an independent seed derived from algorithm seed, iteration and
reflection ID. These are relative ranks without absolute similarity thresholds.

Reflections embed `expected_behavior` followed by `analysis`; span positions, IDs and bullet
assessments are excluded. The encoder is pinned `sentence-transformers/all-MiniLM-L6-v2`,
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, masked mean pooling and L2 normalization,
384 dimensions, CPU only. Its standard 256-wordpiece truncation is recorded for every query
and retrieved example. The full original reflection remains in the mutator evidence.
The CPU HTTP worker uses the existing isolated torch/transformers interpreter; no model
dependency is added to the core environment.

Every mode uses the same additional prompt instructions and evidence format. Examples contain
the task, full action/observation trajectory, public tool contract, official outcome and original
reflection. The prompt treats reflections as hypotheses and requests reusable conditions,
procedures and exceptions. It does not reveal the A/B label, retrieval mode or similarity scores
to the mutator. Scores, IDs, truncation and selections are saved in each candidate's
`proposal_retrieval` audit. The normal request log retains the actual mutator prompt.

The launcher copies the frozen BFCL comparison snapshot, retains its actor, curator, replay,
mutation application, fixed task order and algorithm seed, and overlays the optional retrieval
and prompt-appending code. The 12 outputs and metadata use distinct variants. Final rank-1
playbooks receive frozen official train/val/test evaluations on 50 distinct tasks per split.
The actor-turn-v1 input100k/output2048/max30-turn budget and independent analysis budgets remain.

```bash
uv run python scripts/utils/launch_peace_proposal_retrieval.py prepare \
  --source outputs/qwen_bfcl_temp0_fresh_two_epochs_12runs_20261109 \
  --output outputs/qwen_bfcl_proposal_retrieval_minilm_temp0_1epoch_20261109
```

Start the pinned CPU embedding service before the `check` and `run` commands of the generated
`proposal_runner.py`. `check --output OUT --job JOB_ID` initializes and verifies the fresh
configuration without model/environment execution. An embedding identity or retrieval-mode
change cannot silently resume a checkpoint.

The live journal shows each candidate's retrieved task IDs, A/B/Random roles, exact action F1,
reflection cosine and MiniLM truncation flags. Expand an example to inspect its original task,
official outcome, reflection and full trajectory. These views resolve the saved iteration and
reflection IDs, including the original parent rank; they do not execute an environment or read
private session context. The displayed iteration is the latest committed retrieval record.
If a later task was skipped, its status and error are shown separately from that earlier record.

The follow-up combines all four retrieval modes with `span_policy_shift` evaluation, past=9,
and the same fresh one-epoch seeds, actor bytes, split order and concurrency. Parent training
trajectories still interact with BFCL; candidate evaluation generates actions from saved history
prefixes without restoring an environment. Successful source trajectories reward matching the
saved actions; failed sources reward departing from them. Current/past Pareto selects the beam.

```bash
uv run python scripts/utils/launch_peace_proposal_retrieval.py prepare \
  --source outputs/qwen_bfcl_temp0_fresh_two_epochs_12runs_20261109 \
  --after-output outputs/qwen_bfcl_proposal_retrieval_minilm_temp0_1epoch_20261109 \
  --candidate-evaluation span_policy_shift --past 9 \
  --output outputs/qwen_bfcl_proposal_retrieval_minilm_span_policy_shift_p9_temp0_1epoch_20261109
```

Run the generated frozen `proposal_runner.py run --output OUT` to arm the scheduler. Every
mode/seed has its own matched predecessor. A matching full training cursor and final Test marker
with the correct task list and zero infrastructure errors release only that successor; the other
eleven predecessors need not finish. The controller polls every second, uses an exclusive lock,
records child process identities and adopts existing workers when resumed. Follow-up learning
starts from a fresh playbook and an empty example archive. Final frozen train/val/test evaluations
remain official benchmark measurements and use a separate output/table/journal cohort.

The generator-only rerun queue is separate from these retrieval cohorts. It includes only the
three baseline PEACE seeds and nine policy-shift seeds (past 9 → 3 → 6 per seed), each freshly
trained for two epochs, with at most six experiments running together. Set actor temperature to
0 and reflection/mutation temperatures to 0.7 explicitly; removing the legacy all-role override
alone would leave the original reflection and mutation defaults at zero. These three role
values are pinned in training state and final checkpoint provenance. Existing all-role runs
retain their sampling behavior. Final benchmark evaluations and policy-shift actions use the
same temperature-zero actor.

```bash
uv run python scripts/utils/launch_peace_generator_only.py prepare \
  --baseline outputs/qwen_bfcl_temp0_fresh_two_epochs_12runs_20261109 \
  --shifts outputs/qwen_bfcl_span_policy_shift_temp0_fresh_two_epochs_p9_p3_p6_20261109 \
  --output outputs/qwen_bfcl_peace_generator0_analysis07_fresh_12runs_20261109
```
