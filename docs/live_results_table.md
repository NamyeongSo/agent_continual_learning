# Automatically updated experiment results table

The live monitor at `http://127.0.0.1:8878/#live` includes a publication-style
results table. Each method has Isolated, Sequential and Interleaved rows; ReAct
has one Isolated row. At the user's request, ReAct Offline retains the original
reported means (SD): BFCL 58.00 (2.00), AppWorld 11.33 (6.11), Terminal-Bench
26.44 (1.99), and their macro-average 31.92. These are explicitly configured in
`reported_react_offline`, labeled as previously reported values, and included in
ranking and shading. No per-seed scores or macro-average SD are reconstructed
from those aggregates. ReAct Online and Offline-val remain blank.
PEACE in this table refers to the current PREMiSE implementation.
Other historical numbers from the supplied LaTeX table are not imported.

Each benchmark and the macro-average have three columns:

| Column | Measurement |
| --- | --- |
| Online | Values recorded during actual training: ACE's first task execution, PREMiSE's first pre-update parent, or GEPA/SkillOpt's immutable first-encounter online record. Additional parents, candidates and replay evaluations do not count again. |
| Offline | Full original test split with the final frozen policy. |
| Offline-val | Full common-queue validation evaluation at the final training checkpoint, verified against the frozen policy's exact SHA-256. Candidate-selection validation and earlier CP scores are not substituted. |

Tables can additionally enable **Offline (early)** through `offline_early`
in their table manifest. For each
method/variation/benchmark/seed, it chooses the checkpoint with the highest
complete common-queue validation score (including CP0), breaking exact ties by
the earliest checkpoint. It then reports **that checkpoint's test score**; test
scores and test availability never influence selection. This is a retrospective
best-validation-checkpoint estimate, not a patience-based stopping simulation.
All configured validation checkpoints must be complete and unique. If the
selected test split is incomplete, that seed remains missing and no alternative
checkpoint is substituted. Available complete seeds use the existing `*`
convention, and Avg. includes only seeds complete across every benchmark.
The manifest pins the checkpoint queue and the candidate points separately for
each benchmark. The HTML hover text and JSON retain the selected CP, val score,
test coverage, aliases and missing-data reason; LaTeX uses the same cells.

The user-approved correction on 2026-11-10 configures
`offline_early.reuse_training_val_best_variants = ["gepa_val", "skillopt_val"]`.
Those two variants already freeze the policy selected using their original
training-time validation records. Their early column reuses the exact final
test result displayed in the same table, without selecting another policy from
independent checkpoint-val reevaluations. The live table verifies the final CP
against the run's frozen policy SHA and requires `validation_source="val"`.
The offline four-split figure/table additionally audits all 18 frozen policies
against training results and uses the final-CP queue Test in both columns.
The six-setting live table now uses the same CP-queue reevaluation as the
four-split artifacts, including their Test values. Other table cohorts retain
their existing separate training-final Test source by default.
The Train-as-val variants retain the held-out CP-val selection described above.

The existing **Main Table · Original AppWorld policy · PEACE w/o alignment ·
6 settings** is configured with `offline_splits=true` and
`show_star_markers=false`. Its four columns per benchmark and Avg. are
**Train / Val / Test / Test (early)**; Online is removed from this table only.
All three final-policy splits come from the queue pinned by
`offline_early.checkpoint_queue_id`, at CP50 for BFCL/AppWorld and CP30 for
Terminal-Bench. The final checkpoint must match the completed training run's
exact frozen-policy SHA. Missing/ambiguous checkpoints, a different policy hash,
or incomplete/error-containing splits never fall back to training records or
the separate historical Test run. Completed seed/split measurements contribute
immediately; missing seeds are excluded. Hover text and a coverage note retain
seed counts even though no star markers are drawn. A one-seed cell has no SD;
Avg. uses only seeds with all benchmark splits complete. Ranking, shading,
collapse controls, automatic refresh and LaTeX export share this same live data.
The opt-in layout does not alter other configured tables or raw experiments.

AppWorld Online, Offline and Offline-val use **TGC** (`appworld_tgc_v1`): fully
successful tasks divided by tasks with recorded official outcomes. The official
`success` flag, including budget-policy failures, supplies the binary outcome;
rounded partial test pass rates are never thresholded to reconstruct success.
The monitor projects existing ACE status records and PREMiSE parent feedback
into TGC while retaining their original partial scores in the raw artifacts.
New ACE online metric records store TGC in `score`, with `partial_score`, the
official `success` flag and the metric version alongside it. Restored cumulative
metrics project old records through the same TGC rule before averaging.
Run cards show both TGC and the number of successful/measured tasks. A missing
success flag remains unmeasured, rather than becoming zero. Each seed uses its
own measured-task denominator; the table averages the seed percentages equally.
Other benchmarks retain their recorded task scores. Native GEPA/SkillOpt
already record binary outcomes, as do the shared frozen-checkpoint evaluations.
The fixed-CP evaluation on the train split is a separate measurement and never
supplies the Online column.

Apart from the explicitly retained ReAct Offline baseline, a cell is published
only after the entire corresponding measurement is complete
for all seeds 42, 43 and 44. Values are percentages with the mean and **sample**
standard deviation across seeds. A measured zero remains zero; missing values,
partial splits, ambiguous sources and infrastructure errors remain `--`.
Partial seed readiness is visible in the HTML table's hover text.
Avg. is the equally weighted macro-average over all three benchmarks and remains
blank until all three are available. Its displayed value omits the SD, matching
the supplied table layout. Ranking uses displayed means rounded to two decimal
places, ignores SD, and shares formatting for ties: highest distinct value bold,
second highest underlined. Each column receives independent red-white-blue
min-max shading with a maximum color mixture of 35 percent. No fixed claim about
which method wins or how many cells are blue is included.

The active cohort is configured in
`outputs/live_result_tables_20261023/results_table_manifest.json`. It pins the
original baseline ACE/PREMiSE and current native GEPA/SkillOpt source batches,
seeds, model, temperature, split hash, actor limits and ADD-only editing.
It follows the current job outputs in those manifests, including corrected
`*_harbor_reward_v2` outputs, while excluding invalidated learning and curator
variants. Sequential/Interleaved rows are initially blank. Newly created
compatible per-benchmark queues are discovered after `auto_discover_after`, with
their declared `mode` supplying the setting. A queue must expose the same live
run schema and dataset/limit metadata. Duplicate sources for one seed stay blank
until the cohort configuration explicitly resolves them.

The monitor polls every five seconds, independently of run-card filters.
Filesystem collection runs in the background: a slow scan keeps the previous
complete snapshot visible, and the first scan after a server restart is labeled
as initializing. Completed optimizer tests use their committed summary only
when its policy hash matches the actual frozen instruction.
LaTeX can be downloaded from the table header. The detached
`scripts/utils/publish_live_result_table.py` process also refreshes these derived
files every 30 seconds, even while the browser is closed:

- `main_results.json`: numerical cells, per-seed readiness and provenance.
- `main_results.tex`: a complete compilable LaTeX document.
- `main_results_body.tex`: the table body for `\input{...}` in a paper using
  `booktabs`, `multirow`, `adjustbox`, `xcolor` with table support, and the supplied
  `oursbg`, `scorered` and `scoreblue` colors.

Their output directory is recorded in `/tmp/exgentic_live_results_table_output_path`.
All files are generated atomically from the same API snapshot. The publisher
only reads the monitor and writes derived reports; it does not call a model,
change learning state, or rerun evaluations. Process and heartbeat records live
beside the reports, and it retries when the journal server is temporarily down.

The shared-reflector medium fresh-one-epoch cohort enables `test_only=true` in
`outputs/qwen_shared_reflection_medium_fresh1_four_baselines_20261111/results_table_manifest.json`.
It displays ACE AUD, GEPA Val, SkillOpt Val and PEACE AUD without alignment,
with final Test columns for BFCL, AppWorld, Terminal-Bench and their within-seed
macro-average. The cohort pins the split, actor limits, Original AppWorld policy
and inference-profile SHA (actor temperature 0, analysis 0.7, reasoning medium).
Only complete per-seed Test splits enter the table; current learning/validation
does not supply a final Test score. Partial seed coverage is explained without
star markers, and results update through the existing live collector.

ACE/PEACE's shared-reflector evaluator atomically writes official outcomes to
`evaluation/records/*.json` without a `status` field. The monitor recognizes this
format using its task/index correspondence, boolean official success, matching
binary score, finite official reward and session/run/prompt-hash provenance.
Missing or malformed records remain pending, explicit infrastructure errors
remain errors, and AppWorld still derives TGC from official success. This is a
read-only projection; it does not rewrite records or rerun completed tests.
