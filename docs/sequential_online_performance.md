# Sequential online performance windows

Plot ACE and PREMiSE training scores at every iteration, using recent windows of
5, 10, 15, and 20 tasks:

```bash
.venv/bin/python scripts/utils/plot_sequential_online_performance.py \
  --windows 5 10 15 20 \
  --output outputs/sequential_online_windows_20261009
```

Matplotlib and NumPy are available in the project's optional `analysis` dependencies.
The script only reads training results; it does not run tasks or update experiments.

## Data and score definitions

Defaults use the completed sequential runs from the existing baseline report:

- ACE: `scripts/ace/outputs/ace_appworld_scorefix_resume_bfcl50_20260928`.
- PREMiSE: `outputs/premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928`.
- Seeds: 42, 43, 44, each with 50 BFCL, 50 AppWorld, and 30 Terminal-Bench tasks.

Pass `--ace-root`, `--premise-root`, or `--seeds` to select another experiment.
Each root must contain a `launch_manifest.json`. The methods' task order must match
for each seed, and benchmark boundaries must match across seeds.

To substitute PREMiSE's AppWorld training block from the completion-contract restart:

```bash
.venv/bin/python scripts/utils/plot_sequential_online_performance.py \
  --premise-appworld-root outputs/premise_completion_contract_restart_20261003_0557 \
  --windows 5 10 15 20 \
  --output outputs/sequential_online_windows_20261009_appworld_restart
```

Only PREMiSE AppWorld iterations 51–100 are replaced. The restart root has
sequential results for seeds 43 and 44; seed 42 retains its original sequential
results. Interleaved runs and the separate Terminal-Bench retrain are excluded.
The replacement requires matching task IDs and a completed AppWorld block.
Figures state the seed sources; `task_scores.csv` records each task's `source_run`,
and `metadata.json` records the benchmark-specific source paths.

ACE uses the recorded training session score, including copied metrics from its
resumed BFCL prefix. PREMiSE uses the higher recorded score of its two pre-update
parent rollouts, matching `report_baseline_results.py`; candidate replay scores
are excluded. Pass `--premise-score rank1` to use only the first parent instead.

AppWorld uses task goal completion (TGC): a recorded task score at least
`1 - 1e-9` contributes 1, otherwise 0. Pass `--appworld-metric partial` to plot
partial task scores. BFCL and Terminal-Bench retain their recorded scores.

## Windows and figures

Pass `--cumulative` to average `[benchmark_start, M]` at every iteration, with
the same reset at iterations 51 and 101 and the same source/score definitions:

```bash
.venv/bin/python scripts/utils/plot_sequential_online_performance.py \
  --cumulative \
  --premise-appworld-root outputs/premise_completion_contract_restart_20261003_0557 \
  --output outputs/sequential_online_cumulative_20261009_appworld_restart
```

This produces `sequential_online_cumulative.png` and `.pdf`, `seed_cumulative.csv`,
and `aggregate_cumulative.csv`, together with task scores and source metadata.
The figure uses one axis for all three benchmark regions and seed mean/SD curves.

At iteration M, the window covers `[max(benchmark_start, M-N+1), M]`.
It restarts at iterations 51 and 101. The first N-1 iterations of each benchmark
use the tasks available since that benchmark started; gray shading marks this
window-filling period.

Missing scores keep their original iteration position. They are excluded from
the mean without extending the window further back. An entirely unscored window
has no mean. Recorded ACE execution-error scores remain in the calculation,
matching the existing online report. These outputs describe recorded online
performance, including those execution outcomes.

Compute window means independently for each seed, then average the seed means
with equal weights. Shading shows one sample standard deviation across seeds.
The CSV files record observation counts, missing scores, and contributing seeds.

Each N produces one figure with a single axis spanning iterations 1–130.
Vertical lines at 50.5 and 100.5 divide BFCL, AppWorld, and Terminal-Bench regions.
Curves are drawn separately in each region so no line connects the last score
of one benchmark to the first score of the next. The overview stacks one such
chart per window size.

Every export writes both the labeled figure and a matching `_no_labels` version,
in PNG and PDF. The latter omits method legends, online/checkpoint/initial-value
annotations, their connectors, and explanatory footnotes. Titles, axes, seed
captions, benchmark headings, curves, shading, reference lines, and plot geometry
are preserved. This also applies to cumulative figures and window overviews.

Generated artifacts are kept under the ignored output directory:

- `sequential_online_N05.png` through `sequential_online_N20.png`, with PDF copies.
- `sequential_online_overview.png` and `.pdf`.
- `task_scores.csv`: task IDs, raw scores, plotted scores, and recorded error flags.
- `seed_windows.csv`: every seed's window mean and observation counts.
- `aggregate_windows.csv`: plotted seed means, sample SDs, and coverage.
- `metadata.json`: source runs, score definitions, boundaries, and missing/error iterations.

PREMiSE error flags describe any recorded iteration error, including reflection
or replay failures after a successful parent rollout. They do not automatically
invalidate the parent score.

## Finishing-checkpoint test references

Add `--test-checkpoints` to draw dashed test means at CP50 for BFCL, CP100 for
AppWorld, and CP130 for Terminal-Bench. Each line is restricted to its own
benchmark region. The test checkpoint follows the corresponding online source,
including restarted PREMiSE AppWorld seeds 43 and 44. Test evaluations use the
frozen rank-1 PREMiSE beam, whereas the default online metric uses the best parent.

```bash
.venv/bin/python scripts/utils/plot_sequential_online_performance.py \
  --windows 50 \
  --premise-appworld-root outputs/premise_completion_contract_restart_20261003_0557 \
  --test-checkpoints \
  --initial-context-root scripts/ace/outputs/ace_initial_playbook_test_seed42_44_20261001 \
  --output outputs/sequential_online_N50_test_initial_20261009
```

Replace `--windows 50` with `--cumulative` for the explicitly cumulative figure.
N=50 and per-benchmark cumulative means coincide for these 50/50/30-task blocks.

Add `--by-seed` to write separate figures and CSV/metadata files in `seed42/`,
`seed43/`, and `seed44/`. Each figure uses that seed's online scores, finishing
checkpoint test scores, and initial-context test scores. Across-seed means and
SD bands are omitted; the shared x/y scales and benchmark boundaries are retained.

The black dashed reference uses the shared unchanged `DEFAULT_PLAYBOOK` and
generator prompt. Its existing evaluation was performed with ACE, without
learning; it is not a separate PREMiSE execution. ACE checkpoint and initial
evaluations use an 8192-token limit; PREMiSE BFCL/AppWorld acting uses 2048, and
Terminal-Bench retains its native actor configuration. The source files are retained
in `seed_test_references.csv`, and the equal-weight seed means and sample SDs are
exported to `aggregate_test_references.csv`.

References validate the frozen test task IDs, split manifest hash, and checkpoint
hash. All test tasks contribute to the denominator; execution errors contribute
zero. AppWorld uses TGC by default. ACE Terminal-Bench uses the saved retry results.
References retaining execution errors have an asterisk; they describe the recorded
full-test mean rather than an error-free evaluation. Ordinary unsuccessful tasks,
step-limit outcomes, and unfinished tasks retain their recorded benchmark scores.

## SkillOpt encounter comparison

Pass `--skillopt-csv team_prompt/skillopt_train_encounters.csv` to add green SkillOpt
training curves to any of the preceding commands. The loader selects sequential
rows and requested seeds, divides percentage scores by 100, and applies the same
AppWorld metric and window definition. It requires contiguous, unique iterations
and matching benchmark counts and boundaries. Missing scores retain their positions.

SkillOpt uses the CSV's own task order; scores are never reassigned to ACE task IDs.
This CSV's sequential seeds 43 and 44 match ACE/PREMiSE's task order; seed 42 differs.
Metadata records task-order and task-set agreement, mismatch counts, and the CSV hash;
task exports preserve the CSV's original `source` as `encounter_source`. Those source
rollout files are unavailable locally, so the plot uses the supplied encounter scores
without re-evaluating rollouts or inferring execution-error status.

The supplied CSV contains training scores only. ACE/PREMiSE finishing-checkpoint
and shared initial-context references remain available; no SkillOpt test reference
is inferred from training encounters. Both labeled and `_no_labels` versions are saved.
