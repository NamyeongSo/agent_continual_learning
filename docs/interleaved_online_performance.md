# Interleaved cumulative online performance

Plot three benchmark histories for ACE and PREMiSE on one global iteration axis:

```bash
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --grouping by-seed --output outputs/interleaved_online_by_seed_20261009
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --grouping mean --output outputs/interleaved_online_mean_20261009
```

The first command produces separate `seed42/`, `seed43/`, and `seed44/` figures;
the second produces a `mean/` figure with equal weights for the available seed
means at each iteration. Each figure has PNG/PDF copies and task-level, seed-level,
and plotted CSV files, together with source metadata. The CSV files retain sample
SDs, although the six-line chart displays the mean curves only.

Every export writes both the labeled figure and a matching `_no_labels` version,
in PNG and PDF. The latter omits method legends, endpoint values, annotation
connectors, and explanatory footnotes. Titles, axes, seed captions, curves, colors,
line styles, and plot geometry are preserved. This also applies to window overviews.

For benchmark B at global iteration M, its curve averages recorded online scores
for tasks with benchmark B and iteration at most M. Other benchmarks' tasks leave
that curve unchanged. Switching benchmarks does not reset accumulated histories.
Curves begin when their benchmark has an observed score and use steps to preserve
the update's iteration. Missing scores retain their positions and are omitted
from the denominator; recorded ACE execution-error scores are retained.

ACE uses blue and PREMiSE uses orange. BFCL is solid, AppWorld is dashed, and
Terminal-Bench is dotted. All figures retain common x/y scales and six curves.
AppWorld uses TGC; PREMiSE uses the best pre-update parent, matching the preceding
sequential analysis. Pass `--appworld-metric partial` for partial task scores.

Defaults use the completed ACE experiment and the current dashboard's PREMiSE
interleaved source manifest. PREMiSE seed 42 is the completion-contract restart;
seeds 43 and 44 are the original completed experiments. This uses the dashboard's
interleaved sources independently of the earlier sequential AppWorld-only swap.
All runs have reached iteration 130. Restarted PREMiSE seed 42 has 109 scored
tasks: 40 BFCL, 41 AppWorld, and 28 Terminal-Bench. Those coverage differences are
recorded in CSV and metadata files.

`--grouping best` selects each method's seed with the highest equal average of
its three final benchmark online means. With these sources that is ACE seed 42
and PREMiSE seed 44. Averaging all scored tasks instead gives ACE seed 43 because
the benchmarks have different numbers of tasks. Pass `--ace-root`,
`--premise-root`, and `--seeds` to choose other completed experiments.

The script reads existing results and does not execute benchmark tasks or change
experiments. Matplotlib and NumPy are in the optional `analysis` dependencies.

## Trailing windows

For recent 5/10/15/20/30/40 tasks **of each benchmark**, use:

```bash
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --windows 5 10 15 20 30 40 --window-basis benchmark --grouping by-seed \
  --output outputs/interleaved_online_windows_by_seed_20261009
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --windows 5 10 15 20 30 40 --window-basis benchmark --grouping mean \
  --output outputs/interleaved_online_windows_mean_20261009
```

For benchmark B at global iteration M, select its last N task positions at or
before M, then average the recorded scores of those positions. Other benchmarks'
tasks leave that curve unchanged. Early points use fewer than N tasks. Missing
scores count toward the N task positions but are excluded from the mean, without
fetching additional older tasks. Terminal-Bench has 30 training tasks, so its
N=40 curve equals its benchmark-wise cumulative curve.

Each directory contains `interleaved_online_N05.png` through `N40.png`, PDF
copies, and a six-window overview. `seed_windows.csv` and `aggregate_windows.csv`
record all six windows, and metadata records the window basis.

The alternative `--window-basis global` (CLI default) averages tasks separately
by benchmark within the last N **global iterations** `[max(1, M-N+1), M]`.
A benchmark with no scored task inside that interval has no score. This differs
from the benchmark-task windows explicitly selected for the figures above.

## SkillOpt encounter comparison

Add `--skillopt-csv team_prompt/skillopt_train_encounters.csv` to include green
SkillOpt curves with the same benchmark line styles, cumulative/window calculations,
and seed grouping. For the three-seed cumulative comparison:

```bash
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --grouping mean --skillopt-csv team_prompt/skillopt_train_encounters.csv \
  --output outputs/interleaved_online_mean_skillopt_20261009
```

The supplied CSV has 130 encounters per seed, with 50 BFCL, 50 AppWorld, and 30
Terminal-Bench tasks. Percentage scores are divided by 100 before aggregation.
SkillOpt's interleaved task schedule differs from ACE/PREMiSE for all three seeds.
Its curves therefore use its own recorded global iterations, without moving scores
to ACE's task order. This compares each method's observed learning stream rather
than performance on the same task at a particular iteration.

Metadata records agreement of task orders and sets, mismatch counts, and the CSV
hash. Task exports retain each encounter's original `source`. The referenced rollout
files are unavailable locally; supplied training scores are used without re-evaluation
or an inferred execution-error status. Both labeled and `_no_labels` figures are saved.

## Separate benchmark graphs

Use three horizontal plots, each with its own y scale and global iterations 1–130:

```bash
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --grouping mean --layout benchmark-panels --panel-y-start 20 \
  --skillopt-csv 'team_prompt/skillopt_train_encounters (1).csv' \
  --output outputs/interleaved_online_by_benchmark_20261009
```

BFCL, AppWorld, and Terminal-Bench have separate axes. Each displays only its three
baseline curves, using the same baseline colors and solid step lines. The cumulative
means and their global iteration positions are preserved; standard deviations remain
in the CSV exports and are not shaded, matching the previous interleaved figure.

Each y range covers all plotted means from iteration 20 onward, adds 15% padding
(at least 2.5 percentage points), and rounds outward to multiples of five within
0–100. This expands performance differences after the first few encounters. Early
values outside these zoomed ranges are clipped; x remains 1–130. The labeled figure
states the clipping rule, and metadata records each actual y range. Use
`--panel-y-start 1` to include the entire history when fitting the y axes.

The main exports are `interleaved_online_cumulative_by_benchmark.png` and its
`_no_labels` counterpart, with PDF copies. The layout also supports individual
window figures and other seed groupings; multi-window overviews use the overlay
layout only.

## Benchmark datapoint axes

Replot an existing cumulative export against each benchmark's task encounter count:

```bash
.venv/bin/python scripts/utils/plot_interleaved_benchmark_datapoints.py \
  --source outputs/interleaved_online_by_benchmark_20261009/mean \
  --output acr_oct/interleaved/mean/benchmark_datapoints --linear-regression
```

This reads frozen task scores and metadata from the specified export. At x=k,
each seed's mean covers that benchmark's first k encounters; these means are then
averaged equally across available seeds. Alignment happens before seed averaging.
Missing scores are excluded from the denominator but keep their task positions.
Other benchmarks' encounters do not produce repeated curve points.

BFCL and AppWorld extend to 50 datapoints, and Terminal-Bench to 30. Each panel
retains the source figure's y range, baseline colors, and solid step lines. Values
outside the source y ranges remain clipped. The global-iteration figures remain
unchanged. Task exports retain the original global iterations for provenance.

Both labeled and `_no_labels` PNG/PDF figures are saved. `--linear-regression`
also saves variants with thin, faint OLS lines refitted against benchmark task k,
with one equal weight per available plotted point. The coefficient CSV names
slopes in percentage points per datapoint. Metadata records source hashes,
axis counts, preserved y ranges, and the fit definition. Final scores are checked
against the original export.

Use `--layout benchmark-regions` for the sequential figure's style: one shared
0–100 score axis, three adjacent benchmark regions with vertical dividers, local
datapoint tick labels that restart in each region, and mean ±1 sample SD shading.
Adjacent observed points are connected by ordinary lines rather than steps.
The plotted means, standard deviations, and datapoint regression fits are
unchanged. This layout keeps the benchmarks' 50/50/30 relative region widths.
The plot background is white; the colored shading represents seed standard deviations.

For Pearson correlations between the plotted seed-mean scores and each region's
task steps, run `scripts/utils/report_online_step_correlations.py`. It reads the
frozen sequential and interleaved exports in `acr_oct`, verifies figure hashes,
and saves the 18 coefficients and their source points under
`acr_oct/analysis/update_step_correlation`. Both settings use all available
benchmark-local positions (1–50 or 1–30). Task positions are used as the step
axis; the statistic is a correlation of the mean curves, rather than a mean of
per-seed correlations or a correlation with per-step score differences.
Only descriptive coefficients are reported for the serially dependent cumulative curves.

## Linear regression overlays

Add `--linear-regression` to the overlay layout to draw a separate ordinary least
squares line for every baseline/benchmark curve. For the mean and individual seeds:

```bash
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --grouping mean --layout overlay --linear-regression \
  --skillopt-csv 'team_prompt/skillopt_train_encounters (1).csv' \
  --output outputs/interleaved_online_regression_mean_20261009
.venv/bin/python scripts/utils/plot_interleaved_online_performance.py \
  --grouping by-seed --layout overlay --linear-regression \
  --skillopt-csv 'team_prompt/skillopt_train_encounters (1).csv' \
  --output outputs/interleaved_online_regression_by_seed_20261009
```

The fit predicts the plotted score in percent from the global iteration, with
equal weight for every available curve point in iterations 1–130. This includes
scores carried forward during other benchmarks' tasks. Points before a benchmark
has an observed score are excluded. Each curve is fitted separately, using the
seed-mean curve for the mean figure and individual seed curves for seed figures.
Use `--regression-start` to choose a later fitting interval.

Original step curves retain their full color and width, with thin, faint straight
regression lines (opacity 0.25) in the same baseline colors and benchmark line styles.
Endpoint labels report actual observed scores. No regression confidence bands or hypothesis tests are shown.
`linear_regression.csv` records the slope (percentage points per iteration),
intercept, fitting interval, point count, descriptive R², and fitted endpoints;
metadata preserves the fitting definition. R² is left undefined for a constant
response. All original score and cumulative-curve exports are preserved.

Files use the `_regression` suffix, with paired labeled and `_no_labels` PNG/PDF
exports. Regression mode currently supports individual overlay figures, including
windows, and does not generate a multi-window overview.
