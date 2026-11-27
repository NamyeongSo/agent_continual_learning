# Online performance stabilization

Generate Sequential and Interleaved EWMA and recent-ten-task plots from frozen
raw task scores:

```bash
.venv/bin/python scripts/utils/plot_online_stabilization.py
```

The default output is `acr_oct/stabilization/{sequential,interleaved}/{ewma,rolling10}`.
Each of the four mean figures has labeled and `_no_labels` PNG/PDF exports. Plots
use a white background, ordinary line connections, common 0–100 score axes,
three benchmark regions, and mean ±1 sample SD across seeds 42, 43, 44.
Sequential retains its shared initial-context reference lines and omits checkpoint
test lines. Interleaved ticks count each benchmark's task encounters separately.

Smoothing operates on raw task scores, separately for every seed and benchmark,
before averaging available seed curves equally. EWMA uses alpha=0.2, initializes
at the first recorded score, and pauses its state at missing scores; missing
encounters appear as gaps in seed EWMA curves. Recent-task means use the most
recent ten encounter positions and omit missing scores from the denominator.
Partial windows are plotted, but excluded from stability assessment. Benchmark
switches retain the separate interleaved benchmark histories.

An eligible assessment uses ten consecutive eligible smoothed points. The
absolute ordinary-least-squares slope must be at most 0.2 percentage points per
step, and the score range must be at most 5 percentage points. Both conditions
must hold at five consecutive assessment endpoints. These are configurable
analysis thresholds, not a statistical test of convergence.

Eligibility requires a full warm-up and observed scores. Recent-task smoothing
also requires all ten raw scores to be present in each evaluated smoothing
window. Mean-curve assessments require all three seeds to be eligible at every
point, so changing seed coverage cannot create an apparent plateau.
The earliest eligible assessment endpoint with complete data is task 19, and
five consecutive passes can first confirm an episode at task 23.

A star marks the first passing assessment endpoint of a confirmed final episode
that continues through the observed end. A hollow circle marks an earlier
confirmed episode when the final episode does not meet that requirement. The
graph explicitly labels curves with no confirmed episode as `not stabilized`;
curves with no eligible assessment windows are `insufficient data`.
All episodes, onset/confirmation steps, and pointwise criteria are exported.

Mean-curve decisions are assessed on the displayed means. Individual-seed
decisions are assessed separately and stored under scopes `42`, `43`, `44`.
`stabilization_summary.csv` records both kinds of decisions, and
`smoothed_curves_and_metrics.csv` contains every plotted score and assessment.
Metadata records source hashes, parameter values, eligibility, and full episodes.
A stable interval of online task performance alone does not isolate changes in
the learned agent from changes in task difficulty.

The recent-ten-task Interleaved figure and cumulative Sequential N50 figure can
also be rendered as three panels with independent benchmark y ranges:

```bash
.venv/bin/python scripts/utils/plot_online_independent_y.py
```

This reads the frozen plotted means and sample SDs, preserving smoothing and
stability decisions. Outputs are under `acr_oct/independent_y_ranges/`, with
`*_by_benchmark` labeled and unlabeled PNG/PDF aliases in `acr_oct/`.
Sequential keeps global iteration ranges 1–50, 51–100, and 101–130; Interleaved
keeps benchmark-local datapoint ranges 1–50, 1–50, and 1–30.
Each y range fits mean scores from local task 10 onward and any initial-context
reference, with 15% padding (at least 2.5 percentage points), rounded outward to
multiples of 5 within 0–100. Earlier means and SD-band portions outside these
ranges are clipped. Source curves remain unchanged and are exported alongside
range metadata. `--focus-start` configures the range-selection warm-up only.

For the Interleaved benchmark-datapoint cumulative figure, use:

```bash
.venv/bin/python scripts/utils/plot_online_independent_y.py --setting interleaved --interleaved-curve cumulative
```

This reads `aggregate_cumulative_datapoints.csv` from the frozen sequential-style
export. It preserves the cumulative means and SDs without applying new smoothing
or stability assessments. Its default y ranges are BFCL 40–75, AppWorld 10–45,
and Terminal-Bench 20–45. The four PNG/PDF exports use the basename
`interleaved_benchmark_datapoints_seq_style_independent_y` and are stored under
`acr_oct/independent_y_ranges/interleaved_cumulative/`, with aliases in `acr_oct/`.
