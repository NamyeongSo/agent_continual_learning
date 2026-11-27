# PREMiSE experiments on BFCL, AppWorld, and Terminal-Bench 2.0

## Local alignment judge audit dashboard

```bash
./.venv/bin/python scripts/premise/serve_alignment_dashboard.py --port 8766
# Open http://127.0.0.1:8766/
```

The HTML dashboard discovers runs containing actual alignment judgments under
`outputs/`. It shows each current or past judge result separately, with the
original span, the relevant reflection, replay actions and observations, and
the judge's score, label and rationale. Past evidence comes from the accepted
memory's original parent and reflection; unavailable evidence is flagged.
Task context, public tool schemas, full original trajectories, mutation details,
and official scores can be expanded. The dashboard makes no model calls.
Native commands and code strings are displayed without JSON string escaping;
the exact recorded JSON remains expandable. A prompt panel reconstructs the
current `llm.py` judge template from the case evidence; it is labeled as a
reconstruction, not an original historical request.

Human reliability ratings are binary: **0 = criterion not met, 1 = criterion
met**. Unrated axes are `null`, distinct from zero. Five axes include explicit
pass/fail guidance:

| Axis | Evaluation question |
| --- | --- |
| 근거 충실성 / evidence grounding | Are the rationale's claims supported by recorded actions and observations? |
| 의도 해석 / intent interpretation | Does the judge correctly interpret the reflection's expected behavior? |
| 행동 중심성 / behavioral focus | Does it compare actual behavior rather than rely on final scores or playbook wording? |
| 점수 적정성 / score calibration | Is the judge's original 0 / 0.5 / 1 alignment score appropriate? |
| 설명 일관성 / explanation consistency | Do the score, label and rationale agree without contradictions? |

All five axes must be rated before checking **직접 검증 완료**. Axis scores,
notes, and the completion checkbox automatically save to
`outputs/alignment_judge_audit/reviews.json`, including partial reviews.
Successful file replacement is confirmed in the UI. A failed save retains the
draft and offers retry; browser-local drafts provide recovery when available.
The JSON export includes saved reviews and any remaining browser drafts.
The review file survives browser reloads and server restarts. Changing an
artifact or its past source produces new case IDs, preventing old annotations
from silently attaching to changed evidence.

Filters cover experiment, benchmark, current/past, judge score and review
status. Optional shuffled ordering is deterministic. Judge hiding supports
reading the behavior first. Summary metrics use the current filters: axis
pass rates exclude unrated axes; the all-five pass rate uses only completed
reviews. These are descriptive rates for the reviewed sample, not estimates
of independent repeated-judge reliability. Candidate and task judgments can
be correlated.

The server binds to loopback by default. For a remote workspace, forward the
port from your own computer with `ssh -N -L 8766:127.0.0.1:8766 <ssh-host>`,
then open `http://127.0.0.1:8766/`. `--host`, `--output-root`, and
`--review-file` can be configured. Reviews and server logs remain ignored
experiment outputs.

## Protocol shared with ACE

The default split is [`bfcl_appworld_terminalbench2_seed42_44.json`](../utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json). For each seed (42, 43, 44), BFCL and AppWorld each use 50 train, 50 validation, and 50 test IDs; Terminal-Bench 2.0 uses 30 train, 30 validation, and 29 test IDs. PREMiSE and ACE use the same deterministic per-benchmark task shuffle and the same sequential (BFCL → AppWorld → Terminal-Bench) or interleaved ordering. Each run processes 130 training tasks.

Both methods pass **one full playbook across all three benchmarks within a run**. They use the same seven ACE sections, bullet IDs, helpful/harmful counters, ADD operation, and acting system prompt. Each seed and mode has an independent playbook. PREMiSE retains two playbooks in its beam and does not share either one with an ACE run. Its span reflection, candidate replay, alignment judge, past-task memory, and Pareto selection remain method-specific.

Terminal-Bench uses Harbor Terminus-2, with a 100-turn default. PREMiSE records every Terminus-2 episode (one parsed command batch), including full observations. A reflection selects a contiguous episode span. To score a candidate, PREMiSE starts a fresh Docker task environment, replays the original episode responses through Terminus-2's parser and tmux command executor through `span.start - 1`, and continues from `span.start` using the candidate playbook. Prefix episodes make no model calls; fresh terminal observations form the restored chat history. The original 100-turn budget includes replayed episodes. Stateful or nondeterministic external effects may produce different observations in a new environment; these are recorded, never replaced with the old outputs.

After replay, the first Pareto layer uses current-task and sampled-past-task behavioral alignment (0, 0.5, or 1). The second layer uses **official benchmark scores** from the candidate's actual completed session: BFCL's official accuracy, AppWorld's official task score, or Harbor's Terminal-Bench reward. All three are already in `[0, 1]`; no additional PCT surrogate or language-model grade is used as the performance objective. An execution error is not silently converted to an official zero during training.

## `ps_st_pareto` variant

`ps_st_pareto` keeps PREMiSE's two parents, four span reflections and mutations, candidate replay, benchmark-balanced past-task sampling, and shared ACE-format playbook. It skips the behavioral alignment judge entirely. Each valid candidate has two objectives: its official score on the current task (plasticity) and the mean official replay score on sampled past tasks (stability). The first nondominated Pareto frontier takes priority. If it contains at least three candidates, the two with the highest `0.5 × current + 0.5 × past` score become the next parents. The same score orders parents within smaller frontiers; if the first frontier has only one candidate, the second parent comes from the next frontier. Candidate ID breaks equal-score ties. Before any past task is available, selection uses the current score alone. Rank 1 is evaluated at the same 50/100/130 checkpoints as PREMiSE.

Use `--method ps_st_pareto` with either experiment script. The matrix launcher chooses a separate default output root, `outputs/ps_st_pareto_bfcl_appworld_terminalbench2_seed42_44`, so its checkpoints cannot mix with PREMiSE's:

```bash
./.venv/bin/python scripts/premise/run_matrix.py --method ps_st_pareto --dry-run
./.venv/bin/python -u scripts/premise/run_matrix.py --method ps_st_pareto
```

For the sequential-only seed-42–44 t3 and t6 experiments, use separate output roots and port blocks. Each launcher runs one experiment at a time, so the two launchers together run two training experiments concurrently. `evaluate_early_bfcl.py` watches each run's frozen checkpoint 50 and immediately evaluates its BFCL test IDs while training continues through AppWorld and Terminal-Bench. These results appear in `test_evaluation/cp050_bfcl.json`; the later full checkpoint evaluation resumes them.

```bash
./.venv/bin/python -u scripts/premise/run_matrix.py --method ps_st_pareto \
  --modes sequential --past 3 --max-parallel 1 --port-block-base 48000 \
  --output-root outputs/ps_st_pareto_sequential_t3
./.venv/bin/python -u scripts/premise/run_matrix.py --method ps_st_pareto \
  --modes sequential --past 6 --max-parallel 1 --port-block-base 51000 \
  --output-root outputs/ps_st_pareto_sequential_t6
# In separate shells after each launch_manifest.json appears:
./.venv/bin/python -u scripts/premise/evaluate_early_bfcl.py \
  --launch-manifest outputs/ps_st_pareto_sequential_t3/launch_manifest.json --port-block-base 54000
./.venv/bin/python -u scripts/premise/evaluate_early_bfcl.py \
  --launch-manifest outputs/ps_st_pareto_sequential_t6/launch_manifest.json --port-block-base 57000
```

Context length failures in a parent, reflector, curator, or candidate replay invalidate the affected branch or candidate. Remaining valid candidates can still be selected. If fewer than two valid candidates remain, PREMiSE records a skipped task, preserves the beam and memory, and advances the task cursor. Other infrastructure errors stop the run for diagnosis. No fixed 100k token cutoff or trajectory truncation is applied.

Terminal-Bench task deadlines preserve the partial Terminus-2 trajectory, token usage, and official verifier score. PREMiSE can reflect on completed episodes from that run. A final response whose command batch was interrupted stays in the original recording and in the `interrupted_episodes` field of a separate replay file. A deadline with no usable recorded episodes invalidates that branch; valid candidates from the other parent can continue. Provider timeouts that occur before the task deadline still propagate as infrastructure failures.

An execution deadline reported without an official Terminal-Bench score
invalidates that parent or candidate branch. Provider errors with an error
message still propagate as infrastructure failures. Task networks use the
isolated `/28` allocation described in [the Terminal-Bench adapter docs](../../docs/terminalbench2.md).

## Six-run experiment and checkpoint test

```bash
# Load local model/Docker environment by your usual private setup.
./.venv/bin/python scripts/premise/run_matrix.py --dry-run
./.venv/bin/python -u scripts/premise/run_matrix.py
```

The matrix has six training runs (three seeds × two modes). It starts all six concurrently by default, with disjoint service-port blocks. Each experiment uses two concurrent parents, four reflections, four mutations, and up to 16 concurrent replay jobs for the default `--past 3` setting. `--past 6` raises the replay ceiling to 28. The default has no per-benchmark session semaphore; `--replay-workers`, `--llm-calls`, and `--max-parallel` can cap resource use. Iterations within a run remain sequential because each one depends on the selected beam.

To train both `past=3` and `past=6` matrices with two total running experiments, launch each matrix with `--max-parallel 1` and a distinct 1000-port block range. For example, use the default base `20000` for `past=3` and `--port-block-base 26000` for `past=6`. Their output roots must differ so their six runs and checkpoint evaluations remain separate.

After task ordinals 50, 100, and 130, `state.json` and immutable `checkpoints/checkpoint_*.json` retain the beam and memory. After training, rank 1 of each checkpoint is frozen and evaluated on **all three seed-specific test sets**, giving 54 evaluation groups and 2,322 task evaluations across the matrix. The rank-2 playbook remains in the snapshot for inspection. Test evaluation never updates the training state. It uses the ACE test concurrency of four BFCL/AppWorld tasks or one Terminal-Bench task at a time per experiment. Errors are recorded with a zero in test summaries, as in ACE. Completed results are resumed only when the checkpoint, manifest, and test IDs still match.

Default output: `outputs/premise_bfcl_appworld_terminalbench2_seed42_44`. `launch_manifest.json` fixes the task and evaluation plan; `matrix_status.json` tracks processes; `logs/*.log` contains run output. Individual runs contain `state.json`, `iterations/*.json`, `checkpoints/*.json`, `training_complete.json`, and `test_evaluation/*.json`. Outputs, benchmark installations, and Docker credentials must remain outside Git.

`--adopt-running` allows a replacement launcher to monitor live workers listed
in the previous `matrix_status.json`, preserving their current iterations when
the controller concurrency changes. The previous controller must release its
launcher lock first. Adoption verifies the complete worker command and process
start time; a missing worker resumes from its saved cursor. The frozen launch
manifest and checkpoint evaluation protocol remain the same.

For one run:

```bash
./.venv/bin/python scripts/premise/run_experiment.py \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json \
  --benchmarks bfcl,appworld,terminalbench2 --mode sequential --seed 42 \
  --all-train-tasks --terminalbench2-max-turns 100 --past 3 \
  --output-dir outputs/premise_example --evaluate-test
```

To evaluate an already completed run, add `--evaluation-only --evaluate-test`. `--evaluate-val` separately evaluates the final rank-1 playbook on validation IDs. An older PREMiSE `state.json` with the benchmark-scoped entry schema cannot resume under the ACE-compatible playbook schema; use a new output directory for this protocol.
