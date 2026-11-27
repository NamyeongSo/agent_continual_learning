# Automatic five-task checkpoint inference

The isolated Qwen comparison has one shared checkpoint inference queue. It watches
ACE, PREMiSE, GEPA and SkillOpt streams, including the queued curator variant, and
backfills completed streams. Every five distinct training task IDs defines a CP:
5, 10, 15, and so on. Repeated rollouts, replay tasks and validation examples do
not increase this count. The policy is frozen after the corresponding update or
selection has completed, not when the first rollout of the batch starts.

The active queue is recorded in `/tmp/exgentic_checkpoint_queue_output_path`.
Its reproducibility settings are in `checkpoint_queue_manifest.json`, evaluation
code in `code_snapshot`, and source hashes in `source_hashes.json`. The detached
controller uses `scripts/utils/checkpoint_queue.py --output OUTPUT` from that
snapshot. `--collect-only` verifies and registers checkpoints without inference.

## Checkpoint provenance and selection

Each stream retains its original model, seed, ordered training IDs and disjoint
train/val/test partitions. A stream identity includes its original output path,
so different seeds, benchmarks and curator variants remain separate comparisons.

| Method | Actor policy at the boundary |
| --- | --- |
| ACE | Post-curation per-session checkpoint after the fifth completed training task |
| PREMiSE | Selected beam rank 1, reconstructed from iteration decisions and bullet assessments; verified against parent policies and the saved terminal beam |
| GEPA | Highest full-validation mean among candidates admitted by that boundary; ties retain the earlier candidate |
| SkillOpt | Official trainer's validation-gate best skill at that completed five-task step |

Historical GEPA and SkillOpt snapshots are reconstructed from their native saved
histories, never by applying today's best instruction to earlier checkpoints.
GEPA pickle loading permits plain data only. A missing, inconsistent or changing
source is reported as a discovery issue and retried on a later scan. It is never
silently replaced by a future policy. Source paths and hashes accompany frozen
policies under `frozen/STREAM/cpNNN.json`.

If adjacent checkpoints have identical actor-policy bytes, the later checkpoint
references the earlier canonical checkpoint. It creates no duplicate inference
tasks. Metadata timestamps, optimizer counters and validation scores alone do not
make two policies different. A different policy starts a new evaluation group.

## One bounded inference queue

The durable SQLite queue contains one task per canonical `(CP, split, task_id)`.
Every CP covers the original complete train, val and test partitions. All streams
and splits share a single executor with at most the manifest's `workers` active
inference tasks (default 100; the resumed 8006-port run uses 128). Dispatch,
executor capacity, heartbeat and live summaries use that same limit. All limits
retain the same global controller lock to prevent overlapping queues.
Final-test requests from the existing experiment controllers also wait for this
queue's results. Their waiting processes perform no additional inference. The
original optimizers' internal candidate validation remains part of training.

Already completed final-test results are reused only when the selected policy
and evaluation settings match. The original learning subprocesses continue during
controller handoff. Small final-test dispatch overlays are retained with before
and after hashes in `source_overlay_manifest.json`; their original files remain
under `source_overlays`. Training prompts, actor behavior, splits, budgets and
optimizer decisions are unchanged.

Runtime repairs are referenced by `runtime_recovery.json` and the manifest's
`runtime_overlays` list. Retain the original source hashes and apply these
audited before/after overlays when identifying the actual runtime revision.
An invalid learning stream remains in the database for audit but is excluded
from dispatch, progress totals and score tables. In particular, native
Terminal-Bench histories that used verifier-execution success instead of the
official reward are replaced by fresh `*_harbor_reward_v2` streams with matched
settings. Their earlier checkpoints are explicitly counted as excluded in the UI.

Inference uses temperature 0.7 and shared policy `actor-turn-v1`: at most 100,000
input and 2,048 output tokens per environment turn, 30 BFCL turns and 40 AppWorld
or Terminal-Bench turns. Budget violations are measured zero-score failures and
are not infrastructure retries. Infrastructure failures receive at most three
attempts and remain errors if exhausted. Attempt records, raw sessions, measured
cost reports and original reused-result paths are retained. Evaluation disables
learning and stores results outside the optimizers' feedback inputs.

The controller scans for new boundaries and newly prepared compatible Qwen
streams automatically. Its heartbeat and task states survive terminal
disconnection; SQLite records allow recovery after controller interruption.
`pause_dispatch.json` prevents new submissions while existing tasks finish.
Check worker ownership and use the recorded frozen controller when recovering.

## Resume after teammate-policy final testing

`scripts/utils/schedule_checkpoint_resume.py --output SCHEDULE` reads a pinned
`schedule.json` and waits for all six teammate-policy PEACE contexts (ADD-only and
AUD, seeds 42–44) to commit 50 training tasks and 50 official TGC test outcomes.
It also waits for the shared ACE/PEACE final-test queue to drain all 600 tuples.
Missing outcomes, infrastructure errors, changed policy hashes and incomplete
summary publication keep the checkpoint queue paused.

The reservation selects exactly 36 original-policy streams: ACE/PREMiSE AUD and
GEPA/SkillOpt across the three benchmarks and seeds. `selected_stream_ids` limits
dispatch, interrupted-task recovery and summary counts. `auto_discover: false`
prevents new cohorts from entering this resumption. Other streams and their
historical charts remain intact; their pending work is shown as paused. Completed
results are reused, including measured zeros. A changed pause owner or source
configuration prevents automatic release. No training or model server is started.
The scoped queue uses one executor with the manifest's worker limit and exits
when its work is drained.

## Live monitoring

Open the local research journal at `http://127.0.0.1:8878/#live`. The
**5 task CP 평가 · 공용 큐** panel shows the active count out of 100, waiting,
completed and failed tasks, checkpoint counts and unchanged-policy aliases.
Each stream expands into CP rows with train/val/test progress bars and completed
outcome means. Method, benchmark, seed, state and CP filters apply to the display.
Partial means are labeled as such; task progress is separate from success rate.
Timestamps are rendered in KST regardless of the browser timezone. The page
refreshes progress every five seconds without replacing the editable notebook.
