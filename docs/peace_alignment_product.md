# PEACE alignment × benchmark reward

`scripts/premise/run_experiment.py --candidate-selection alignment_product` enables
a separate search variant. Default `pareto` runs retain their existing two-parent
behavior and checkpoint configuration.

At each training task, one pre-update parent produces one environment trajectory.
Four independent reflection/mutation samples generate four children from that
same parent. Each child replays the current span and the same sample of up to
three past memory spans. Existing reflection, mutation, alignment judging, replay,
official benchmark scoring, and actor budget semantics are retained.

For a child evaluated on the current replay and m past replays:

```
A = sum(alignment_j for j in current_and_past) / (m + 1)
B = sum(candidate_benchmark_score_j for j in current_and_past) / (m + 1)
selection_score = A * B
```

Each replay has equal weight. With no memory, only the current replay is used.
The separate AppWorld PCT comparison uses PCT (official individual-test pass fraction) for
candidate selection, enabled by `--appworld-candidate-metric pct`. Online and
final train/val/test reporting retain TGC, the official full-success flag. PCT is
read from `replay.partial_score`; TGC remains in `replay.score` and `official_score`.
Missing/invalid PCT never falls back to TGC or zero. Candidate rewards and metric
names are recorded separately. The generic CLI defaults to historical TGC for
compatibility; the separate comparison launcher explicitly opts AppWorld into PCT. BFCL and
Terminal-Bench keep their existing benchmark scores.

`prepare-appworld-pct` creates a separate AppWorld-only queue with two variants:
alignment-first nested Pareto (two parents, two children per parent, two survivors)
and alignment product (one parent, four children, one survivor). Both use AUD,
PCT candidate rewards, TGC reporting, seeds 42/43/44, and matched fixed splits,
Original code-generation environment policy, common reflector, and inference
profile. Three additional training slots run independently of existing cohorts.
Already scheduled cohorts retain their original manifests, code snapshots,
controllers, metrics, and queues. The existing nine-job product cohort therefore
retains its originally scheduled TGC selection in AppWorld.
The highest product child becomes the only next parent. The previous parent is
not a competing candidate. Exact ties use ascending child ID. Invalid candidates
are excluded; if none remain, the existing task-error policy applies. Valid no-op
mutations remain eligible. All candidate means, products, sample counts, selected
IDs, and detailed replay/judge evidence are saved in iteration artifacts.

`scripts/utils/run_peace_alignment_product.py` prepares an isolated code snapshot
and clones the previous cohort's split membership/order, environment policy,
inference profile, actor context, edit mode, temperatures, and internal workers.
The source cohort is provenance only; its checkpoints/results are not reused.
Seeds 42/43/44 start with BFCL, followed in the queue by AppWorld, then
Terminal-Bench. There is no benchmark completion barrier.

The controller has three training slots and three independent final Test slots.
After training finishes, it freezes the selected context and persists a completion
marker. The next controller tick refills the training slot and dispatches Test
without waiting for evaluation to finish. Each evaluation uses at most 50 task
workers, preserving the source cohort's setting. Process identity, stage, port
allocation, final context hashes, and queue state are persisted for restart.
Jobs can pin a separate immutable `snapshot`, allowing a pending benchmark's code
to be revised without replacing the files used by already running experiments.
Infrastructure failures remain errors, never measured zeroes. Evaluation retains
the existing three-attempt task recovery; failed jobs remain visible for diagnosis.
