# BFCL checkpoint 50 test evaluation

Run `scripts/utils/evaluate_bfcl_checkpoints.py` with the three current launch manifests:

```bash
.venv/bin/python -u scripts/utils/evaluate_bfcl_checkpoints.py \
  --launch-manifest <ace-output>/launch_manifest.json \
  --launch-manifest <premise-t3-output>/launch_manifest.json \
  --launch-manifest <ps-st-pareto-t3-output>/launch_manifest.json \
  --output outputs/bfcl50_test_all_methods
```

The batch selects sequential seeds 42, 43, and 44 whose first 50 training tasks are BFCL. It evaluates each
checkpoint on the seed's BFCL, AppWorld, and Terminal-Bench 2.0 test IDs from the nine-key split JSON:
27 groups and 1,161 tasks. Add `--prepare-only` to archive checkpoints and report existing results before execution.

Nine checkpoint workers run concurrently. Each processes the three benchmarks in order, using four task workers
for BFCL and AppWorld and one for Terminal-Bench. Terminal-Bench uses Terminus-2 with the checkpoint's 100-turn
setting. Evaluation performs no reflection, curation, or learning.

ACE archives checkpoint 50 in its normal `test_evaluation/snapshots` tree and shares a result lock with its
post-training evaluation controller. PREMiSE variants copy checkpoint 50 into separate evaluation runs, retaining
the complete saved configuration and evaluating beam rank 1. Completed compatible results are reused; newly completed
PREMiSE results are also published to the training run for reuse by its later evaluation. An existing partial native
result is preserved. No live training state is read as the evaluation playbook or replaced by this batch.

`evaluation_plan.json` records checkpoint hashes, task IDs, source runs, and independent port blocks.
`status.json` tracks all 27 groups, completed tasks, scores, errors, and worker PIDs.
`logs/*.log` contains checkpoint worker output. Resume with the same `--output` to reuse finished results and continue
partial PREMiSE evaluations. Configure the model endpoint, Docker connection, cache, and virtual environment before
launching; credentials belong in local ignored configuration.
