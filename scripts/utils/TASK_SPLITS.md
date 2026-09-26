# Shared train/validation task split

For AppWorld (`test_challenge`), BFCL (`multi_turn_base`), and SWE-bench
(`princeton-nlp/SWE-bench_Verified`), use the manifest created by
`create_task_splits.py` as the task-ID source for training and validation.

```bash
.venv/bin/python scripts/utils/create_task_splits.py \
  --seed 42 --train-count 50 --val-count 50 \
  --output scripts/utils/task_splits/seed42_train50_val50.json
```

The policy is implemented in `task_ordering.split_task_ids`: sort the complete
task-ID list, derive an independent seed from the master seed and benchmark
slug, shuffle once, assign positions 1–50 to `train` and 51–100 to `val`.
The generator fails on duplicate IDs or fewer than 100 available tasks. The
manifest stores the exact IDs and benchmark subset, and refuses to overwrite
a different split. GEPA, SkillOpt, and other methods should load those IDs
directly rather than generating a new split for each run.

Validation is read-only with respect to learned agent state: load a frozen
training checkpoint, evaluate the 50 `val` IDs, record scores, and do not feed
validation scores back into memory/playbook/skill updates. Official scoring is
still needed to report validation performance; it is the *learning update*
that must be disabled. Do not select or revise the split based on validation
results.

Existing ACE and baseline experiment scripts still use their legacy selection
policy. In particular, the old ACE `seed=42, num_tasks=50` task set is **not**
assumed to equal the new manifest's `train` set because the new policy sorts
the source IDs before shuffling. Compare old and new runs only after checking
their saved task ID lists.

For the BFCL + AppWorld + Terminal-Bench 2.0 experiment, use
`seed42_train50_val39_terminalbench2.json`. Terminal-Bench 2.0 has 89 tasks,
so its frozen split has 50 training and 39 validation IDs. The two existing
benchmarks keep the same 50 training IDs. See [the Terminal-Bench setup guide](../../docs/terminalbench2.md).
