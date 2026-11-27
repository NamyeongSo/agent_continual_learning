# ACE and PREMiSE playbook edit modes

Both training scripts accept `--playbook-edit-mode`:

| Mode | Allowed operations |
| --- | --- |
| `add_only` (default) | ADD |
| `add_update` | ADD, UPDATE |
| `add_update_delete` | ADD, UPDATE, DELETE |

The default retains the historical ADD-only curator prompt. Existing running jobs
continue using their frozen source snapshots and original settings. Enabling an
edit mode requires a fresh experiment output directory; changing policy while
resuming or continuing an existing run is rejected. An old checkpoint without an
edit-mode field means `add_only`.

UPDATE and DELETE are supported in both the prompt and the shared operation
application code. UPDATE replaces the entire entry while keeping its ID, section,
and helpful/harmful counters. It removes any old continuation lines before writing
the replacement. DELETE removes the entry and its continuation lines while leaving
section headers and unrelated entries intact. Neither operation consumes an ID;
later ADDs do not recycle removed IDs. PREMiSE edits a deep copy of its parent and
uses the existing replay, judging and selection procedure to evaluate the child.

Unknown or ambiguous IDs, unsupported operations, and empty UPDATE contents are
ignored. A target must occur exactly once in the parent playbook. Edits are applied
in order; an UPDATE after deleting its target cannot resurrect it. PREMiSE retains
`proposed_operations` and `applied` in candidate artifacts. ACE logs skipped edits
through `ace_failures.jsonl` with component `curator_operations_ignored`.

The existing response envelope remains a JSON object:

```json
{
  "reasoning": "Brief evidence-based edit summary",
  "operations": [
    {"type": "UPDATE", "entry_id": "str-00001", "content": "Complete rewritten rule", "rationale": "Existing coverage, evidenced gap, behavior change"},
    {"type": "DELETE", "entry_id": "str-00002", "rationale": "Superseded by the rewritten rule"}
  ]
}
```

The application code also accepts `op` as an alias for `type`, and `bullet_id` as
an alias for `entry_id`. ACE's editable curator accepts a bare operations array as
well; PREMiSE keeps its single-object analysis response contract. IDs are read from
the current playbook. The section names remain the existing ACE section names.

Editable curator prompts incorporate the supplied evidence-to-edit principles:
verify reflection claims against visible evidence, prefer coherent rewrites over
correction appendices, preserve valid scope and examples, avoid evaluator-answer
leakage, and permit no change. They request 0–3 focused operations. ACE supplies
its execution trace and, in training-time mode, post-session grading; PREMiSE
retains its span, full trajectory and public tool contract evidence.

For example, prepare **separate new runs** using the current Qwen server:

```bash
.venv/bin/python scripts/ace/run_experiment.py \
  --mode isolated --seed 42 --benchmarks bfcl \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json \
  --all-train-tasks --model openai/Qwen/Qwen3.6-35B-A3B --temperature 0.7 \
  --training-time --disable-thinking --max-tokens 8192 \
  --playbook-edit-mode add_update_delete \
  --output-dir outputs/qwen36_edit_modes/add_update_delete/bfcl/ace/s42/train

.venv/bin/python scripts/premise/run_experiment.py \
  --mode isolated --seed 42 --benchmarks bfcl \
  --task-split-manifest scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json \
  --all-train-tasks --model openai/Qwen/Qwen3.6-35B-A3B --temperature 0.7 \
  --base-url http://127.0.0.1:18006/v1 --past 3 \
  --parent-workers 2 --reflection-workers 4 --mutation-workers 4 \
  --replay-workers 28 --llm-calls 32 --unlimited-benchmark-sessions \
  --playbook-edit-mode add_update_delete \
  --output-dir outputs/qwen36_edit_modes/add_update_delete/bfcl/premise/s42/train
```

For ACE, set `OPENAI_API_BASE=http://127.0.0.1:18006/v1` and the local server's API
key in the environment as in the original run. These examples do not start final
test evaluation or copy the running comparison into a new variant.

Actor budgets stay at 100,000 input / 2,048 output tokens per environment turn,
30 BFCL turns, and 40 AppWorld/Terminal-Bench turns. All-role temperature, analysis
budgets, task splits, seeds, internal worker counts, and ADD-only baseline artifacts
are unaffected by adding this capability.

No ablation results have been measured for the editable modes. Future comparisons
must match splits, seeds, models, budgets and the remaining training settings. The
editable modes include additional evidence and concision instructions compared
with the legacy prompt; control that wording as well before attributing a measured
difference solely to the operation set. `add_update` versus `add_update_delete`
shares those instructions and differs in whether DELETE is advertised and applied.
