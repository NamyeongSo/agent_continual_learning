# Sequential BFCL root versus final policy shift

Run `scripts/premise/run_bfcl_root_final_policy_shift.py` to compare ACE, GEPA,
SkillOpt, and PREMiSE using Qwen, seeds 42/43/44, and the fifty frozen BFCL training
tasks per seed. Only two policies are probed per method and seed: the original
root and the policy selected at the end of the BFCL stage (CP50). Intermediate
policies and later AppWorld/Terminal-Bench training are excluded.

The roots are the initial ACE/PREMiSE playbook, GEPA candidate zero, and SkillOpt
`skill_v0000.md`. Final policies are the original ACE CP50 snapshot, the exported
GEPA BFCL validation winner, the exported SkillOpt BFCL best skill, and the
post-selection PREMiSE CP50 rank-1 beam. Source models, task splits, selected
exports, and checkpoint hashes are verified before freezing these prompts.

Reuse the 1,873 saved Qwen BFCL acting inputs from
`outputs/bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/qwen`.
Replace only the first system message with the exact full method prompt; preserve
all later messages, previous tool-call IDs, and tools. Query the current Qwen
endpoint with temperature zero, thinking disabled, and 8,192 output tokens.
No training, benchmark environments, or predicted API actions are executed.

The controller uses 150 concurrent requests within each seed/method cohort.
Identical prompts share cached predictions under the same saved input; the
24 policy instances require 13,111 distinct input/prompt requests. Successful
responses are reused on resume and failed responses are retried. Each cohort
finishes inference before the next starts, and the current bar chart is updated
after every completed cohort. Intermediate plots identify the completed count.

Compare predicted root and final tool-call sequences using loose names and strict
names plus parsed JSON arguments. Accuracy requires the exact ordered sequence;
precision/recall/F1 use multiset overlap, retaining duplicate calls. Every completed
method uses the same successful-input intersection within a seed. Average acting
inputs within task, tasks equally within seed, then seeds equally. Error bars show
one sample seed standard deviation. These are conditional policy agreement scores,
not BFCL task-success scores. Correlations are not estimated from the two policies.

The model server is port 8006. The client-side Qwen parser bridge on 18018 preserves
the chat-template input tokens and restores tool calls through the installed
official `qwen3_coder` parser. `/tokenize` confirmed identical auto/none prompts,
and a live ping call confirmed tool parsing with no reasoning output.

Outputs are under
`outputs/bfcl_sequential_root_final_policy_shift_qwen_queue150_20261013`.
`status.json` reports the current cohort's attempted/total requests and completed
cohorts; `runtime_manifest.json` records the complete study request count.
`analysis/root_final_policy_shift.pdf/png/svg` and the matching LaTeX are updated
automatically, with `seed_metrics.csv`, `method_metrics.csv`, and `summary.json`.

The earlier ancestry-only preparation was superseded before issuing policy probes.
Its status file records the replacement; intermediate policies are never queried
by this root/final study.

## Root versus CP130 on the same BFCL inputs

Pass `--checkpoint 130` to use the policies after the complete sequential
BFCL-50, AppWorld-50, Terminal-Bench-30 curriculum. GEPA and SkillOpt take their
exported Terminal-Bench-stage selected winner; ACE and PREMiSE use CP130 snapshots.
The reference remains each method's original pre-BFCL root. In particular,
Terminal-Bench candidate/step zero is an inherited learned policy, not the original
root. The evaluation benchmark and all recorded inputs remain BFCL.

The default output becomes
`outputs/bfcl_sequential_root_cp130_policy_shift_qwen_queue150_20261013`.
CP50 predictions are reused only when the input, full policy text, model, and
complete request hashes match. `cache_import.json` records every copied response
and its source hash. The current CP130 study imports 6,232 predictions and issues
6,879 new requests. This includes the identical SkillOpt seed-43 CP50/CP130 selected
policy, as well as all root predictions. The CP50 study files remain unchanged.

Plots, CSV files, and LaTeX use the same filenames in the CP130 output directory;
their metadata and captions explicitly identify CP130. The same 150-request
concurrency, thinking setting, token budget, and common-input scoring apply.
