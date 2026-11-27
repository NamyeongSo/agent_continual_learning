# Qwen ACE AppWorld policy shift from saved inputs

Use the completed isolated Qwen ACE AppWorld learning for seeds 42, 43, and 44.
The collection is `outputs/appworld_isolated_checkpoints_cp005_050_ace_premise_20261008`.
Verify its thirty CP5..50 snapshots against the original ACE checkpoints, final playbooks,
learning configuration, and all fifty recorded training tasks per seed. CP0 uses the
initial playbook. No new learning, AppWorld servers, or environment sessions start.

Restore the first **twenty valid acting inputs per task** from the original ACE training
LiteLLM traces. Preserve all messages, observations, previous tool-call IDs, and the actual
available tools. Exclude tool-shortlisting requests and malformed/missing recorded actions;
report missing traces and per-seed task coverage. Source hashes and immutable history hashes
are saved and checked again on resume. The inherited learning used fifty steps per task.

For each saved input, query Qwen at CP0,5,10,...,50, replacing only the generator's first
system context with the corresponding ACE playbook. Call only the existing inference endpoint;
save its predicted API calls without executing them or updating histories. Use temperature zero,
8,192 output tokens, reasoning disabled, and **150 concurrent requests**. The shuffled queue
mixes seeds and checkpoints. The exact request count is eleven times the restored history count.

AppWorld in Exgentic exposes native API functions such as `gmail__search_emails`, so score
these function-call sequences directly. This differs from the old `agent_context` Python parser.
Loose metrics compare names; strict metrics also compare parsed JSON arguments. Accuracy
requires the full ordered sequence to match. Precision/recall/F1 use multiset overlap, retaining
duplicate calls. Transport/protocol errors are missing results, never zero-filled agreement.
All checkpoints use the same successful-history intersection. Average steps within task,
tasks equally within seed, then seeds equally; bands show seed sample SD.

The reference actions were produced by the **task-specific learned policy during original
training**, rather than by one shared initial context. The agreement figure labels this reference
explicitly. CP0 is an initial-context query on those recorded histories and is not a repeatability
control. The pairwise figure compares predictions between every CP pair under identical saved
inputs and reports `1 - ACC` and `1 - F1` versus checkpoint distance. It needs no executed API calls.
These scores measure conditional policy agreement, rather than AppWorld task success.

Prepare inputs offline with `uv run python scripts/ace/run_appworld_policy_shift.py --output
outputs/appworld_ace_qwen_policy_shift_cp5_restored_history_queue150_20261013 --prepare-only`.
Run the same command without `--prepare-only` to start the resumable inference queue.
The default endpoint is `http://127.0.0.1:18006/v1`; use `--base-url` and `--workers` when needed.

Outputs include `status.json`, `prepared.json`, `contexts.json`, `history_bank.json`,
`history_restore_audit.json`, and per-history predictions. Final analysis saves
`checkpoint_agreement.pdf/png`, `pairwise_policy_shift.pdf/png`, `checkpoint_metrics.csv`,
`seed_summary.csv`, `task_metrics.csv`, `pairwise_metrics.csv`, `checkpoint_correlations.csv`,
`coverage.json`, and `summary.json`. Source learning stays untouched. File locks prevent
duplicate controllers; restarts reuse successful predictions and retry failed probes.

The earlier single-task environment check was stopped on clarification and is isolated under
`outputs/appworld_ace_qwen_policy_shift_cp5_steps20_queue150_20261013`.
Its data are not used by this study.

## OSS own-model AppWorld study

Use `scripts/ace/run_appworld_own_model_policy_shift.py --output
outputs/appworld_ace_oss_policy_shift_cp5_steps20_queue150_20261013 --base-url
http://127.0.0.1:18010/v1 --workers 150 --max-steps 20 --reasoning-effort low`.
The pipeline trains ACE independently for seeds 42, 43, and 44 on the same fifty frozen
AppWorld training tasks and the same seeded ordering as the Qwen ACE source. Actor,
tool shortlisting, reflector, curator, and policy probes all use OSS; reasoning effort
is low for each LLM stage. Training uses the ACE default actor temperature and at most
twenty steps per task; policy probes use temperature zero and 8,192 output tokens.
The earlier Qwen source used fifty training steps, so training budgets differ and must
be reported when comparing models.

This run creates its own checkpoints and recorded histories. After all three training
runs finish, it freezes CP5..50 states, retaining the previous context when an update
was skipped. It restores up to twenty valid acting inputs per task from the own-model
traces and automatically evaluates CP0/5/…/50 with 150 concurrent inference requests.
AppWorld environments are used during the initial learning stage; the subsequent
policy-shift evaluation only queries the inference endpoint and executes no API actions.
Agreement is measured against the recorded task-specific OSS training actions.

The root `status.json` distinguishes `own_model_training`, `exporting_checkpoints`,
`restoring_training_inputs`, `same_history_probes`, and completion. Learning subprocess
fingerprints are saved under `training_workers/` and adopted on restart to prevent
duplicate learning. `launch_manifest.json` fixes the splits, model, reasoning, and step
budget. The checkpoint bundle validates model identity and source hashes before probing;
it rejects using Qwen-learned contexts as OSS-learned contexts. Plots and metrics use
the same output filenames and common-history scoring as the Qwen study.

Endpoint changes preserve the original `launch_manifest.json`; current routing is
recorded in `runtime_route.json` and `pipeline_runtime_manifest.json`. The 8006 OSS
server hides tool calls unless auto-tools are enabled. The `gpt_oss` mode of
`scripts/utils/serve_vllm_tool_parser_bridge.py` requests completion token IDs and
uses the installed official Harmony parser to recover these calls. It preserves
the original tool choice, prompt, sampling settings, reasoning, and truncation
status. The local bridge listens on 18017 and forwards inference to 8006. Pass
`--base-url http://127.0.0.1:18017/v1 --inference-upstream http://127.0.0.1:8006/v1`
to the own-model pipeline when using this route.

The migration archives the connection-failed suffixes and their derived snapshots
under `recovery/`, retains healthy prefixes CP12/12/13 for seeds 42/43/44, and
resumes their original task order. Infrastructure failures stop owned learning
processes instead of being counted as learned checkpoints. The history reader
accepts both the configured LiteLLM model name and its logged model ID; in OSS's
case these are `openai/openai/gpt-oss-20b` and `openai/gpt-oss-20b` respectively.

If a saved compact trace is missing a definition, the history restorer searches the
other recorded traces for the same content hash. It can also reconstruct the initial
user message from that session's saved task and context, accepting it only when its
hash matches the original reference. Every recovery is recorded in
`history_restore_audit.json`, and hashes of the supporting source files are included
in the frozen history bank. A reference that cannot be recovered is reported and
excluded; it does not stop evaluation of the remaining completed training runs.
The OSS study resumed policy probes on 2026-10-13 after all three seeds completed,
with 1,130 valid saved inputs and 12,430 CP0/5/.../50 requests at concurrency 150.
