# Shared experiment actor limits

All future BFCL, AppWorld, and Terminal-Bench runs use the policy in
[`core/experiment_limits.py`](../src/exgentic/core/experiment_limits.py), regardless
of experiment method, training/evaluation split, or replay role.

| Limit | BFCL | AppWorld | Terminal-Bench 2.0 |
| --- | ---: | ---: | ---: |
| Input tokens per environment turn | 100,000 | 100,000 | 100,000 |
| Output tokens per environment turn | 2,048 | 2,048 | 2,048 |
| Turns per full trajectory | 30 | 40 | 40 |

Limits are inclusive. A completed response with exactly 2,048 output tokens or
a trajectory completed on its last allowed turn remains eligible for success.
The benchmark's grading still determines task success within these budgets.

One turn is one agent reaction/environment action round, or one native Terminus-2
episode. Parallel tool calls form one round. Tool selection, acting requests, and
measured retries in that round share the same input/output budget. Inputs include
the history submitted in each request, including cached prompt tokens. Token usage
resets on the next round; it is not accumulated across the whole trajectory.
Replay prefixes count toward the full trajectory's turn limit and consume no fresh
model tokens. The continuation receives only the remaining turns.

Actor request `max_tokens` is capped at the remaining output budget for that turn.
Provider-reported prompt/completion usage is checked before returned actions execute;
this avoids guessing token counts with a different model's tokenizer. An excessive
input can therefore reach the provider, but its response cannot execute an action.
Missing token usage fails closed as `missing_token_usage`. `finish_reason=length`
and Harbor's output-truncation signal fail the trajectory immediately, without
resending, recovering, or executing the partial response. Native Terminus-2 retains
Harbor's completion confirmation, with both confirmation episodes inside the limit.

Budget failures are completed, unsuccessful results with `score=0`, `success=false`,
and `is_finished=true`. `session_metadata.experiment_budget_exceeded` and
`experiment_budget_failure` preserve the reason, limit, observed/requested value,
turn index, and available usage. Provider context rejections also terminate the
trajectory. Truncation/context errors that expose no usage use `actual=0` to indicate
an unavailable count. Budget failures bypass the official
verifier and are retained as task failures rather than infrastructure retry/skip events.

The shared helper covers PREMiSE, `ps_st_pareto`, tool-calling/prompt baselines, ACE,
A-Mem, AutoSkill, Harness, ReasoningBank, and native Terminal-Bench Terminus-2.
New actor implementations must use the same accounting helper. Reflection, mutation,
curation, and judging retain independent budgets. Smaller explicit runtime limits
are allowed; a launcher cannot raise the shared caps.

The `actor-turn-v1` policy is included in canonical run/session configuration and
session identity. PREMiSE checkpoints record it and reject incompatible resumed
configurations. Historical outputs remain unchanged. Start fresh worker processes
to load the new runtime; existing Python processes retain their loaded implementation.
