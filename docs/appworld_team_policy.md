# AppWorld teammate-policy learning

`scripts/utils/run_appworld_team_policy.py` prepares a fresh isolated cohort:
ACE and PREMiSE each use ADD-only and ADD+UPDATE+DELETE with seeds 42, 43, 44;
GEPA and SkillOpt each use their existing optimizer with those three seeds.
There are 18 runs, with the original frozen 50/50/50 AppWorld splits and orders.

The environment policy is frozen from
`scripts/premise/policies/appworld_teammate_completion.txt`. Its exact bytes and
SHA-256 are recorded in training configurations, checkpoint stream provenance,
and frozen evaluations. Actor execution, parent replay and checkpoint train,
val and test evaluations use the same environment policy. Learned playbooks
remain separate. Reporting uses TGC and the original PREMiSE first-parent Online
definition. Result-table cohorts require matching environment-policy hashes.

The existing GPU 0 Qwen services are explicitly shared. All roles retain
temperature 0.7 and the actor-turn-v1 limits of input100000, output2048 and
40 turns. PREMiSE retains parent2/reflection4/mutation4/replay28/LLM32;
GEPA/SkillOpt retain 50 internal workers and unlimited cumulative training.
The three child controllers run six experiments each concurrently.

Prepare with `--output`, `--service-owner` and `--queue-output`; then use
`--output ... --start`. Preparation freezes source code and creates planned
journal records without invoking a model. Starting creates the final-test proxy
markers and launches all 18 training runs. Repeated start is rejected.

The shared checkpoint queue keeps its maximum of 100 active evaluations.
When its loaded workers need the new policy-propagation support, new dispatch
temporarily pauses while all active evaluations finish. A recorded handoff
process then replaces only that queue controller and its idle workers, adds the
new nested source roots, and resumes dispatch with the new frozen code. It
preserves completed results and retry counters. Five-task checkpoints generated
during the handoff are recovered automatically from the training artifacts.
Its `queue_handoff_status.json` reports the remaining active count and new PID.

Private execution environments are stored only in the ignored cohort output
with mode 0600. Existing experiments, models and Docker networks are not modified
by the cohort launcher.

To prioritize existing ACE/PREMiSE learning without checkpoint or final-test
inference, `scripts/utils/resume_appworld_training.py --root CHILD --resume-output
RESUME` accepts a prepared batch with `evaluation_deferred=true` and an explicit
`base_url`. The resume descriptor's `services.json` records the forwarded model
endpoint and matching parser bridge process identities. The batch retains its
frozen actor code, policy hash, splits, learning state, budgets and internal
concurrency. Completed training remains `test_pending`; no evaluation workers
are started. A later normal evaluation run can consume those frozen results.
The previous seed-wave scheduler must remain stopped when all three seeds are
resumed together, and unrelated optimizer/benchmark batches remain paused.

Final ACE testing can run independently with
`scripts/utils/evaluate_appworld_final_queue.py --prepare --cohort COHORT --output OUT`,
then `--run --output OUT`. It freezes six ADD-only/AUD seed42–44 contexts and
places their 300 test tuples in one persistent queue with a global concurrency
of 50. Workers use the existing cohort's frozen runtime and exact environment
policy. Results return to each original run's evaluation directory, and the
journal shows separate final-test activity while the learner remains untouched.
Intermediate checkpoint evaluation is not resumed. Official budget failures
remain measured zeroes; only infrastructure errors receive up to three attempts.

After ACE testing, `--enable-peace --output OUT` enables automatic final-context
registration for the same two curator cohorts and all three seeds. Restart the
same queue with `--run --output OUT`. It checks every five seconds for both a
full PREMiSE training cursor and the training completion marker, adds each
rank-1 final context exactly once, and keeps waiting while other seeds learn.
ACE's existing 300 results remain in the queue; the eventual total is 600 test
tuples. ACE and PREMiSE share one global concurrency limit of 50, and intermediate
checkpoint evaluation remains paused. The tuple CSV updates when contexts join.
