# ACE reflection accumulation

This experiment measures whether native ACE's ADD-only Curator accumulates a fixed bank of reflections across tasks.
It waits for the matched AppWorld reflection experiment to complete, including all 500 candidate contexts and their
1,500 evaluations. It uses all original 50 training tasks, including turn-capped trajectories scored by official PCT.
The filtered plotting cohort is not a learning cohort.

For each original task-order seed (42, 43, 44), train six independent empty playbooks: without/with GT crossed with
random/best/worst reflection selection. Each task contributes exactly one of its five saved reflections.
Best and worst use the exact mean of the candidate's three official PCT evaluations. Select uniformly among exact
ties using a recorded deterministic RNG seed, independent of task ordering and process scheduling. Random selection
uses the same source trajectory index across both GT arms. Best/worst tie RNGs are separate per arm.

Learning invokes only the native Curator, sequentially over the seed's 50 pinned training tasks. It does not generate
new reflections or trajectories. Save the empty checkpoint C0 and every subsequent checkpoint C1 through C50,
including no-op edits. Six contexts may train concurrently; each context has its own playbook, IDs, calls and checkpoints.
Empty or malformed edits keep native ACE's no-op behavior. Infrastructure exceptions stop progression and preserve
the last durable checkpoint for resumption. The 900 Curator steps are logical steps; record any actual retries separately.

Finish these phases before advancing to the next seed:

1. Train all six contexts (300 Curator steps).
2. Evaluate each final C50 on the 50 training tasks (300 trajectories).
3. Evaluate each task Tt under its corresponding post-learning checkpoint Ct (300 trajectories).
4. Evaluate each final C50 on the 50 fixed validation tasks (300 trajectories).

Use one independent rollout per evaluation job, with concurrency 50. BWT is the task mean of
`PCT(C50, Tt) - PCT(Ct, Tt)`. Report both all 50 tasks and the conventional prior 49 tasks. The last-task checkpoint
evaluation remains an independent repeat. Source candidate scores are for independent updates from C0 and cannot
replace measurements under the accumulated Ct. Total planned cost is 900 Curator steps and 2,700 actor trajectories.
Validation is frozen evaluation only; no validation/test task enters learning or reflection selection.

Pin the predecessor's complete Qwen3.8 27B call settings, endpoint 8006 on physical GPU 1, teammate environment policy
bytes/hash, empty actor context, unlimited output, 100k actor input limit and 40-turn actual-PCT scoring exception.
Freeze runtime code and the SQLModel dependency overlay in a separate output directory. Native ADD-only Curator
prompts and edit application are unchanged.

Prepare and launch the detached watcher with the project Python environment:

```bash
uv run --no-sync python scripts/ace/run_reflection_accumulation.py \
  --prepare --source <reflection-run> --output <new-run>
uv run --no-sync python <new-run>/source_snapshot/scripts/ace/run_reflection_accumulation.py \
  --run --output <new-run>
```

`progress.log` and `status.json` show the current seed, phase, completed jobs, running jobs and errors.
The journal remains planned while waiting and becomes running only when learning begins. Any unresolved evaluation
error stops the next phase and seed. Relaunching the same frozen command resumes saved checkpoints and evaluations;
the scheduler lock prevents duplicate controllers. A `STOP` file prevents new work at the next scheduling/Curator boundary.

Analyze completed evaluation phases without new model calls:

```bash
uv run --no-sync python -m scripts.ace.analyze_reflection_accumulation --run <new-run> --seed 42
```

The analyzer accepts a phase only after its complete results file exists, checks all six contexts and 50 pinned tasks,
and verifies each evaluation's policy and checkpoint hash. BWT must use the task's own post-learning checkpoint.
It reports one-rollout task means, paired comparisons, task bootstrap intervals and budget failures separately;
the intervals do not estimate within-task repeatability. Full-seed reports also pair final and post-learning scores
for BWT all50/prior49. Preserve intermediate reports as clearly labelled phase results.

PCT and TGC (`appworld_tgc_v1`) are reported together. TGC uses recorded boolean official success flags;
missing outcomes are excluded and never inferred from a rounded PCT. Approved budget terminations retain their
recorded failure outcome. TGC BWT uses matched final/post-learning success flags and reports its paired denominator.

Compare the actual seed-specific selected reflections against their independent update scores:

```bash
uv run --no-sync python -m scripts.ace.compare_reflection_accumulation --run <new-run> --seed 42
```

The independent expectation averages each selected reflection's three saved evaluations on its own training task.
The accumulated score evaluates C50 once per same training task. Keep the PCT selection/tie decisions unchanged when
computing TGC. A separate uniform-random expectation averages all five candidates; it differs from the realized seed's
selected-random expectation. Same-bank best/worst selection bias, different repeat counts, fresh Curator generation,
budget outcomes and evaluation noise prevent interpreting this difference as a pure causal interference effect.
No matched independent-reflection or empty-context held-out baseline exists in this study; report held-out actuals
without substituting training scores or generating unrequested inference calls.

### Scheduled non-thinking replication

`scripts/ace/schedule_nonthinking_replication.py` prepares a separate frozen pipeline and waits for the current
thinking experiment's three seed completion markers and all 2,700 distinct evaluations before starting inference.
It regenerates the Reflection study (250 initial actors, 500 proposal chains, 1,500 updated-context actors), then
runs the same 18-context accumulation with 2,700 actor evaluations. Thinking rollouts, reflections and learned
contexts are not reused. Verified GT references, policy bytes, task orders, budgets and frozen core code are retained.

All actor, reflector and curator requests use Qwen3.8-27B with thinking/preserve-thinking disabled, temperature 0.7,
top_p 0.80, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0 and no reasoning_effort request.
Output remains unlimited. Serving stays on physical GPU1 and port8006; the existing template supports per-request
thinking disablement. Explicit ACE role overrides read the protocol so hidden T1/thinking defaults cannot override it.
Earlier frozen running cohorts retain their original code and settings.

```bash
.venv/bin/python scripts/ace/schedule_nonthinking_replication.py --prepare \
  --predecessor <thinking-accumulation-run> --output <new-pipeline>
.venv/bin/python <new-pipeline>/scheduler_runtime/scripts/ace/schedule_nonthinking_replication.py \
  --run --output <new-pipeline>
```

The scheduler saves its dependency, settings, source hashes, child commands/PIDs and current phase. Failure in
Reflection prevents accumulation from starting. Resuming the same prepared pipeline reuses only its own saved work.

### Matched AUD accumulation after both ADD-only modes

`scripts/ace/schedule_aud_accumulation.py` waits for the entire non-thinking ADD-only pipeline to finish, then runs
thinking AUD followed by non-thinking AUD. Each mode retains its own model sampler and uses its corresponding
ADD-only Reflection bank and all 18 seed-specific selection files byte for byte. It never generates new Reflections,
reranks candidates, redraws ties, or transfers an ADD-only learned Context. Candidate rankings therefore remain the
original PCT rankings of independent ADD-only proposals, holding selection fixed for the accumulation comparison.

Each AUD cohort starts from the same empty Context, calls only native Curator during learning, and saves C0~C50.
Thinking AUD retains seed42 learning/evaluation before seed43 and seed44. Non-thinking AUD uses the user's
2026-11-18 phase-major amendment: learn all three seeds' 18 Contexts first, then finish all 900 train evaluations,
all 900 BWT evaluations, and all 900 held-out evaluations. One shared queue interleaves seeds with global
concurrency 50. Training has 18 independent sequential Context chains, so its actual pool has 18 workers.
Both the native prompt and operation
application enable ADD, UPDATE and DELETE. Other task, policy, model, budget and evaluation settings match ADD-only.
The native AUD prompt differs from the ADD-only prompt, so this comparison measures the complete native edit-mode
variant rather than separating prompt wording from allowed operations. Plan: 36 new Contexts, 1,800 logical Curator
steps, 5,400 new actor evaluations and zero new Reflections. Report PCT and official TGC for each mode separately.

```bash
.venv/bin/python scripts/ace/schedule_aud_accumulation.py --prepare \
  --thinking <thinking-add-only-accumulation> --nonthinking-pipeline <nonthinking-pipeline> --output <new-aud-pipeline>
.venv/bin/python <new-aud-pipeline>/scheduler_runtime/scripts/ace/schedule_aud_accumulation.py \
  --run --output <new-aud-pipeline>
```

The selected bank is imported only after its ADD-only source completes. Its manifest pins bank evidence and exact
selection hashes. The native learner refuses to regenerate choices when a matched import is required. A failure
in thinking AUD blocks non-thinking AUD; an interrupted cohort resumes its own durable checkpoints.

With `execution_order=phase_major` and `global_concurrency=50`, progress reports phase totals of 900 and a
`by_seed` breakdown in `status.json`. Per-seed task orders, exact Reflection selections, Contexts, checkpoints,
and evaluation files are retained. Each Context still applies its 50 Reflections sequentially. BWT always uses
that task's own post-learning Context Ct. Results already saved under the same protocol are validated and reused;
an unresolved job blocks the next phase. The default standalone accumulation order remains `seed_major`.
Only the not-yet-started non-thinking AUD cohort is amended; running/completed snapshots stay unchanged.
