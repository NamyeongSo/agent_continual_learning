# PReMISE rubric judge for trajectory preferences

This independent implementation follows [PReMISE, arXiv:2605.30803v1](https://arxiv.org/html/2605.30803v1).
No official repository was found. It is distinct from our original PREMiSE beam/playbook reward trainer.
The judge model stays fixed; learned artifacts are reusable rubric definitions.

## Learning path

`scripts/premise/run_rubric_judge.py` runs the paper's batch discovery and preference-repair path:

1. Extract criteria and A/B criterion scores from each **training** preference pair (Appendix F.1).
2. Embed definitions; average-link cosine agglomerative clustering; recursively split clusters larger than 2T.
3. An LLM partitions each cluster into operationally equivalent subgroups of 2–5 criteria. Every ID must occur
   exactly once. Describe each subgroup as one criterion; sum source support and union source pair IDs.
4. Repeat up to ten levels, stopping at insufficient reduction or a small selection pool (Appendix A).
5. Filter by minimum support and a reusable/observable genericity check. Select a diverse discovered rubric
   using MMR with normalized log-support relevance and maximum-cosine redundancy.
6. Re-score every supported candidate against **all discovery training pairs**, with labels hidden.
   Keep criteria whose preferred response scores strictly higher on at least eta of applicable pairs (§3.3).
   Ties count as non-wins; inapplicable pairs are excluded. Rank by preference fit, support, then ID;
   take up to K criteria. This is a strict subset/reordering: no textual rewrite during preference selection.
7. Evaluate no-rubric, discovered, and preference-repaired judges on the fixed validation split (Appendix F.10).
   If no criterion passes the training threshold, record `no_qualifying_criteria` and an empty selected
   rubric; evaluate only no-rubric and discovered variants. The empty selection is a measured outcome,
   not an execution failure or a successful learned judge. No threshold is lowered, criterion substituted,
   or empty repaired variant assigned a synthetic validation score. Optional repaired-rubric audits are skipped.
   Each held-out pair is tested in both A/B orders. Report per-benchmark ordinary accuracy, swapped accuracy,
   and order-flip rate. These are preference accuracies, not benchmark task success rates.

The existing seed-42 frozen data has 565 train and 141 validation pairs. Its task-group split, labels,
common-prefix reconstruction, public API contracts and **500 model tokens per feedback turn** are preserved.
AppWorld labels compare the existing stored partial official SessionScore, **not binary TGC**.
Validation labels never enter discovery, selection, audit failure packets or repair acceptance.
The learning path pools all benchmarks; sequential/interleaved online updates are not part of this paper's algorithm.

## Optional audits and robustness repair

`--audit` reserves entire training task groups before discovery, then divides reserved tasks into
disjoint calibration/development pools. It runs structure checks, per-criterion ordinal ratings,
five stochastic re-runs (flip rate), five paraphrases (Spearman stability), and synthetic attacker →
judge → verifier probes with score threshold 7. All six attack strategies from Appendix F.6 are represented
when at least six pairs are available. Synthetic records execute no environments and are labeled as synthetic.
The verifier receives the reference and candidate but no attack strategy, judge score or judge rationale.
Failures abort the audit; failed requests are never counted as agreement or as failed attacks.

One available model supports a limited single-model audit. Independent-model reliability alpha remains
unavailable in this case. Configure at least three distinct model/endpoint identities to measure ordinal
Krippendorff alpha and to enable `--repair`; repeated stochastic calls to one model do not substitute for an ensemble.

`--repair` implements singleton operator calibration, proposed operator combinations up to configurable
depth, a predicted-VFR beam, measured dev acceptance, and a reliability/VFR Pareto frontier (Appendix C.1).
Predictions only prune candidates; acceptance requires measured VFR improvement, sufficient alpha and
preference fit on dev. Undefined alpha or preference fit prevents acceptance. Repair uses the paper's seven
operator types, 60-word definition / two 40-word clauses / 150-word total budget. The final dev winner is
the lowest-VFR frontier member, with alpha as the tie breaker. Audits additionally need manual examination
of synthetic exploit validity before interpreting them as real deployment robustness.
Budget-invalid proposals are rejected without terminating the search; each valid revision also passes
a task-independence genericity check before metric measurement.

## Explicit adaptations and unresolved paper details

- The body and Algorithm 1 say **5–15 criteria per pair**, while the actual Appendix F.1 prompt says **3–5**.
  Default 3–5 follows the deployed prompt; `--criteria-min 5 --criteria-max 15` follows the body.
- K=20, ten levels, subgroup maximum five, five stochastic runs/paraphrases, and attack threshold seven
  come from the paper. The HTML lists parameter symbols but omits numerical defaults for several values.
  T=25, K_pool=100, min_support=10, lambda=0.7, eta=0.6, and min_applicable=10 are explicit implementation
  settings, not claimed exact original hyperparameters. Minimum support 10 also appears in its Figure 3.
- Embedding uses the user-selected cached **BAAI/bge-m3** dense vectors (1,024 dimensions), CLS pooling
  from its cached sentence-transformers configuration, and L2 normalization on CPU. Criterion inputs
  retain the 256-token limit, with batches of eight. The original paper's exact encoder is unspecified.
  Earlier MiniLM runs remain preserved separately. A hub cache root without `refs/main` is supported
  when it contains exactly one usable snapshot; multiple snapshots require an explicit selection.
- Current LLM defaults to local Qwen3.6-35B-A3B instead of the paper's DeepSeek/external model suite.
  All original response-oriented prompts are paraphrased and adapted to complete recorded trajectories.
  JSON output replaces the paper's mixed JSON, XML and free-text transport formats.
- Scores for up to five criteria share one request, with each criterion evaluated independently.
  The genericity filter is an explicit LLM check because no exact filter implementation is published.
- Model-facing batch criterion IDs use short local aliases (`r1`, `r2`, etc.), mapped exactly back to
  immutable lineage IDs. Unknown and duplicated IDs are rejected. Omitted IDs are preserved unchanged
  in the unmerged set, with per-group provenance in `partition_recovery/`; no merge is inferred.
  Singleton groups are carried as unmerged criteria; oversized equivalent groups are split into groups
  of at most five. This preserves all IDs while enforcing the paper's merge information-loss budget.
- The paper's full 500/1000/3000-pair audit protocol cannot be reproduced with this 706-pair dataset.
  Our optional audit sample counts are configurable and written into the manifest. The BT nested-CV
  diagnostic and foreign-dataset specificity audit are not part of this training implementation.
- Structural word/count limits are enforced; semantic atomicity, nested-condition avoidance and
  prompt independence are instructed and audited rather than guaranteed by a lexical parser.
- The robustness search uses the generated original-criterion attacks as a fixed candidate comparison set.
  Fresh attacks against final revised criteria can be evaluated in a separate audit run.
- Repair failure packets are clipped to 16,384 Qwen tokens with an explicit omission flag, to fit multiple
  long trajectories in the repair context. The complete calibration evidence remains in the audit artifact.

## Running

Embeddings require a separate interpreter with `torch`, `transformers`, and `numpy`; no core or benchmark
dependency changes are made. The model must already be cached, or supply a local snapshot directory.

```bash
.venv/bin/python -u scripts/premise/run_rubric_judge.py \
  --dataset outputs/premise_reward_agent_seed42_nodel_past10_beams_20261005_004426/dataset \
  --output outputs/premise_rubric_judge_seed42_RUN \
  --embedding-python /home/thskadud/.vllm/bin/python \
  --embedding-model /home/thskadud/.cache/huggingface/hub/models--BAAI--bge-m3 \
  --workers 50
```

Use a separate output directory for each configuration. `--prepare-only` validates and snapshots inputs
without LLM calls. A small end-to-end smoke test can use `--train-limit-per-benchmark 2`
`--val-limit-per-benchmark 1 --min-support 1 --min-applicable 1 --target-size 3`.
Those overrides are for mechanics only and cannot establish training quality.

Resume with the same arguments plus `--resume`; `--retry-errors` explicitly retries cached failed jobs.
Successful requests are reused. Changed code, prompts, model settings, data hashes or audit configuration
reject resumption into an existing run. An exclusive controller lock prevents concurrent writers.
When parsed JSON fails a validation constraint, the original request
and invalid response are sent with explicit validation feedback for at most two correction calls.
The same validator must accept the result; limits are never relaxed or silently truncated. Invalid
responses and correction requests remain recorded. Transport failures retain their existing bounded
recovery policy, and exhausted validation recovery stops the stage for explicit resumption.
Explicit `--retry-errors` reuses a recorded invalid response to start its correction directly.
Every stage has cached requests, source snapshots, trace files and a progress status. The global request
semaphore covers discovery, scoring, validation and optional audit models **within this controller**.
Each call uses max output 2048, timeout 600 seconds, one transient recovery retry, and five extra identical
calls for an output truncation. Exhaustion fails the stage; it cannot silently produce a trained rubric.
Validation errors are visible and official validation accuracy stays null until all requests succeed.

For optional multi-model repair, `--audit-judges FILE` takes a JSON list of three distinct
`{"base_url": "http://host:port/v1", "model": "model-name"}` objects, without keys.
Add `--audit --repair`; the default reserves two whole train tasks per benchmark.
API keys are read from the environment and never written into launch manifests.

Outputs include `discovery/level_*.json`, `supported_pool.json`, `preference_fit.json`, `rubrics/*.json`,
`validation.json`, optional `audit.json` / `repair_decisions.json`, and `status.json`.
