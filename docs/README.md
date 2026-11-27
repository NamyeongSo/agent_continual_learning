# Exgentic Documentation

Welcome to the Exgentic docs. Use the table below to find what you need.

---

## Using Exgentic

| Document | Description |
|----------|-------------|
| [CLI Reference](./cli-reference.md) | Every command, flag, and environment variable |
| [Python API](./python-api.md) | `evaluate()`, `execute()`, `aggregate()`, `status()`, `list_*()`, and all other library functions |
| [Custom Models](./custom-models.md) | Use any LLM provider (OpenAI, Anthropic, Azure, Bedrock, Ollama, and more) via LiteLLM |
| [Batch Runs](./batch.md) | Run parameter sweeps, manage large evaluations, export to CSV, publish to HuggingFace |
| [Runners](./runners.md) | `direct`, `venv`, `docker` — isolation levels, configuration, Docker-in-Docker |
| [Output Format](./output-format.md) | Schema for `results.json`, `trajectory.jsonl`, session results, and cost reports |
| [Observers](./observers.md) | Hook into the evaluation lifecycle for custom logging, monitoring, and early stopping |
| [Terminal-Bench 2.0 / 2.1](./terminalbench2.md) | Run the corrected 2.1 release, or train ACE/PREMiSE on frozen task splits |
| [AppWorld Code Generation](./appworld-code-generation.md) | Interactive Python execution, persistent shell behavior, and the legacy function-calling mode |
| [Checkpoint Policy Shift](./checkpoint_policy_shift.md) | Wait for checkpoint evaluation, then compare fixed-history actor policies across four methods and three benchmarks |
| [PEACE Span Policy Shift](./peace_span_policy_shift.md) | Evaluate candidates from saved training actions with strict F1, outcome-directed preservation/change, and current/past Pareto selection |
| [PEACE Group-relative Optimization](./peace_group_relative.md) | Collect five fixed rollouts per task, propose five prompts from best/worst evidence, and select by advantage-weighted saved-prefix agreement |
| [PEACE Proposal Retrieval](./peace_proposal_retrieval.md) | Compare A/B/AB/random historical evidence using action strict F1 and MiniLM reflections, with real BFCL replay |
| [PEACE Grounded Contrast Proposal](./peace_grounded_contrast_proposal.md) | Review reflection evidence, preserve strict A/B priorities, and generate conditional rules; queue matched BFCL runs |
| [SWE-Milestone](./swemilestone.md) | Persistent milestone sessions, Terminus-2 overflow recovery, ACE updates, and replay on an SSH Docker server |
| [Task Splits](../scripts/utils/TASK_SPLITS.md) | Train/validation/test sizes, deterministic shuffle rules, and task IDs for seeds 42–44 |
| [Playbook Edits](./playbook_edit_modes.md) | Optional ADD, UPDATE and DELETE policies for ACE and PREMiSE |
| [Shared Actor Context](./shared_actor_context.md) | Match the initial actor context across ACE, PEACE, GEPA and SkillOpt with empty learned policies and versioned hashes |

## Extending Exgentic

| Document | Description |
|----------|-------------|
| [Adding Agents](./adding-agents.md) | Write a new agent adapter — design principles, required methods, file layout, and validation checklist |
| [Adding Benchmarks](./adding-benchmarks.md) | Write a new benchmark adapter — design principles, contract rules, and validation checklist |
| [Replay Testing](./replay-testing.md) | Test benchmark and agent adapters end-to-end without API calls, using recorded sessions |

## Observability

| Document | Description |
|----------|-------------|
| [Quick Start](./observability/quickstart.md) | Set up OpenTelemetry tracing with Jaeger in five minutes |
| [Semantic Conventions](./observability/semantic-conventions.md) | Full reference of every span and attribute Exgentic emits |

## Other resources

- [GEPA / SkillOpt Baselines](./gepa_skillopt_baselines.md) — pinned official sources, isolated environments, prompt provenance, and experiment design decisions
- [GEPA / SkillOpt Protocol](./gepa_skillopt_protocol.md) — accepted stream/validation settings, measured PREMiSE rollout budgets, and optimizer terminology
- [Research Journal](./research_journal.md) — local experiment records, editable notes, artifact snapshots, comparisons, and portable reports
- [README.md](../README.md) — project overview, quick start, CLI reference, and available benchmarks/agents
- [DEVELOPMENT.md](../DEVELOPMENT.md) — local setup, running tests, and linting
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contribution workflow, legal requirements, and PR guidelines
