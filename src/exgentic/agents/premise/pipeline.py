# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Two-parent span-replay Pareto optimization across ExGentic benchmarks."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ...actor_context import (
    ActorContextMode,
    actor_context_metadata,
    shared_actor_prompt,
    validate_actor_context,
)
from ...core.experiment_limits import ACTOR_OUTPUT_TOKENS, BENCHMARK_MAX_TURNS, experiment_limits
from ...core.types import RunConfig
from ...interfaces.lib.api import execute
from ...utils.context_errors import is_context_length_error
from ...utils.task_splits import select_task_split_sections, validate_task_split
from ..ace.playbook_store import DEFAULT_PLAYBOOK
from ..ace.playbook_utils import (
    PlaybookEditMode,
    allowed_playbook_operations,
    apply_playbook_edits,
    update_bullet_counts,
)
from ..litellm_tool_calling.instance import CompletionRetriesExhaustedError
from .llm import PremiseLLM, public_tool_contract
from .proposal_retrieval import MODES as PROPOSAL_RETRIEVAL_MODES
from .proposal_retrieval import MiniLMClient, ProposalRetriever
from .ablation_selection import ABLATION_SELECTION_RULES, select_ablation_beam
from .selection import (
    sample_past,
    select_alignment_product,
    select_beam,
    select_policy_shift_beam,
    select_ps_st_pareto_beam,
    select_validation_beam,
)
from .span_policy_shift import HISTORY_FILE, SCHEDULING_VERSION, SpanPolicyShiftEvaluator
from .span_policy_shift import VERSION as POLICY_SHIFT_VERSION


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


class TaskTimeLimitError(TimeoutError):
    """The benchmark deadline left no usable recorded action span."""


def is_tool_parser_configuration_error(exc: Exception) -> bool:
    """A missing server parser invalidates every tool request, not an individual task."""
    message = str(exc).lower()
    return "--tool-call-parser" in message and any(marker in message for marker in (
        "requires", "must be set", "--enable-auto-tool-choice",
    ))


def skip_task_after_llm_error(exc: Exception) -> bool:
    """Abandon this training update after request recovery fails or output is truncated."""
    if is_tool_parser_configuration_error(exc):
        return False
    message = str(exc).lower()
    return isinstance(exc, CompletionRetriesExhaustedError) or any(marker in message for marker in (
        "completionretriesexhaustederror", "litellm completion truncated (finish_reason=length)",
        # Native Terminus-2 surfaces provider errors after Harbor's retry loop.
        "apitimeouterror", "apiconnectionerror", "ratelimiterror", "litellm.timeout",
        "serviceunavailableerror", "internalservererror",
    ))


def is_nontransient_task_error(exc: Exception) -> bool:
    """Only skip failures attributable to this task, not service outages."""
    message = str(exc).lower()
    context_markers = (
        "maximum context length", "context length exceeded", "context_length_exceeded",
        "context window exceeded", "prompt is too long", "input tokens exceeds",
        "litellm completion truncated (finish_reason=length)",
        "litellm completion missing assistant content",
    )
    return isinstance(exc, TaskTimeLimitError) or is_context_length_error(exc) or any(
        marker in message for marker in context_markers
    ) or (
        isinstance(exc, RuntimeError) and message.startswith("reflection_") and " failed:" in message
    )


def is_replay_restore_error(exc: Exception) -> bool:
    """A fresh terminal could not execute the full recorded prefix; no score exists."""
    return "Terminus-2 ended before the replay prefix was restored" in str(exc)


def playbook_text(playbook: dict[str, Any], benchmark: str = "") -> str:
    """Every benchmark receives the entire ACE-format playbook."""
    return playbook.get("playbook", DEFAULT_PLAYBOOK)


def apply_operations(parent: dict[str, Any], operations: list[dict[str, Any]],
                     benchmark: str = "", *,
                     edit_mode: PlaybookEditMode = "add_only") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    child = copy.deepcopy(parent)
    child["playbook"], child["next_global_id"], applied = apply_playbook_edits(
        playbook_text(parent), operations, parent.get("next_global_id", 1), edit_mode=edit_mode,
    )
    return child, applied


BENCHMARK_DEFAULTS = {
    "bfcl": {"subset": "multi_turn_base"},
    "appworld": {"subset": "test_challenge", "interaction_mode": "code_generation"},
    "terminalbench2": {
        "subset": "2.0", "actor": "terminus2", "max_interactions": BENCHMARK_MAX_TURNS["terminalbench2"],
    },
}


def manifest_order(
    path: Path, benchmarks: list[str], seed: int, count: int | None, mode: str
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sections = select_task_split_sections(manifest, benchmarks, seed)
    pools: dict[str, deque[str]] = {}
    configs: dict[str, Any] = {}
    for slug in benchmarks:
        section = sections[slug]
        train = validate_task_split(section, slug)["train"]
        if count is not None and not 0 < count <= len(train):
            raise ValueError(f"Train manifest for {slug} must have at least {count} tasks")
        order_seed = int(hashlib.md5(f"{seed}_{slug}".encode()).hexdigest(), 16) % (2**31)
        if manifest.get("seed_scope") != "within_split_order_only":
            random.Random(order_seed).shuffle(train)
        pools[slug] = deque(train[:count])
        configs[slug] = {**BENCHMARK_DEFAULTS.get(slug, {}), **(section.get("benchmark_kwargs") or {})}
        if slug == "terminalbench2":
            configs[slug]["actor"] = "terminus2"
    if mode in ("isolated", "sequential"):
        slugs = sorted(benchmarks) if mode == "isolated" else [
            s for s in ("bfcl", "appworld", "swebench", "terminalbench2") if s in benchmarks
        ]
        slugs += [s for s in benchmarks if s not in slugs]
        return [(slug, task) for slug in slugs for task in pools[slug]], configs
    if mode != "interleaved":
        raise ValueError(f"Unknown mode {mode}")
    rng = random.Random(seed)
    order = []
    while any(pools.values()):
        slug = rng.choice(sorted(s for s, tasks in pools.items() if tasks))
        order.append((slug, pools[slug].popleft()))
    return order, configs


def trajectory_from_session(root: Path, run_id: str, session_result: Any, benchmark: str) -> dict[str, Any]:
    session_dir = root / run_id / "sessions" / session_result.session_id
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line) for line in (session_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observations = {
        event["step"]: event.get("observation") for event in events
        if event.get("event") == "observation" and not event.get("initial")
    }
    actions = [{
        "action": event["action"],
        "action_class": event.get("action_class", "SingleAction"),
        "observation": observations.get(event["step"]),
    } for event in events if event.get("event") == "action"]
    score = session_result.score
    if score is None:
        raise ValueError(f"{benchmark} session returned no official score")
    partial_score = score
    if benchmark == "appworld":
        if type(getattr(session_result, "success", None)) is not bool:
            raise ValueError("AppWorld TGC requires the official success flag")
        score = float(session_result.success)
    if not 0 <= float(score) <= 1:
        raise ValueError(f"{benchmark} score {score} is outside [0,1]; add an explicit normalization adapter")
    prefix_length = 0
    terminus_episode_path = None
    timed_out = False
    if benchmark == "terminalbench2":
        def find_trace_path(value):
            if isinstance(value, dict):
                if "trace_path" in value:
                    return value["trace_path"]
                for item in value.values():
                    found = find_trace_path(item)
                    if found:
                        return found
            elif isinstance(value, list):
                for item in value:
                    found = find_trace_path(item)
                    if found:
                        return found
            elif isinstance(value, str) and value.lstrip().startswith("{"):
                try:
                    return find_trace_path(json.loads(value))
                except json.JSONDecodeError:
                    pass
            return None

        metadata = (session_result.details or {}).get("session_metadata") or {}
        if metadata.get("agent_error"):
            raise RuntimeError(f"Terminus-2 failed: {metadata['agent_error']}")
        timed_out = bool(metadata.get("agent_timed_out")) or any(
            isinstance(result := event.get("observation", {}).get("result"), dict) and result.get("agent_timed_out")
            for event in events
        )
        trace_path = next((found for event in events if (found := find_trace_path(event))), None)
        trace_path = trace_path or find_trace_path(session_result.details)
        if trace_path is None:
            # Older timeout sessions saved Harbor's files but lost the return
            # value when ActionsHandler converted TimeoutError to an observation.
            saved_trace = session_dir / "benchmark/harbor/agent/trajectory.json"
            error_log = session_dir / "benchmark/session.log"
            if saved_trace.is_file() and error_log.is_file() and "TimeoutError" in error_log.read_text():
                trace_path = str(saved_trace)
                timed_out = True
        if trace_path is None:
            if timed_out:
                raise TaskTimeLimitError("Terminus-2 time limit reached before recording a trajectory")
            raise ValueError("Missing Terminus-2 internal trajectory path")
        if timed_out and not Path(trace_path).is_file():
            raise TaskTimeLimitError("Terminus-2 time limit reached before recording a trajectory")
        atif = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        terminus_episode_path = str(Path(trace_path).with_name("premise_episodes.json"))
        if timed_out and not Path(terminus_episode_path).is_file():
            raise TaskTimeLimitError("Terminus-2 time limit reached before recording replay episodes")
        recorded = json.loads(Path(terminus_episode_path).read_text(encoding="utf-8"))
        steps = [step for step in atif["steps"] if step["source"] == "agent"]
        episodes = recorded["episodes"]
        if timed_out and len(episodes) == len(steps) + 1:
            # The last response was recorded before its command batch finished.
            # Keep that response in a separate field and replay completed spans.
            completed_path = Path(trace_path).with_name("premise_completed_episodes.json")
            write_json(completed_path, {
                **recorded, "episodes": episodes[:len(steps)],
                "interrupted_episodes": episodes[len(steps):],
                "source_recording": terminus_episode_path,
                "agent_timed_out": True,
            })
            episodes = episodes[:len(steps)]
            terminus_episode_path = str(completed_path)
        if len(steps) != len(episodes):
            if timed_out:
                raise TaskTimeLimitError("Terminus-2 time limit left inconsistent episode records")
            raise ValueError("Terminus-2 trajectory and episode records do not match")
        prefix_length = int(recorded["prefix_length"])
        if timed_out and len(steps) <= prefix_length:
            raise TaskTimeLimitError("Terminus-2 time limit left no completed continuation episodes")
        actions = [{"action": {"name": "terminus2_episode", "arguments": episode},
                    "action_class": "Terminus2Episode", "message": step.get("message"),
                    "observation": step.get("observation"), "tool_calls": step.get("tool_calls"),
                    "episode": index}
                   for index, (step, episode) in enumerate(zip(steps, episodes, strict=True)) if index >= prefix_length]
    return {
        "benchmark": benchmark, "task_id": str(manifest["task_id"]),
        "task": manifest["task"], "context": manifest.get("context", {}),
        "actions": actions, "score": float(score), "details": session_result.details,
        **({"partial_score": partial_score, "score_metric": "appworld_tgc_v1"} if benchmark == "appworld" else {}),
        "success": getattr(session_result, "success", None),
        "public_tool_contract": public_tool_contract(manifest.get("actions", []), actions),
        "session_dir": str(session_dir), "prefix_length": prefix_length,
        "terminus_episode_path": terminus_episode_path,
        **({"agent_timed_out": timed_out} if benchmark == "terminalbench2" else {}),
        "score_source": "official_benchmark_session_score",
    }


class PremisePipeline:
    def __init__(self, *, output: Path, manifest: Path, benchmarks: list[str], mode: str,
                 seed: int, count: int | None, past_count: int, model: str, base_url: str,
                 parent_workers: int = 2, reflection_workers: int = 4,
                 mutation_workers: int = 4, replay_workers: int = 28,
                 replay_retries: int = 2,
                 llm_timeout: float = 600.0, llm_retries: int = 1,
                 reasoning_effort: str | None = None,
                 tool_calling_max_tokens: int = ACTOR_OUTPUT_TOKENS,
                 terminalbench2_version: str | None = None,
                 appworld_policy_override: str | None = None,
                 execution_tag: str = "",
                 llm_calls: int = 32, benchmark_limits: dict[str, int | None] | None = None,
                 skip_task_errors: str = "nontransient",
                 max_steps: dict[str, int] | None = None, method: str = "premise",
                 candidate_validation: str = "current_past", validation_scope: str = "current",
                 candidate_evaluation: str = "environment_replay", policy_shift_temperature: float = 0.0,
                 proposal_retrieval: str = "none", proposal_embedding_url: str = "http://127.0.0.1:18886",
                 grounded_proposals: bool = False, grounded_retrieval: str = "ab",
                 alignment_scoring: bool = True,
                 candidate_selection: str = "pareto",
                 appworld_candidate_metric: str = "tgc",
                 benchmark_candidate_metric: str = "official",
                 actor_temperature: float | None = None, reflection_temperature: float | None = None,
                 mutation_temperature: float | None = None,
                 temperature: float | None = None, playbook_edit_mode: PlaybookEditMode = "add_only",
                 actor_context: ActorContextMode = "legacy"):
        validate_actor_context(actor_context)
        self.actor_context = actor_context
        if past_count not in (3, 6, 9):
            raise ValueError("past_count must be 3, 6 or 9")
        if method not in ("premise", "ps_st_pareto"):
            raise ValueError("method must be premise or ps_st_pareto")
        if candidate_validation not in ("current_past", "random_val4"):
            raise ValueError("candidate_validation must be current_past or random_val4")
        if validation_scope not in ("current", "seen"):
            raise ValueError("validation_scope must be current or seen")
        if candidate_validation == "random_val4" and mode != "sequential":
            raise ValueError("random_val4 candidate validation requires sequential mode")
        if candidate_evaluation not in ("environment_replay", "span_policy_shift"):
            raise ValueError("candidate_evaluation must be environment_replay or span_policy_shift")
        if candidate_evaluation == "span_policy_shift" and (
            method != "premise" or candidate_validation != "current_past"
        ):
            raise ValueError("span_policy_shift requires method=premise and candidate_validation=current_past")
        if candidate_evaluation == "span_policy_shift" and any(slug not in BENCHMARK_MAX_TURNS for slug in benchmarks):
            raise ValueError("span_policy_shift supports BFCL, AppWorld and native Terminal-Bench")
        if not 0 <= policy_shift_temperature <= 2:
            raise ValueError("policy_shift_temperature must be between 0 and 2")
        self.candidate_evaluation = candidate_evaluation
        self.policy_shift_temperature = policy_shift_temperature
        if proposal_retrieval not in PROPOSAL_RETRIEVAL_MODES:
            raise ValueError("Invalid proposal_retrieval mode")
        if proposal_retrieval != "none" and (
            benchmarks != ["bfcl"] or method != "premise" or candidate_validation != "current_past"
        ):
            raise ValueError("Proposal retrieval requires BFCL PEACE with current/past candidate selection")
        self.proposal_retrieval = proposal_retrieval
        if grounded_proposals and (
            benchmarks != ["bfcl"] or method != "premise" or candidate_validation != "current_past"
            or alignment_scoring or proposal_retrieval != "none" or temperature is not None
        ):
            raise ValueError("Grounded proposals require BFCL PEACE current/past without alignment or legacy retrieval")
        self.candidate_validation = candidate_validation
        self.validation_scope = validation_scope
        self.appworld_policy_override = appworld_policy_override
        self.method = method
        self.alignment_scoring = (
            bool(alignment_scoring) and method == "premise" and candidate_evaluation == "environment_replay"
        )
        if candidate_selection not in ("pareto", "alignment_product", *ABLATION_SELECTION_RULES):
            raise ValueError("Unknown candidate_selection rule")
        if candidate_selection in ABLATION_SELECTION_RULES and (
            method != "premise" or candidate_validation != "current_past"
            or candidate_evaluation != "environment_replay" or grounded_proposals or proposal_retrieval != "none"
            or (candidate_selection != "random" and not self.alignment_scoring)
        ):
            raise ValueError("Selection ablations require PEACE current/past environment replay")
        if candidate_selection == "alignment_product" and (
            not self.alignment_scoring or candidate_validation != "current_past"
            or grounded_proposals or proposal_retrieval != "none"
        ):
            raise ValueError("alignment_product requires PEACE current/past alignment environment replay")
        self.candidate_selection = candidate_selection
        if appworld_candidate_metric not in ("tgc", "pct"):
            raise ValueError("appworld_candidate_metric must be tgc or pct")
        if appworld_candidate_metric == "pct" and (
            method != "premise" or candidate_evaluation != "environment_replay"
            or candidate_validation != "current_past"
        ):
            raise ValueError("PCT candidate scoring requires PEACE current/past environment replay")
        self.appworld_candidate_metric = appworld_candidate_metric
        if benchmark_candidate_metric not in {"official", "auxiliary"}:
            raise ValueError("benchmark_candidate_metric must be official or auxiliary")
        if benchmark_candidate_metric == "auxiliary" and (
            method != "premise" or candidate_evaluation != "environment_replay"
            or candidate_validation != "current_past" or appworld_candidate_metric != "tgc"
            or any(slug not in {"bfcl", "terminalbench2"} for slug in benchmarks)
        ):
            raise ValueError("Auxiliary rewards require BFCL/Terminal-Bench PEACE current/past environment replay")
        self.benchmark_candidate_metric = benchmark_candidate_metric
        self.beam_size = 1 if candidate_selection == "alignment_product" else 2
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        # AppWorld writes outside this output tree, so concurrent experiments
        # must not use the same experiment_name for a matching replay label.
        namespace_source = str(self.output) + (f"\n{execution_tag}" if execution_tag else "")
        self.run_namespace = hashlib.sha256(namespace_source.encode()).hexdigest()[:10]
        self.manifest = manifest.resolve()
        self.benchmarks = benchmarks
        self.mode, self.seed, self.count = mode, seed, count
        self.past_count = past_count
        self.model, self.base_url = model, base_url
        self.reasoning_effort = reasoning_effort
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self.temperature = temperature
        roles = {"actor": actor_temperature, "reflection": reflection_temperature, "mutation": mutation_temperature}
        if any(value is not None and not 0 <= value <= 2 for value in roles.values()):
            raise ValueError("Role temperatures must be between 0 and 2")
        if temperature is not None and any(value is not None for value in roles.values()):
            raise ValueError("Use either all-role temperature or separate role temperatures")
        self.actor_temperature = actor_temperature
        self.role_temperatures = {key: value for key, value in roles.items() if value is not None}
        allowed_playbook_operations(playbook_edit_mode)
        self.playbook_edit_mode = playbook_edit_mode
        self.parent_workers = parent_workers
        self.reflection_workers = reflection_workers
        self.mutation_workers = mutation_workers
        self.replay_workers = replay_workers
        if replay_retries < 0:
            raise ValueError("replay_retries must be nonnegative")
        if skip_task_errors not in ("none", "nontransient", "all"):
            raise ValueError("skip_task_errors must be none, nontransient, or all")
        self.replay_retries = replay_retries
        self.skip_task_errors = skip_task_errors
        self.llm = PremiseLLM(
            model, base_url, max_calls=llm_calls, timeout=llm_timeout,
            num_retries=llm_retries, trace_dir=self.output / "llm_calls",
            reasoning_effort=reasoning_effort,
            temperature_override=temperature,
            reflection_temperature=reflection_temperature, mutation_temperature=mutation_temperature,
            playbook_edit_mode=playbook_edit_mode,
        )
        self.llm_timeout = llm_timeout
        self.llm_retries = llm_retries
        if tool_calling_max_tokens < 1:
            raise ValueError("tool_calling_max_tokens must be positive")
        self.tool_calling_max_tokens = min(tool_calling_max_tokens, ACTOR_OUTPUT_TOKENS)
        self.policy_shift_evaluator = SpanPolicyShiftEvaluator(
            model=model, base_url=base_url, output=self.output, temperature=policy_shift_temperature,
            max_tokens=self.tool_calling_max_tokens, timeout=llm_timeout, num_retries=llm_retries,
        ) if candidate_evaluation == "span_policy_shift" else None
        self.rng = random.Random(seed)
        self.proposal_retriever = ProposalRetriever(
            self.output, proposal_retrieval, MiniLMClient(proposal_embedding_url),
            getattr(self, "algorithm_seed", seed),
        ) if proposal_retrieval != "none" else None
        self.grounding = None
        if grounded_proposals:
            from .grounded_proposals import CompleteMiniLMClient, GroundedProposals

            self.grounding = GroundedProposals(
                self.output, self.llm, CompleteMiniLMClient(proposal_embedding_url),
                getattr(self, "algorithm_seed", seed), mode=grounded_retrieval,
                temperature=reflection_temperature if reflection_temperature is not None else 0.7,
                mutation_temperature=mutation_temperature if mutation_temperature is not None else 0.7)
        self.order, self.benchmark_configs = manifest_order(self.manifest, benchmarks, seed, count, mode)
        if terminalbench2_version is not None:
            if terminalbench2_version not in ("2.0", "2.1"):
                raise ValueError(f"Unsupported Terminal-Bench release: {terminalbench2_version}")
            if "terminalbench2" in self.benchmark_configs:
                self.benchmark_configs["terminalbench2"]["subset"] = terminalbench2_version
        if benchmark_candidate_metric == "auxiliary" and "bfcl" in self.benchmark_configs:
            self.benchmark_configs["bfcl"]["auxiliary_turn_metrics"] = True
        if appworld_policy_override is not None and "appworld" in self.benchmark_configs:
            self.benchmark_configs["appworld"]["policy_override"] = appworld_policy_override
        self.validation_pools: dict[str, list[str]] = {}
        if candidate_validation == "random_val4":
            sections = select_task_split_sections(json.loads(self.manifest.read_text()), benchmarks, seed)
            for slug in benchmarks:
                split = validate_task_split(sections[slug], slug)
                if len(split["val"]) < 4:
                    raise ValueError(f"Need at least four held-out validation tasks for {slug}")
                self.validation_pools[slug] = split["val"]
        limits = {slug: None for slug in benchmarks} if benchmark_limits is None else benchmark_limits
        self.benchmark_slots = {
            slug: nullcontext() if limits.get(slug) is None else threading.BoundedSemaphore(limits.get(slug, 4))
            for slug in benchmarks
        }
        self._replay_lock = threading.Lock()
        self._active_replays = 0
        self._peak_replays = 0
        self.max_steps = {"swebench": 100, **BENCHMARK_MAX_TURNS, **(max_steps or {})}
        for slug, cap in BENCHMARK_MAX_TURNS.items():
            self.max_steps[slug] = min(self.max_steps[slug], cap)
        self.states = {slug: self._empty_state(self.beam_size)
                       for slug in (benchmarks if mode == "isolated" else ["global"])}
        self.cursor = 0
        self._restore_or_initialize()
        if self.cursor in (50, 100, 130, len(self.order)) and not (
            self.output / "checkpoints" / f"checkpoint_{self.cursor:03d}.json"
        ).exists():
            self._save()

    @staticmethod
    def _empty_state(beam_size: int = 2) -> dict[str, Any]:
        initial = {"playbook": DEFAULT_PLAYBOOK, "next_global_id": 1}
        return {"beam": [copy.deepcopy(initial) for _ in range(beam_size)], "memory": []}

    def _restore_or_initialize(self):
        state_path = self.output / "state.json"
        config = {
            "schema_version": 2, "manifest": str(self.manifest),
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "benchmarks": self.benchmarks, "mode": self.mode, "seed": self.seed,
            "count": self.count, "past_count": self.past_count, "model": self.model,
            "order": self.order, "benchmark_configs": self.benchmark_configs,
            "max_steps": self.max_steps, "replay_retries": self.replay_retries,
            "experiment_limits": {slug: experiment_limits(slug) for slug in self.benchmarks},
        }
        # Budget policy is part of provenance; do not silently mix old results.
        if self.actor_context != "legacy":
            config["actor_context"] = self.actor_context
            config["actor_context_metadata"] = actor_context_metadata(self.actor_context, self.benchmarks)
        if self.temperature is not None:
            config["temperature_all_roles"] = self.temperature
        if self.role_temperatures:
            config["role_temperatures"] = self.role_temperatures
        if self.appworld_policy_override is not None:
            config["appworld_policy_sha256"] = hashlib.sha256(self.appworld_policy_override.encode()).hexdigest()
        if self.playbook_edit_mode != "add_only":
            config["playbook_edit_mode"] = self.playbook_edit_mode
        if self.method != "premise":
            config["method"] = self.method
        if self.method == "premise" and not self.alignment_scoring:
            config["alignment_scoring"] = False
        if self.appworld_candidate_metric != "tgc":
            config["appworld_candidate_metric"] = self.appworld_candidate_metric
            config["appworld_reporting_metric"] = "appworld_tgc_v1"
        if self.benchmark_candidate_metric != "official":
            config["benchmark_candidate_metric"] = self.benchmark_candidate_metric
            config["auxiliary_metric_versions"] = {
                "bfcl": "bfcl_turn_pass_rate_v1", "terminalbench2": "terminalbench_test_pass_rate_v1"}
        if self.candidate_selection != "pareto":
            config["candidate_selection"] = self.candidate_selection
            config["search_topology"] = {"parents": self.beam_size,
                                         "children_per_parent": 4 if self.beam_size == 1 else 2,
                                         "survivors": self.beam_size,
                                         "parent_in_selection_pool": False}
            if self.candidate_selection == "alignment_product":
                config["product_aggregation"] = "mean_over_current_and_each_past_replay_then_product"
            else:
                config["selection_rule_version"] = "submission-ablation-v1"
                config["candidate_scoring"] = self.candidate_selection != "random"
        if self.proposal_retriever is not None:
            config["proposal_retrieval"] = self.proposal_retriever.config
        if self.grounding is not None:
            config["grounded_proposals"] = self.grounding.config
        if self.candidate_evaluation == "span_policy_shift":
            config["candidate_evaluation"] = self.candidate_evaluation
            config["policy_shift"] = {
                "version": POLICY_SHIFT_VERSION, "metric": "strict_f1", "reference": "recorded_training_action",
                "temperature": self.policy_shift_temperature, "environment_execution": False,
                "outcome_source": "official_trajectory_success", "aggregation": "mean_over_span_actions",
                "scheduling": SCHEDULING_VERSION,
            }
        if self.candidate_validation != "current_past":
            config["candidate_validation"] = self.candidate_validation
            config["validation_scope"] = self.validation_scope
            config["validation_sample_size"] = 4
        if self.reasoning_effort is not None:
            config["reasoning_effort"] = self.reasoning_effort
        if state_path.exists():
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if saved["config"] != json.loads(json.dumps(config)):
                raise ValueError("Existing checkpoint configuration differs from this run")
            self.cursor = int(saved["cursor"])
            self.states = saved["states"]
            self.rng.setstate(ast.literal_eval(saved["rng_state"]))
        else:
            write_json(state_path, {"config": config, "cursor": 0, "states": self.states, "rng_state": repr(self.rng.getstate())})

    def _save(self):
        saved = json.loads((self.output / "state.json").read_text(encoding="utf-8"))
        saved.update({"cursor": self.cursor, "states": self.states, "rng_state": repr(self.rng.getstate())})
        write_json(self.output / "state.json", saved)
        if self.cursor in (50, 100, 130, len(self.order)):
            checkpoint = self.output / "checkpoints" / f"checkpoint_{self.cursor:03d}.json"
            if not checkpoint.exists():
                write_json(checkpoint, saved)

    def _run_agent(self, *, benchmark: str, task_id: str, playbook: dict[str, Any],
                   label: str, replay_source: dict[str, Any] | None = None,
                   span_start: int | None = None, temperature: float = 0.0) -> dict[str, Any]:
        if getattr(self, "temperature", None) is not None:
            temperature = self.temperature
        if getattr(self, "actor_temperature", None) is not None:
            temperature = self.actor_temperature
        capture_history = (getattr(self, "policy_shift_evaluator", None) is not None
                           or (getattr(self, "grounding", None) is not None and replay_source is None))
        run_id = f"premise_{self.run_namespace}_{label}_{benchmark}".replace("/", "_")
        kwargs = dict(self.benchmark_configs[benchmark])
        kwargs["use_cache"] = False
        # AppWorld's experiment directory is per job, including concurrent
        # executions of the same task under different candidates.
        if benchmark == "appworld":
            env = dict(kwargs.get("env_kwargs") or {})
            env["experiment_name"] = run_id
            kwargs["env_kwargs"] = env
        replay_spec = None
        terminus_prefix_path = None
        terminus_prefix_count = 0
        prefix_action_count = 0
        if replay_source is not None:
            assert span_start is not None
            prefix = replay_source["actions"][:span_start]
            if benchmark != "terminalbench2":
                replay_spec = {
                    "task_id": str(task_id),
                    "actions": [
                        {"action": item["action"], "action_class": item["action_class"]}
                        for item in prefix
                    ],
                }
                prefix_action_count = sum(
                    len(item["action"].get("actions", [])) if "actions" in item["action"] else 1
                    for item in prefix
                )
        if benchmark == "terminalbench2":
            kwargs.update({"actor": "terminus2", "max_interactions": self.max_steps.get(benchmark, 100)})
            if replay_source is not None:
                prefix_offset = int(replay_source.get("prefix_length", 0))
                if prefix_offset:
                    raise ValueError("Memory sources must be original parent trajectories, not continuations")
                terminus_prefix_path = replay_source["terminus_episode_path"]
                terminus_prefix_count = span_start
            replay_spec = None
            prefix_action_count = 0
        step_limit = self.max_steps.get(benchmark, 100)
        continuation_limit = step_limit - (span_start or 0) if benchmark != "terminalbench2" else step_limit
        continuation_actions = max(step_limit, 100) - prefix_action_count
        if continuation_limit < 1 or continuation_actions < 1:
            raise ValueError("Replay prefix consumes the entire action or step budget")
        reasoning_effort = getattr(self, "reasoning_effort", None)
        config = RunConfig(
            benchmark=benchmark, agent="premise", task_ids=[str(task_id)],
            output_dir=str(self.output / "sessions"), run_id=run_id,
            model=self.model, benchmark_kwargs=kwargs,
            agent_kwargs={
                "model": self.model,
                "actor_context": getattr(self, "actor_context", "legacy"),
                "actor_benchmark": benchmark,
                "temperature_override": (
                    self.actor_temperature if getattr(self, "actor_temperature", None) is not None
                    else getattr(self, "temperature", None)),
                "terminus_replay_path": terminus_prefix_path,
                "terminus_replay_count": terminus_prefix_count,
                "playbook": playbook_text(playbook, benchmark),
                **({"record_policy_shift": True} if capture_history else {}),
                "context_policy_override": (
                    self.appworld_policy_override if benchmark == "appworld" else None
                ),
                "runner": "direct", "enable_thinking": False,
                "model_settings": {
                    "temperature": temperature,
                    "max_tokens": self.tool_calling_max_tokens,
                    "timeout": self.llm_timeout, "num_retries": self.llm_retries,
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
                },
                "enable_tool_shortlisting": (
                    benchmark == "appworld" and getattr(self, "actor_context", "legacy") == "legacy"
                ),
                "max_selected_tools": 30,
            },
            replay_prefix=replay_spec, max_steps=continuation_limit,
            max_actions=continuation_actions,
            overwrite_sessions=True,
        )
        actor_settings = getattr(self, "actor_model_settings", None)
        if actor_settings is not None:
            config.agent_kwargs["model_settings"] = config.agent_kwargs["model_settings"].model_copy(
                update=copy.deepcopy(actor_settings),
            )
            config.agent_kwargs["temperature_override"] = actor_settings["temperature"]
            config.agent_kwargs["enable_thinking"] = bool(
                (actor_settings.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking")
            )
            config.agent_kwargs["allow_shared_thinking"] = True
            config.agent_kwargs["native_model_settings"] = copy.deepcopy(actor_settings)
        result = execute(config=config)
        if len(result.session_results) != 1:
            raise RuntimeError(f"{label}: expected one completed session, got {len(result.session_results)}")
        session_result = result.session_results[0]
        if session_result.status.value in ("error", "cancelled"):
            metadata = (session_result.details or {}).get("session_metadata") or {}
            if ("gpt-oss" in self.model and session_result.status.value == "error"
                    and session_result.score == 0
                    and "LiteLLM completion missing assistant content" in str(metadata.get("error", ""))):
                # Malformed Harmony tool headers are model failures with a zero
                # task score. Keep their partial trajectory for PREMiSE feedback.
                trace = trajectory_from_session(self.output / "sessions", run_id, session_result, benchmark)
                trace["agent_protocol_error"] = metadata["error"]
                return trace
            if (
                benchmark == "terminalbench2"
                and metadata.get("error_source") == "execution"
                and metadata.get("error_type") == "TimeoutError"
                and not metadata.get("error")
            ):
                raise TaskTimeLimitError(f"{label}: Terminal-Bench session deadline expired without an official score")
            raise RuntimeError(f"{label}: {session_result.status}: {session_result.details}")
        trace = trajectory_from_session(self.output / "sessions", run_id, session_result, benchmark)
        if getattr(self, "benchmark_candidate_metric", "official") == "auxiliary":
            from .auxiliary_scores import candidate_metric

            trace["auxiliary_metric"] = candidate_metric(trace, benchmark)
        if capture_history:
            from ..ace.prompts.generator import build_generator_system_prompt

            context_mode = getattr(self, "actor_context", "legacy")
            trace["actor_context"] = context_mode
            trace["actor_policy"] = (
                shared_actor_prompt(benchmark, playbook_text(playbook)) if context_mode != "legacy" else
                playbook_text(playbook) if benchmark == "terminalbench2" else
                build_generator_system_prompt(playbook_text(playbook))
            )
            directory = Path(trace["terminus_episode_path"]).parent if benchmark == "terminalbench2" else \
                Path(trace["session_dir"]) / "agent"
            trace["policy_shift_history"] = str(directory / HISTORY_FILE)
            if type(trace["success"]) is not bool:
                raise ValueError("Span policy shift requires an official trajectory success flag")
            if benchmark == "appworld":
                trace.setdefault("partial_score", trace["score"])
                trace["score"] = float(trace["success"])
                trace["score_source"] = "appworld_tgc_v1"
        return trace

    def _one_reflection(self, rank: int, index: int, trace: dict[str, Any], playbook: dict[str, Any]):
        if getattr(self, "grounding", None) is not None:
            value = self.grounding.reflect(trace, playbook)
            value.update({"parent_rank": rank, "id": f"reflection_{index}"})
            return value
        last = None
        for _ in range(3):
            try:
                value = self.llm.reflect(trace, playbook, correction=str(last) if last else None)
                value.update({"parent_rank": rank, "id": f"reflection_{index}"})
                return value
            except (ValueError, KeyError) as exc:
                if is_context_length_error(exc):
                    raise
                last = exc
        raise RuntimeError(f"reflection_{index} failed: {last}")

    def _one_candidate(self, reflection: dict[str, Any], trace: dict[str, Any], parent: dict[str, Any],
                       proposal_evidence: dict[str, Any] | None = None):
        grounding = getattr(self, "grounding", None)
        if grounding is not None and not reflection.get("grounding", {}).get("verified"):
            return {"id": f"candidate_{reflection['id'].rsplit('_', 1)[-1]}",
                    "parent_rank": reflection["parent_rank"], "reflection": reflection,
                    "playbook": copy.deepcopy(parent), "valid": False,
                    "errors": ["Unverified reflection: retain parent, no policy-shift reward or mutation"]}
        try:
            response = grounding.mutate(
                trace, parent, reflection, proposal_evidence or {}, self.playbook_edit_mode,
            ) if grounding is not None else self.llm.mutate(
                trace, parent, reflection, proposal_examples=proposal_evidence["prompt"],
            ) if proposal_evidence is not None else self.llm.mutate(trace, parent, reflection)
        except Exception as exc:
            if skip_task_after_llm_error(exc):
                raise
            if not is_context_length_error(exc) and not isinstance(exc, (KeyError, ValueError)):
                raise
            return {"id": f"candidate_{reflection['id'].rsplit('_', 1)[-1]}",
                    "parent_rank": reflection["parent_rank"], "reflection": reflection,
                    "playbook": copy.deepcopy(parent), "valid": False,
                    "errors": [f"mutator: {type(exc).__name__}: {exc}"],
                    **({"proposal_retrieval": proposal_evidence["audit"]} if proposal_evidence else {})}
        operations = response.get("operations")
        child, applied = apply_operations(
            parent, operations if isinstance(operations, list) else [], trace["benchmark"],
            edit_mode=getattr(self, "playbook_edit_mode", "add_only"),
        )
        return {"id": f"candidate_{reflection['id'].rsplit('_', 1)[-1]}",
                "parent_rank": reflection["parent_rank"], "reflection": reflection,
                "playbook": child, "applied": applied,
                "proposed_operations": operations,
                "mutation_summary": response.get("mutation_summary", ""), "valid": True,
                **({"grounded_proposal": response["grounded_proposal"]} if grounding is not None else {}),
                **({"proposal_retrieval": proposal_evidence["audit"]} if proposal_evidence else {})}

    def _score_job(self, candidate: dict[str, Any], source: dict[str, Any], reflection: dict[str, Any], label: str):
        if getattr(self, "candidate_evaluation", "environment_replay") == "span_policy_shift":
            return self.policy_shift_evaluator.evaluate(candidate, source, reflection, label)
        benchmark = source["benchmark"]
        for attempt in range(self.replay_retries + 1):
            try:
                with self.benchmark_slots[benchmark]:
                    with self._replay_lock:
                        self._active_replays += 1
                        self._peak_replays = max(self._peak_replays, self._active_replays)
                    try:
                        replay = self._run_agent(
                            benchmark=benchmark, task_id=source["task_id"],
                            playbook=candidate["playbook"], label=f"{label}_try{attempt + 1}",
                            replay_source=source, span_start=reflection["span"]["start"],
                        )
                    finally:
                        with self._replay_lock:
                            self._active_replays -= 1
                break
            except (KeyError, ValueError):
                raise
            except Exception as exc:
                # Let the outer training loop skip the update once request
                # recovery is exhausted, without restarting the trajectory.
                if (
                    is_tool_parser_configuration_error(exc) or skip_task_after_llm_error(exc)
                    or is_nontransient_task_error(exc) or attempt >= self.replay_retries
                ):
                    raise
        judge = self.llm.judge(source, reflection, replay) if getattr(
            self, "alignment_scoring", getattr(self, "method", "premise") == "premise") else None
        auxiliary = replay.get("auxiliary_metric")
        # The official replay score is retained, while detailed grader output
        # remains available in the session files instead of every iteration.
        replay.pop("details", None)
        result = {"replay": replay, "official_score": replay["score"], "replay_attempts": attempt + 1}
        if getattr(self, "benchmark_candidate_metric", "official") == "auxiliary":
            if not auxiliary:
                raise ValueError("Auxiliary candidate evidence is missing; refusing official-score substitution")
            result.update(candidate_score=auxiliary["score"], candidate_score_metric=auxiliary["metric"])
        if getattr(self, "appworld_candidate_metric", "tgc") == "pct":
            # Keep the trajectory and official reporting score as TGC. PCT is
            # an explicit, independent objective for selecting candidate edits.
            value = replay["partial_score"] if benchmark == "appworld" else replay["score"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("Candidate score must be a finite number in [0, 1]")
            result["candidate_score"] = float(value)
            result["candidate_score_metric"] = "appworld_pct_v1" if benchmark == "appworld" else "official_score"
        if judge is not None:
            result["alignment"] = judge
        return result

    def _memory_trace(self, item: dict[str, Any]) -> dict[str, Any]:
        if "trace" in item:  # Checkpoints created before trace references.
            return item["trace"]
        artifact = json.loads((self.output / item["trace_artifact"]).read_text(encoding="utf-8"))
        return artifact["parents"][item["parent_index"]]

    def _score_candidates(self, iteration: int, candidates: list[dict[str, Any]],
                          parent_traces: list[dict[str, Any]], past: list[dict[str, Any]]):
        jobs = []
        for candidate in candidates:
            if not candidate.get("valid"):
                continue
            rank = candidate["parent_rank"]
            jobs.append((candidate["id"], "current", None, parent_traces[rank - 1], candidate["reflection"]))
            for index, item in enumerate(past):
                jobs.append((candidate["id"], "past", index, item["trace"], item["reflection"]))
        by_id = {candidate["id"]: candidate for candidate in candidates}
        self._peak_replays = 0
        outcomes: dict[tuple[str, str, int | None], dict[str, Any]] = {}

        def evaluation_error(exc):
            if skip_task_after_llm_error(exc) or (
                not is_nontransient_task_error(exc) and not is_replay_restore_error(exc)
                and not isinstance(exc, (KeyError, ValueError))
            ):
                raise exc
            return {"error": f"{type(exc).__name__}: {exc}"}

        if getattr(self, "candidate_evaluation", "environment_replay") == "span_policy_shift":
            evaluations = [
                ((cid, kind, index), by_id[cid], source, reflection,
                 f"i{iteration:04d}_{cid}_{kind}{index if index is not None else ''}")
                for cid, kind, index, source, reflection in jobs
            ]
            results = self.policy_shift_evaluator.evaluate_batch(evaluations, max_workers=self.replay_workers)
            outcomes = {key: evaluation_error(result) if isinstance(result, Exception) else result
                        for key, result in results.items()}
        else:
            with ThreadPoolExecutor(max_workers=min(self.replay_workers, len(jobs)) or 1) as pool:
                futures = {}
                for candidate_id, kind, past_index, source, reflection in jobs:
                    label = f"i{iteration:04d}_{candidate_id}_{kind}{past_index if past_index is not None else ''}"
                    future = pool.submit(self._score_job, by_id[candidate_id], source, reflection, label)
                    futures[future] = (candidate_id, kind, past_index)
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        outcomes[key] = future.result()
                    except Exception as exc:
                        if skip_task_after_llm_error(exc):
                            for pending in futures:
                                pending.cancel()
                        outcomes[key] = evaluation_error(exc)
        for candidate in candidates:
            if not candidate.get("valid"):
                continue
            cid = candidate["id"]
            current = outcomes[(cid, "current", None)]
            past_results = [outcomes[(cid, "past", i)] for i in range(len(past))]
            if "error" in current or any("error" in item for item in past_results):
                candidate["valid"] = False
                candidate["errors"] = [item["error"] for item in [current, *past_results] if "error" in item]
                continue
            score_key = "evaluation_score" if getattr(self, "candidate_evaluation", "environment_replay") == \
                "span_policy_shift" else "official_score"
            candidate.update({
                f"current_{score_key}": current[score_key],
                f"past_{score_key}": (
                    sum(item[score_key] for item in past_results) / len(past_results) if past_results else None
                ),
                "current_result": current,
                "past_results": past_results,
            })
            if "candidate_score" in current:
                candidate["current_candidate_score"] = current["candidate_score"]
                candidate["past_candidate_score"] = (
                    sum(item["candidate_score"] for item in past_results) / len(past_results)
                    if past_results else None
                )
                candidate["candidate_score_metric"] = current["candidate_score_metric"]
            if getattr(self, "alignment_scoring", getattr(self, "method", "premise") == "premise"):
                candidate["current_alignment"] = current["alignment"]["score"]
                candidate["past_alignment"] = (
                    sum(item["alignment"]["score"] for item in past_results) / len(past_results)
                    if past_results else None
                )

    def _validation_sample(self, iteration: int) -> list[dict[str, str]]:
        benchmark = self.order[iteration - 1][0]
        slugs = [benchmark] if self.validation_scope == "current" else [
            slug for slug in self.benchmarks if slug in {name for name, _ in self.order[:iteration]}
        ]
        pool = [(slug, task_id) for slug in slugs for task_id in self.validation_pools[slug]]
        if len(pool) < 4:
            raise ValueError("Need at least four validation tasks in the selected pool")
        return [{"benchmark": slug, "task_id": task_id} for slug, task_id in self.rng.sample(pool, 4)]

    def _score_validation_job(self, candidate: dict[str, Any], task: dict[str, str], label: str) -> dict[str, Any]:
        benchmark = task["benchmark"]
        for attempt in range(self.replay_retries + 1):
            try:
                with self.benchmark_slots[benchmark]:
                    with self._replay_lock:
                        self._active_replays += 1
                        self._peak_replays = max(self._peak_replays, self._active_replays)
                    try:
                        result = self._run_agent(
                            benchmark=benchmark, task_id=task["task_id"],
                            playbook=candidate["playbook"], label=f"{label}_try{attempt + 1}",
                        )
                    finally:
                        with self._replay_lock:
                            self._active_replays -= 1
                return {**task, "official_score": result["score"],
                        "session_dir": result.get("session_dir"), "attempts": attempt + 1}
            except (KeyError, ValueError):
                raise
            except Exception as exc:
                if (
                    is_tool_parser_configuration_error(exc) or skip_task_after_llm_error(exc)
                    or is_nontransient_task_error(exc) or attempt >= self.replay_retries
                ):
                    raise
        raise AssertionError("Validation retry loop ended without a result")

    def _score_validation_candidates(self, iteration: int, candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
        sample = self._validation_sample(iteration)
        by_id = {candidate["id"]: candidate for candidate in candidates if candidate.get("valid")}
        jobs = [(candidate_id, index, task) for candidate_id in by_id for index, task in enumerate(sample)]
        self._peak_replays = 0
        outcomes: dict[tuple[str, int], dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(self.replay_workers, len(jobs)) or 1) as pool:
            futures = {
                pool.submit(self._score_validation_job, by_id[candidate_id], task,
                            f"i{iteration:04d}_{candidate_id}_val{index}"): (candidate_id, index)
                for candidate_id, index, task in jobs
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    outcomes[key] = future.result()
                except Exception as exc:
                    if skip_task_after_llm_error(exc):
                        for pending in futures:
                            pending.cancel()
                        raise
                    if not is_nontransient_task_error(exc) and not isinstance(exc, (KeyError, ValueError)):
                        raise
                    outcomes[key] = {"error": f"{type(exc).__name__}: {exc}"}
        for candidate_id, candidate in by_id.items():
            results = [outcomes[(candidate_id, index)] for index in range(len(sample))]
            if any("error" in result for result in results):
                candidate["valid"] = False
                candidate["errors"] = [result["error"] for result in results if "error" in result]
                continue
            candidate["validation_results"] = results
            candidate["validation_mean_official_score"] = sum(
                result["official_score"] for result in results
            ) / len(results)
        return sample

    def step(self) -> dict[str, Any]:
        if self.cursor >= len(self.order):
            raise StopIteration
        iteration = self.cursor + 1
        iteration_started = time.monotonic()
        benchmark, task_id = self.order[self.cursor]
        state_key = benchmark if self.mode == "isolated" else "global"
        state = self.states[state_key]
        parents = [copy.deepcopy(item) for item in state["beam"]]

        def run_parent(rank: int):
            with self.benchmark_slots[benchmark]:
                return self._run_agent(
                    benchmark=benchmark, task_id=task_id,
                    playbook=parents[rank], label=f"i{iteration:04d}_parent{rank + 1}",
                )

        errors = []
        traces = [None] * len(parents)
        with ThreadPoolExecutor(max_workers=min(self.parent_workers, len(parents))) as pool:
            futures = {pool.submit(run_parent, rank): rank for rank in range(len(parents))}
            for future in as_completed(futures):
                rank = futures[future]
                try:
                    traces[rank] = future.result()
                except Exception as exc:
                    if skip_task_after_llm_error(exc):
                        for pending in futures:
                            pending.cancel()
                        raise
                    if not is_nontransient_task_error(exc):
                        raise
                    errors.append(f"parent_{rank + 1}: {type(exc).__name__}: {exc}")
        parent_seconds = time.monotonic() - iteration_started
        children_per_parent = 4 // len(parents)
        specs = [(rank, index, traces[rank - 1], parents[rank - 1])
                 for rank in range(1, len(parents) + 1)
                 for index in range(children_per_parent * (rank - 1) + 1, children_per_parent * rank + 1)
                 if traces[rank - 1] is not None]
        reflections = []
        with ThreadPoolExecutor(max_workers=min(self.reflection_workers, 4)) as pool:
            futures = [pool.submit(self._one_reflection, *spec) for spec in specs]
            for future in futures:
                try:
                    reflections.append(future.result())
                except Exception as exc:
                    if skip_task_after_llm_error(exc):
                        for pending in futures:
                            pending.cancel()
                        raise
                    if not is_context_length_error(exc) and not is_nontransient_task_error(exc):
                        raise
                    errors.append(f"reflection: {type(exc).__name__}: {exc}")
        reflection_seconds = time.monotonic() - iteration_started - parent_seconds
        proposal_evidence = self.proposal_retriever.prepare(iteration, [
            (traces[reflection["parent_rank"] - 1], reflection) for reflection in reflections
        ]) if self.proposal_retriever is not None else {}
        if self.grounding is not None:
            proposal_evidence = self.grounding.prepare(iteration, [
                (traces[reflection["parent_rank"] - 1], reflection) for reflection in reflections])
        with ThreadPoolExecutor(max_workers=min(self.mutation_workers, 4)) as pool:
            futures = []
            for reflection in reflections:
                args = (reflection, traces[reflection["parent_rank"] - 1], parents[reflection["parent_rank"] - 1])
                if self.proposal_retriever is not None or self.grounding is not None:
                    args += (proposal_evidence.get(reflection["id"], {}),)
                futures.append(pool.submit(self._one_candidate, *args))
            candidates = [future.result() for future in futures]
        mutation_seconds = time.monotonic() - iteration_started - parent_seconds - reflection_seconds
        validation_sample = None
        if self.candidate_validation == "random_val4":
            past = []
            validation_sample = self._score_validation_candidates(iteration, candidates)
        else:
            eligible_memory = [item for item in state["memory"]
                               if not (item["benchmark"] == benchmark and item["task_id"] == task_id)]
            past = [dict(item, trace=self._memory_trace(item))
                    for item in sample_past(eligible_memory, self.past_count, self.rng)]
            if getattr(self, "candidate_selection", "pareto") == "random":
                self._peak_replays = 0
            else:
                self._score_candidates(iteration, candidates, traces, past)
        replay_seconds = time.monotonic() - iteration_started - parent_seconds - reflection_seconds - mutation_seconds
        if sum(bool(item.get("valid")) for item in candidates) < len(parents):
            failures = errors + [error for item in candidates for error in item.get("errors", [])]
            artifact = {
                "iteration": iteration, "benchmark": benchmark, "task_id": task_id,
                "status": "skipped", "train_update_applied": False,
                "reason": "insufficient_valid_candidates", "errors": failures,
                "parents": traces, "reflections": reflections, "candidates": candidates,
                **({"validation_sample": validation_sample} if validation_sample is not None else {}),
            }
            if self.grounding is not None:
                artifact["reason"] = "no_update_insufficient_grounded_candidates"
            write_json(self.output / "iterations" / f"iter_{iteration:04d}.json", artifact)
            if self.skip_task_errors == "none" and self.grounding is None:
                raise RuntimeError(
                    f"Iteration {iteration} has fewer than {len(parents)} valid candidates: {failures[:3]}")
            self.cursor = iteration
            self._save()
            return artifact
        selected, selection = (
            select_ablation_beam(
                candidates, self.candidate_selection, seed=self.seed, iteration=iteration,
                score_key="candidate_score"
                if (getattr(self, "appworld_candidate_metric", "tgc") == "pct"
                    or getattr(self, "benchmark_candidate_metric", "official") == "auxiliary") else "official_score",
            ) if getattr(self, "candidate_selection", "pareto") in ABLATION_SELECTION_RULES else
            select_alignment_product(
                candidates, score_key="candidate_score"
                if (getattr(self, "appworld_candidate_metric", "tgc") == "pct"
                    or getattr(self, "benchmark_candidate_metric", "official") == "auxiliary") else "official_score"
            ) if getattr(self, "candidate_selection", "pareto") ==
            "alignment_product" else
            select_validation_beam(candidates) if self.candidate_validation == "random_val4" else
            select_policy_shift_beam(candidates) if self.candidate_evaluation == "span_policy_shift" else
            select_ps_st_pareto_beam(candidates) if self.method == "ps_st_pareto" else select_beam(
                candidates, alignment_scoring=getattr(self, "alignment_scoring", True),
                score_key="candidate_score"
                if (getattr(self, "appworld_candidate_metric", "tgc") == "pct"
                    or getattr(self, "benchmark_candidate_metric", "official") == "auxiliary") else "official_score")
        )
        new_state = copy.deepcopy(state)
        new_state["beam"] = [copy.deepcopy(item["playbook"]) for item in selected]
        for rank, chosen in enumerate(selected):
            tags = [{"id": item.get("entry_id", item.get("id")), "tag": item.get("tag", "neutral")}
                    for item in chosen["reflection"].get("entry_assessments", [])]
            new_state["beam"][rank]["playbook"] = update_bullet_counts(
                new_state["beam"][rank]["playbook"], tags,
            )
        primary = selected[0]
        shift_evaluation = self.candidate_evaluation == "span_policy_shift"
        result_score = "evaluation_score" if shift_evaluation else "official_score"
        memory_score = "policy_shift_score" if shift_evaluation else "replay_official_score"
        candidate_key = "latest_selected_candidate_id" if shift_evaluation else "latest_selected_replay_candidate_id"
        iteration_key = "latest_selected_iteration" if shift_evaluation else "latest_selected_replay_iteration"
        for sampled, result in zip(past, primary.get("past_results", [])):
            memory_id = sampled.get("memory_id")
            for entry in new_state["memory"]:
                if entry.get("memory_id") == memory_id:
                    entry[f"latest_selected_{memory_score}"] = result[result_score]
                    entry[candidate_key] = primary["id"]
                    entry[iteration_key] = iteration
                    break
        if primary["applied"]:
            memory_entry = {
                "memory_id": f"{iteration}|{benchmark}|{task_id}",
                "benchmark": benchmark, "task_id": task_id,
                "trace_artifact": f"iterations/iter_{iteration:04d}.json",
                "parent_index": primary["parent_rank"] - 1,
                "reflection": primary["reflection"],
                "accepted_candidate_id": primary["id"],
            }
            if self.candidate_validation == "random_val4":
                memory_entry["selected_validation_mean_official_score"] = primary["validation_mean_official_score"]
            elif getattr(self, "candidate_selection", "pareto") != "random":
                selected_key = "selected_policy_shift_score" if shift_evaluation else "selected_official_score"
                memory_entry[selected_key] = primary[f"current_{result_score}"]
                memory_entry[f"latest_selected_{memory_score}"] = primary[f"current_{result_score}"]
            else:
                memory_entry["candidate_score_measured"] = False
            new_state["memory"].append(memory_entry)
        artifact = {
            "iteration": iteration, "benchmark": benchmark, "task_id": task_id,
            "parents": traces, "reflections": reflections, "candidates": candidates,
            "past_sample": [{"benchmark": item["benchmark"], "task_id": item["task_id"]} for item in past],
            **({"validation_sample": validation_sample} if validation_sample is not None else {}),
            "selection": selection,
            "errors": errors,
            "parallelism": {"replay_jobs": sum(bool(c.get("valid")) for c in candidates) * (1 + len(past))
                            if (validation_sample is None and not shift_evaluation
                                and getattr(self, "candidate_selection", "pareto") != "random") else 0,
                            **({"policy_shift_span_jobs": (
                                    sum(bool(c.get("valid")) for c in candidates) * (1 + len(past))
                                ),
                                "policy_shift_action_generations": sum(
                                    result.get("action_generations", len(result.get("actions", [])))
                                    for candidate in candidates
                                    for result in ([candidate["current_result"], *candidate["past_results"]]
                                                   if candidate.get("valid") else [])
                                ),
                                "policy_shift_queue": self.policy_shift_evaluator.last_parallelism,
                                "policy_shift_action_workers": self.replay_workers,
                            } if shift_evaluation else {}),
                            **({"validation_jobs": sum(bool(c.get("valid")) for c in candidates) * 4}
                               if validation_sample is not None else {}),
                            "peak_replay_sessions": self._peak_replays,
                            "replay_workers": self.replay_workers},
            "timings_seconds": {"parents": round(parent_seconds, 2), "reflections": round(reflection_seconds, 2),
                                "mutations": round(mutation_seconds, 2), "replays_and_judges": round(replay_seconds, 2)},
        }
        write_json(self.output / "iterations" / f"iter_{iteration:04d}.json", artifact)
        self.states[state_key] = new_state
        self.cursor = iteration
        self._save()
        return artifact

    def run(self, stop_after: int | None = None):
        target = len(self.order) if stop_after is None else stop_after
        if not self.cursor <= target <= len(self.order):
            raise ValueError("Training stop must be between the current cursor and the full task count")
        while self.cursor < target:
            try:
                artifact = self.step()
            except Exception as exc:
                if is_tool_parser_configuration_error(exc):
                    raise
                if self.skip_task_errors == "none" or (
                    not skip_task_after_llm_error(exc) and (
                        self.skip_task_errors == "nontransient" and not is_nontransient_task_error(exc)
                    )
                ):
                    raise
                iteration = self.cursor + 1
                benchmark, task_id = self.order[self.cursor]
                artifact = {
                    "iteration": iteration, "benchmark": benchmark, "task_id": task_id,
                    "status": "skipped", "train_update_applied": False,
                    "reason": "llm_request_failed_or_output_truncated" if skip_task_after_llm_error(exc)
                              else "nontransient_task_error",
                    "error_type": type(exc).__name__, "error_message": str(exc)[:4000],
                    "rng_state_after": repr(self.rng.getstate()),
                }
                write_json(self.output / "iterations" / f"iter_{iteration:04d}.json", artifact)
                # step() commits beam and memory only after successful selection.
                # A skipped task advances the frozen order without training on it.
                self.cursor = iteration
                self._save()
                print(f"[{iteration}/{len(self.order)}] SKIPPED {benchmark}::{task_id} "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            if artifact.get("status") == "skipped":
                print(f"[{artifact['iteration']}/{len(self.order)}] SKIPPED {artifact['benchmark']}::"
                      f"{artifact['task_id']} insufficient valid candidates", flush=True)
                continue
            print(f"[{artifact['iteration']}/{len(self.order)}] {artifact['benchmark']}::{artifact['task_id']} "
                  f"selected={artifact['selection']['selected']} past={len(artifact['past_sample'])}", flush=True)

        write_json(self.output / "training_complete.json", {
            "completed_task_count": self.cursor, "expected_task_count": target,
        })

    def evaluate_validation(self):
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        sections = select_task_split_sections(manifest, self.benchmarks, getattr(self, "seed", 42))
        jobs = []
        for slug in self.benchmarks:
            state = self.states[slug if self.mode == "isolated" else "global"]
            for index, task_id in enumerate(sections[slug].get("val", [])):
                jobs.append((slug, str(task_id), index, copy.deepcopy(state["beam"][0])))

        def one(job):
            slug, task_id, index, playbook = job
            with self.benchmark_slots[slug]:
                result = self._run_agent(benchmark=slug, task_id=task_id,
                                         playbook=playbook, label=f"validation_{slug}_{index:04d}")
            return {"benchmark": slug, "task_id": task_id, "score": result["score"]}

        result_path = self.output / "validation.json"
        results = [None] * len(jobs)
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(previous, list) and len(previous) == len(jobs):
                for index, (record, job) in enumerate(zip(previous, jobs)):
                    if isinstance(record, dict) and (record.get("benchmark"), record.get("task_id")) == job[:2]:
                        results[index] = record
        pending = [(index, job) for index, job in enumerate(jobs) if results[index] is None]
        with ThreadPoolExecutor(max_workers=min(self.replay_workers, len(pending)) or 1) as pool:
            futures = {pool.submit(one, job): index for index, job in pending}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                write_json(result_path, results)
        return results

    def evaluate_test_checkpoints(self, checkpoints=(50, 100, 130),
                                  benchmarks: tuple[str, ...] | None = None,
                                  allow_incomplete: bool = False):
        """Evaluate rank 1 without learning, on all three test sets at each checkpoint."""
        if not allow_incomplete and self.cursor != len(self.order):
            raise ValueError("Complete training before evaluating test checkpoints")
        target_benchmarks = tuple(self.benchmarks) if benchmarks is None else benchmarks
        if not target_benchmarks or any(slug not in self.benchmarks for slug in target_benchmarks):
            raise ValueError("Test benchmarks must be a nonempty subset of trained benchmarks")
        sections = select_task_split_sections(
            json.loads(self.manifest.read_text(encoding="utf-8")), self.benchmarks, self.seed,
        )
        splits = {slug: validate_task_split(sections[slug], slug) for slug in self.benchmarks}
        state_before = (self.output / "state.json").read_bytes()
        if hashlib.sha256(self.manifest.read_bytes()).hexdigest() != json.loads(state_before)["config"]["manifest_sha256"]:
            raise ValueError("Task split manifest changed after training")
        summaries = []
        for point in checkpoints:
            snapshot_path = self.output / "checkpoints" / f"checkpoint_{point:03d}.json"
            snapshot_bytes = snapshot_path.read_bytes()
            snapshot = json.loads(snapshot_bytes)
            if snapshot["cursor"] != point or snapshot["config"] != json.loads(state_before)["config"]:
                raise ValueError(f"Checkpoint {point} does not match this experiment")
            for slug in target_benchmarks:
                ids = splits[slug]["test"]
                if not ids:
                    raise ValueError(f"No test tasks for {slug}")
                beam = snapshot["states"][slug if self.mode == "isolated" else "global"]["beam"]
                frozen = copy.deepcopy(beam[0])
                identity = {
                    "checkpoint": point, "benchmark": slug, "task_ids": ids,
                    "checkpoint_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                    "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                    "beam_rank": 1, "learning": False,
                }
                path = self.output / "test_evaluation" / f"cp{point:03d}_{slug}.json"
                records = [None] * len(ids)
                if path.exists():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if old.get("identity") != identity:
                        raise ValueError(f"Existing evaluation identity differs: {path}")
                    records = old["results"]
                    if len(records) != len(ids):
                        raise ValueError(f"Existing evaluation result length differs: {path}")
                    for index, record in enumerate(records):
                        if record is not None and record.get("task_id") != ids[index]:
                            raise ValueError(f"Existing evaluation task differs: {path}")

                def one(index, task_id):
                    try:
                        with self.benchmark_slots[slug]:
                            result = self._run_agent(
                                benchmark=slug, task_id=task_id, playbook=copy.deepcopy(frozen),
                                label=f"test_cp{point:03d}_{slug}_{index:04d}",
                            )
                        return {"task_id": task_id, "score": result["score"], "status": "completed"}
                    except Exception as exc:
                        # Match ACE: failed test execution is recorded and contributes zero.
                        return {"task_id": task_id, "score": 0.0, "status": "error",
                                "context_error": is_context_length_error(exc),
                                "error": f"{type(exc).__name__}: {exc}"}

                pending = [(index, task_id) for index, task_id in enumerate(ids) if records[index] is None]
                workers = 1 if slug == "terminalbench2" else min(4, self.replay_workers)
                with ThreadPoolExecutor(max_workers=min(workers, len(pending)) or 1) as pool:
                    futures = {pool.submit(one, *job): job[0] for job in pending}
                    for future in as_completed(futures):
                        records[futures[future]] = future.result()
                        write_json(path, {"identity": identity, "results": records, "complete": False})
                summary = {
                    "identity": identity, "results": records, "complete": True,
                    "mean_official_score": sum(record["score"] for record in records) / len(records),
                    "execution_errors": sum(record["status"] == "error" for record in records),
                }
                write_json(path, summary)
                summaries.append(summary)
                print(f"TEST checkpoint={point} benchmark={slug} tasks={len(ids)} "
                      f"score={summary['mean_official_score']:.4f} errors={summary['execution_errors']}", flush=True)
        if not allow_incomplete and (self.output / "state.json").read_bytes() != state_before:
            raise RuntimeError("Test evaluation changed the training state")
        if not allow_incomplete:
            write_json(self.output / "test_evaluation" / "complete.json", {
                "evaluation_groups": len(summaries), "checkpoints": list(checkpoints),
                "total_tasks": sum(len(item["results"]) for item in summaries),
            })
        return summaries
