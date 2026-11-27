# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Group-relative prompt search using a frozen bank of original action prefixes."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean, pstdev

from ...actor_context import DEFAULT_ACTOR_CONTEXT, actor_context_metadata
from ...core.experiment_limits import experiment_limits
from .action_agreement import METRICS as AGREEMENT_METRICS
from .action_agreement import CanonicalActionAgreement
from .action_nli import DEFAULT_MODEL, DEFAULT_REVISION, LocalTextEquivalence
from .auxiliary_scores import METRICS, candidate_metric
from .llm import render_actions, render_official_feedback, render_public_tool_contract
from .pipeline import PremisePipeline, write_json
from .span_policy_shift import SCHEDULING_VERSION, VERSION, strict_f1

METHOD = "group-relative-policy-shift-v2"
REWARD_METRICS = {**METRICS, "appworld": "appworld_pct_v1"}
CANONICAL_SCORE_MODES = ("lower", "interval")


def partial_reward(trace: dict) -> dict:
    """Keep dense optimization rewards separate from official success reporting."""
    benchmark = trace["benchmark"]
    metadata = (trace.get("details") or {}).get("session_metadata") or {}
    if metadata.get("experiment_budget_exceeded") or metadata.get("experiment_budget_failure"):
        return {"metric": REWARD_METRICS[benchmark], "score": 0.0, "source": "actor_budget_failure"}
    if benchmark != "appworld":
        return candidate_metric(trace, benchmark)
    value = trace.get("partial_score")
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Missing or invalid official AppWorld individual-test pass fraction")
    return {"metric": REWARD_METRICS[benchmark], "score": float(value), "source": "official_partial_score"}


def digest(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def relative_advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    """Normalize within a task; equal rewards contribute exactly zero."""
    if not rewards or any(not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError("Rewards must be finite official scores in [0, 1]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Advantage epsilon must be positive and finite")
    mean, std = fmean(rewards), pstdev(rewards)
    return [(reward - mean) / (std + epsilon) for reward in rewards] if std else [0.0] * len(rewards)


def agreement(predicted: list[dict], reference: list[dict], metric: str) -> float:
    if metric == "strict_f1":
        return strict_f1(predicted, reference)
    if metric == "strict_accuracy":
        return float(predicted == reference)
    raise ValueError("Metric must be strict_f1 or strict_accuracy")


class GroupRelativePipeline(PremisePipeline):
    """Reuse PEACE acting and proposals, replacing replay with signed agreement."""

    def __init__(
        self, *, rollout_count: int = 5, candidate_count: int = 5, epochs: int = 1,
        past_count: int = 0, metric: str = "canonical_f1", shift_penalty: float = 0.0,
        max_policy_shift: float | None = None, advantage_epsilon: float = 1e-8,
        model_call_settings: dict | None = None, model_call_source: dict | None = None,
        action_comparator=None, agreement_text_model=DEFAULT_MODEL, agreement_text_revision=DEFAULT_REVISION,
        agreement_text_threshold: float = .95, agreement_text_max_tokens: int = 512,
        agreement_cpu_threads: int = 4, canonical_score_mode: str = "lower",
        **kwargs,
    ):
        if any(type(n) is not int or n < 1 for n in (rollout_count, candidate_count, epochs)):
            raise ValueError("Rollout, candidate and epoch counts must be positive integers")
        if rollout_count < 2 or type(past_count) is not int or past_count < 0:
            raise ValueError("Need at least two rollouts and a nonnegative past task count")
        if metric not in AGREEMENT_METRICS:
            raise ValueError(f"Metric must be one of {AGREEMENT_METRICS}")
        if canonical_score_mode not in CANONICAL_SCORE_MODES:
            raise ValueError(f"Canonical score mode must be one of {CANONICAL_SCORE_MODES}")
        self.canonical_score_mode = canonical_score_mode
        self.canonical_agreement = metric.startswith("canonical_")
        self.action_comparator = None
        if self.canonical_agreement:
            self.action_comparator = action_comparator or CanonicalActionAgreement(
                Path(kwargs["output"]) / "agreement_cache",
                text=LocalTextEquivalence(model=agreement_text_model, revision=agreement_text_revision,
                                          threshold=agreement_text_threshold, max_tokens=agreement_text_max_tokens,
                                          cpu_threads=agreement_cpu_threads),
            )
        elif action_comparator is not None:
            raise ValueError("An action comparator requires a canonical metric")
        relative_advantages([0.0], advantage_epsilon)
        if not math.isfinite(shift_penalty) or shift_penalty < 0:
            raise ValueError("Shift penalty must be nonnegative and finite")
        if max_policy_shift is not None and not 0 <= max_policy_shift <= 1:
            raise ValueError("Maximum policy shift must be in [0, 1]")
        self.rollout_count, self.candidate_count, self.epochs = rollout_count, candidate_count, epochs
        self.group_past_count, self.agreement_metric = past_count, metric
        self.shift_penalty, self.max_policy_shift = shift_penalty, max_policy_shift
        self.advantage_epsilon = advantage_epsilon
        self.actor_model_settings = copy.deepcopy(model_call_settings)
        self.model_call_source = model_call_source
        if model_call_settings is not None:
            from ...core.types.model_settings import ModelSettings

            ModelSettings(**model_call_settings)
            kwargs.setdefault("actor_temperature", model_call_settings["temperature"])
            kwargs.setdefault("policy_shift_temperature", model_call_settings["temperature"])
            kwargs.setdefault("reflection_temperature", model_call_settings["temperature"])
            kwargs.setdefault("mutation_temperature", model_call_settings["temperature"])
            if model_call_settings.get("max_tokens") is None:
                import os

                if os.environ.get("EXGENTIC_ACTOR_OUTPUT_UNLIMITED") != "1":
                    raise ValueError("Unlimited actor output requires EXGENTIC_ACTOR_OUTPUT_UNLIMITED=1")
        kwargs.setdefault("actor_context", DEFAULT_ACTOR_CONTEXT)
        kwargs.setdefault("actor_temperature", 1.0)
        kwargs.setdefault("policy_shift_temperature", 1.0)
        kwargs.setdefault("reflection_temperature", 0.7)
        kwargs.setdefault("mutation_temperature", 0.7)
        if kwargs["actor_temperature"] != kwargs["policy_shift_temperature"]:
            raise ValueError("Rollout and F1 probe temperature must match; only context should differ")
        from ...inference_profile import profile

        active_profile = profile()
        if active_profile and active_profile["actor_temperature"] != kwargs["actor_temperature"]:
            raise ValueError("The active inference profile overrides the requested rollout/probe temperature")
        # The current GPU1 serving endpoint admits 128 sequences. Keep prefix
        # workers and the shared model-call gate aligned so neither retains the
        # former 28/32-call bottleneck. Explicit per-run limits still take priority.
        kwargs.setdefault("llm_calls", 128)
        for name, default in (("parent_workers", 5), ("mutation_workers", 5), ("replay_workers", 128)):
            value = kwargs.setdefault(name, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # The parent class supplies the actor/session harness, not its optimizer.
        super().__init__(
            **kwargs, past_count=3, candidate_evaluation="span_policy_shift", alignment_scoring=False,
        )

    @staticmethod
    def _empty_state(beam_size: int = 1) -> dict:
        return PremisePipeline._empty_state(1)

    def _restore_or_initialize(self):
        if "bfcl" in self.benchmark_configs:
            self.benchmark_configs["bfcl"]["auxiliary_turn_metrics"] = True
        self.policy_shift_evaluator.match_source_settings = True
        if self.actor_model_settings is not None:
            self.policy_shift_evaluator.max_tokens = self.actor_model_settings.get("max_tokens")
            # PEACE analysis prompts are retained; model generation follows the chosen live profile.
            self.llm.generation_overrides = copy.deepcopy(self.actor_model_settings)
        self.collection_order = list(self.order)
        self.order = self.collection_order * self.epochs
        prompt_paths = [
            Path(__file__).with_name("llm.py"), Path(__file__).with_name("instance.py"),
            Path(__file__).parents[1] / "ace/prompts/generator.py",
            Path(__file__).parents[1] / "ace/prompts/curator.py",
        ]
        self.config = {
            "method": METHOD, "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "benchmarks": self.benchmarks, "mode": self.mode, "seed": self.seed,
            "collection_order": self.collection_order, "benchmark_configs": self.benchmark_configs,
            "rollout_count": self.rollout_count, "candidate_count": self.candidate_count,
            "epochs": self.epochs, "past_task_count": self.group_past_count,
            "metric": self.agreement_metric, "shift_penalty": self.shift_penalty,
            "max_policy_shift": self.max_policy_shift, "advantage_epsilon": self.advantage_epsilon,
            "advantage_std": "population",
            "reward_metrics": {b: REWARD_METRICS[b] for b in self.benchmarks},
            "refresh": "all_zero_advantage_tasks_after_each_full_epoch_using_epoch_end_context",
            "sampling_contract": "recorded_actor_generation_settings_context_only_change_v1",
            "model_call_settings": self.actor_model_settings, "model_call_source": self.model_call_source,
            "aggregation": "mean_actions_then_advantage_weighted_rollouts_then_equal_task_mean",
            "prefixes": "all_original_acting_prefixes", "zero_advantage": "excluded",
            "model": self.model, "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature, "role_temperatures": self.role_temperatures,
            "policy_shift_temperature": self.policy_shift_temperature,
            "actor_context": self.actor_context,
            "actor_context_metadata": actor_context_metadata(self.actor_context, self.benchmarks),
            "max_steps": self.max_steps,
            "actor_output_tokens": (self.actor_model_settings.get("max_tokens")
                                    if self.actor_model_settings is not None else self.tool_calling_max_tokens),
            "experiment_limits": {slug: experiment_limits(slug) for slug in self.benchmarks},
            "llm_timeout": self.llm_timeout, "llm_retries": self.llm_retries,
            "playbook_edit_mode": self.playbook_edit_mode,
            "evaluator": VERSION, "scheduling": SCHEDULING_VERSION,
            "prompt_sources": {str(p.relative_to(Path(__file__).parents[2])):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in prompt_paths},
        }
        if self.canonical_agreement:
            self.config.update(method="group-relative-canonical-action-v1",
                               action_agreement=self.action_comparator.identity,
                               sampling_ratio="canonical_agreement_proxy_not_likelihood_ratio",
                               unresolved_comparisons="interval_bounds_fixed_prefix_denominators",
                               selection="child_lower_bound_must_exceed_incumbent_upper_bound")
            if self.canonical_score_mode == "lower":
                self.config.update(method="group-relative-canonical-action-lower-v1",
                                   canonical_score_mode="lower", sampling_ratio_bound="lower",
                                   unresolved_comparisons="lower_agreement_before_signed_advantage_fixed_denominators",
                                   selection="maximize_lower_agreement_surrogate_keep_incumbent_on_ties",
                                   policy_shift="one_minus_lower_agreement")
        self.config.update(self._configuration_extensions())
        self.config = json.loads(json.dumps(self.config))
        if self.actor_model_settings is not None:
            template = (self.actor_model_settings.get("extra_body") or {}).get("chat_template_kwargs", {})
            self.config["actor_context_metadata"]["enable_thinking"] = bool(
                template.get("enable_thinking")
            )
        self.config_sha256 = digest(self.config)
        self.bank = {}
        self.bank_history, self.pending_refresh, self.refreshed_epochs = {}, set(), set()
        path = self.output / "state.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["config"] != self.config:
                raise ValueError("Existing checkpoint configuration differs from this group-relative run")
            self.cursor, self.states, self.bank = saved["cursor"], saved["states"], saved["bank"]
            self.bank_history = saved["bank_history"]
            self.pending_refresh = set(saved["pending_refresh"])
            self.refreshed_epochs = set(saved["refreshed_epochs"])
            self.rng.setstate(ast.literal_eval(saved["rng_state"]))
        else:
            self._save()

    def _configuration_extensions(self) -> dict:
        """Pin subclass-specific source/protocol settings before validating resume."""
        return {}

    def _save(self):
        write_json(self.output / "state.json", {
            "config": self.config, "cursor": self.cursor, "states": self.states,
            "bank": self.bank, "rng_state": repr(self.rng.getstate()),
            "bank_history": self.bank_history, "pending_refresh": sorted(self.pending_refresh),
            "refreshed_epochs": sorted(self.refreshed_epochs),
        })

    def _reference(self, path: Path) -> dict:
        return {"path": str(path.relative_to(self.output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _read_reference(self, reference: dict) -> dict:
        path = self.output / reference["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"Frozen rollout artifact changed: {path}")
        return json.loads(path.read_text())

    @staticmethod
    def _check_history(record: dict):
        path = Path(record["trace"]["policy_shift_history"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["history_sha256"]:
            raise ValueError(f"Frozen action history changed: {path}")

    def _collect_one(self, index: int, rollout: int, generation: int, playbook: dict) -> dict:
        benchmark, task_id = self.collection_order[index]
        stem = f"task_{index + 1:04d}_gen_{generation:04d}_rollout_{rollout + 1:02d}"
        path = self.output / "bank/rollouts" / f"{stem}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["config_sha256"] != self.config_sha256:
                raise ValueError("Partial rollout bank belongs to a different configuration")
            if record["playbook_sha256"] != digest(playbook):
                raise ValueError("Partial rollout belongs to a different collection context")
            self._check_history(record)
            return self._reference(path)
        with self.benchmark_slots[benchmark]:
            trace = self._run_agent(
                benchmark=benchmark, task_id=task_id, playbook=playbook,
                label=f"bank_t{index + 1:04d}_g{generation:04d}_r{rollout + 1:02d}",
            )
        # Infrastructure exceptions propagate; only measured official outcomes enter the bank.
        reward = partial_reward(trace)
        relative_advantages([reward["score"]], self.advantage_epsilon)
        if trace["benchmark"] != benchmark or trace["task_id"] != task_id or type(trace.get("success")) is not bool:
            raise ValueError("Rollout identity or official outcome is missing")
        history = Path(trace["policy_shift_history"])
        if not history.exists() and not trace["actions"]:
            # A first-turn budget failure is a measured zero with no prefix evidence.
            # Preserve the missing source path and create an explicit empty bank record.
            trace["original_policy_shift_history"] = str(history)
            history = self.output / "bank/empty_histories" / f"{stem}.jsonl"
            history.parent.mkdir(parents=True, exist_ok=True)
            history.write_text("")
            trace["policy_shift_history"] = str(history)
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        indices = [row["action_index"] for row in rows]
        if len(indices) != len(set(indices)) or not set(range(len(trace["actions"]))).issubset(indices):
            raise ValueError("Original trajectory does not have exactly one saved prefix per action")
        record = {
            "config_sha256": self.config_sha256, "trace": trace,
            "reward": reward, "playbook_sha256": digest(playbook),
            "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
        }
        write_json(path, record)
        return self._reference(path)

    def collect_rollouts(self):
        """Finish all initial-policy environment runs before any prompt update."""
        if self.cursor and len(self.bank) != len(self.collection_order):
            raise ValueError("An optimization checkpoint cannot have an incomplete rollout bank")
        for index, (benchmark, task_id) in enumerate(self.collection_order):
            key = str(index)
            if key in self.bank:
                self._load_group(index)
                continue
            self.bank[key] = self._collect_group(index, 0, self._empty_state()["beam"][0])
            self.bank_history[key] = [self.bank[key]]
            self._load_group(index)
            self._save()
            print(f"[bank {index + 1}/{len(self.collection_order)}] {benchmark}::{task_id}", flush=True)

    def _collect_group(self, index: int, generation: int, playbook: dict) -> dict:
        path = self.output / "bank/groups" / f"task_{index + 1:04d}_gen_{generation:04d}.json"
        if not path.exists():
            references = [None] * self.rollout_count
            with ThreadPoolExecutor(max_workers=min(self.parent_workers, self.rollout_count)) as pool:
                futures = {pool.submit(self._collect_one, index, r, generation, playbook): r
                           for r in range(self.rollout_count)}
                for future in as_completed(futures):
                    references[futures[future]] = future.result()
            records = [self._read_reference(ref) for ref in references]
            rewards = [record["reward"]["score"] for record in records]
            benchmark, task_id = self.collection_order[index]
            write_json(path, {
                "config_sha256": self.config_sha256, "index": index, "generation": generation,
                "collection_playbook": playbook, "benchmark": benchmark, "task_id": task_id,
                "rollouts": references, "reward_metric": REWARD_METRICS[benchmark],
                "rewards": rewards, "advantages": relative_advantages(rewards, self.advantage_epsilon),
                "has_prefixes": [bool(record["trace"]["actions"]) for record in records],
            })
        group = json.loads(path.read_text())
        if group["config_sha256"] != self.config_sha256 or group["collection_playbook"] != playbook:
            raise ValueError("Collected group configuration or context differs")
        return self._reference(path)

    def refresh_zero_advantage(self, epoch: int):
        """Replace zero-signal groups once per epoch, preserving every old generation."""
        if epoch in self.refreshed_epochs:
            return
        path = self.output / "bank/refresh" / f"epoch_{epoch:04d}.json"
        if path.exists():
            plan = json.loads(path.read_text())
            if plan["config_sha256"] != self.config_sha256:
                raise ValueError("Refresh plan belongs to a different configuration")
        else:
            plan = {"config_sha256": self.config_sha256, "epoch": epoch, "tasks": []}
            for index in sorted(self.pending_refresh):
                benchmark = self.collection_order[index][0]
                key = benchmark if self.mode == "isolated" else "global"
                plan["tasks"].append({"index": index, "generation": self._load_group(index)["generation"] + 1,
                                      "playbook": copy.deepcopy(self.states[key]["beam"][0])})
            write_json(path, plan)
        for task in plan["tasks"]:
            index, generation = task["index"], task["generation"]
            reference = self._collect_group(index, generation, task["playbook"])
            self.bank[str(index)] = reference
            if reference not in self.bank_history[str(index)]:
                self.bank_history[str(index)].append(reference)
            self._load_group(index)
            self._save()
            print(f"[refresh epoch {epoch}] task={index + 1} generation={generation}", flush=True)
        self.pending_refresh.difference_update(task["index"] for task in plan["tasks"])
        self.refreshed_epochs.add(epoch)
        self._save()

    def _load_group(self, index: int) -> dict:
        group = self._read_reference(self.bank[str(index)])
        if group["config_sha256"] != self.config_sha256 or group["index"] != index:
            raise ValueError("Frozen group belongs to another run or task")
        records = [self._read_reference(ref) for ref in group["rollouts"]]
        for record in records:
            self._check_history(record)
        traces = [record["trace"] for record in records]
        rewards = [record["reward"]["score"] for record in records]
        if (len(traces) != self.rollout_count or rewards != group["rewards"]
                or [bool(trace["actions"]) for trace in traces] != group["has_prefixes"]
                or relative_advantages(rewards, self.advantage_epsilon) != group["advantages"]):
            raise ValueError("Frozen group rewards or advantages changed")
        return {**group, "traces": traces}

    def _proposal(self, index: int, group: dict, parent: dict, best: int, worst: int) -> dict:
        traces = [dict(trace, official_score=trace["score"], score=group["rewards"][r],
                       score_metric=group["reward_metric"], score_source=group["reward_metric"])
                  for r, trace in enumerate(group["traces"])]
        best_reflection = self._one_reflection(1, index, traces[best], parent)
        worst_reflection = self._one_reflection(1, index, traces[worst], parent)
        evidence = [
            "Compare these two original rollouts of the SAME training task. Their rewards and relative "
            "advantages are fixed observations, not new candidate outcomes. Preserve supported high-reward "
            "procedures and correct supported low-reward mistakes using the current playbook. "
            "Each reflection uses the PEACE reflector. Neither rollout was generated by this candidate.",
        ]
        for role, r, reflection in (("best", best, best_reflection), ("worst", worst, worst_reflection)):
            trace = traces[r]
            evidence.append(
                f"\n{role.upper()} rollout {r + 1}; reward={group['rewards'][r]}; "
                f"advantage={group['advantages'][r]}; metric={group['reward_metric']}\n"
                f"Public tool contract: {render_public_tool_contract(trace)}\n"
                f"Recorded trajectory:\n{render_actions(trace['actions'], max_total_chars=80000)}\n"
                f"Official feedback: {render_official_feedback(trace)}\n"
                f"PEACE reflection: {json.dumps(reflection, ensure_ascii=False)}"
            )
        candidate = self._one_candidate(worst_reflection, traces[worst], parent, {
            "prompt": "\n".join(evidence), "audit": {"source": "same_task_best_worst", "best": best, "worst": worst},
        })
        candidate["best_reflection"] = best_reflection
        return candidate

    def _past_groups(self, current: int) -> list[dict]:
        seen = sorted({i % len(self.collection_order) for i in range(self.cursor)} - {current})
        if self.mode == "isolated":
            seen = [i for i in seen if self.collection_order[i][0] == self.collection_order[current][0]]
        eligible = [self._read_reference(self.bank[str(i)]) for i in seen] if self.group_past_count else []
        eligible = [g for g in eligible if any(
            a != 0 and has_prefix for a, has_prefix in zip(g["advantages"], g["has_prefixes"], strict=True)
        )]
        sampled = self.rng.sample(eligible, min(len(eligible), self.group_past_count))
        return [self._load_group(g["index"]) for g in sampled]

    def _score(self, iteration: int, candidates: list[dict], groups: list[dict]):
        active = [(g, r, a, trace) for g in groups
                  for r, (a, trace) in enumerate(zip(g["advantages"], g["traces"], strict=True))
                  if a != 0 and trace["actions"]]
        jobs, representatives = [], {}
        for candidate in candidates:
            if not candidate.get("valid"):
                continue
            policy = candidate["playbook"]["playbook"]
            if policy in representatives:
                continue
            representatives[policy] = candidate["id"]
            for group, r, _, trace in active:
                key = (candidate["id"], group["index"], r)
                jobs.append((key, candidate, trace, {"span": {"start": 0, "end": len(trace["actions"]) }},
                             f"gr_i{iteration:04d}_{candidate['id']}_g{group['index']:04d}_r{r:02d}"))
        results = self.policy_shift_evaluator.evaluate_batch(jobs, max_workers=self.replay_workers)
        if self.canonical_agreement:
            self._score_canonical(candidates, groups, active, representatives, results)
            return
        for candidate in candidates:
            if not candidate.get("valid"):
                continue
            representative = representatives[candidate["playbook"]["playbook"]]
            candidate["probe_representative"] = representative
            evaluations, task_scores, task_shifts = [], {}, {}
            for group, r, advantage, _ in active:
                result = results[(representative, group["index"], r)]
                if isinstance(result, Exception) or result.get("status") != "completed":
                    candidate["valid"] = False
                    candidate.setdefault("errors", []).append(
                        f"prefix evaluation g{group['index']} r{r}: {result}"
                    )
                    # Retain measured usage and partial output even on actor budget failure.
                    evaluations.append({"group": group["index"], "rollout": r,
                                        "result": str(result) if isinstance(result, Exception) else result})
                    continue
                baseline = results[("incumbent", group["index"], r)]
                if isinstance(baseline, Exception) or baseline.get("status") != "completed":
                    candidate["valid"] = False
                    candidate.setdefault("errors", []).append("Incumbent prefix evaluation failed")
                    continue
                actions = result["actions"]
                similarity = fmean(agreement(a["predicted"], a["reference"], self.agreement_metric) for a in actions)
                shift = fmean(1 - agreement(a["predicted"], b["predicted"], self.agreement_metric)
                              for a, b in zip(actions, baseline["actions"], strict=True))
                task_scores.setdefault(group["index"], []).append(advantage * similarity)
                task_shifts.setdefault(group["index"], []).append(shift)
                evaluations.append({"group": group["index"], "rollout": r, "advantage": advantage,
                                    "agreement": similarity, "shift_from_incumbent": shift, "result": result})
            candidate["prefix_evaluations"] = evaluations
            if candidate["valid"]:
                # Keep the original group size in the denominator; zero advantages add zero.
                objective = fmean(sum(task_scores.get(g["index"], [])) / self.rollout_count for g in groups)
                shift = fmean(fmean(values) for values in task_shifts.values())
                candidate.update(surrogate=objective, shift_from_incumbent=shift,
                                 selection_score=objective - self.shift_penalty * shift)
                if self.max_policy_shift is not None and shift > self.max_policy_shift:
                    candidate["valid"] = False
                    candidate.setdefault("errors", []).append("Maximum policy shift exceeded")

    def _score_canonical(self, candidates, groups, active, representatives, results):
        """Score by lower agreement, retaining signed interval diagnostics and fixed denominators."""
        histories = {}
        for _, _, _, trace in active:
            path = trace["policy_shift_history"]
            if path in histories:
                continue
            rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
            index = {row["action_index"]: row for row in rows}
            if len(index) != len(rows):
                raise ValueError("Duplicate original comparison prefix")
            histories[path] = index
        for candidate in candidates:
            if not candidate.get("valid"):
                continue
            representative = representatives[candidate["playbook"]["playbook"]]
            candidate["probe_representative"] = representative
            evaluations, task_scores, task_shifts, task_lower_scores = [], {}, {}, {}
            for group, rollout, advantage, trace in active:
                result = results[(representative, group["index"], rollout)]
                baseline = results[("incumbent", group["index"], rollout)]
                if any(isinstance(value, Exception) or value.get("status") != "completed"
                       for value in (result, baseline)):
                    candidate["valid"] = False
                    candidate.setdefault("errors", []).append("Candidate or incumbent prefix evaluation failed")
                    evaluations.append({"group": group["index"], "rollout": rollout,
                                        "result": str(result) if isinstance(result, Exception) else result})
                    continue
                comparisons = []
                try:
                    for action, parent in zip(result["actions"], baseline["actions"], strict=True):
                        if action["action_index"] != parent["action_index"]:
                            raise ValueError("Candidate/incumbent original prefix positions differ")
                        request = histories[trace["policy_shift_history"]][action["action_index"]]["request"]
                        ratio = self.action_comparator.compare(
                            action["predicted"], action["reference"], benchmark=trace["benchmark"],
                            request=request, metric=self.agreement_metric,
                        )
                        trust = self.action_comparator.compare(
                            action["predicted"], parent["predicted"], benchmark=trace["benchmark"],
                            request=request, metric=self.agreement_metric,
                        )
                        comparisons.append({"action_index": action["action_index"], "sampling_ratio": ratio,
                                            "incumbent_agreement": trust})
                        if self.canonical_score_mode == "lower":
                            comparisons[-1]["sampling_ratio_value"] = ratio["lower"]
                except Exception as exc:
                    candidate["valid"] = False
                    candidate.setdefault("errors", []).append(f"Canonical comparison failed: {exc}")
                    evaluations.append({"group": group["index"], "rollout": rollout, "result": result,
                                        "comparisons": comparisons})
                    continue
                lo = fmean(c["sampling_ratio"]["lower"] for c in comparisons)
                hi = fmean(c["sampling_ratio"]["upper"] for c in comparisons)
                shift_lo = fmean(1 - c["incumbent_agreement"]["upper"] for c in comparisons)
                shift_hi = fmean(1 - c["incumbent_agreement"]["lower"] for c in comparisons)
                weighted = (advantage * lo, advantage * hi) if advantage > 0 else (advantage * hi, advantage * lo)
                task_scores.setdefault(group["index"], []).append(weighted)
                task_lower_scores.setdefault(group["index"], []).append(advantage * lo)
                task_shifts.setdefault(group["index"], []).append((shift_lo, shift_hi))
                evaluations.append({"group": group["index"], "rollout": rollout, "advantage": advantage,
                                    "agreement": lo if self.canonical_score_mode == "lower" or lo == hi else None,
                                    "sampling_ratio_lower": lo,
                                    "sampling_ratio_upper": hi, "shift_lower": shift_lo, "shift_upper": shift_hi,
                                    "resolved_prefixes": sum(c["sampling_ratio"]["resolved"] for c in comparisons),
                                    "total_prefixes": len(comparisons), "comparisons": comparisons, "result": result})
            candidate["prefix_evaluations"] = evaluations
            if not candidate["valid"]:
                continue
            objective = [fmean(sum(v[side] for v in task_scores.get(g["index"], [])) / self.rollout_count
                               for g in groups) for side in (0, 1)]
            shift = [fmean(fmean(v[side] for v in values) for values in task_shifts.values()) for side in (0, 1)]
            selection = [objective[0] - self.shift_penalty * shift[1],
                         objective[1] - self.shift_penalty * shift[0]]
            surrogate = (fmean(sum(task_lower_scores.get(g["index"], [])) / self.rollout_count for g in groups)
                         if self.canonical_score_mode == "lower" else objective[0])
            score = surrogate - self.shift_penalty * shift[1]
            candidate.update(surrogate=surrogate, surrogate_lower=objective[0], surrogate_upper=objective[1],
                             shift_from_incumbent=shift[1], shift_lower=shift[0], shift_upper=shift[1],
                             selection_score=score, selection_lower=selection[0], selection_upper=selection[1])
            if self.max_policy_shift is not None and shift[1] > self.max_policy_shift:
                candidate["valid"] = False
                candidate.setdefault("errors", []).append("Maximum policy shift bound is not satisfied")

    def _commit(self, artifact: dict) -> dict:
        write_json(self.output / "iterations" / f"iter_{artifact['iteration']:04d}.json", artifact)
        self.states[artifact["state_key"]] = artifact["state_after"]
        self.pending_refresh = set(artifact["pending_refresh_after"])
        self.rng.setstate(ast.literal_eval(artifact["rng_state_after"]))
        self.cursor = artifact["iteration"]
        checkpoint = self.output / "checkpoints" / f"checkpoint_{self.cursor:04d}.json"
        write_json(checkpoint, {"config": self.config, "cursor": self.cursor, "states": self.states})
        self._save()
        return artifact

    def step(self) -> dict:
        if self.cursor >= len(self.order):
            raise StopIteration
        if len(self.bank) != len(self.collection_order):
            raise ValueError("Collect the complete initial rollout bank before optimization")
        iteration, index = self.cursor + 1, self.cursor % len(self.collection_order)
        benchmark, task_id = self.collection_order[index]
        state_key = benchmark if self.mode == "isolated" else "global"
        state = self.states[state_key]
        parent = copy.deepcopy(state["beam"][0])
        path = self.output / "iterations" / f"iter_{iteration:04d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["config_sha256"] != self.config_sha256 or saved["parent_sha256"] != digest(parent):
                raise ValueError("Uncommitted iteration differs from this run or current policy")
            return self._commit(saved)
        group = self._load_group(index)
        artifact = {
            "iteration": iteration, "epoch": self.cursor // len(self.collection_order) + 1,
            "benchmark": benchmark, "task_id": task_id, "state_key": state_key,
            "config_sha256": self.config_sha256, "parent_sha256": digest(parent),
            "reward_group": group["rewards"], "advantages": group["advantages"],
            "group_generation": group["generation"], "group_reference": self.bank[str(index)],
            "reward_metric": group["reward_metric"], "pending_refresh_after": sorted(self.pending_refresh),
            "candidates": [], "train_update_applied": False, "state_after": copy.deepcopy(state),
            "environment_replay_sessions": 0,
        }
        usable = [r for r, trace in enumerate(group["traces"]) if trace["actions"] and group["advantages"][r] != 0]
        if all(a == 0 for a in group["advantages"]):
            artifact["pending_refresh_after"] = sorted(self.pending_refresh | {index})
        if not usable or len({group["rewards"][r] for r in usable}) < 2:
            artifact.update(status="skipped", reason="no_relative_advantage_or_contrasting_prefixes",
                            rng_state_after=repr(self.rng.getstate()))
            return self._commit(artifact)
        best = min(usable, key=lambda r: (-group["rewards"][r], r))
        worst = min(usable, key=lambda r: (group["rewards"][r], r))
        candidates = [{"id": "incumbent", "valid": True, "playbook": parent}]
        with ThreadPoolExecutor(max_workers=min(self.mutation_workers, self.candidate_count)) as pool:
            futures = [pool.submit(self._proposal, c, group, parent, best, worst)
                       for c in range(1, self.candidate_count + 1)]
            candidates.extend(future.result() for future in futures)
        groups = [group, *self._past_groups(index)]
        self._score(iteration, candidates, groups)
        incumbent = candidates[0]
        selected = incumbent
        if incumbent["valid"]:
            for candidate in candidates[1:]:
                if self.canonical_agreement and self.canonical_score_mode == "interval" and candidate.get("valid") and (
                    candidate["selection_lower"] <= incumbent["selection_upper"] + 1e-12
                ):
                    continue
                if candidate["valid"] and candidate["selection_score"] > selected["selection_score"] + 1e-12:
                    selected = candidate
        artifact.update(
            status="completed" if incumbent["valid"] else "skipped",
            reason="selected_by_surrogate" if incumbent["valid"] else "incumbent_evaluation_failed",
            best_rollout=best, worst_rollout=worst, candidates=candidates,
            evaluated_groups=[{"index": g["index"], "benchmark": g["benchmark"], "task_id": g["task_id"],
                               "generation": g["generation"], "group_reference": self.bank[str(g["index"])]}
                              for g in groups],
            selection={"selected": selected["id"], "incumbent_in_pool": True},
            train_update_applied=selected["id"] != "incumbent",
            rng_state_after=repr(self.rng.getstate()),
            prefix_queue=self.policy_shift_evaluator.last_parallelism,
        )
        artifact["state_after"]["beam"] = [copy.deepcopy(selected["playbook"])]
        return self._commit(artifact)

    def run(self, stop_after: int | None = None):
        target = len(self.order) if stop_after is None else stop_after
        if not self.cursor <= target <= len(self.order):
            raise ValueError("Stop must be between the current cursor and total optimization steps")
        if self.canonical_agreement and self.cursor < target:
            self._prepare_action_comparator()
        self.collect_rollouts()
        while self.cursor < target:
            if self.cursor and self.cursor % len(self.collection_order) == 0:
                self.refresh_zero_advantage(self.cursor // len(self.collection_order))
            artifact = self.step()
            selected = artifact.get("selection", {}).get("selected", "incumbent")
            print(f"[{self.cursor}/{len(self.order)}] {artifact['benchmark']}::{artifact['task_id']} "
                  f"{artifact['status']} selected={selected}", flush=True)
        if self.cursor and self.cursor % len(self.collection_order) == 0:
            self.refresh_zero_advantage(self.cursor // len(self.collection_order))
        if self.cursor == len(self.order):
            write_json(self.output / "training_complete.json", {
                "method": self.config["method"], "completed_task_count": self.cursor,
                "initial_rollouts": len(self.collection_order) * self.rollout_count,
                "total_rollouts": sum(map(len, self.bank_history.values())) * self.rollout_count,
                "environment_replay_sessions": 0, "states": self.states,
            })

    def _prepare_action_comparator(self):
        manifest = self.action_comparator.prepare(self.benchmarks)
        path = self.output / "agreement_runtime.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Pinned canonical comparator model/parser runtime changed")
        write_json(path, manifest)
