# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm import (
    ChatCompletionAssistantMessage,
    ChatCompletionSystemMessage,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
)

from ...actor_context import ActorContextMode, shared_actor_prompt, validate_actor_context
from ...core.agent_instance import AgentInstance
from ...core.experiment_limits import ExperimentBudgetExceededError, actor_completion
from ...core.types import (
    Action,
    ActionType,
    Message,
    MessageAction,
    MessageObservation,
    MessagePayload,
    ModelSettings,
    Observation,
)
from ...utils.context_errors import is_context_length_error
from ...utils.cost import LiteLLMCostReport
from ...utils.settings import get_settings
from ..tool_shortlisting import shortlist_tools
from .bulletpoint_analyzer import DEDUP_AVAILABLE, BulletpointAnalyzer
from .playbook_store import PlaybookStore
from .playbook_utils import (
    PlaybookEditMode,
    allowed_playbook_operations,
    apply_playbook_edits,
    extract_json_from_text,
    get_playbook_stats,
    update_bullet_counts,
)
from .prompts.curator import get_curator_prompt
from .prompts.reflector import REFLECTOR_PROMPT_NO_GT, REFLECTOR_PROMPT_TRAINING_TIME, REFLECTOR_PROMPT_WITH_GT

try:
    from ...agents.litellm_tool_calling.utils import ToolsActionsRegistry
except ImportError:
    ToolsActionsRegistry = None

_BULLET_ID_RE = re.compile(r"\[([a-z]{2,5}-\d{5})\]")

settings = get_settings()


class ACEAgentInstance(AgentInstance):
    actor_context: ActorContextMode = "legacy"

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4o",
        curator_model: str = "gpt-4o",
        max_num_rounds: int = 3,
        curator_frequency: int = 1,
        playbook_token_budget: int = 80000,
        shuffle_mode: str = "isolated",
        initial_playbook: Optional[str] = None,
        actor_context: ActorContextMode = "legacy",
        use_json_mode: bool = True,
        training_time: bool = False,
        evaluation_mode: bool = False,
        model_settings: Optional[ModelSettings] = None,
        enable_thinking: Optional[bool] = None,
        allow_shared_thinking: bool = False,
        benchmark_id: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        enable_tool_shortlisting: bool = False,
        max_selected_tools: int = 30,
        temperature_override: float | None = None,
        analysis_temperature_override: float | None = None,
        playbook_edit_mode: PlaybookEditMode = "add_only",
    ) -> None:
        super().__init__(session_id)

        self.model = model
        if actor_context != "legacy" and enable_thinking is True and not allow_shared_thinking:
            raise ValueError("Shared actor context requires enable_thinking=false")
        self.enable_thinking = enable_thinking if actor_context == "legacy" or allow_shared_thinking else False
        self.temperature_override = temperature_override
        self.analysis_temperature_override = analysis_temperature_override
        self.curator_model = curator_model
        self.max_num_rounds = max_num_rounds
        self.curator_frequency = curator_frequency
        self.playbook_token_budget = playbook_token_budget
        allowed_playbook_operations(playbook_edit_mode)
        self.playbook_edit_mode = playbook_edit_mode
        self.shuffle_mode = shuffle_mode
        self.initial_playbook = initial_playbook
        validate_actor_context(actor_context)
        self.actor_context = actor_context
        self.use_json_mode = use_json_mode
        self.training_time = training_time
        self.evaluation_mode = evaluation_mode
        self._training_feedback: Optional[Dict[str, Any]] = None
        self.benchmark_id = benchmark_id
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        self.enable_tool_shortlisting = enable_tool_shortlisting if actor_context == "legacy" else False
        self.max_selected_tools = max_selected_tools

        if model_settings is None:
            self._model_settings = ModelSettings()
        elif isinstance(model_settings, ModelSettings):
            self._model_settings = model_settings
        else:
            self._model_settings = ModelSettings()

        self._cost = LiteLLMCostReport.initialize_empty(model_name=self.model)
        self._store: Optional[PlaybookStore] = None

        self.messages: list[
            Union[
                ChatCompletionAssistantMessage,
                ChatCompletionToolMessage,
                ChatCompletionUserMessage,
                ChatCompletionSystemMessage,
            ]
        ] = []
        self._step_count: int = 0

        self._registry: Optional[ToolsActionsRegistry] = None
        self._all_actions: list[ActionType] = []

        self._observation_log: List[Dict[str, Any]] = []
        self._action_log: List[Dict[str, Any]] = []
        self._terminus_started = False
        self._task_context_error: Optional[Dict[str, str]] = None
        self._active_llm_stage = "generator"

    def _log_failure(
        self, component: str, error: Exception, context: Dict[str, Any]
    ) -> None:
        try:
            log_path = self.paths.agent_dir / "ace_failures.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "component": component,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                **{k: str(v)[:2000] if isinstance(v, str) else v
                   for k, v in context.items()},
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_bullet_usage(self, bullet_ids: List[str]) -> None:
        store = self._store
        if store is None:
            return
        try:
            log_path = self.paths.agent_dir / "bullet_usage_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            from .playbook_utils import extract_playbook_bullets
            bullets_text = extract_playbook_bullets(store.playbook, bullet_ids)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "task_id": str(
                    self.context.get("task_id", "") if self.context else ""
                ),
                "benchmark_id": self.benchmark_id or "",
                "store_id": store.store_id,
                "session_count": store.session_count,
                "bullet_ids_used": bullet_ids,
                "bullet_count": len(bullet_ids),
                "bullets_detail": bullets_text,
                "total_steps": self._step_count,
                "total_observations": len(self._observation_log),
                "total_actions": len(self._action_log),
                "question_preview": (
                    self.task if self.task else ""
                ),
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def start(
        self,
        task: str,
        context: Dict[str, Any],
        actions: list[ActionType],
    ) -> None:
        super().start(task, context, actions)

        self._all_actions = list(self.actions)
        if ToolsActionsRegistry is not None:
            self._registry = ToolsActionsRegistry(self._all_actions)

        task_group = str(
            context.get("task_group")
            or context.get("task_id")
            or context.get("task_name")
            or "default"
        )
        self._store = PlaybookStore.get_or_create(
            shuffle_mode=self.shuffle_mode,
            task_group=task_group,
            initial_playbook=self.initial_playbook,
            benchmark_id=self.benchmark_id,
        )
        if not self.evaluation_mode:
            self._store.increment_session()

        playbook = self._store.playbook
        system_content = self._build_system_prompt(playbook)
        self._add_message(
            ChatCompletionSystemMessage(role="system", content=system_content)
        )

        content_parts: list[Any] = []
        ctx = ""
        if self.context:
            for k, v in self.context.items():
                if isinstance(v, dict) and v.get("type") == "image_url":
                    content_parts.append({"type": "image_url", "image_url": {"url": v["data"], "detail": "high"}})
                else:
                    ctx += f"\n<{k}>\n{v}\n</{k}>"

        text_content = f"{self.task}\n{ctx}"
        if content_parts:
            content_parts.insert(0, {"type": "text", "text": text_content})
            self._add_message(ChatCompletionUserMessage(role="user", content=content_parts))
        else:
            self._add_message(
                ChatCompletionUserMessage(role="user", content=text_content)
            )

        self.logger.info(
            "ACE v2 instance started  store=%s  session_count=%d  "
            "playbook_bullets=%d  benchmark=%s  tools=%d",
            self._store.store_id,
            self._store.session_count,
            get_playbook_stats(playbook)["total_bullets"],
            self.benchmark_id or "(none)",
            len(self._all_actions),
        )

    def react(self, observation: Optional[Observation]) -> Optional[Action]:

        self._step_count += 1
        self._observe(observation)
        self._log_observation(observation)

        if any(action.name == "run_terminus2" for action in self._all_actions):
            if self._terminus_started:
                finish = next((action for action in self._all_actions if action.name == "finish"), None)
                if finish is not None:
                    return finish.build_action({})
                return None
            self._terminus_started = True
            assert self._store is not None
            self._action_log.append({
                "step": self._step_count,
                "action": "run_terminus2",
                "arguments": {
                    "model": self.model,
                    "playbook_bullets": get_playbook_stats(self._store.playbook)["total_bullets"],
                },
            })
            action_type = next(action for action in self._all_actions if action.name == "run_terminus2")
            return action_type.build_action({
                "model": self.model, "playbook": self._store.playbook,
                **({"actor_context": self.actor_context} if self.actor_context != "legacy" else {}),
                **({"temperature": self.temperature_override}
                   if getattr(self, "temperature_override", None) is not None else {}),
            })

        tools = self._assistant_tools()
        if self._task_context_error:
            raise RuntimeError("context length exceeded during generator tool selection")
        response = self._completion(
            model=self.model,
            messages=self.messages,
            tools=tools if tools else None,
        )

        if response is None:
            self.logger.error("ACE v2: LLM returned None response")
            return None

        if response.usage:
            self._cost.update_cost_from_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")

        if finish_reason == "tool_calls" and self._registry is not None:
            tool_calls = self._extract_tool_calls(message)
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                )
            )
            actions = self._registry.tool_calls_to_action(tool_calls)

            for tc in tool_calls:
                self._action_log.append({
                    "step": self._step_count,
                    "action": tc["name"],
                    "arguments": tc["arguments"],
                })

            self.logger.info("ACE v2 step %d: tool_calls=%s", self._step_count,
                             [tc["name"] for tc in tool_calls])
            return actions
        else:
            content = message.content if message.content else ""
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant", content=content
                )
            )
            self._action_log.append({
                "step": self._step_count,
                "action": "message",
                "content": content,
            })
            self.logger.info("ACE v2 step %d: message response", self._step_count)
            return MessageAction(arguments=Message(content=content))

    def receive_training_feedback(self, feedback: Dict[str, Any]) -> None:
        metadata = feedback.get("session_metadata") or {}
        if metadata.get("skip_reason") == "context_length_exceeded":
            self._task_context_error = {"stage": "generator", "error": str(metadata.get("error", ""))}
            return
        if self.training_time:
            self._training_feedback = feedback
            if self.benchmark_id == "terminalbench2":
                usage = (feedback.get("session_metadata") or {}).get("agent_usage") or {}
                self._cost.update_cost_from_tokens(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            self.logger.info(
                "ACE training_time feedback received: score=%s success=%s fields=%s",
                feedback.get("score"),
                feedback.get("success"),
                sorted(feedback),
            )

    def close(self) -> None:
        store = self._store
        if store is None:
            return

        bullet_ids = _BULLET_ID_RE.findall(
            " ".join(
                str(m.get("content", "")) if isinstance(m, dict)
                else str(getattr(m, "content", ""))
                for m in self.messages
                if (isinstance(m, dict) and m.get("role") == "assistant")
                or getattr(m, "role", "") == "assistant"
            )
        )
        self._log_bullet_usage(bullet_ids)

        if self.evaluation_mode:
            # Validation is inference-only: no reflection, curator call,
            # learning event, or checkpoint write may affect later val tasks.
            self._write_task_skip()
            return

        reflection_content = "(empty)"
        previous_playbook, previous_id = store.playbook, store.next_global_id
        if not getattr(self, "_task_context_error", None) and (
            self._observation_log or (self.training_time and self._training_feedback is not None)
        ):
            self._active_llm_stage = "reflector"
            try:
                reflection_content = self._run_post_session_reflection()
            except Exception as exc:
                if is_context_length_error(exc):
                    self._task_context_error = {"stage": "reflector", "error": str(exc)}
                self.logger.warning(
                    "ACE v2: post-session reflection failed: %s", exc
                )
                self._log_failure("reflector", exc, {
                    "observation_count": len(self._observation_log),
                    "action_count": len(self._action_log),
                })

        if not getattr(self, "_task_context_error", None) and store.session_count % self.curator_frequency == 0:
            self._active_llm_stage = "curator"
            try:
                self._run_curator(reflection_content)
            except Exception as exc:
                if is_context_length_error(exc):
                    self._task_context_error = {"stage": "curator", "error": str(exc)}
                self.logger.warning(
                    "ACE v2: curator failed: %s", exc
                )
                self._log_failure("curator", exc, {
                    "reflection_preview": reflection_content[:500],
                })

        skip = getattr(self, "_task_context_error", None)
        if skip:
            store.playbook, store.next_global_id = previous_playbook, previous_id
            self.logger.warning("ACE: skipping current task after %s context length error", skip["stage"])
            self._write_task_skip()

        store.record_learning(
            session_id=self.session_id,
            task_id=str(self.context.get("task_id", "") if self.context else ""),
            was_correct_before=False,
            was_correct_after=False,
            summary=(
                (f"skipped=context_length_exceeded stage={skip['stage']} " if skip else "") +
                f"steps={self._step_count}  "
                f"observations={len(self._observation_log)}  "
                f"actions={len(self._action_log)}"
            ),
            benchmark_id=self.benchmark_id or "",
        )

        try:
            cp = str(self.paths.agent_dir / "playbook_checkpoint.json")
            store.save_checkpoint(cp)
            pb = str(self.paths.agent_dir / "playbook.txt")
            store.save_playbook_text(pb)
        except Exception as exc:
            self.logger.warning("ACE v2: failed to save checkpoint: %s", exc)

    def get_cost(self) -> LiteLLMCostReport:
        return self._cost

    def _write_task_skip(self) -> None:
        skip = getattr(self, "_task_context_error", None)
        if not skip:
            return
        skip_path = self.paths.agent_dir / "task_skip.json"
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        skip_path.write_text(json.dumps({
            "reason": "context_length_exceeded", "stage": skip["stage"], "error": skip["error"],
            "session_id": self.session_id,
            "task_id": str(self.context.get("task_id", "") if self.context else ""),
            "playbook_updated": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_system_prompt(self, playbook: str) -> str:
        if self.actor_context != "legacy":
            return shared_actor_prompt(self.benchmark_id or "", playbook)
        from .prompts.generator import build_generator_system_prompt

        return build_generator_system_prompt(playbook)

    def _add_message(self, message: Any) -> None:
        self.logger.debug("Adding message: role=%s", getattr(message, "role", "?"))
        self.messages.append(message)

    def _observe(self, observation: Optional[Observation]) -> None:
        if self.actor_context != "legacy":
            from ..litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance

            return LiteLLMToolCallingAgentInstance._observe(self, observation)
        if observation is None:
            return

        observations = observation.to_observation_list()
        if observation.is_empty():
            if not any(obs.invoking_actions for obs in observations):
                return

        for obs in observations:
            if isinstance(obs, MessageObservation) and isinstance(
                obs.result, MessagePayload
            ):
                self._add_message(
                    ChatCompletionUserMessage(
                        role="user", content=obs.result.message
                    )
                )
                continue

            if len(obs.invoking_actions) > 0:
                invoking = obs.invoking_actions[0]
                if invoking.name == "message":
                    self._add_message(
                        ChatCompletionUserMessage(
                            role="user", content=str(obs)
                        )
                    )
                    continue

                action_id = invoking.id
                tool_call_id = invoking.id
                if not (
                    isinstance(tool_call_id, str)
                    and tool_call_id.startswith("call_")
                ):
                    if self._registry is not None:
                        tool_call_id = (
                            self._registry.action_id_to_tool_call_id.get(
                                action_id, tool_call_id
                            )
                        )

                value = obs.result
                try:
                    content = json.dumps(
                        value, ensure_ascii=False, separators=(",", ":")
                    )
                except TypeError:
                    content = str(value)

                if tool_call_id is not None:
                    self._add_message(
                        ChatCompletionToolMessage(
                            role="tool",
                            tool_call_id=tool_call_id,
                            content=content,
                        )
                    )
                else:
                    self._add_message(
                        ChatCompletionUserMessage(
                            role="user",
                            content=f"Tool result: {content}",
                        )
                    )
            else:
                self._add_message(
                    ChatCompletionUserMessage(
                        role="user", content=str(obs)
                    )
                )

    def _log_observation(self, observation: Optional[Observation]) -> None:
        if observation is None or observation.is_empty():
            return

        for obs in observation.to_observation_list():
            result = obs.result
            if result is None:
                continue

            entry: Dict[str, Any] = {"step": self._step_count}
            if isinstance(result, str):
                entry["content"] = result
            elif isinstance(result, dict):
                if obs.invoking_actions and obs.invoking_actions[0].name == "run_terminus2":
                    self._cost.update_cost_from_tokens(
                        result.get("input_tokens", 0), result.get("output_tokens", 0)
                    )
                entry["content"] = json.dumps(result, ensure_ascii=False)
            else:
                entry["content"] = str(result)

            if obs.invoking_actions:
                entry["action"] = obs.invoking_actions[0].name

            self._observation_log.append(entry)

    def _assistant_tools(self) -> list[dict[str, Any]]:
        if self._registry is None:
            return []
        tools = self._registry.openai_tools()
        if not self.enable_tool_shortlisting:
            return tools

        def _cost_cb(usage):
            if usage:
                self._cost.update_cost_from_tokens(
                    usage.prompt_tokens, usage.completion_tokens
                )

        return shortlist_tools(
            tools=tools,
            max_selected=self.max_selected_tools,
            messages=self.messages,
            completion_fn=self._completion,
            model=self.model,
            logger=self.logger,
            cost_callback=_cost_cb,
        )

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[dict[str, str]]:
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []
        tool_calls = []
        for tc in message.tool_calls:
            tool_calls.append({
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "id": tc.id,
            })
        return tool_calls

    def _completion(self, **kwargs) -> Any:
        call_kwargs = self._model_settings.model_dump(
            exclude_none=True,
            exclude={"num_retries", "retry_after", "retry_strategy"},
        )
        call_kwargs.update(kwargs)
        if getattr(self, "temperature_override", None) is not None:
            call_kwargs["temperature"] = self.temperature_override
        if self.enable_thinking is not None:
            extra_body = call_kwargs.get("extra_body") or {}
            call_kwargs["extra_body"] = {
                **extra_body,
                "chat_template_kwargs": {
                    **(extra_body.get("chat_template_kwargs") or {}),
                    "enable_thinking": self.enable_thinking,
                },
            }
        if call_kwargs.get("tools") is None:
            call_kwargs.pop("tools", None)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = actor_completion(self, call_kwargs)
                return response
            except ExperimentBudgetExceededError:
                raise
            except Exception as exc:
                if is_context_length_error(exc):
                    self._task_context_error = {"stage": "generator", "error": str(exc)}
                    raise
                self.logger.warning(
                    "ACE LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                if attempt + 1 >= max_attempts:
                    raise
                time.sleep(2 ** attempt)
        return None

    def _llm_call_simple(
        self,
        model: str,
        prompt: str,
        *,
        json_mode: bool = False,
        system_prompt: str | None = None,
    ) -> str:
        kwargs: Dict[str, Any] = self._model_settings.model_dump(
            exclude_none=True,
            exclude={"num_retries", "retry_after", "retry_strategy"},
        )
        kwargs.update({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": getattr(self, "temperature_override", None) or 0.0,
        })
        if getattr(self, "analysis_temperature_override", None) is not None:
            kwargs["temperature"] = self.analysis_temperature_override
        if system_prompt is not None:
            kwargs["messages"].insert(0, {"role": "system", "content": system_prompt})
        if self._model_settings.max_tokens is not None:
            kwargs["max_tokens"] = self._model_settings.max_tokens
        if self._model_settings.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._model_settings.reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.enable_thinking is not None:
            extra_body = kwargs.get("extra_body") or {}
            kwargs["extra_body"] = {
                **extra_body,
                "chat_template_kwargs": {
                    **(extra_body.get("chat_template_kwargs") or {}),
                    "enable_thinking": self.enable_thinking,
                },
            }
        from ...inference_profile import apply_request, profile

        kwargs = apply_request(kwargs, "analysis")
        if profile() or kwargs.get("reasoning_effort") is not None:
            kwargs["allowed_openai_params"] = ["reasoning_effort"]

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = litellm.completion(**kwargs)
                if profile():
                    import os
                    import uuid

                    directory = self.paths.agent_dir / "analysis_calls"
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f"{uuid.uuid4().hex}.json"
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(descriptor, "w") as stream:
                        json.dump({"stage": getattr(self, "_active_llm_stage", "analysis"),
                                   "request": kwargs, "response": resp.model_dump(mode="json")},
                                  stream, ensure_ascii=False, default=str)
                if resp.usage:
                    self._cost.update_cost_from_tokens(
                        resp.usage.prompt_tokens,
                        resp.usage.completion_tokens,
                    )
                if getattr(resp.choices[0], "finish_reason", None) == "length":
                    raise ValueError("Analysis completion truncated")
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned None content")
                return content
            except Exception as exc:
                if is_context_length_error(exc):
                    self._task_context_error = {
                        "stage": getattr(self, "_active_llm_stage", "reflector"), "error": str(exc),
                    }
                    raise
                self.logger.warning(
                    "ACE simple LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                if attempt + 1 >= max_attempts:
                    self._log_failure(
                        "llm_call", exc, {
                            "model": model,
                            "prompt_length": len(prompt),
                            "prompt_preview": prompt[:500],
                            "attempts": max_attempts,
                        },
                    )
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _shared_reflection(self, playbook: str, variant: str) -> str:
        from ...reflection import SYSTEM, ace_tags, build_prompt, validate

        feedback = self._training_feedback or {}
        actions = []
        native = (feedback.get("session_metadata") or {}).get("agent_trace")
        if native:
            for line in native.splitlines():
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(row, dict) and row.get("source", "agent") == "agent":
                    actions.append({"action": {"native_commands": row.get("tool_calls"),
                                               "message": row.get("message")},
                                    "observation": row.get("observation")})
        else:
            calls = {}
            for message in self.messages:
                if message.get("role") == "assistant":
                    group = None
                    for call in message.get("tool_calls") or []:
                        if group is None:
                            group = {"action": {"actions": []}, "observation": []}
                            actions.append(group)
                        arguments = call["function"]["arguments"]
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except ValueError:
                                pass
                        group["action"]["actions"].append({"name": call["function"]["name"], "arguments": arguments})
                        calls[call["id"]] = (group, len(group["observation"]))
                        group["observation"].append(None)
                    if not message.get("tool_calls"):
                        actions.append({"action": {"name": "message", "content": message.get("content")},
                                        "observation": None})
                elif message.get("role") == "tool" and message.get("tool_call_id") in calls:
                    row, index = calls[message["tool_call_id"]]
                    row["observation"][index] = message.get("content")
                elif message.get("role") == "user" and actions:
                    row = actions[-1]
                    row.setdefault("subsequent_user_messages", []).append(message.get("content"))
            for row in actions:
                if isinstance(row["observation"], list) and len(row["observation"]) == 1:
                    row["observation"] = row["observation"][0]
                if "actions" in row["action"] and len(row["action"]["actions"]) == 1:
                    row["action"] = row["action"]["actions"][0]
                if row.get("subsequent_user_messages"):
                    row["observation"] = {"tool_results": row["observation"],
                                          "subsequent_user_messages": row.pop("subsequent_user_messages")}
        if not actions:
            raise ValueError("Shared reflector has no chronological actor decisions")
        from ..premise.llm import public_tool_contract

        definitions = [{"name": item.name, "description": item.description, "is_finish": item.is_finish,
                        "is_hidden": item.is_hidden, "is_message": item.is_message,
                        "arguments_schema": item.arguments.model_json_schema()}
                       for item in getattr(self, "_all_actions", [])]
        observed = []
        for row in actions:
            observed.extend({"action": item} for item in row["action"].get("actions", [row["action"]]))
        trace = {"task": self.task, "context": self.context, "actions": actions,
                 "score": feedback.get("score"), "success": feedback.get("success"), "details": feedback,
                 "public_tool_contract": public_tool_contract(definitions, observed)}
        error = None
        for _ in range(3):
            prompt = build_prompt(trace, playbook, variant=variant, correction=error)
            raw = self._llm_call_simple(self.model, prompt, json_mode=self.use_json_mode, system_prompt=SYSTEM)
            try:
                value = validate(extract_json_from_text(raw), len(actions), playbook)
                break
            except (TypeError, AttributeError, ValueError) as exc:
                error = str(exc)
        else:
            raise ValueError(f"Shared reflection validation failed: {error}")
        self._store.playbook = update_bullet_counts(self._store.playbook, ace_tags(value))
        path = self.paths.agent_dir / "shared_reflection.json"
        path.write_text(json.dumps({"reflection": value, "variant": variant, "actions": actions,
                                    "prompt": prompt, "system_prompt": SYSTEM},
                                   ensure_ascii=False, default=str, indent=2) + "\n")
        return json.dumps(value, ensure_ascii=False)

    def _run_post_session_reflection(self, *, gt_trajectory: str | None = None) -> str:
        store = self._store
        assert store is not None

        playbook = store.playbook
        from ...inference_profile import profile

        active = profile()
        if active and active.get("reflection_variant"):
            return self._shared_reflection(playbook, active["reflection_variant"])
        session_trace = self._build_session_trace()

        prompt_kwargs = {
            "question": self.task if self.task else "",
            "reasoning_trace": session_trace,
            "predicted_answer": "(see session trace above)",
            "bullets_used": playbook,
        }
        if self.training_time and self._training_feedback is not None:
            feedback = {
                key: value for key, value in self._training_feedback.items()
                if key in {
                    "score", "success", "is_finished", "session_metrics",
                    "session_metadata", "evaluation", "summary", "patch", "container",
                }
            }
            if isinstance(feedback.get("session_metadata"), dict):
                feedback["session_metadata"] = {
                    key: value for key, value in feedback["session_metadata"].items()
                    if key != "agent_trace"
                }
            template = REFLECTOR_PROMPT_WITH_GT if gt_trajectory is not None else REFLECTOR_PROMPT_TRAINING_TIME
            prompt = template.format(
                **prompt_kwargs,
                grading_feedback=json.dumps(feedback, ensure_ascii=False, default=str),
                gt_trajectory=gt_trajectory,
            )
        else:
            if gt_trajectory is not None:
                raise ValueError("GT reflection requires training-time grading feedback")
            prompt = REFLECTOR_PROMPT_NO_GT.format(**prompt_kwargs)

        raw = self._llm_call_simple(
            self.model, prompt, json_mode=self.use_json_mode
        )

        bullet_tags: List[Dict] = []
        reflection_text = raw
        parsed = extract_json_from_text(raw)
        if parsed and isinstance(parsed, dict):
            bullet_tags = parsed.get("bullet_tags", [])
            reflection_text = parsed.get("reasoning", raw)
        else:
            self.logger.warning(
                "ACE Reflector: JSON parse failed, raw length=%d", len(raw)
            )
            self._log_failure(
                "reflector_parse", ValueError("JSON parse failed"), {
                    "raw_response_preview": raw[:1000],
                },
            )

        if bullet_tags:
            store.playbook = update_bullet_counts(store.playbook, bullet_tags)

        self.logger.info(
            "ACE post-session reflection: %d bullet tags updated",
            len(bullet_tags),
        )

        return reflection_text

    def _build_session_trace(self) -> str:
        events: List[Dict[str, Any]] = []
        for entry in self._action_log:
            events.append({"type": "action", **entry})
        for entry in self._observation_log:
            events.append({"type": "observation", **entry})

        events.sort(key=lambda e: (e.get("step", 0), 0 if e["type"] == "action" else 1))

        lines: List[str] = []
        for event in events:
            step = event.get("step", "?")
            if event["type"] == "action":
                action = event.get("action", "?")
                args = event.get("arguments", event.get("content", ""))
                lines.append(f"[Step {step}] Action: {action}")
                if args:
                    lines.append(f"  Args: {str(args)}")
            else:
                action = event.get("action", "env")
                content = event.get("content", "")
                lines.append(f"[Step {step}] Observation from {action}:")
                lines.append(f"  {content}")

        if self._training_feedback is not None:
            agent_trace = (self._training_feedback.get("session_metadata") or {}).get("agent_trace")
            if agent_trace:
                lines.append("Terminus-2 trajectory:\n" + agent_trace)
        return "\n".join(lines) if lines else "(No session trace recorded)"

    def _run_curator(self, reflection_content: str) -> None:
        store = self._store
        assert store is not None

        playbook = store.playbook
        stats = get_playbook_stats(playbook)

        question_context = self.task if self.task else ""
        if self.context:
            question_context += "".join(
                f"\n<{k}>\n{v}\n</{k}>"
                for k, v in self.context.items()
            )

        edit_mode = getattr(self, "playbook_edit_mode", "add_only")
        if edit_mode != "add_only":
            question_context += "\nExecution trace:\n" + self._build_session_trace()
            if self.training_time and self._training_feedback is not None:
                feedback = {key: value for key, value in self._training_feedback.items() if key != "session_metadata"}
                metadata = {key: value for key, value in (self._training_feedback.get("session_metadata") or {}).items()
                            if key != "agent_trace"}
                question_context += "\nOfficial post-execution feedback:\n" + json.dumps(
                    {**feedback, "session_metadata": metadata}, ensure_ascii=False, default=str,
                )
        prompt = get_curator_prompt(edit_mode).format(
            token_budget=self.playbook_token_budget,
            current_step=store.session_count,
            total_samples="ongoing",
            playbook_stats=json.dumps(stats, indent=2),
            recent_reflection=reflection_content,
            current_playbook=playbook,
            question_context=question_context,
        )
        raw = self._llm_call_simple(
            self.curator_model, prompt, json_mode=self.use_json_mode
        )

        if raw.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
            self.logger.warning("ACE Curator: skipping due to empty response")
            self._log_failure(
                "curator_empty_response", ValueError("empty LLM response"), {
                    "session_count": store.session_count,
                },
            )
            return

        parsed = extract_json_from_text(raw)
        if edit_mode != "add_only" and isinstance(parsed, list):
            parsed = {"operations": parsed}
        if parsed and isinstance(parsed, dict):
            if "operations" not in parsed or not isinstance(parsed["operations"], list):
                self.logger.warning("ACE Curator: missing or invalid 'operations' field")
                self._log_failure(
                    "curator_schema", ValueError("missing 'operations' list"), {
                        "raw_response_preview": raw[:1000],
                        "parsed_keys": list(parsed.keys()),
                    },
                )
                return

            ops = parsed["operations"]
            new_playbook, new_id, applied = apply_playbook_edits(
                playbook, ops, store.next_global_id, edit_mode=edit_mode,
            )
            if len(applied) != len(ops):
                self._log_failure("curator_operations_ignored", ValueError("Invalid or ineffective playbook edits"), {
                    "playbook_edit_mode": edit_mode, "proposed_operations": ops, "applied_operations": applied,
                })
            if applied:
                store.playbook = new_playbook
                store.next_global_id = new_id
                self.logger.info("ACE Curator: applied %d operations (%s)", len(applied), edit_mode)

            if self.use_bulletpoint_analyzer and DEDUP_AVAILABLE:
                self.logger.info(
                    "ACE BulletpointAnalyzer: running (threshold=%.2f)",
                    self.bulletpoint_analyzer_threshold,
                )
                analyzer = BulletpointAnalyzer(
                    llm_merge_fn=lambda p: self._llm_call_simple(
                        self.curator_model, p, json_mode=False
                    ),
                )
                store.playbook = analyzer.analyze(
                    playbook=store.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True,
                )
        else:
            self.logger.warning("ACE Curator: failed to parse response")
            self._log_failure(
                "curator_parse", ValueError("JSON parse failed"), {
                    "raw_response_preview": raw[:1000],
                },
            )
