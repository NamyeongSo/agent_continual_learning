# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Prompted reflection, playbook mutation, and replay alignment judging."""

from __future__ import annotations

import json
import os
import threading
from typing import Any


_slots_lock = threading.Lock()
_shared_slots = threading.BoundedSemaphore(32)


def configure_llm_slots(max_calls: int) -> None:
    """Share one in-process limit across acting, reflection, and judging calls."""
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    global _shared_slots
    with _slots_lock:
        _shared_slots = threading.BoundedSemaphore(max_calls)


def shared_llm_slots() -> threading.BoundedSemaphore:
    with _slots_lock:
        return _shared_slots


def parse_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("LLM response has no JSON object")


def render_actions(actions: list[dict[str, Any]]) -> str:
    parts = []
    for index, item in enumerate(actions):
        observation = json.dumps(item.get("observation"), ensure_ascii=False, default=str)
        parts.append(
            f"### action[{index}]\n{json.dumps(item['action'], ensure_ascii=False, default=str)}"
            f"\nObservation: {observation}"
        )
    return "\n\n".join(parts) or "(no actions)"


class PremiseLLM:
    def __init__(self, model: str, base_url: str, max_calls: int = 32):
        self.model = model.removeprefix("openai/")
        self.base_url = base_url
        configure_llm_slots(max_calls)
        self._local = threading.local()

    def _client(self):
        if not hasattr(self._local, "client"):
            from openai import OpenAI
            self._local.client = OpenAI(
                base_url=self.base_url,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                timeout=None,
            )
        return self._local.client

    def ask(self, prompt: str, *, temperature: float = 0.0) -> dict[str, Any]:
        with shared_llm_slots():
            response = self._client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        return parse_json_object(response.choices[0].message.content or "")

    def reflect(self, trace: dict[str, Any], playbook: dict[str, Any], *, correction: str | None = None) -> dict[str, Any]:
        action_count = len(trace["actions"])
        if action_count == 0:
            raise ValueError("Cannot choose a reflection span from an empty action trace")
        prompt = f"""Analyze this completed benchmark task using only the task, recorded actions, observations, and official post-session feedback. Read the entire visible trajectory before choosing exactly one contiguous action span. For a failure choose where a decisive mistake began or recovery was missed; for success choose the decisive correct behavior. The feedback was unavailable to the acting agent: never turn private answers or hidden target IDs into a playbook rule. Distinguish missing guidance, wrong guidance, ambiguous guidance, and correct guidance that was ignored. Do not infer missing data from truncated output.

Benchmark: {trace['benchmark']}
Task: {trace['task']}
Context: {json.dumps(trace['context'], ensure_ascii=False, default=str)}
There are exactly {action_count} recorded actions, indexed 0 through {action_count - 1}.
Actions (0-based):
{render_actions(trace['actions'])}
Official score: {trace['score']}
Official feedback: {json.dumps(trace['details'], ensure_ascii=False, default=str)}
Current playbook: {json.dumps(playbook, ensure_ascii=False)}

Output rules (mandatory):
- Return exactly one JSON object, with no Markdown fences, prose, or trailing text.
- Required keys: "span", "analysis", "expected_behavior", "entry_assessments".
- "span" must be {{"start": integer, "end": integer, "type": "success" or "failure"}}.
- Indices refer ONLY to the displayed action[i] records. Observations and dialogue turns do not get separate indices.
- Choose an existing contiguous action range: 0 <= start < end <= {action_count}. End is exclusive. For a single action at index i, use start=i and end=i+1. Never return -1 or a placeholder span.
- The replay restores actions [0, start) and then asks the agent to continue. Therefore start must be the first action whose behavior needs to change or be reinforced. An error in an action before start cannot be corrected by the replay; move start back to that action. Do not include unrelated setup actions merely to make the span longer.
- If an earlier setup action succeeded and its observation is consistent with the task, exclude it from a failure span. In particular, do not default to start=0 merely because it begins the trajectory.
- For failure, locate the earliest visible causal mistake or missed recovery opportunity, not merely the final failed submission. For success, choose the smallest contiguous sequence that demonstrates the successful decision. Use the action and its observation together as evidence; official grading confirms the outcome but does not create an action index.
- The action named as the first mistake or decisive success in "analysis" MUST be action[start]. If your analysis says action[1] is the first mistake, start must equal 1; a span [0,1) would be wrong because it excludes action[1]. Check this consistency before returning JSON.
- If several spans are plausible, prefer the shortest one that contains the decisive behavior and has enough visible evidence. Never infer a hidden test's expected answer or blame a step without evidence. If an observation is truncated, say what is uncertain in the analysis; do not invent its missing contents.
- "analysis" and "expected_behavior" must be non-empty strings. "entry_assessments" is an array; use [] if no playbook entry is relevant. Each assessment has "entry_id" and "tag" (helpful, harmful, or neutral).
- Keep the explanation concise and use only visible evidence; do not reproduce hidden answers.
{f'Previous output was rejected: {correction}. Correct the JSON and bounds now.' if correction else ''}
Few-shot example A (failure; example only):
  action[0] inspect repository -> observation: relevant file found
  action[1] edit unrelated file -> observation: patch applied
  action[2] submit patch -> observation: submission accepted, official score 0
  The first causal mistake is action[1], not the final submission at action[2]. Output:
  {{"span":{{"start":1,"end":2,"type":"failure"}},"analysis":"The edit targeted a file unrelated to the reported issue; the zero score confirms the attempt failed, not which hidden test failed.","expected_behavior":"Inspect the relevant source file and edit that file before submitting.","entry_assessments":[]}}
Few-shot example B (success; example only):
  action[0] inspect inputs -> observation: requirements found
  action[1] call the matching tool -> observation: requested change made
  action[2] verify result -> observation: matches request, official score 1
  The decisive successful behavior spans actions[1:3]; end=3 is legal for three actions. Output:
  {{"span":{{"start":1,"end":3,"type":"success"}},"analysis":"The tool applied the requested change and the next action verified it.","expected_behavior":"Apply the matching tool to the observed inputs, then verify the result.","entry_assessments":[]}}
Few-shot example C (correct setup followed by wrong target; example only):
  action[0] lookup Alex -> observation: Alex found (correct setup)
  action[1] update Sam -> observation: Sam updated (wrong target)
  action[2] finish -> observation: turn ended, official score 0
  The first wrong behavior is action[1]. The correct action[0] can be replayed unchanged, and [0,1) would contain only that correct action. Output:
  {{"span":{{"start":1,"end":2,"type":"failure"}},"analysis":"Action[1] changed Sam after action[0] found Alex; the selected action is the wrong-target update.","expected_behavior":"Update the contact returned by the lookup, then verify the change.","entry_assessments":[]}}
Now analyze ONLY the real task above. Before answering, internally check that start and end are integer action boundaries and satisfy 0 <= start < end <= {action_count}."""
        value = self.ask(prompt, temperature=0.0)
        span = value.get("span") or {}
        start, end = int(span.get("start", -1)), int(span.get("end", -1))
        if not 0 <= start < end <= len(trace["actions"]):
            raise ValueError(f"Invalid reflection span [{start}, {end})")
        if span.get("type") not in ("success", "failure"):
            raise ValueError(f"Invalid reflection span type: {span.get('type')!r}")
        if not str(value.get("analysis", "")).strip() or not str(value.get("expected_behavior", "")).strip():
            raise ValueError("Reflection lacks analysis or expected_behavior")
        value["span"] = {"start": start, "end": end, "type": span["type"]}
        return value

    def mutate(self, trace: dict[str, Any], playbook: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""You are a playbook curator. Treat the reflection as a claim and compare it with the task, visible action evidence, official feedback, and current playbook. Propose only a justified concrete procedural edit. Do not copy hidden evaluator answers into guidance. Avoid duplicating sufficient guidance merely because it was ignored. A no-change decision is valid. Preserve benchmark scope and exceptions.\n\nBenchmark: {trace['benchmark']}\nTask: {trace['task']}\nOriginal span: {render_actions(trace['actions'][reflection['span']['start']:reflection['span']['end']])}\nFeedback: {json.dumps(trace['details'], ensure_ascii=False, default=str)}\nReflection: {json.dumps(reflection, ensure_ascii=False)}\nPlaybook: {json.dumps(playbook, ensure_ascii=False)}\n\nReturn JSON {{\"operations\":[{{\"op\":\"ADD|UPDATE|DELETE\",\"entry_id\":\"existing id for update/delete\",\"content\":\"actionable advice for add/update\",\"benchmark\":\"{trace['benchmark']} or global\",\"rationale\":\"evidence-to-edit reasoning\"}}],\"mutation_summary\":\"brief summary\"}}. Use an empty operations list if no edit is warranted."""
        prompt += (
            '\n\nOutput exactly one JSON object; no Markdown or commentary. '
            'Required keys are "operations" (an array) and "mutation_summary" '
            '(a brief string). Use {"operations":[],"mutation_summary":"No justified change"} '
            'when no edit is warranted. For each operation, "op" is one of '
            'ADD, UPDATE, DELETE; "benchmark" is exactly the current benchmark '
            'or "global". Do not use a literal pipe-separated list as a value.'
        )
        return self.ask(prompt)

    def judge(self, source: dict[str, Any], reflection: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
        span = reflection["span"]
        prompt = f"""Judge whether the candidate playbook changed the agent's actual replay behavior in the direction stated by the reflection. Judge behavior, not playbook wording or final score. Compare the original span with the continuation from the restored state.\n\nBenchmark: {source['benchmark']}\nTask: {source['task']}\nReflection analysis: {reflection['analysis']}\nExpected behavior: {reflection['expected_behavior']}\nOriginal span: {render_actions(source['actions'][span['start']:span['end']])}\nCandidate continuation: {render_actions(replay['actions'])}\n\nReturn JSON {{\"score\":0|0.5|1,\"label\":\"not_aligned|partial|aligned\",\"rationale\":\"brief reason\"}}. 1 means aligned, 0.5 partial, 0 not aligned."""
        prompt += (
            '\n\nOutput exactly one JSON object; no Markdown or commentary. '
            'Use the numeric score 0, 0.5, or 1 and the matching label '
            'not_aligned, partial, or aligned. Example: '
            '{"score":0.5,"label":"partial","rationale":"brief evidence-based reason"}. '
            'Do not use a literal pipe-separated list as a value.'
        )
        value = self.ask(prompt)
        score = float(value.get("score", 0))
        value["score"] = 1.0 if score >= 0.75 else 0.5 if score >= 0.25 else 0.0
        return value
