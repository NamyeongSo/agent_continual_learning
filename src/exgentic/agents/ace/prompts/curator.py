# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json

from ..playbook_utils import PlaybookEditMode, allowed_playbook_operations

CURATOR_PROMPT_NO_GT = """\
You are a master curator of knowledge. Your job is to identify what new insights \
should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using environment feedback that will NOT be available \
when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id \
in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your reasoning here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}
"""


def get_curator_prompt(edit_mode: PlaybookEditMode = "add_only") -> str:
    """Keep the historical baseline prompt unchanged; enable targeted edits explicitly."""
    operations = allowed_playbook_operations(edit_mode)
    if edit_mode == "add_only":
        return CURATOR_PROMPT_NO_GT
    prompt = CURATOR_PROMPT_NO_GT.split("**Your Task:**", 1)[0]
    replacements = {
        "identify what new insights should be added": "identify what targeted edits should be made",
        "- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook":
            "- Distinguish missing knowledge, wrong guidance, ambiguous procedures, and correct guidance not applied",
        ("- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement "
         "to the existing playbook"):
            "- Avoid redundancy; prefer a coherent UPDATE when the same trigger family already has an entry",
        "- Do NOT regenerate the entire playbook - only provide the additions needed":
            "- Do NOT regenerate the entire playbook; propose only justified targeted edits",
        "- For any operation if no new content to add, return an empty list for the operations field":
            ("- Return an empty operations list when no justified edit remains; "
             "correct guidance ignored is not a new lesson"),
        "- Be concise and specific - each addition should be actionable":
            "- Be concise and specific; each entry needs a clear trigger and a concrete procedure or verification gate",
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    prompt += """
**Evidence-to-edit check:**
Treat the reflection as a claim, not ground truth. Compare it with the available task,
trace, public tool contract, feedback and current playbook. For every edit, put a brief
rationale in the operation: existing coverage (entry IDs or missing), the evidenced gap,
and the concrete before-to-after behavior change. Do not duplicate sufficient guidance
merely because it was ignored. Clarify wording only when you can name the ambiguity.
Preserve task scope, exceptions and valid executable examples. Do not promote hypotheses,
private evaluator answers or task-specific solutions into unconditional rules.

**Entry design:**
Prefer explicit triggers, short procedural stages, concrete checkpoints, and recovery steps.
State WHAT to do, WHEN, and WHY; use ALWAYS / TYPICALLY / WATCH FOR / UNCERTAIN when appropriate.
Generalize without removing actionable anchors. Quality matters more than quantity.
Propose 0-3 operations: usually zero or one, two for independent gaps, three only rarely.
Never force novelty or propose several edits for the same underlying issue.

**Available operations:**
- ADD: create a missing entry, using section and content; IDs are assigned by the system.
- UPDATE: reference an existing entry_id and supply its complete replacement content.
  Rewrite the entry from scratch as one coherent rule integrating old and new evidence;
  do not append a correction paragraph. Preserve unrelated valid guidance and examples.
  The entry keeps its ID, section and helpful/harmful counters. A different trigger family
  belongs in a separate ADD instead. Reference only IDs shown in the current playbook.
"""
    examples = [
        {"type": "ADD", "section": "strategies_and_insights", "content": "New guidance...",
         "rationale": "Missing coverage; evidenced gap; concrete behavior change"},
        {"type": "UPDATE", "entry_id": "<existing ID>", "content": "Complete rewritten entry...",
         "rationale": "Existing entry coverage; evidenced ambiguity; before-to-after change"},
    ]
    if "DELETE" in operations:
        prompt += """- DELETE: reference an existing entry_id to remove a wrong, harmful, redundant or superseded entry.
  Explain why removal preserves useful coverage. Split an overloaded entry using ADD plus
  DELETE only when the replacement preserves its valid scope, exceptions and examples.
  Repeated confusing or contradictory evidence can justify rewriting or replacing an entry.
"""
        examples.append({"type": "DELETE", "entry_id": "<existing ID>",
                         "rationale": "Why this entry is wrong, harmful, redundant or superseded"})
    response = json.dumps({"reasoning": "Brief evidence-based edit summary", "operations": examples}, indent=2)
    prompt += "\n**Your Task:**\nOutput ONLY one valid JSON object (no markdown):\n"
    prompt += response.replace("{", "{{").replace("}", "}}")
    prompt += '\nIf no justified change remains, return {{"reasoning":"No justified edit","operations":[]}}.\n'
    return prompt
