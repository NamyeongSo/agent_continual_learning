# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Capture the exact implementation excerpts used by the browser-only guide."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
BASE = "src/exgentic/agents/"
SPECS = {
    "a_mem": {
        "a_mem/a_mem_instance.py": ["close", "_store_session_memories", "_build_system_prompt_with_memories"],
        "a_mem/memory_note.py": ["MemoryNote", "analyze_content"],
        "a_mem/memory_store.py": ["_process_and_add", "_run_evolution", "_consolidate", "find_related_with_neighbors"],
        "a_mem/a_mem_agent.py": ["AMemAgent"],
        "a_mem/prompts.py": ["ANALYZE_CONTENT_PROMPT", "EVOLUTION_DECISION_PROMPT", "STRENGTHEN_DETAILS_PROMPT",
                             "UPDATE_NEIGHBORS_PROMPT", "GENERATE_QUERY_PROMPT"],
    },
    "autoskill": {
        "autoskill/autoskill_instance.py": ["close", "_retrieve_skills"],
        "autoskill/skill_extraction.py": ["extract_skills_from_trace"],
        "autoskill/skill_maintenance.py": ["maintain_skill", "judge_skill", "merge_skills"],
        "autoskill/skill_retrieval.py": ["hybrid_search"],
        "autoskill/skill_store.py": ["bump_version"],
        "autoskill/autoskill_agent.py": ["AutoSkillAgent"],
        "autoskill/prompts.py": ["SKILL_EXTRACTION_PROMPT", "SKILL_JUDGE_PROMPT", "SKILL_MERGE_PROMPT",
                                "QUERY_REWRITE_PROMPT", "SKILL_CONTEXT_TEMPLATE"],
    },
    "harness": {
        "harness/harness_instance.py": ["close", "start"],
        "harness/evolver.py": ["_EvolverToolExecutor", "_run_evolver_loop", "run_evolver"],
        "harness/harness_store.py": ["commit_version", "snapshot", "rollback"],
        "harness/retriever.py": ["retrieve_skills"],
        "harness/harness_agent.py": ["HarnessAgent"],
        "harness/prompts/evolver.py": ["EVOLVER_SYSTEM_PROMPT", "build_evolution_user_message"],
        "harness/prompts/inject.py": ["build_system_message"],
    },
    "reasoning_bank": {
        "reasoning_bank/rb_instance.py": ["close", "_run_post_session_learning", "start"],
        "reasoning_bank/evaluator.py": ["evaluate"],
        "reasoning_bank/induce_memory.py": ["induce_memory", "format_trajectory"],
        "reasoning_bank/rb_store.py": ["MemoryEntry", "add_entry"],
        "reasoning_bank/memory_management.py": ["select_memory", "format_memories_for_prompt"],
        "reasoning_bank/rb_agent.py": ["ReasoningBankAgent"],
        "reasoning_bank/prompts/memory_instruction.py": ["SUCCESSFUL_SI", "FAILED_SI", "MEMORY_INJECTION_INSTRUCTION"],
        "reasoning_bank/prompts/eval_prompts.py": ["build_text_eval_prompt"],
    },
}


def main() -> None:
    methods = {}
    for method, files in SPECS.items():
        excerpts = []
        for relative, names in files.items():
            path = ROOT / BASE / relative
            source = path.read_text()
            tree = ast.parse(source)
            lines = source.splitlines()
            for name in names:
                found = None
                value = None
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                        found = node
                        break
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == name for target in node.targets
                    ):
                        found = node
                        value = ast.literal_eval(node.value)
                        break
                if found is None:
                    raise ValueError(f"Missing source: {relative}:{name}")
                excerpts.append({
                    "name": name,
                    "path": BASE + relative,
                    "line": found.lineno,
                    "code": textwrap.dedent("\n".join(lines[found.lineno - 1:found.end_lineno])),
                    "prompt": value,
                    "file_sha256": hashlib.sha256(source.encode()).hexdigest(),
                })
        methods[method] = excerpts
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Current working-tree source; file hashes identify local modifications independently of HEAD.",
        "license": "Apache-2.0. Google-origin ReasoningBank excerpts retain Google copyright attribution.",
        "methods": methods,
    }
    target = PROJECT / "dist/source-data.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("/* Apache-2.0; see source attribution in the guide. */\nwindow.SOURCE_DATA = "
                      + json.dumps(snapshot, ensure_ascii=False, indent=2) + ";\n")
    print(f"Captured {sum(map(len, methods.values()))} excerpts into {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
