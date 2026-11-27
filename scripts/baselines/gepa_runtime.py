# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Official GEPA engine with an ordered stream and an Exgentic trajectory adapter.

DSPy's convenience compile() requires a finite metric budget. The pinned official
engine exposes a supported uncapped stop callback and external-system adapter;
use those directly so the accepted unlimited-budget protocol has no sentinel cap.
"""

from __future__ import annotations

import json
import math
import re

from scripts.baselines.runtime import digest, final_test, initial_instruction, prompt, unfence, write


class StreamLoader:
    """Expose only the current batch; the controller owns unrevealed task IDs."""

    def __init__(self, items, batch_size=5):
        self.__controller_items = items
        self.batch_size = batch_size
        self.current_batch = 0
        self.current = items[:batch_size]

    def __len__(self):
        return len(self.current)

    def all_ids(self):
        return [item["task_id"] for item in self.current]

    def fetch(self, ids):
        visible = {item["task_id"]: item for item in self.current}
        return [visible[task_id] for task_id in ids]

    def reveal(self, iteration):
        self.current_batch = iteration
        start = iteration * self.batch_size
        self.current = self.__controller_items[start:start + self.batch_size]
        return self.all_ids()


class OrderedSampler:
    def next_minibatch_ids(self, loader, state):
        return loader.reveal(state.i)


def train(runtime, upstream_dir):
    from gepa import EvaluationBatch, optimize

    config = runtime.config
    loader = StreamLoader(config["train"])
    batches = math.ceil(len(config["train"]) / 5)

    class Adapter:
        evaluations = 0

        def __init__(self):
            self.online_batches = set()

        def evaluate(self, batch, candidate, capture_traces=False):
            learned = candidate["shared"]
            split = batch[0]["split"]
            self.evaluations += 1
            label = f"{split}_evaluation_{self.evaluations:04d}"
            rows = runtime.batch(batch, learned, label)
            if split == "train" and loader.current_batch not in self.online_batches:
                runtime.online(batch, rows, learned, loader.current_batch)
                self.online_batches.add(loader.current_batch)
            # Validation outputs deliberately contain scores only, excluding traces/answers from reflection.
            outputs = rows if split == "train" else [{"hard": row["hard"]} for row in rows]
            return EvaluationBatch(outputs=outputs, scores=[float(row["hard"]) for row in rows],
                                   trajectories=rows if capture_traces and split == "train" else None)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            records = []
            for row in eval_batch.trajectories or []:
                trace = row["trace"]
                records.append({"Inputs": {"task": trace["task"], "tools": trace["public_tool_contract"]},
                                "Generated Outputs": {"trajectory": trace["actions"]},
                                "Feedback": {"success": row["hard"], "evaluation": trace["details"]}})
            return dict.fromkeys(components_to_update, records)

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            proposals = {}
            for name in components_to_update:
                body = prompt("gepa", "reflection/instruction_proposal.txt")
                # Replace placeholders in one pass; never interpret placeholders in task/tool text.
                replacements = {"<curr_instructions>": candidate[name], "<inputs_outputs_feedback>":
                                json.dumps(reflective_dataset[name], ensure_ascii=False)}
                body = re.sub(r"<curr_instructions>|<inputs_outputs_feedback>",
                              lambda match, replacements=replacements: replacements[match.group()], body)
                try:
                    text, _ = runtime.chat("", body, "instruction_proposal")
                    proposals[name] = unfence(text) or candidate[name]
                except Exception as exc:
                    runtime.append("skipped_updates.jsonl", {"batch": loader.current_batch,
                                   "error": f"{type(exc).__name__}: {exc}"})
                    proposals[name] = candidate[name]
            return proposals

    result = optimize(
        seed_candidate={"shared": initial_instruction("gepa", config.get("actor_context", "legacy"))}, trainset=loader,
        valset=config["validation"], adapter=Adapter(), candidate_selection_strategy="pareto",
        skip_perfect_score=True, perfect_score=1.0, batch_sampler=OrderedSampler(),
        use_merge=False, max_metric_calls=None, stop_callbacks=lambda state: state.i + 1 >= batches,
        run_dir=str(upstream_dir), seed=config.get("optimizer_seed", config["seed"]), raise_on_exception=True,
        cache_evaluation=False, display_progress_bar=False,
    )
    learned = result.best_candidate["shared"]
    write(runtime.output / "training_result.json", {
        "entrypoint": "gepa.optimize", "reason": "official uncapped external-system API",
        "best_idx": result.best_idx, "val_aggregate_scores": result.val_aggregate_scores,
        "metric_calls": result.total_metric_calls, "prompt_sha256": digest(learned),
        "unique_train_encounters": len(list((runtime.output / "online").glob("*.json"))),
        "total_training_limit": None,
        "analysis_token_summary": runtime.token_summary(),
    })
    return final_test(runtime, learned)
