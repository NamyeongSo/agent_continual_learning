# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Exercise official optimizer code offline; never contact an LLM or benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import tempfile


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def gepa_smoke() -> dict:
    import dspy
    import gepa
    from dspy.utils.dummies import DummyLM

    class Answer(dspy.Signature):
        """Return the expected answer."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=float(gold.answer == pred.answer), feedback="Expected ok.")

    def propose(candidate, reflective_dataset, components_to_update):
        return {name: candidate[name] + " Return ok." for name in components_to_update}

    # A local proposer and scripted LM keep even a complete compile call offline.
    with tempfile.TemporaryDirectory(prefix="gepa-smoke-") as folder:
        with dspy.context(lm=DummyLM([{"answer": "ok"} for _ in range(100)])):
            optimizer = dspy.GEPA(
                metric=metric,
                max_metric_calls=8,
                instruction_proposer=propose,
                skip_perfect_score=False,
                use_merge=False,
                num_threads=1,
                track_stats=True,
                log_dir=folder,
            )
            train = [dspy.Example(question=f"train-{index}", answer="ok").with_inputs("question") for index in range(2)]
            val = [dspy.Example(question=f"val-{index}", answer="ok").with_inputs("question") for index in range(2)]
            optimized = optimizer.compile(dspy.Predict(Answer), trainset=train, valset=val)
            details = optimized.detailed_results
            check(details.total_metric_calls > 0, "GEPA did not evaluate the program")
            check(details.val_aggregate_scores[details.best_idx] == 1.0, "Unexpected scripted evaluation score")
    return {
        "method": "gepa",
        "offline": True,
        "versions": {name: importlib.metadata.version(name) for name in ("dspy", "gepa")},
        "source_files": {"dspy": inspect.getfile(dspy.GEPA), "gepa": inspect.getfile(gepa.optimize)},
        "checks": ["official dspy.GEPA.compile", "scripted prediction", "metric calls", "selected result"],
        "metric_calls": details.total_metric_calls,
    }


def skillopt_smoke() -> dict:
    from skillopt.config import flatten_config, load_config
    from skillopt.engine.trainer import ReflACTTrainer
    from skillopt.evaluation.gate import evaluate_gate
    from skillopt.optimizer.skill import apply_patch_with_report

    source = inspect.getfile(ReflACTTrainer)
    from pathlib import Path

    root = Path(source).resolve().parents[2]
    config = flatten_config(load_config(str(root / "configs/_base_/default.yaml")))
    check(config["use_gate"] is True, "Official default gate unexpectedly disabled")
    original = "# Skill\nOld rule.\n<!-- SLOW_UPDATE_START -->\nProtected rule.\n<!-- SLOW_UPDATE_END -->\n"
    updated, reports = apply_patch_with_report(
        original,
        {
            "edits": [
                {"op": "replace", "target": "Old rule.", "content": "Revised rule."},
                {"op": "append", "content": "New rule."},
                {"op": "delete", "target": "Protected rule."},
            ]
        },
    )
    check("Revised rule." in updated and "Old rule." not in updated, "Official replace failed")
    check("New rule." in updated, "Official append failed")
    check("Protected rule." in updated, "Protected region was modified")
    check(reports[-1]["status"] == "skipped_protected_region", "Missing protected-region report")
    same = evaluate_gate(updated, 0.5, original, 0.5, original, 0.5, 0, 1)
    improved = evaluate_gate(updated, 0.75, original, 0.5, original, 0.5, 0, 1)
    check(same.action == "reject" and improved.action == "accept_new_best", "Official strict gate differs")
    return {
        "method": "skillopt",
        "offline": True,
        "versions": {"skillopt": importlib.metadata.version("skillopt")},
        "source_files": {"skillopt": source},
        "checks": [
            "official trainer import",
            "config load",
            "replace/append operations",
            "protected region",
            "strict validation gate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("gepa", "skillopt"))
    args = parser.parse_args()
    result = gepa_smoke() if args.method == "gepa" else skillopt_smoke()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
