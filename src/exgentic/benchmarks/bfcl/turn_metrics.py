# SPDX-License-Identifier: Apache-2.0
"""Turn-level diagnostic using BFCL's own state and response predicates."""

from __future__ import annotations

import re
import uuid


def turn_metrics(model_turns, ground_truth, entry, category):
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as runtime
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import response_checker, state_checker

    namespace = "exgentic_turn_metric_" + uuid.uuid4().hex
    results, accumulated = [], []
    evaluated = 0
    try:
        for index, gold_calls in enumerate(ground_truth):
            calls = model_turns[index] if index < len(model_turns) else []
            model_outputs = []
            options = {"initial_config": entry["initial_config"], "involved_classes": entry["involved_classes"],
                       "test_entry_id": entry["id"],
                       "long_context": "long_context" in category or "composite" in category,
                       "is_evaL_run": True}
            _, model_instances = runtime.execute_multi_turn_func_call(
                func_call_list=[], model_name=namespace, **options)
            for step in calls:
                outputs, model_instances = runtime.execute_multi_turn_func_call(
                    func_call_list=step, model_name=namespace, **options)
                model_outputs.extend(outputs)
            gold_outputs, gold_instances = runtime.execute_multi_turn_func_call(
                func_call_list=gold_calls, model_name=namespace + "_ground_truth", **options)
            accumulated.extend(model_outputs)
            if not gold_calls:
                results.append({"turn": index, "scored": False, "reason": "official_checker_skips_empty_gold"})
                continue
            evaluated += 1
            state_ok = bool(state_checker(model_instances, gold_instances)["valid"])
            response_ok = bool(response_checker(accumulated, gold_outputs, index)["valid"])
            acted = bool(calls) and not runtime.is_empty_execute_response(calls)
            results.append({"turn": index, "scored": True, "state_pass": state_ok,
                            "response_pass": response_ok, "passed": acted and state_ok and response_ok})
        if not evaluated:
            raise ValueError("BFCL turn metric requires at least one officially scored turn")
        passed = sum(row.get("passed", False) for row in results)
        return {"metric": "bfcl_turn_pass_rate_v1", "score": passed / evaluated,
                "passed": passed, "total": evaluated, "turns": results,
                "definition": "Mean official state-and-response pass over nonempty-gold user turns; "
                              "continue the actual trajectory after earlier failures; skipped gold turns excluded"}
    finally:
        prefix = re.sub(r"[-./:]", "_", namespace)
        for key in list(vars(runtime)):
            if key.startswith(prefix) and key.endswith("_instance"):
                delattr(runtime, key)
