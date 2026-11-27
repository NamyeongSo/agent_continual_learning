# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Refresh the frozen trajectory judge audit without making any model calls."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
import shutil
from datetime import UTC, datetime

import numpy as np

METRICS = ["unsupported_claims", "unnecessary_actions", "repetition", "constraint_violations"]


def render_markdown(audit: dict) -> str:
    total = audit["successful_trace_judgments"]
    full = sum(
        data["all_six_attempts"]["fulfillment"]["distribution"].get("1.0", 0) for data in audit["by_model"].values()
    )
    percentage = 100 * full / total if total else 0
    lines = [
        "# Frozen trajectory judge: quality audit",
        "",
        f"Updated {audit['created_at']}. The audit uses {audit['successful_task_groups']} accepted groups "
        f"({total} attempts). It makes no new model call, changes no prompt or reward, and reads no "
        "full-set train/validation target. Official task labels are joined after judging for these checks.",
        "",
        "| Model | Accepted/planned groups | Complete n=10 CPs | q=1, all attempts | q=1 on failures | Varying q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, data in audit["by_model"].items():
        attempts = data["all_six_attempts"]
        fail = attempts["by_official_success_for_posthoc_audit_only"]["0"]
        saturated = attempts["fulfillment"]["distribution"].get("1.0", 0)
        lines.append(
            f"| {model} | {data['successful_task_groups']}/{data['planned_task_groups']} | "
            f"{data['complete_checkpoint_prefixes_by_n']['10']} | {saturated}/{attempts['trace_count']} | "
            f"{fail['fulfillment_one']}/{fail['count']} | "
            f"{data['candidate_group_variation']['fulfillment_varies']}/{data['successful_task_groups']} |"
        )
    lines += [
        "",
        f"Fulfillment is q=1 for **{full}/{total} attempts ({percentage:.2f}%)**. "
        "These are repeated-task descriptive counts, not independent observations. Public turn "
        "fulfillment is not a calibrated probability of official success.",
        "",
    ]
    for model, data in audit["by_model"].items():
        pairs, attempts = data["within_group_candidate_pairs"], data["all_six_attempts"]
        lines.append(
            f"- {model}: q differs in {pairs.get('same_binary_fulfillment_differs', 0)}/"
            f"{pairs.get('same_binary_label', 0)} candidate pairs tied on official success. "
            f"Material h: {attempts['material_violation']}; unknown attainment "
            f"{attempts['unknown_turns']}/{attempts['shared_turns']} turns; truncated attempts "
            f"{attempts['traces_with_truncation']}; unreached-turn values "
            f"{attempts['unreached_turn_attainment']}."
        )
    lines += ["", "| Nonzero severity | Qwen | Gemma |", "|---|---:|---:|"]
    for metric in METRICS:
        values = []
        for model in ("Qwen", "Gemma"):
            dist = audit["by_model"][model]["all_six_attempts"]["severities"][metric]["distribution"]
            values.append(sum(count for value, count in dist.items() if value != "null" and float(value) > 0))
        lines.append(f"| {metric} | {values[0]} | {values[1]} |")
    lines += [
        "",
        "## Mechanical acceptance and accuracy",
        "",
        "Strict parsing checks schema shape, ranges, complete IDs, shared turn denominators, "
        "evidence presence/index bounds and unknown reasons. It does **not** establish that cited "
        "evidence semantically entails the judgment. Score variation does not itself demonstrate "
        "better held-out selection. Missing judgments remain missing, not zero rewards.",
        "",
        "Judge input excludes official labels, actor/model IDs, private initial state and reference answers. "
        "Six anonymous attempts share a randomized request. Comparison effects and model-family "
        "self-preference remain possible. Frozen v1 omits action call_number but retains step, turn, "
        "name, chronology and observation calls.",
        "",
        "## Independent public toy",
        "",
    ]
    toy = audit["controlled_public_toy"]
    lines += [
        f"Accepted: {toy['accepted']}; status `{toy['status']}`; error: {toy['error']}. {toy['interpretation']}",
        "",
    ]
    for key, value in toy["diagnostic_checks"].items():
        lines.append(f"- {key}: {value}")
    reliability = audit.get("reliability_audit_summary", {})
    lines += ["", "## Reversed-position reliability", ""]
    if reliability:
        lines.append(
            f"Accepted groups {reliability.get('accepted_groups', 0)}; "
            f"q mean absolute difference {reliability.get('fulfillment_mean_absolute_difference')}; "
            f"known h flips {reliability.get('material_known_flips')}. "
            "See reliability_audit/summary.json. Primary judgments are unchanged."
        )
    else:
        lines.append("No completed reversed-position summary is available. A prepared manifest is not a retest.")
    lines += [
        "",
        "## Dated initial-snapshot qualitative appendix",
        "",
        "These cases were hand-inspected in the original 129-group snapshot. Aggregate claims "
        "within case notes refer to that dated snapshot; current aggregate counts appear above.",
        "",
    ]
    for example in audit["qualitative_examples"]:
        lines.append(f"- **{example['group_id']}**: {example['interpretation']}")
    lines += ["", "## Recorded transport evidence", ""]
    for error in audit["errors_by_attempt"]:
        lines.append(f"- {error['status']}: {error['count']} attempts; {error['error']}")
    lines += [
        "",
        "Completion timestamps are not precise server shutdown timestamps. This audit does not "
        "contact or restart a server. Full distributions, timestamp ranges, hashes and event excerpts "
        "are in judge_quality_audit.json. Original 129-group reports are preserved in "
        "judge_quality_audit_snapshot_129_20261021/.",
        "",
    ]
    return "\n".join(lines)


def run(folder: pathlib.Path) -> None:
    base = folder.resolve()
    snapshot_dir = base / "judge_quality_audit_snapshot_129_20261021"
    previous_path = base / "judge_quality_audit.json"
    if previous_path.exists() and json.loads(previous_path.read_text()).get("successful_task_groups") == 129:
        snapshot_dir.mkdir(exist_ok=True)
        for filename in ("judge_quality_audit.json", "judge_quality_audit.md"):
            source, target = base / filename, snapshot_dir / filename
            if source.exists() and not target.exists():
                shutil.copyfile(source, target)
    reliability_path = base / "reliability_audit/summary.json"
    reliability = json.loads(reliability_path.read_text()) if reliability_path.exists() else {}
    groups = {g["id"]: g for g in map(json.loads, (base / "group_features.jsonl").read_text().splitlines())}
    paths = sorted((base / "judge_results").glob("*.json"))
    results = [json.loads(path.read_text()) for path in paths]
    rows = []
    by_group = {}
    for result in results:
        if not result["ok"]:
            continue
        group = groups[result["id"]]
        traces = {trace["context"]: trace for trace in group["traces"]}
        local = []
        for context, judgment in result["result"]["judgments"].items():
            trace = traces[context]
            metrics = judgment["metrics"]
            fulfillment = metrics["fulfillment"]
            requests = {event["turn"] for event in trace["compact_trajectory"]["events"] if event["type"] == "request"}
            row = {
                "group_id": group["id"],
                "model": group["model"],
                "seed": group["seed"],
                "checkpoint": group["checkpoint"],
                "task_id": group["task_id"],
                "context": context,
                "official_success": trace["official_success"],
                "fulfillment_lower_bound": fulfillment["lower_bound"],
                "fulfillment_upper_bound": fulfillment["upper_bound"],
                "unknown_turns": fulfillment["unknown_turns"],
                "shared_turn_count": fulfillment["shared_turn_count"],
                "attainment": judgment["attainment"],
                "unreached_attainment": [a for a in judgment["attainment"] if a["turn"] not in requests],
                "material_violation": metrics["material_violation"]["value"],
                "severities": {metric: metrics[metric]["value"] for metric in METRICS},
                "truncated_fields": trace["compact_trajectory"]["truncated_fields"],
                "lost_evidence": trace["compact_trajectory"]["lost_evidence"],
            }
            local.append(row)
            rows.append(row)
        by_group[group["id"]] = local

    def counts(values):
        return {
            ("null" if key is None else str(key)): value
            for key, value in sorted(
                collections.Counter(values).items(), key=lambda item: (item[0] is None, str(item[0]))
            )
        }

    def descriptive(values):
        finite = [value for value in values if value is not None]
        return {
            "count": len(values),
            "known": len(finite),
            "unknown": len(values) - len(finite),
            "mean": float(np.mean(finite)) if finite else None,
            "std": float(np.std(finite)) if finite else None,
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "distribution": counts(values),
        }

    def summarize(subset):
        by_success = {}
        for success in (0, 1):
            selected = [row for row in subset if row["official_success"] == success]
            by_success[str(success)] = {
                "count": len(selected),
                "fulfillment": descriptive([row["fulfillment_lower_bound"] for row in selected]),
                "fulfillment_nonzero": sum(row["fulfillment_lower_bound"] > 0 for row in selected),
                "fulfillment_one": sum(row["fulfillment_lower_bound"] == 1 for row in selected),
                "material_violation": counts(row["material_violation"] for row in selected),
            }
        return {
            "trace_count": len(subset),
            "fulfillment": descriptive([row["fulfillment_lower_bound"] for row in subset]),
            "material_violation": counts(row["material_violation"] for row in subset),
            "severities": {metric: descriptive([row["severities"][metric] for row in subset]) for metric in METRICS},
            "unknown_turns": sum(row["unknown_turns"] for row in subset),
            "shared_turns": sum(row["shared_turn_count"] for row in subset),
            "unreached_turn_attainment": counts(a["value"] for row in subset for a in row["unreached_attainment"]),
            "traces_with_truncation": sum(row["truncated_fields"] > 0 for row in subset),
            "traces_with_lost_evidence": sum(row["lost_evidence"] for row in subset),
            "by_official_success_for_posthoc_audit_only": by_success,
        }

    models = {}
    for model in ("Qwen", "Gemma"):
        subset = [row for row in rows if row["model"] == model]
        model_results = [result for result in results if result["model"] == model]
        accepted = [result for result in model_results if result["ok"]]
        cp_groups = collections.defaultdict(set)
        variation = collections.Counter()
        pairs = collections.Counter()
        for result in accepted:
            cp_groups[(result["seed"], result["checkpoint"])].add(result["query_position"])
            candidates = [row for row in by_group[result["id"]] if row["context"] != "parent"]
            q = [row["fulfillment_lower_bound"] for row in candidates]
            h = [row["material_violation"] for row in candidates]
            variation["fulfillment_varies"] += len(set(q)) > 1
            variation["material_violation_varies"] += len(set(h)) > 1
            variation["all_five_fulfillment_one"] += all(value == 1 for value in q)
            variation["any_severity_varies"] += any(
                len({row["severities"][metric] for row in candidates}) > 1 for metric in METRICS
            )
            for first, second in itertools.combinations(candidates, 2):
                if first["official_success"] == second["official_success"]:
                    key = "both_fail" if first["official_success"] == 0 else "both_pass"
                    pairs["same_binary_label"] += 1
                    pairs[key] += 1
                    if first["fulfillment_lower_bound"] != second["fulfillment_lower_bound"]:
                        pairs["same_binary_fulfillment_differs"] += 1
                        pairs[key + "_fulfillment_differs"] += 1
                    if first["material_violation"] != second["material_violation"]:
                        pairs["same_binary_material_differs"] += 1
        prefix_coverage = {}
        for n in range(1, 11):
            prefix_coverage[str(n)] = sum(set(range(n)) <= positions for positions in cp_groups.values())
        models[model] = {
            "planned_task_groups": sum(group["model"] == model for group in groups.values()),
            "successful_task_groups": len(accepted),
            "status_counts": dict(collections.Counter(result["status"] for result in model_results)),
            "successful_groups_by_seed": dict(collections.Counter(str(result["seed"]) for result in accepted)),
            "successful_groups_by_checkpoint": {
                f"s{seed}_cp{cp:03d}": len(positions) for (seed, cp), positions in sorted(cp_groups.items())
            },
            "complete_checkpoint_prefixes_by_n": prefix_coverage,
            "all_six_attempts": summarize(subset),
            "five_candidates_only": summarize([row for row in subset if row["context"] != "parent"]),
            "candidate_group_variation": dict(variation),
            "within_group_candidate_pairs": dict(pairs),
        }

    errors = collections.Counter()
    timestamps = collections.defaultdict(list)
    usage = collections.Counter()
    for result in results:
        timestamps[result["status"]].append(result["completed_at"])
        for attempt in result["attempts"]:
            item = attempt["result"]
            if not item["ok"]:
                errors[(item["status"], item.get("error", ""))] += 1
            for key, value in item.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value

    controlled = json.loads((base / "controlled_judge_audit.json").read_text())
    raw_controlled = json.loads(controlled["result"]["raw_content"])
    mapping = controlled["result"]["position_map"]
    controlled_rows = {}
    for item in raw_controlled["items"]:
        key = mapping[item["id"]]["key"]
        controlled_rows[key] = {
            "attainment": item["a"],
            "severities": dict(zip(METRICS, item["v"], strict=True)),
            "material_violation": item["h"],
            "unknown_reasons": item["u"],
        }

    # Hand-inspected, publicly observable examples. No validation outcomes are read.
    example_specs = [
        (
            "qwen_s42_cp011_q02",
            ["candidate_01", "candidate_02", "candidate_03"],
            "All three receive official failure. The judge distinguishes completed public request turns: "
            "candidate_01 completes move, grep and sort (3/4); candidate_02 only the initial move (1/4); "
            "candidate_03 completes move and grep (2/4). Source events directly support this distinction.",
            {"candidate_01": [10, 31, 36], "candidate_02": [12, 25, 29, 41]},
        ),
        (
            "qwen_s42_cp011_q07",
            ["candidate_01", "candidate_02"],
            "The tire-pressure observation is 35 PSI. candidate_01 claims that 35 is above the requested "
            "37 PSI threshold and says no detour is needed; h=1 is supported by this explicit contradiction. "
            "candidate_02 is h=0. In the initial 129-group snapshot, this group contained every positive h.",
            {"candidate_01": [26, 29, 31, 32]},
        ),
        (
            "gemma_s42_cp014_q03",
            ["candidate_01", "candidate_04"],
            "User requests backup_tests. candidate_01 copies files to backup_tests_new instead; the judge "
            "assigns the second turn 0 and constraint severity 1 while keeping material h=0. The finish "
            "accurately names the new folder, so a wrong destination need not imply a false completion claim.",
            {"candidate_01": [5, 16, 19, 21, 32]},
        ),
        (
            "gemma_s43_cp013_q06",
            ["candidate_01", "candidate_04"],
            "Both receive official failure, but q is 1 versus 11/12. candidate_04 posts #TravelGoals in "
            "content with an empty structured tags array. The judge marks that request partially fulfilled. "
            "This exposes sensitivity to API-field interpretation: the visible tweet contains the requested "
            "hashtag, so the half-credit judgment is debatable without a declared semantic requirement.",
            {"candidate_04": [22, 25, 26, 27]},
        ),
    ]
    examples = []
    for group_id, contexts, interpretation, indices in example_specs:
        group = groups[group_id]
        result = next(result for result in results if result["id"] == group_id)
        traces = {trace["context"]: trace for trace in group["traces"]}
        examples.append(
            {
                "group_id": group_id,
                "task_id": group["task_id"],
                "interpretation": interpretation,
                "public_task": group["public_task"],
                "attempts": [
                    {
                        "context": context,
                        "official_success_for_audit_only": traces[context]["official_success"],
                        "judgment": result["result"]["judgments"][context],
                        "selected_observable_events": [
                            {"index": index, **traces[context]["compact_trajectory"]["events"][index]}
                            for index in indices.get(context, [])
                        ],
                    }
                    for context in contexts
                ],
            }
        )

    source_files = [
        base / name
        for name in (
            "group_features.jsonl",
            "judge_preflight.json",
            "controlled_judge_audit.json",
            "judge_audit_notes.md",
        )
    ]
    source_files += [
        pathlib.Path("scripts/premise/bfcl_trajectory_reward_judge.py"),
        pathlib.Path("scripts/premise/run_bfcl_trajectory_reward_judging.py"),
        pathlib.Path(__file__),
    ]
    result_hashes = [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    audit = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Read-only utility and observability audit of frozen v1 judge; no new LLM calls or reward tuning.",
        "scope": "Only training-task trajectories and official train labels for posthoc calibration checks; "
        "no full-train or validation targets read.",
        "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "judge_result_files_count": len(paths),
        "judge_results_sorted_name_digest_sha256": hashlib.sha256(json.dumps(result_hashes).encode()).hexdigest(),
        "successful_task_groups": len(by_group),
        "successful_trace_judgments": len(rows),
        "overall": summarize(rows),
        "by_model": models,
        "errors_by_attempt": [
            {"status": status, "error": error, "count": count} for (status, error), count in errors.most_common()
        ],
        "final_result_timestamp_ranges_utc": {
            status: {"earliest": min(values), "latest": max(values)} for status, values in timestamps.items()
        },
        "usage_from_recorded_attempts_only": dict(usage),
        "controlled_public_toy": {
            "accepted": controlled["result"]["ok"],
            "status": controlled["result"]["status"],
            "error": controlled["result"]["error"],
            "raw_values_for_diagnostic_inspection_only": controlled_rows,
            "diagnostic_checks": {
                "legitimate_complete_fulfillment_one": controlled_rows["legitimate_complete"]["attainment"] == [1, 1],
                "false_completion_material_one": controlled_rows["false_completion"]["material_violation"] == 1,
                "prohibited_deletion_material_one": controlled_rows["prohibited_deletion"]["material_violation"] == 1,
                "needless_repetition_severity_one": controlled_rows["needless_repetition"]["severities"]["repetition"]
                == 1,
                "partial_early_stop_unreached_turn_zero": controlled_rows["partial_early_stop"]["attainment"][1] == 0,
                "unverifiable_material_null_with_reason": (
                    controlled_rows["unverifiable"]["material_violation"] is None
                    and any(reason["field"] == "h" for reason in controlled_rows["unverifiable"]["unknown_reasons"])
                ),
            },
            "interpretation": "Not a passed sanity audit. Raw content separates several easy semantic cases, "
            "but assigns unreached turns null instead of 0 and omits the required h-null reason. "
            "No repair or score import was performed.",
        },
        "qualitative_examples": examples,
        "input_boundary_review": {
            "official_labels_in_prompt": False,
            "basis": "Queue judge_inputs passes only key/events/lost_evidence/truncated_fields. "
            "Builder retains key only in private position_map, and public_task is allowlisted to "
            "user_turns/tool_classes/public tool definitions. Official labels are joined only in this audit.",
            "hidden_initial_state_or_reference_answers_in_prompt": False,
            "full_validation_targets_used_in_audit": False,
            "six_attempt_shared_group": True,
            "position_bias_retest_available": bool(reliability.get("accepted_groups", 0)),
            "call_number_limitation": "v1 EVENT_FIELDS omits action call_number; step, turn, name, event order "
            "and observation calls remain. No v1 input was retroactively changed.",
        },
        "decision": "Report matched coverage, score saturation and reliability. "
        "Mechanical acceptance is not semantic accuracy.",
    }
    audit["reliability_audit_summary"] = reliability
    audit["initial_129_snapshot_directory"] = str(snapshot_dir) if snapshot_dir.exists() else None
    (base / "judge_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")

    (base / "judge_quality_audit.md").write_text(render_markdown(audit))
    print(
        json.dumps(
            {
                "successful_task_groups": len(by_group),
                "successful_trace_judgments": len(rows),
                "report": str(base / "judge_quality_audit.md"),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=pathlib.Path, required=True)
    args = parser.parse_args()
    run(args.folder)


if __name__ == "__main__":
    main()
