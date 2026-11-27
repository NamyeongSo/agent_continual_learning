# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Summarize a complete assistant close-reading census of frozen BFCL rules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def labels_for(rows: list[dict], labels: list[dict]) -> dict:
    indexed = {r["audit_id"]: r for r in labels}
    if len(indexed) != len(labels) or set(indexed) != {r["audit_id"] for r in rows}:
        raise ValueError("Every reviewed rule requires exactly one annotation")
    if any(r["scope"] not in {"G", "P", "A", "I"} or not set(r["flags"]) <= {"B", "T"} for r in labels):
        raise ValueError("Unknown semantic coding category or review flag")
    return indexed


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    p = args.input
    added, parents = read(p / "added_rules.json"), read(p / "parent_rules.json")
    labels = labels_for(added, read(p / "manual_added_annotations.json"))
    parent_labels = labels_for(parents, read(p / "manual_parent_annotations.json"))
    candidates = read(p / "candidates.json")
    grouped = defaultdict(list)
    for row in added:
        grouped[row["model"], row["key"], row["candidate"]].append(labels[row["audit_id"]])
    duplicates = defaultdict(list)
    candidate_rows = []
    for row in candidates:
        codes = grouped[row["model"], row["key"], row["candidate"]]
        count = len(codes)
        text = read(Path(row["source"]))["playbook"]["playbook"]
        enriched = {**row, "domain_rules": sum(r["scope"] == "A" for r in codes),
                    "instance_rules": sum(r["scope"] == "I" for r in codes),
                    "protocol_rules": sum(r["scope"] == "P" for r in codes),
                    "general_rules": sum(r["scope"] == "G" for r in codes),
                    "review_flag_B": any("B" in r["flags"] for r in codes),
                    "protocol_flag_T": any("T" in r["flags"] for r in codes),
                    "task_family_fraction": sum(r["scope"] in {"A", "I"} for r in codes) / count if count else None,
                    "playbook_sha256": hashlib.sha256(text.encode()).hexdigest()}
        candidate_rows.append(enriched)
        duplicates[row["model"], row["key"], enriched["playbook_sha256"]].append(enriched)
    duplicate_rows = []
    for (model, key, digest), group in duplicates.items():
        if len(group) > 1:
            duplicate_rows.append({"model": model, "key": key, "playbook_sha256": digest,
                                   "candidates": ",".join(str(r["candidate"]) for r in group), "count": len(group),
                                   "validation40_score_range_pp": float(
                                       100 * np.ptp([r["validation40_score"] for r in group]))})
    model_summaries = []
    for model in ("Qwen", "Gemma"):
        rules = [r for r in added if r["model"] == model]
        cp = [r for r in candidate_rows if r["model"] == model]
        scopes = Counter(labels[r["audit_id"]]["scope"] for r in rules)
        groups = [r for r in duplicate_rows if r["model"] == model]
        model_summaries.append({
            "model": model, "checkpoints": len({r["key"] for r in cp}), "candidates": len(cp),
            "new_rules": len(rules), "scope_counts": {k: scopes[k] for k in "GPAI"},
            "domain_or_instance_pct": 100 * (scopes["A"] + scopes["I"]) / len(rules),
            "B_review_flags": sum("B" in labels[r["audit_id"]]["flags"] for r in rules),
            "T_protocol_flags": sum("T" in labels[r["audit_id"]]["flags"] for r in rules),
            "no_op_candidates": sum(r["no_op"] for r in cp),
            "mean_parent_words": float(np.mean([r["parent_words"] for r in cp])),
            "mean_new_words_with_formatting": float(np.mean([r["candidate_words"] - r["parent_words"] for r in cp])),
            "mean_new_word_fraction": float(np.mean([(r["candidate_words"] - r["parent_words"])
                                                       / r["candidate_words"] for r in cp])),
            "identical_playbook_groups": len(groups),
            "identical_playbook_groups_with_different_val40": sum(r["validation40_score_range_pp"] > 1e-10
                                                                  for r in groups),
            "max_val40_score_range_pp_identical_playbooks": max((r["validation40_score_range_pp"] for r in groups),
                                                                default=None),
        })
    # Same-parent contrasts exclude no-op candidates and require semantic scope variation.
    pairs = []
    cp_groups = defaultdict(list)
    for row in candidate_rows:
        cp_groups[row["model"], row["key"]].append(row)
    for (model, key), group in cp_groups.items():
        narrow = [r for r in group if r["task_family_fraction"] is not None and r["task_family_fraction"] > 0]
        broad = [r for r in group if r["task_family_fraction"] == 0]
        if narrow and broad:
            pairs.append({"model": model, "key": key, "seed": group[0]["seed"],
                          "domain_candidates": len(narrow), "general_or_protocol_candidates": len(broad),
                          "val40_difference_pp": 100 * (np.mean([r["validation40_lift"] for r in narrow])
                                                         - np.mean([r["validation40_lift"] for r in broad]))})
    # Compare concrete inherited/new statements only where they actually co-occur.
    lineages = read(p / "parent_occurrences.json")
    inherited = defaultdict(set)
    for r in lineages:
        inherited[r["model"], r["key"]].add(r["audit_id"])
    tensions = []
    for first, second, note in (
        ("P003", "P007", "Do not retry/correct any verification or tool error vs repair interlocks and retry"),
        ("P095", "P112", "Prefer absolute cd paths vs cd may reject absolute paths"),
        ("P105", "P109", "Priority=4 for standard complaints vs priority=1 for ordinary bug/sync issues"),
    ):
        found = [{"model": m, "key": k} for (m, k), ids in inherited.items() if {first, second} <= ids]
        tensions.append({"first": first, "second": second, "note": note,
                         "cooccurring_model_checkpoints": found,
                         "unique_checkpoints": len({r["key"] for r in found})})
    parent_counts = Counter(r["scope"] for r in parent_labels.values())
    weighted = Counter()
    for r in parents:
        weighted[parent_labels[r["audit_id"]]["scope"]] += len(r["occurrences"])
    result = {"new_model_calls": 0, "new_environment_calls": 0, "models": model_summaries,
              "parent_unique_rule_counts": dict(parent_counts), "parent_rule_occurrence_counts": dict(weighted),
              "parent_rules_shared_between_models": True,
              "cooccurring_policy_tensions": tensions, "within_checkpoint_scope_contrasts": pairs,
              "within_checkpoint_contrast_summary": [
                  {"model": m, "checkpoints": sum(r["model"] == m for r in pairs),
                   "mean_val40_difference_pp_domain_minus_general_or_protocol":
                   float(np.mean([r["val40_difference_pp"] for r in pairs if r["model"] == m]))
                   if any(r["model"] == m for r in pairs) else None} for m in ("Qwen", "Gemma")],
              "interpretation": "G=general, P=benchmark-wide protocol, A=domain/API family, I=instance answer/output "
                                "prescription. A/P are not inherently harmful. B is a review flag, not an error label. "
                                "Single assistant semantic coding; no independent annotator or causal intervention.",
              "source_files": [{"path": str(x.resolve()), "sha256": hashlib.sha256(x.read_bytes()).hexdigest()}
                               for x in (p / "added_rules.json", p / "parent_rules.json",
                                         p / "manual_added_annotations.json", p / "manual_parent_annotations.json")]}
    (p / "audit_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(p / "candidate_scope_outcomes.csv", candidate_rows)
    write_csv(p / "added_rules_labeled.csv", [{**r, **labels[r["audit_id"]],
                                              "flags": ",".join(labels[r["audit_id"]]["flags"])} for r in added])
    write_csv(p / "parent_rules_labeled.csv", [{"audit_id": r["audit_id"], "text": r["text"],
                                               "scope": parent_labels[r["audit_id"]]["scope"],
                                               "flags": ",".join(parent_labels[r["audit_id"]]["flags"]),
                                               "occurrences": len(r["occurrences"]),
                                               "example_source": r["occurrences"][0]["path"]} for r in parents])
    write_csv(p / "identical_context_score_variation.csv", duplicate_rows)
    write_csv(p / "within_checkpoint_scope_contrasts.csv", pairs)
    for row in model_summaries:
        print(json.dumps(row), flush=True)
    print(json.dumps(result["within_checkpoint_contrast_summary"]), flush=True)
    print("Parent policy tensions:", [(r["first"], r["second"], r["unique_checkpoints"]) for r in tensions])


if __name__ == "__main__":
    main()
