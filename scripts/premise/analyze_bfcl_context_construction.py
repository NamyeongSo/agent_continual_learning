# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Audit frozen BFCL parent and ADD candidate contexts without new inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def playbook(record: dict) -> str:
    value = record["playbook"]
    return value["playbook"] if isinstance(value, dict) else value


def entries(text: str) -> dict[str, str]:
    pattern = r"^\[([a-z]+-\d+)\][^\n]*? :: (.*?)(?=\n\[|\n##|\Z)"
    return {match.group(1): match.group(2).strip() for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL)}


def words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def lexical_overlap(added: list[str], inherited: list[str]) -> float | None:
    """Descriptive lexical overlap, not a semantic contradiction or plagiarism detector."""
    if not added or not inherited:
        return None
    vectors = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform([*added, *inherited])
    cross = (vectors[:len(added)] @ vectors[len(added):].T).toarray()
    return float(cross.max(axis=1).mean())


def csv_file(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(raw: dict) -> tuple[dict, list[dict], list[dict]]:
    candidates, checkpoints, parent_hashes = [], [], {}
    for model in raw["models"]:
        for point in model["points"]:
            root = Path(point["source"]) / "frozen" / point["key"]
            parent = playbook(read(root / "parent.json"))
            parent_entries = entries(parent)
            parent_hash = hashlib.sha256(parent.encode()).hexdigest()
            parent_hashes.setdefault(point["key"], {})[model["label"]] = parent_hash
            trace = read(root / "proposal_trace.json")
            protocol_error = trace.get("agent_protocol_error")
            hashes = []
            row_words, row_overlap = [], []
            for index in range(1, 6):
                path = root / f"candidate_{index:02d}.json"
                candidate = read(path)
                text = playbook(candidate)
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                expected_hash = point["context_sha256"][index - 1]
                if actual_hash != expected_hash:
                    raise ValueError(f"Candidate differs from evaluated context: {path}")
                hashes.append(text_hash)
                content = entries(text)
                removed = set(parent_entries) - set(content)
                changed = [key for key in parent_entries if key in content and content[key] != parent_entries[key]]
                new_keys = set(content) - set(parent_entries)
                new_text = [content[key] for key in sorted(new_keys)]
                overlap = lexical_overlap(new_text, list(parent_entries.values()))
                added_words = sum(words(value) for value in new_text)
                row_words.append(added_words)
                if overlap is not None:
                    row_overlap.append(overlap)
                normalized = {re.sub(r"\s+", " ", value).strip().casefold() for value in parent_entries.values()}
                exact_duplicates = sum(re.sub(r"\s+", " ", value).strip().casefold() in normalized
                                       for value in new_text)
                candidates.append({
                    "model": model["label"], "key": point["key"], "seed": point["seed"],
                    "checkpoint": point["checkpoint"], "candidate": index, "path": str(path),
                    "valid": candidate["valid"], "parent_sha256": parent_hash, "context_sha256": actual_hash,
                    "playbook_sha256": text_hash,
                    "parent_words": words(parent), "candidate_words": words(text),
                    "parent_entries": len(parent_entries), "new_entries": len(new_keys), "added_words": added_words,
                    "removed_inherited_entries": len(removed), "changed_inherited_entries": len(changed),
                    "applied_operations": ";".join(operation["type"] for operation in candidate["applied"]),
                    "exact_duplicate_entries": exact_duplicates, "lexical_overlap_with_parent": overlap,
                    "added_protocol_marker": any("<|" in value for value in new_text),
                    "no_op": text_hash == parent_hash, "proposal_score": trace.get("score"),
                    "proposal_protocol_error": bool(protocol_error),
                    "full_train": point["scores"]["full_train"][index - 1],
                    "full_validation": point["scores"]["full_validation"][index - 1],
                })
            checkpoints.append({
                "model": model["label"], "key": point["key"], "seed": point["seed"],
                "checkpoint": point["checkpoint"], "parent_sha256": parent_hash,
                "unique_candidate_contexts": len(set(hashes)), "parent_words": words(parent),
                "parent_entries": len(parent_entries), "mean_added_words": float(np.mean(row_words)),
                "mean_lexical_overlap": float(np.mean(row_overlap)) if row_overlap else None,
                "proposal_score": trace.get("score"), "proposal_actions": len(trace.get("actions", [])),
                "proposal_protocol_error": bool(protocol_error),
                "proposal_protocol_error_message": str(protocol_error) if protocol_error else "",
                "proposal_trace_path": str(root / "proposal_trace.json"),
            })
    models = []
    duplicate_scores = []
    for model in raw["models"]:
        values = [row for row in candidates if row["model"] == model["label"]]
        points = [row for row in checkpoints if row["model"] == model["label"]]
        numeric = ("added_words", "new_entries", "candidate_words", "parent_words", "lexical_overlap_with_parent")
        descriptions = {}
        for key in numeric:
            numbers = np.asarray([row[key] for row in values if row[key] is not None])
            descriptions[key] = ({"mean": float(numbers.mean()), "median": float(np.median(numbers)),
                                  "min": float(numbers.min()), "max": float(numbers.max())} if numbers.size else
                                 {"mean": None, "median": None, "min": None, "max": None})
        groups = {}
        for row in values:
            groups.setdefault((row["key"], row["playbook_sha256"]), []).append(row)
        duplicates = [group for group in groups.values() if len(group) > 1]
        for group in duplicates:
            duplicate_scores.append({
                "model": model["label"], "key": group[0]["key"], "playbook_sha256": group[0]["playbook_sha256"],
                "candidates": ";".join(str(row["candidate"]) for row in group), "count": len(group),
                "no_op": group[0]["no_op"],
                **{f"{name}_range": max(row[name] for row in group) - min(row[name] for row in group)
                   for name in ("full_train", "full_validation")},
            })
        duplicate_metrics = {f"duplicate_context_{name}_range_mean": float(np.mean([
            max(row[name] for row in group) - min(row[name] for row in group) for group in duplicates]))
            if duplicates else None for name in ("full_train", "full_validation")}
        models.append({
            "model": model["label"], "checkpoints": len(points), "candidates": len(values),
            "all_context_hashes_match_evaluated_candidates": True,
            "invalid_candidates": sum(not row["valid"] for row in values),
            "no_op_candidates": sum(row["no_op"] for row in values),
            "changed_inherited_entries": sum(row["changed_inherited_entries"] for row in values),
            "removed_inherited_entries": sum(row["removed_inherited_entries"] for row in values),
            "exact_duplicate_added_entries": sum(row["exact_duplicate_entries"] for row in values),
            "candidates_with_added_protocol_markers": sum(row["added_protocol_marker"] for row in values),
            "proposal_protocol_error_checkpoints": sum(row["proposal_protocol_error"] for row in points),
            "proposal_success_checkpoints": sum(row["proposal_score"] == 1 for row in points),
            "mean_unique_contexts_per_checkpoint": float(np.mean([row["unique_candidate_contexts"] for row in points])),
            "duplicate_context_groups": len(duplicates),
            "candidates_in_duplicate_groups": sum(len(group) for group in duplicates),
            **duplicate_metrics,
            **descriptions,
        })
    return {"models": models, "duplicate_context_scores": duplicate_scores,
            "shared_parent_checkpoints": sum(len(set(value.values())) == 1
                                                                for value in parent_hashes.values()),
            "checkpoint_parent_hashes": parent_hashes,
            "lexical_overlap_definition": "Mean maximum TF-IDF unigram/bigram cosine between each added rule "
            "and inherited parent rules, fit within that candidate. Descriptive text similarity only; "
            "this does not measure semantic redundancy or contradictions."}, candidates, checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, candidates, checkpoints = analyze(read(args.source))
    summary.update({"source": str(args.source.resolve()),
                    "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(), "new_model_calls": 0})
    (args.output / "context_construction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_file(args.output / "candidate_context_metrics.csv", candidates)
    csv_file(args.output / "checkpoint_context_metrics.csv", checkpoints)
    csv_file(args.output / "duplicate_context_scores.csv", summary["duplicate_context_scores"])
    for model in summary["models"]:
        print(json.dumps(model), flush=True)


if __name__ == "__main__":
    main()
