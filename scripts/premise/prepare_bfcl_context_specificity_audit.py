# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Extract evaluated rules for close reading, without automatic semantic labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def entries(text: str) -> dict[str, str]:
    pattern = r"^\[([a-z]+-\d+)\][^\n]*? :: (.*?)(?=\n\[|\n##|\Z)"
    return {m.group(1): m.group(2).strip() for m in re.finditer(pattern, text, re.MULTILINE | re.DOTALL)}


def clean(text: str) -> str:
    prefix = r"^\[[a-z]+-\d+\]\s+helpful=\d+\s+harmful=\d+\s+::\s*"
    while re.match(prefix, text):
        text = re.sub(prefix, "", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source / "summary.json").read_text())
    cache_path = Path(source["source_files"][1]["path"])
    cache = json.loads(cache_path.read_text())
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != source["source_files"][1]["sha256"]:
        raise ValueError("Outcome cache changed")
    added, candidates, parents, lineage = [], [], {}, []
    for model in cache["models"]:
        if model["label"] not in ("Qwen", "Gemma"):
            continue
        for point in sorted(model["points"], key=lambda p: p["key"]):
            if point["key"] not in source["fixed_checkpoint_keys"]:
                continue
            frozen = Path(point["provenance"]["source"]) / "frozen" / point["key"]
            parent_path = frozen / "parent.json"
            parent_blob = parent_path.read_bytes()
            if hashlib.sha256(parent_blob).hexdigest() != point["provenance"]["context_sha256"][0]:
                raise ValueError("Parent differs from evaluated context")
            parent = entries(json.loads(parent_blob)["playbook"]["playbook"])
            for entry_id, text in parent.items():
                canonical = clean(text)
                if canonical not in parents:
                    parents[canonical] = {"audit_id": f"P{len(parents) + 1:03d}", "text": canonical,
                                          "occurrences": []}
                parents[canonical]["occurrences"].append({"model": model["label"], "key": point["key"],
                                                         "entry_id": entry_id, "path": str(parent_path.resolve())})
                lineage.append({"model": model["label"], "key": point["key"], "entry_id": entry_id,
                                "audit_id": parents[canonical]["audit_id"]})
            for i in range(1, 6):
                path = frozen / f"candidate_{i:02d}.json"
                blob = path.read_bytes()
                if hashlib.sha256(blob).hexdigest() != point["provenance"]["context_sha256"][i]:
                    raise ValueError("Candidate differs from evaluated context")
                candidate = json.loads(blob)
                child = entries(candidate["playbook"]["playbook"])
                if any(child.get(k) != text for k, text in parent.items()):
                    raise ValueError("Expected ADD-only preservation of inherited entries")
                new = [(k, text) for k, text in child.items() if k not in parent]
                if len(new) != sum(op["type"] == "ADD" for op in candidate["applied"]):
                    raise ValueError("Applied operation count and actual rule delta differ")
                identity = {"model": model["label"], "key": point["key"], "seed": point["seed"],
                            "checkpoint": point["checkpoint"], "candidate": i}
                for entry_id, text in new:
                    added.append({"audit_id": f"N{len(added) + 1:03d}", **identity, "entry_id": entry_id,
                                  "text": clean(text), "source": str(path.resolve()),
                                  "source_sha256": hashlib.sha256(blob).hexdigest(),
                                  "proposal_task": point["train_order"][point["current_index"]]})
                success = np.asarray(point["success"], dtype=float)
                past = slice(0, point["current_index"])
                val40, val50 = (np.asarray(point["targets"][t])
                                for t in ("validation_remaining40", "full_validation"))
                candidates.append({**identity, "new_rules": len(new), "inherited_rules": len(parent),
                                   "no_op": not new, "source": str(path.resolve()),
                                   "parent_source": str(parent_path.resolve()),
                                   "parent_words": len(json.loads(parent_blob)["playbook"]["playbook"].split()),
                                   "candidate_words": len(candidate["playbook"]["playbook"].split()),
                                   "gain_past": float((success[i, past] - success[0, past]).mean()),
                                   "validation40_score": float(val40[i - 1]),
                                   "validation40_lift": float(val40[i - 1] - val40.mean()),
                                   "validation50_lift": float(val50[i - 1] - val50.mean())})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("added_rules", added), ("parent_rules", list(parents.values())),
                       ("parent_occurrences", lineage), ("candidates", candidates)):
        (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    write_csv(args.output / "candidates.csv", candidates)
    rubric = {
        "G": "Cross-task principle; applies without a particular application or API contract.",
        "P": "Benchmark-wide protocol/completion guidance applying across BFCL task families; "
             "not task-specific within BFCL.",
        "A": "Domain/API/tool-specific but reusable across tasks within that family; not instance memorization.",
        "I": "Instance-specific prescription: a concrete answer, entity, fixed input/output or replayed sequence "
             "is prescribed beyond a clearly optional example.",
        "flags": {"B": "Review flag: potentially overbroad default, unsupported evaluator assumption, "
                       "missing condition or protocol confusion; not a validated harmfulness judgment",
                  "T": "Mentions turn/finish/next-prompt semantics, whether correct or incorrect; "
                       "not an error label"},
        "annotator": "Single Codex assistant close reading of verbatim evaluated rule deltas; "
                     "no external judge or model-server calls; not independent human annotation.",
        "causality": "Descriptive scope coding is separate from causal effect; narrow API knowledge may be useful.",
    }
    (args.output / "rubric.json").write_text(json.dumps(rubric, indent=2) + "\n")
    for name, rows in (("added", added), ("parent", list(parents.values()))):
        batch = 40
        for start in range(0, len(rows), batch):
            lines = []
            for row in rows[start:start + batch]:
                origin = (f"{row['model']} {row['key']} candidate{row['candidate']}" if name == "added"
                          else f"{len(row['occurrences'])} parent occurrences")
                lines.append(f"{row['audit_id']} [{origin}] {row['text']}")
            (args.output / f"review_{name}_{start // batch + 1:02d}.txt").write_text("\n\n".join(lines) + "\n")
    counts = defaultdict(int)
    for row in added:
        counts[row["model"]] += 1
    print(json.dumps({"candidates": len(candidates), "added_rules": dict(counts), "parent_unique": len(parents),
                      "parent_words_unique": sum(len(t.split()) for t in parents),
                      "new_words": sum(len(r["text"].split()) for r in added),
                      "new_model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
