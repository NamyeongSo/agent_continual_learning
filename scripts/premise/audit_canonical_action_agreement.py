# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""CPU regression on the frozen manual-review corpus; no actor or environment calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.agents.premise.action_agreement import CanonicalActionAgreement  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    sample_path = args.review / "sample_private.json"
    manifest_path = args.review / "sample_manifest.json"
    frozen_path = args.review / "annotations_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if sha(manifest_path) != frozen["sample_manifest_sha256"]:
        raise ValueError("Frozen review sample changed")
    manifest = json.loads(manifest_path.read_text())
    if sha(sample_path) != manifest["sample_private.json"]:
        raise ValueError("Private sample differs from frozen sample manifest")
    annotation_path = args.review / "annotations.json"
    if sha(annotation_path) != frozen["annotations_sha256"]:
        raise ValueError("Frozen annotations changed")
    sample = json.loads(sample_path.read_text())
    annotations = {r["review_id"]: r for r in json.loads(annotation_path.read_text())}
    previous = {r["review_id"]: r for r in json.loads((args.review / "review_results.json").read_text())}
    if len(sample) != len(previous) or len({s["review_id"] for s in sample}) != len(sample):
        raise ValueError("Review sample/annotation join differs")
    if any(r["review_id"] not in annotations or annotations[r["review_id"]]["label"] !=
           previous[r["review_id"]]["label"] for r in sample):
        raise ValueError("Frozen labels and metric join differ")
    comparator = CanonicalActionAgreement(args.output / "agreement_cache")
    protocol = {"review": str(args.review.resolve()), "samples": len(sample),
                "source_sha256": {p.name: sha(p) for p in (
                    sample_path, manifest_path, frozen_path, annotation_path, args.review / "review_results.json",
                )},
                "comparator": comparator.identity, "external_actor_calls": 0, "environment_execution": False,
                "device": "cpu", "purpose": "development_regression_not_independent_validation"}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Use a separate output for a changed comparator or corpus")
    write(protocol_path, protocol)
    started = time.perf_counter()
    runtime = comparator.prepare(sorted({r["benchmark"] for r in sample}))
    write(args.output / "agreement_runtime.json", runtime)
    load_seconds = time.perf_counter() - started
    cache_files_before = len(list((args.output / "agreement_cache").glob("*.json")))
    results = []
    verified_files = set()
    for row in sample:
        for path, expected in [(row["history"], row["history_sha256"]),
                               *[(d["path"], d["sha256"]) for d in row["draws"]]]:
            if path not in verified_files:
                if sha(path) != expected:
                    raise ValueError("Frozen source file changed")
                verified_files.add(path)
        context = json.loads(Path(row["history"]).read_text())
        old = previous[row["review_id"]]
        started = time.perf_counter()
        scores = {metric: comparator.compare(row["actions_A"], row["actions_B"], benchmark=row["benchmark"],
                                             request=context, metric=metric)
                  for metric in ("canonical_f1", "canonical_accuracy")}
        results.append({"review_id": row["review_id"], "benchmark": row["benchmark"], "label": old["label"],
                        "strict_f1": old["strict_f1"], "strict_accuracy": old["strict_accuracy"],
                        "seconds": time.perf_counter() - started,
                        **{metric: {k: score[k] for k in ("lower", "upper", "reason", "approximate", "resolved")}
                           for metric, score in scores.items()}})
    summaries = {}
    for benchmark in ("all", *sorted({r["benchmark"] for r in results})):
        rows = [r for r in results if benchmark == "all" or r["benchmark"] == benchmark]
        summaries[benchmark] = {"pairs": len(rows), "by_label": {}}
        for label, count in Counter(r["label"] for r in rows).items():
            subset = [r for r in rows if r["label"] == label]
            summaries[benchmark]["by_label"][label] = {
                "pairs": count,
                "resolved_equal": sum(r["canonical_f1"]["lower"] == 1 for r in subset),
                "resolved_different": sum(r["canonical_f1"]["upper"] < 1 for r in subset),
                "point_score": sum(r["canonical_f1"]["resolved"] for r in subset),
                "unresolved": sum(not r["canonical_f1"]["resolved"] for r in subset),
                "approximate": sum(r["canonical_f1"]["approximate"] for r in subset),
            }
    report = {"summaries": summaries, "source_files_verified": len(verified_files),
              "comparison_cache_files_before": cache_files_before,
              "count_definitions": {"resolved_equal": "lower == 1", "resolved_different": "upper < 1",
                                    "point_score": "lower == upper", "unresolved": "lower < upper",
                                    "overlap": "resolved_different can overlap unresolved: magnitude remains unknown"},
              "load_seconds": load_seconds, "scoring_seconds": sum(r["seconds"] for r in results),
              "external_actor_calls": 0, "environment_execution": False,
              "limitations": "Same development corpus used to design comparator; labels are static review, "
                             "not execution-grounded truth. NLI estimates can be wrong; "
                             "intervals are not confidence CIs."}
    write(args.output / "results.json", results)
    write(args.output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
