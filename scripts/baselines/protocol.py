# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Build a reproducible controller plan and fixed validation pools, without inference.

The resulting plan contains the controller's complete order. Never pass the
controller plan to an optimizer as its trainset: release one batch at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from exgentic.agents.premise.pipeline import manifest_order
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_suffix(".toml")


def validation_pools(manifest: dict, seed: int, quotas: dict[str, int]) -> dict:
    sections = select_task_split_sections(manifest, list(quotas), seed)
    full = {}
    mixed = []
    for benchmark, quota in quotas.items():
        ids = validate_task_split(sections[benchmark], benchmark)["val"]
        if not 0 < quota <= len(ids):
            raise ValueError(f"{benchmark}: validation quota {quota} exceeds the available {len(ids)} tasks")
        full[benchmark] = [{"benchmark": benchmark, "task_id": task_id} for task_id in ids]
        ranked = sorted(
            ids,
            key=lambda task_id: (
                hashlib.sha256(f"baseline-fixed-val-v1\0{seed}\0{benchmark}\0{task_id}".encode()).digest(),
                task_id,
            ),
        )
        mixed.extend({"benchmark": benchmark, "task_id": task_id} for task_id in ranked[:quota])
    return {"interleaved": mixed, "by_benchmark": full}


def build_plan(protocol_path: Path, manifest_path: Path, seed: int, mode: str) -> dict:
    config = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    if mode not in ("isolated", "sequential", "interleaved"):
        raise ValueError(f"Unknown mode: {mode}")
    if seed not in config["seeds"]:
        raise ValueError(f"Seed {seed} is not in the protocol")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    quotas = config["validation"]["interleaved_counts"]
    if set(quotas) != set(config["benchmarks"]) or sum(quotas.values()) != config["validation"]["interleaved_total"]:
        raise ValueError("Validation quotas must cover the benchmarks and sum to interleaved_total")
    pools = validation_pools(manifest, seed, quotas)
    order, benchmark_configs = manifest_order(manifest_path, config["benchmarks"], seed, None, mode)
    return {
        "schema_version": 1,
        "role": "controller_plan_not_optimizer_trainset",
        "method_independent": True,
        "seed": seed,
        "mode": mode,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "validation_sha256": hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest(),
        "unique_encounters": len(order),
        "controller_train_order": [{"benchmark": slug, "task_id": task_id} for slug, task_id in order],
        "manifest_benchmark_configs": benchmark_configs,
        "validation": pools,
        "accepted": {key: value for key, value in config.items() if key != "proposed_not_accepted"},
        "pending_proposals": config["proposed_not_accepted"],
        "runnable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("isolated", "sequential", "interleaved"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New controller plan; refuses to overwrite")
    args = parser.parse_args()
    config = tomllib.loads(args.protocol.read_text(encoding="utf-8"))
    manifest_path = args.manifest or ROOT / config["split_manifest"]
    plan = build_plan(args.protocol, manifest_path, args.seed, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output), "unique_encounters": plan["unique_encounters"],
        "mixed_validation_tasks": len(plan["validation"]["interleaved"]), "runnable": plan["runnable"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
