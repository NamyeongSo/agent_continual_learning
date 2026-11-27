# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Count observed representative-policy transitions and plot mean updates across seeds."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from exgentic.agents.ace.playbook_utils import update_bullet_counts  # noqa: E402
from scripts.utils import checkpoint_sources  # noqa: E402
from scripts.utils.plot_pairwise_policy_shift_interim import now, write_csv  # noqa: E402
from scripts.utils.policy_shift_metrics import digest, load_json, write_json  # noqa: E402
from scripts.utils.run_add_only_policy_shift import frozen_renderer  # noqa: E402
from scripts.utils.run_checkpoint_policy_shift import initial_content  # noqa: E402

ORDER = ("ace_add", "ace", "premise_add", "premise", "gepa", "skillopt")
LABELS = {
    "ace_add": "ACE\nADD",
    "ace": "ACE\nAUD",
    "premise_add": "PEACE\nADD",
    "premise": "PEACE\nAUD",
    "gepa": "GEPA",
    "skillopt": "SkillOpt",
}
COLORS = {
    "ace_add": "#7753A6",
    "ace": "#7753A6",
    "premise_add": "#2877B6",
    "premise": "#2877B6",
    "gepa": "#D58A21",
    "skillopt": "#2B9472",
}


def content_key(text):
    # Retain entry IDs, ordering and guidance; exclude only usage-counter values and trailing whitespace.
    text = re.sub(r"\bhelpful=\d+\s+harmful=\d+", "helpful=<usage> harmful=<usage>", text)
    return digest("\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip())


class CheckedReader(checkpoint_sources.Reader):
    def __init__(self, hashes):
        super().__init__()
        self.hashes = hashes

    def raw(self, path):
        data = super().raw(path)
        self.hashes[str(Path(path).resolve())] = digest(data)
        return data


def following_actor_policy(config_path, spec, reader):
    """Recover the carried policy from the next recorded acting request after a missing task snapshot."""
    from exgentic.integrations.litellm.trace_logger import iter_expanded_trace_rows

    trace = config_path.parent / "agent/litellm/trace.jsonl"
    if not trace.exists():
        # Native Terminal-Bench actors store their pre-reflection playbook in the framework reflector request.
        trace = config_path.parent / "framework/litellm/trace.jsonl"
        if not trace.exists():
            return None, None
        reader.raw(trace)
        template = Path(spec["snapshot"]) / "src/exgentic/agents/ace/prompts/reflector.py"
        reader.raw(template)
        for row in iter_expanded_trace_rows(trace):
            for message in (row.get("request") or {}).get("messages", []):
                text = message.get("content")
                if not isinstance(text, str) or not text.startswith("You are an expert analyst and educator."):
                    continue
                _, marker, remainder = text.rpartition("**Full Playbook:**\n")
                content, suffix, _ = remainder.partition("\n\n**Answer in this exact JSON format:**")
                if marker and suffix:
                    return content, trace
        return None, None
    reader.raw(trace)
    render = frozen_renderer(Path(spec["snapshot"]), reader.hashes)
    marker = "[str-99999] helpful=0 harmful=0 :: __UPDATE_COUNT_POLICY_MARKER__"
    prefix, _, suffix = render(marker).partition(marker)
    initial = initial_content(spec, reader.hashes)
    for row in iter_expanded_trace_rows(trace):
        request = row.get("request") or {}
        if not request.get("tools"):
            continue
        messages = request.get("messages") or []
        if not messages or messages[0].get("role") != "system":
            continue
        text = messages[0].get("content")
        if not isinstance(text, str):
            continue
        if text == render(initial):
            return initial, trace
        if text.startswith(prefix) and (not suffix or text.endswith(suffix)):
            content = text[len(prefix) : -len(suffix) if suffix else None]
            if render(content) != text:
                raise ValueError("Recovered ACE input playbook does not reproduce the recorded actor prompt")
            return content, trace
    return None, None


def ace_states(spec, reader):
    folder = Path(spec["train_dir"])
    config = reader.json(folder / "experiment_config.json")
    order = config["task_order"]
    metrics = [json.loads(line) for line in reader.raw(folder / "online_metrics.jsonl").decode().splitlines()]
    records = {}
    for path in sorted(folder.glob("*/sessions/*/config.json"), key=lambda p: p.stat().st_mtime_ns):
        metadata = reader.json(path)
        result = path.parent / "results.json"
        if not result.exists() or reader.json(result).get("status") in {"error", "cancelled"}:
            continue
        records[(metadata["benchmark"], str(metadata["task_id"]))] = path
    states = []
    previous = initial_content(spec, reader.hashes)
    for index, metric in enumerate(metrics):
        point = index + 1
        task = order[index]
        if (metric["benchmark_slug"], str(metric["task_id"])) != (task[0], str(task[1])):
            raise ValueError("ACE task order differs from metrics")
        if metric.get("status") in {"error", "cancelled"}:
            states.append(
                {
                    "point": point,
                    "content": previous,
                    "source": str(folder / "online_metrics.jsonl"),
                    "event": "infrastructure_error_no_learning",
                }
            )
            continue
        session_config = records.get((task[0], str(task[1])))
        checkpoint = session_config.parent / "agent/playbook_checkpoint.json" if session_config else None
        if checkpoint and checkpoint.exists():
            content = reader.json(checkpoint)["playbook"]
            source = checkpoint
        else:
            following = records.get(tuple(order[point])) if point < len(order) else None
            if following:
                content = reader.json(following).get("agent_kwargs", {}).get("initial_playbook")
                source = following
                if not isinstance(content, str):
                    content, source = following_actor_policy(following, spec, reader)
            elif point == len(order):
                source = folder / f"playbook_ace_isolated_{spec['benchmark']}.json"
                content = reader.json(source)["playbook"] if source.exists() else None
            else:
                content = None
            if not isinstance(content, str):
                raise ValueError(f"Missing exact post-task ACE policy: {spec['id']}/{point}")
        states.append({"point": point, "content": content, "source": str(source), "event": "completed_task"})
        previous = content
    return states, len(metrics)


def premise_states(spec, reader):
    folder = Path(spec["train_dir"])
    state = reader.json(folder / "state.json")
    initial = initial_content(spec, reader.hashes)
    beam = [initial, initial]
    result = []
    for point, (benchmark, task) in enumerate(state["config"]["order"][: state["cursor"]], 1):
        path = folder / "iterations" / f"iter_{point:04d}.json"
        artifact = reader.json(path)
        if (artifact["iteration"], artifact["benchmark"], artifact["task_id"]) != (point, benchmark, task):
            raise ValueError("PREMiSE order differs from saved iterations")
        if "selection" in artifact:
            candidates = {item["id"]: item for item in artifact["candidates"]}
            selected = [candidates[key] for key in artifact["selection"]["selected"]]
            beam = []
            for candidate in selected:
                assessments = candidate["reflection"].get("entry_assessments", [])
                tags = [
                    {"id": item.get("entry_id", item.get("id")), "tag": item.get("tag", "neutral")}
                    for item in assessments
                ]
                beam.append(update_bullet_counts(candidate["playbook"]["playbook"], tags))
            event = "selected_beam_rank_1"
        elif artifact.get("status") == "skipped" and artifact.get("train_update_applied") is False:
            event = "skipped_no_learning"
        else:
            raise ValueError("PREMiSE iteration lacks selection or explicit unchanged-policy evidence")
        result.append({"point": point, "content": beam[0], "source": str(path), "event": event})
    if beam != [item["playbook"] for item in state["states"][spec["benchmark"]]["beam"]]:
        raise ValueError("PREMiSE reconstruction differs from committed beam")
    return result, state["cursor"]


def count_stream(spec, reference, hashes):
    reader = CheckedReader(hashes)
    initial = initial_content(spec, hashes)
    method = spec["method"]
    if method == "ace":
        states, tasks = ace_states(spec, reader)
    elif method == "premise":
        states, tasks = premise_states(spec, reader)
    else:
        states = checkpoint_sources.checkpoints(spec, reader)
        tasks = max((row["point"] for row in states), default=0)
    if not states:
        raise ValueError(f"No observed policy transitions for {spec['id']}")
    variant = method + ("_add" if method in ("ace", "premise") and spec["playbook_edit_mode"] == "add_only" else "")
    previous = initial
    previous_cp = initial
    changes = full_changes = cp_changes = 0
    events = []
    visited = set()
    while not (reference / "policies" / spec["id"]).exists():
        if str(reference) in visited:
            raise ValueError("Cycle in frozen policy provenance")
        visited.add(str(reference))
        parent = reader.json(reference / "plan.json").get("source_history_bank")
        if not parent:
            raise ValueError(f"Missing frozen policy provenance for {spec['id']}")
        reference = Path(parent)
    reference_files = {
        int(path.stem.removeprefix("cp")): path for path in (reference / "policies" / spec["id"]).glob("cp*.json")
    }
    if 0 not in reference_files:
        raise ValueError("Missing frozen reference CP0")
    if reader.json(reference_files[0])["content"] != initial:
        raise ValueError("Initial policy differs from policy-shift reference")
    verified_cps = 0
    for state in states:
        content, point = state["content"], state["point"]
        changed = content_key(content) != content_key(previous)
        full_changed = digest(content) != digest(previous)
        changes += changed
        full_changes += full_changed
        if point % 5 == 0:
            cp_changes += content_key(content) != content_key(previous_cp)
            previous_cp = content
        if point in reference_files:
            if reader.json(reference_files[point])["content"] != content:
                raise ValueError(f"Policy reconstruction differs from frozen CP: {spec['id']}/{point}")
            verified_cps += 1
        events.append(
            {
                "stream": spec["id"],
                "benchmark": spec["benchmark"],
                "variant": variant,
                "seed": spec["seed"],
                "training_tasks_seen": point,
                "content_changed": int(changed),
                "full_policy_changed": int(full_changed),
                "policy_sha256": digest(content),
                "content_sha256": content_key(content),
                "source": state["source"],
                "event": state.get("event", state.get("selection", "")),
            }
        )
        previous = content
    return {
        "stream": spec["id"],
        "benchmark": spec["benchmark"],
        "variant": variant,
        "seed": spec["seed"],
        "content_updates": changes,
        "full_policy_updates": full_changes,
        "cp5_content_updates": cp_changes,
        "observed_training_events": len(states),
        "training_tasks_seen": tasks,
        "training_tasks_planned": spec["train_total"],
        "training_complete": tasks == spec["train_total"],
        "frozen_checkpoint_matches": verified_cps,
        "train_dir": spec["train_dir"],
    }, events


def plot(rows, aggregates, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.7), sharey=True)
    names = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "Terminal-Bench"}
    lookup = {(row["benchmark"], row["variant"]): row for row in aggregates}
    max_value = max(row["content_updates"] for row in rows)
    upper = max(10, max_value * 1.24)
    for ax, benchmark in zip(axes, names, strict=True):
        for index, variant in enumerate(ORDER):
            group = lookup[(benchmark, variant)]
            ax.bar(
                index,
                group["mean_updates"],
                width=0.67,
                color=COLORS[variant],
                edgecolor="white",
                hatch="//" if variant.endswith("_add") else None,
                alpha=0.87,
                yerr=group["sample_seed_sd"],
                capsize=3,
                error_kw={"linewidth": 1.1, "ecolor": "#333333"},
            )
            seed_rows = sorted(
                (row for row in rows if row["benchmark"] == benchmark and row["variant"] == variant),
                key=lambda row: row["seed"],
            )
            ax.scatter(
                [index + offset for offset in (-0.11, 0, 0.11)],
                [row["content_updates"] for row in seed_rows],
                s=20,
                color="#262626",
                alpha=0.65,
                zorder=3,
            )
            mark = "*" if group["incomplete_runs"] else ""
            ax.text(
                index,
                group["mean_updates"] + group["sample_seed_sd"] + upper * 0.025,
                f"{group['mean_updates']:.1f}{mark}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set(
            title=names[benchmark],
            xticks=range(len(ORDER)),
            xticklabels=[LABELS[key] for key in ORDER],
            ylim=(0, upper),
            xlabel="Method / curator mode",
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Observed policy-content updates per run")
    fig.suptitle("Representative-policy updates during training · mean ± seed SD (seeds 42-44)", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "One content-changing training event = one update; usage-counter-only changes excluded. "
        "Dots: individual seeds. *: includes an unfinished run.\n"
        "ACE/PEACE: per task; GEPA/SkillOpt: per five-task batch (highest-val / validation-gate policy).\n"
        "Training tasks: BFCL/AppWorld up to 50; Terminal-Bench 30.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"mean_policy_update_counts.{extension}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-study", type=Path, required=True)
    parser.add_argument("--add-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, events, hashes = [], [], {}
    for root in (args.baseline_study.resolve(), args.add_study.resolve()):
        prepared = root / "prepared.json"
        hashes[str(prepared)] = digest(prepared.read_bytes())
        for spec in load_json(prepared)["streams"]:
            row, changes = count_stream(spec, root, hashes)
            rows.append(row)
            events.extend(changes)
    expected = {(b, m, s) for b in ("bfcl", "appworld", "terminalbench2") for m in ORDER for s in (42, 43, 44)}
    if len(rows) != len(expected) or {(r["benchmark"], r["variant"], r["seed"]) for r in rows} != expected:
        raise ValueError("Expected exactly 54 unique method/benchmark/seed runs")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["variant"])].append(row)
    aggregates = []
    for (benchmark, variant), group in sorted(groups.items()):
        values = [row["content_updates"] for row in group]
        aggregates.append(
            {
                "benchmark": benchmark,
                "variant": variant,
                "seeds": len(group),
                "mean_updates": statistics.mean(values),
                "sample_seed_sd": statistics.stdev(values),
                "min_updates": min(values),
                "max_updates": max(values),
                "incomplete_runs": sum(not row["training_complete"] for row in group),
            }
        )
    write_csv(output / "per_run_updates.csv", rows)
    write_csv(output / "mean_updates.csv", aggregates)
    write_csv(output / "update_events.csv", events)
    write_json(output / "source_hashes.json", hashes)
    write_json(
        output / "definition.json",
        {
            "generated_at": now(),
            "metric": "Observed representative policy-content transitions per training run",
            "counting": "Compare each completed training event's representative policy with the previous one. "
            "Repeated content counts zero; reverting to an older policy counts as an update. "
            "Exclude helpful/harmful counter values and trailing whitespace; retain entry IDs and policy text.",
            "units": "ACE/PEACE task; GEPA/SkillOpt five-task optimizer batch; individual bullet edits are not counted",
            "aggregation": "Mean and sample standard deviation over seeds42/43/44; "
            "incomplete runs are current observed counts",
            "secondary_counts": "full_policy_updates retains counters; "
            "cp5_content_updates compares only five-task boundaries",
            "scope": "ACE/PEACE ADD/AUD plus GEPA/SkillOpt from original-policy isolated Qwen training runs",
            "source_studies": [str(args.baseline_study.resolve()), str(args.add_study.resolve())],
            "source_script_sha256": digest(Path(__file__).read_bytes()),
            "incomplete_runs": [r for r in rows if not r["training_complete"]],
        },
    )
    plot(rows, aggregates, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "aggregates": aggregates,
                "incomplete": [r for r in rows if not r["training_complete"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
