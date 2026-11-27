# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Rebuild publication tables from compatible reruns and explicitly retained reported baselines."""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

METHODS = (("react", "ReAct"), ("ace", "ACE"), ("gepa", "GEPA"), ("skillopt", "SkillOpt"), ("premise", "PEACE"))
SETTINGS = ("isolated", "sequential", "interleaved")
BENCHMARKS = (("bfcl", "BFCL"), ("appworld", "AppWorld"), ("terminalbench2", "Terminal-Bench"))
METRICS = (("online", "Online"), ("offline", "Offline"), ("offline_val", "Offline-val"))


def rounded(value):
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def ready(score):
    value = score.get("value")
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and 0 <= value <= 1 and score.get("complete") is True and score.get("total", 0) > 0
            and score.get("count") == score["total"] and not score.get("errors"))


def checkpoint_score(checkpoint, split):
    group = checkpoint.get("splits", {}).get(split, {})
    return {"value": group.get("score"), "count": group.get("done", 0), "total": group.get("total", 0),
            "errors": group.get("errors", 0),
            "complete": bool(group.get("total") and group.get("done") == group["total"])}


def early_sample(result, checkpoints, points):
    """Choose using complete validation only; inspect the chosen test split afterwards."""
    by_point = {point: [cp for cp in checkpoints if cp["point"] == point] for point in points}
    if any(len(matches) != 1 for matches in by_point.values()):
        return {**result, "reason": "checkpoint_missing_or_ambiguous"}
    candidates = [matches[0] for matches in by_point.values()]
    if not candidates or any(not ready(checkpoint_score(cp, "val")) for cp in candidates):
        return {**result, "reason": "checkpoint_validation_incomplete"}
    chosen = min(candidates, key=lambda cp: (-checkpoint_score(cp, "val")["value"], cp["point"]))
    score = checkpoint_score(chosen, "test")
    return {**result, "value": score["value"] if ready(score) else None,
            "reason": "completed" if ready(score) else "selected_test_incomplete",
            "count": score["count"], "total": score["total"], "errors": score["errors"],
            "checkpoint": chosen["point"], "checkpoint_id": chosen.get("id"),
            "checkpoint_stream": chosen.get("stream"), "checkpoint_alias": chosen.get("alias"),
            "val_score": checkpoint_score(chosen, "val")["value"], "candidate_points": points}


def cell(samples, expected, *, average=False, allow_partial=False, show_stars=True):
    valid = [sample for sample in samples if sample.get("value") is not None]
    result = {"mean": None, "std": None, "display": "--", "rank": 0, "shade": "white", "color": "#ffffff",
              "ready": len(valid), "expected": expected, "samples": samples, "average": average,
              "provisional": False}
    if len(valid) == expected or (allow_partial and valid):
        values = [sample["value"] * 100 for sample in valid]
        result.update(mean=float(rounded(statistics.mean(values))),
                      std=float(rounded(statistics.stdev(values))) if len(values) > 1 else None,
                      provisional=len(valid) < expected)
        result["display"] = f"{result['mean']:.2f}" + (
            "" if average or result["std"] is None else f" ({result['std']:.2f})") + (
            "*" if result["provisional"] and show_stars else "")
    return result


def reported_cell(mean, std, source, expected, *, average=False):
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           or not 0 <= value <= 100 for value in (mean, std) if value is not None):
        raise ValueError("Reported baseline scores must be finite percentages")
    result = cell([], expected, average=average)
    result.update(mean=float(rounded(mean)), std=float(rounded(std)) if std is not None else None,
                  historical=True, source=source)
    result["display"] = f"{result['mean']:.2f}" + ("" if average else f" ({result['std']:.2f})")
    return result


def compatible(config, batch):
    explicit = {source["batch"]: source for source in config.get("sources", [])}
    source = explicit.get(batch["id"], {})
    if batch["id"] not in explicit:
        if config.get("explicit_sources_only"):
            return False
        try:
            created = datetime.fromisoformat(batch.get("created_at", ""))
            cutoff = datetime.fromisoformat(config.get("auto_discover_after", ""))
            if not created.tzinfo or not cutoff.tzinfo or created < cutoff:
                return False
        except (TypeError, ValueError):
            return False
    return (batch.get("model", "").removeprefix("openai/") == config["model"].removeprefix("openai/")
            and batch.get("temperature") == config.get("temperature", 0.7)
            and batch.get("appworld_policy_sha256") == source.get(
                "appworld_policy_sha256", config.get("appworld_policy_sha256"))
            and (not config.get("inference_profile_sha256") or batch.get(
                "inference_profile_sha256") == config["inference_profile_sha256"])
            and batch.get("playbook_edit_mode", "add_only") == config.get("playbook_edit_mode", "add_only")
            and batch.get("setting", "isolated") in SETTINGS)


def build_table(config, batches, queues):
    seeds = config.get("seeds", [42, 43, 44])
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("A results table needs distinct seeds and sample-standard-deviation support")
    requested = config.get("benchmarks", [key for key, _ in BENCHMARKS])
    if (not isinstance(requested, (list, tuple)) or not requested or len(set(requested)) != len(requested)
            or set(requested) - {key for key, _ in BENCHMARKS}):
        raise ValueError("Results-table benchmarks must be known and distinct")
    benchmarks = tuple((key, label) for key, label in BENCHMARKS if key in requested)
    include_average = config.get("include_average", True)
    early = config.get("offline_early")
    offline_splits = config.get("offline_splits", False)
    test_only = config.get("test_only", False)
    show_stars = config.get("show_star_markers", True)
    if test_only and (offline_splits or early):
        raise ValueError("Test-only tables cannot enable checkpoint split or early-selection columns")
    if offline_splits and not early:
        raise ValueError("Offline split columns need an explicit checkpoint queue and early-selection configuration")
    metrics = (*METRICS[:2], ("offline_early", "Offline (early)"), *METRICS[2:]) if early else METRICS
    if offline_splits:
        metrics = (("offline_train", "Train"), ("offline_val", "Val"),
                   ("offline", "Test"), ("offline_early", "Test (early)"))
    if test_only:
        metrics = (("offline", "Test"),)
    if early:
        if not early.get("checkpoint_queue_id"):
            raise ValueError("Offline (early) needs an explicit checkpoint queue")
        for benchmark, _ in benchmarks:
            points = early.get("points", {}).get(benchmark, [])
            if (0 not in points or any(type(point) is not int or point < 0 for point in points)
                    or len(points) != len(set(points))):
                raise ValueError("Offline (early) needs distinct checkpoint points for each benchmark")
    candidates = {}
    custom_rows = config.get("rows")
    partial_seeds = config.get("show_partial_seeds", False)
    specs = custom_rows or [{"id": f"{method}.{setting}", "method": method, "label": label,
                            "setting": setting, "variant": None}
                           for method, label in METHODS
                           for setting in (("isolated",) if method == "react" else SETTINGS)]
    if len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("Duplicate results-table row IDs")
    source_ids = set()
    explicit = {source["batch"]: source for source in config.get("sources", [])}
    for batch in batches:
        if not compatible(config, batch):
            continue
        allowed_methods = explicit.get(batch["id"], {}).get("methods", [key for key, _ in METHODS])
        allowed_benchmarks = explicit.get(batch["id"], {}).get("benchmarks", requested)
        for run in batch["runs"]:
            if (run["method"] not in allowed_methods or run["benchmark"] not in allowed_benchmarks
                    or run.get("invalidated")
                    or run.get("split_manifest_sha256") != config["split_manifest_sha256"]
                    or (config.get("actor_limits") and run.get("actor_limits") != config["actor_limits"].get(
                        run["benchmark"]))):
                continue
            source_ids.add(batch["id"])
            key = (run["method"], batch.get("setting", "isolated"),
                   run.get("variant") if custom_rows else None, run["benchmark"], run["seed"])
            candidates.setdefault(key, []).append((batch, run))
    checkpoints = {}
    early_checkpoints = {}
    for queue in queues:
        for cp in queue["rows"]:
            key = (cp["variant"], cp["method"], cp["benchmark"], cp["seed"], cp["point"], cp.get("run_variant"))
            checkpoints.setdefault(key, []).append(cp)
            if early and queue.get("id") == early["checkpoint_queue_id"] and cp.get("selected") is not False:
                early_key = (cp["variant"], cp["method"], cp["benchmark"], cp["seed"], cp.get("run_variant"))
                early_checkpoints.setdefault(early_key, []).append(cp)

    def sample(method, setting, variant, benchmark, seed, metric):
        choices = candidates.get((method, setting, variant, benchmark, seed), [])
        result = {"seed": seed, "value": None, "count": 0, "total": 0,
                  "reason": "no_source" if not choices else "ambiguous_sources"}
        if len(choices) != 1:
            return result
        batch, run = choices[0]
        result.update(batch=batch["id"], run=run["id"], reason="incomplete")
        if metric == "offline_early":
            if not run["train"].get("complete"):
                return result
            if variant in early.get("reuse_training_val_best_variants", []):
                if method not in {"gepa", "skillopt"} or run.get("validation_source") != "val":
                    return {**result, "reason": "training_val_selection_unverified"}
                key = (batch["id"], method, benchmark, seed, variant)
                matches = [cp for cp in early_checkpoints.get(key, []) if cp["point"] == run["train"]["total"]]
                if len(matches) != 1 or not run.get("frozen_policy_sha256") or (
                    matches[0].get("policy_sha256") != run["frozen_policy_sha256"]
                ):
                    return {**result, "reason": "policy_unverified"}
                final = matches[0]
                return {**sample(method, setting, variant, benchmark, seed, "offline"),
                        "checkpoint": final["point"], "checkpoint_id": final.get("id"),
                        "checkpoint_stream": final.get("stream"), "checkpoint_alias": final.get("alias"),
                        "val_score": final.get("selected_val_score"),
                        "selection_source": "training_val_best_final_policy", "test_result_reused": True,
                        "checkpoint_queue": early["checkpoint_queue_id"]}
            key = (batch["id"], method, benchmark, seed, variant)
            return {**early_sample(result, early_checkpoints.get(key, []), early["points"][benchmark]),
                    "checkpoint_queue": early["checkpoint_queue_id"]}
        if metric == "offline_val" or (offline_splits and metric in {"offline_train", "offline"}):
            if offline_splits:
                key = (batch["id"], method, benchmark, seed, variant)
                matches = [cp for cp in early_checkpoints.get(key, []) if cp["point"] == run["train"]["total"]]
            else:
                key = (batch["id"], method, benchmark, seed, run["train"]["total"])
                matches = checkpoints.get((*key, variant), checkpoints.get((*key, None), []))
            if not run["train"].get("complete") or len(matches) != 1:
                return result
            cp = matches[0]
            if not run.get("frozen_policy_sha256") or cp.get("policy_sha256") != run["frozen_policy_sha256"]:
                result["reason"] = "policy_unverified"
                return result
            split = {"offline_train": "train", "offline_val": "val", "offline": "test"}[metric]
            score = checkpoint_score(cp, split)
            result.update(checkpoint=cp["point"], checkpoint_id=cp.get("id"),
                          checkpoint_stream=cp.get("stream"), checkpoint_alias=cp.get("alias"))
            if offline_splits:
                result.update(checkpoint_queue=early["checkpoint_queue_id"], evaluation_split=split)
        else:
            name = "train" if metric == "online" else "test"
            score = dict(run.get("scores", {}).get(name, {}))
            if not run[name].get("complete") or (metric == "offline" and not run.get("frozen_policy_sha256")):
                score["complete"] = False
        result.update(count=score.get("count", 0), total=score.get("total", 0), errors=score.get("errors", 0))
        if ready(score):
            result.update(value=score["value"], reason="completed")
        return result

    rows = []
    reported = config.get("reported_react_offline")
    for spec in specs:
        method, label, setting = spec["method"], spec["label"], spec.get("setting", "isolated")
        variant = spec.get("variant")
        cells = {}
        for benchmark, _ in benchmarks:
            for metric, _ in metrics:
                cells[f"{benchmark}.{metric}"] = cell(
                    [sample(method, setting, variant, benchmark, seed, metric) for seed in seeds],
                    len(seeds), allow_partial=partial_seeds, show_stars=show_stars)
            if method == "react" and reported:
                score = reported["benchmarks"][benchmark]
                cells[f"{benchmark}.offline"] = reported_cell(
                    score["mean"], score["std"], reported["source"], len(seeds))
        for metric, _ in (metrics if include_average else ()):
            components = [cells[f"{benchmark}.{metric}"] for benchmark, _ in benchmarks]
            samples = []
            historical = method == "react" and metric == "offline" and reported
            if (partial_seeds or all(component["mean"] is not None for component in components)) and not historical:
                samples = [{"seed": seed, "value": statistics.mean(
                    component["samples"][index]["value"] for component in components)}
                           for index, seed in enumerate(seeds)
                           if all(component["samples"][index]["value"] is not None for component in components)]
            average = cell(samples, len(seeds), average=True, allow_partial=partial_seeds, show_stars=show_stars)
            if historical:
                average = reported_cell(statistics.mean(component["mean"] for component in components),
                                        None, reported["source"], len(seeds), average=True)
            average["benchmarks_ready"] = sum(component["mean"] is not None for component in components)
            cells[f"avg.{metric}"] = average
        rows.append({"id": spec["id"], "method": method, "label": label, "setting": setting,
                     "setting_label": spec.get("setting_label", setting.title()),
                     "variant": variant, "cells": cells})
    groups = (*benchmarks, ("avg", "Avg.")) if include_average else benchmarks
    columns = [{"id": f"{benchmark}.{metric}", "benchmark": title, "metric": heading}
               for benchmark, title in groups for metric, heading in metrics]
    for column in columns:
        values = sorted({row["cells"][column["id"]]["mean"] for row in rows
                         if row["cells"][column["id"]]["mean"] is not None}, reverse=True)
        for row in rows:
            value = row["cells"][column["id"]]
            if value["mean"] is None:
                continue
            index = values.index(value["mean"])
            value["rank"] = index + 1 if index < 2 else 0
            if len(values) > 1:
                normalized = (value["mean"] - values[-1]) / (values[0] - values[-1])
                amount = int(Decimal(str(35 * abs(normalized - 0.5) * 2)).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP))
                name, rgb = ("scorered", (183, 62, 41)) if normalized < 0.5 else ("scoreblue", (7, 92, 173))
                value["shade"] = f"{name}!{amount}!white"
                value["color"] = "#" + "".join(f"{round(255 + (channel - 255) * amount / 100):02x}" for channel in rgb)
    return {"id": config["id"], "title": config["title"], "model": config["model"], "seeds": seeds,
            "priority": config.get("priority", 0), "row_layout": "variants" if custom_rows else "methods",
            "partial_seeds": partial_seeds, "benchmark_count": len(benchmarks), "include_average": include_average,
            "offline_splits": offline_splits, "show_star_markers": show_stars,
            "test_only": test_only, "inference_profile_label": config.get("inference_profile_label"),
            "environment_policy_label": config.get("environment_policy_label"),
            "appworld_metric": "appworld_tgc_v1",
            "playbook_edit_mode": config.get("playbook_edit_mode", "add_only"),
            "journal_id": config.get("journal_id"), "sources": sorted(source_ids), "columns": columns, "rows": rows,
            "reported_react_offline": bool(reported),
            "offline_early": early,
            "displayed_cells": sum(c["mean"] is not None for row in rows for c in row["cells"].values()),
            "provisional_cells": sum(c["provisional"] for row in rows for c in row["cells"].values()),
            "completed_cells": sum(c["mean"] is not None and not c["provisional"]
                                   for row in rows for c in row["cells"].values())}


def latex(table, *, standalone=True):
    def escape(text):
        mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                   "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        return "".join(mapping.get(char, char) for char in str(text))

    groups = list(dict.fromkeys(column["benchmark"] for column in table["columns"]))
    widths = [sum(column["benchmark"] == group for column in table["columns"]) for group in groups]
    rules, start = [], 3
    for width in widths:
        rules.append(r"\cmidrule(lr){" + str(start) + "-" + str(start + width - 1) + "}")
        start += width
    scope = "across three benchmarks. " if table.get("benchmark_count", 3) == 3 else (
        "on " + ", ".join(escape(label) for label in groups if label != "Avg.") + ". ")
    average_caption = "Avg. is the benchmark macro-average. " if table.get("include_average", True) else ""
    early_heading = "Test (early)" if table.get("offline_splits") else "Offline (early)"
    lines = [r"\documentclass[twocolumn]{article}", r"\usepackage[table]{xcolor}",
             r"\usepackage{booktabs,multirow,adjustbox}", r"\definecolor{oursbg}{RGB}{241,246,252}",
             r"\definecolor{scorered}{HTML}{B73E29}", r"\definecolor{scoreblue}{HTML}{075CAD}",
             r"\begin{document}", r"\begin{table*}[t]\centering\small",
             r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.18}",
             r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{@{}ll*{" + str(len(table["columns"])) + r"}{c}@{}}\toprule",
             r"\textbf{Method} & \textbf{Setting} & " + " & ".join(
                 r"\multicolumn{" + str(width) + r"}{c}{\textbf{" + label + "}}"
                 for label, width in zip(groups, widths, strict=True)) + r" \\",
             "".join(rules),
             " & & " + " & ".join(column["metric"] for column in table["columns"]) + r" \\\midrule"]
    for row in table["rows"]:
        if row["setting"] == "isolated" and row["method"] != "react":
            lines.append(r"\midrule")
        prefix = r"\rowcolor{oursbg}" if row["method"] == "premise" else ""
        if table.get("row_layout") == "variants":
            method = r"\textbf{" + escape(row["label"]) + "}"
        elif row["method"] == "react":
            method = r"\textbf{ReAct}"
        elif row["method"] == "premise":
            method = r"\multirow{-3}{*}{\textbf{PEACE}}" if row["setting"] == "interleaved" else ""
        else:
            method = r"\multirow{3}{*}{\textbf{" + escape(row["label"]) + "}}" if row["setting"] == "isolated" else ""
        cells = []
        for column in table["columns"]:
            value = row["cells"][column["id"]]
            text = "--"
            if value["mean"] is not None:
                text = f"{value['mean']:.2f}" + ("" if value["average"] or value["std"] is None
                        else rf"\,{{\scriptsize ({value['std']:.2f})}}")
                if value.get("provisional") and table.get("show_star_markers", True):
                    text += r"\textsuperscript{*}"
                if value["rank"] == 1:
                    text = r"{\bfseries\boldmath " + text + "}"
                elif value["rank"] == 2:
                    text = r"\underline{" + text + "}"
                text = r"\cellcolor{" + value["shade"] + "}" + text
            cells.append(text)
        lines.append(prefix + method + " & " + escape(row.get("setting_label", row["setting"].title()))
                     + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              (r"\caption{Final frozen-policy Test performance (\%) " if table.get("test_only") else
               r"\caption{Offline Train, Val, Test and Test (early) performance (\%) "
               r"from the configured checkpoint queue; Train, Val and Test evaluate the final frozen policy, "
               if table.get("offline_splits") else
               r"\caption{Online training performance, Offline test performance and Offline-val performance (\%) ") +
              scope + (
              (r"Completed seeds are included immediately; * marks fewer than three completed seeds. "
               if table.get("show_star_markers", True) else
               r"Only complete per-seed splits are included; missing seeds are excluded, with no star markers. ") +
              r"A single seed has no sample standard deviation. " if table.get("partial_seeds") else
              (r"ReAct Offline retains the previously reported baseline scores. "
              r"Other cells are filled only after all three seeds complete the corresponding "
              if table.get("reported_react_offline") else
              r"Cells are filled only after all three seeds complete the corresponding ") +
              r"measurement. ") + r"AppWorld uses task goal completion (TGC): fully successful tasks divided by "
              r"tasks with recorded official outcomes. Values are mean "
              r"(sample standard deviation). " + average_caption +
              (("For the configured GEPA/SkillOpt Val settings, " + early_heading + r" reuses the final Test result "
                r"of the policy selected using training-time validation. For other settings, "
                if table.get("offline_early", {}).get("reuse_training_val_best_variants") else "") +
               early_heading + r" selects the highest validation score among the configured checkpoints "
               r"(including CP0), breaking ties by the earliest checkpoint, and reports that checkpoint's test score. "
               r"Test scores never affect selection; incomplete selected tests remain missing. "
               if table.get("offline_early") else "") +
              r"Colors are column-wise min--max normalized (red: low; blue: high). Highest and second-highest "
              r"distinct displayed means are bold and underlined; ties share formatting.}",
              r"\label{tab:main_results}\end{table*}", r"\end{document}"]
    document = "\n".join(lines) + "\n"
    return document if standalone else document.split("\\begin{document}\n", 1)[1].rsplit("\\end{document}", 1)[0]
