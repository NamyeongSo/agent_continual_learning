# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import copy
import json

import pytest

from scripts.research_journal.result_tables import BENCHMARKS, build_table, latex


def config():
    return {"id": "rerun", "title": "Fresh rerun", "model": "Qwen", "temperature": 0.7,
            "seeds": [42, 43, 44], "sources": [{"batch": "current"}],
            "auto_discover_after": "2026-10-23T00:00:00+00:00", "split_manifest_sha256": "split"}


def batch(method="ace", values=(0.0, 0.5, 1.0)):
    runs = []
    cps = []
    for benchmark, _ in BENCHMARKS:
        for seed, value in zip((42, 43, 44), values, strict=True):
            score = {"value": value, "count": 5, "total": 5, "complete": True, "errors": 0}
            runs.append({"id": f"{method}_{benchmark}_{seed}", "method": method, "benchmark": benchmark,
                         "seed": seed, "train": {"total": 5, "complete": True}, "test": {"complete": True},
                         "scores": {"train": dict(score), "test": dict(score),
                                    "val": {**score, "value": 0.99}},
                         "frozen_policy_sha256": "policy", "split_manifest_sha256": "split"})
            cps.append({"variant": "current", "method": method, "benchmark": benchmark, "seed": seed,
                        "point": 5, "policy_sha256": "policy", "alias": "earlier-identical-cp",
                        "splits": {"train": {"score": 1},
                                   "val": {"done": 5, "total": 5, "score": value, "errors": 0}}})
    return {"id": "current", "model": "openai/Qwen", "temperature": 0.7, "created_at": "2026-10-22",
            "setting": "isolated", "playbook_edit_mode": "add_only", "runs": runs}, [{"rows": cps}]


def row(table, method="ace", setting="isolated"):
    return next(r for r in table["rows"] if r["method"] == method and r["setting"] == setting)


def test_online_is_training_offline_is_test_and_offline_val_is_exact_final_policy():
    source, queues = batch()
    table = build_table(config(), [source], queues)
    cells = row(table)["cells"]
    for metric in ("online", "offline", "offline_val"):
        assert cells[f"bfcl.{metric}"]["display"] == "50.00 (50.00)"
        assert cells[f"avg.{metric}"]["display"] == "50.00"
    assert all(cell["mean"] is None for r in table["rows"] if r["setting"] != "isolated"
               for cell in r["cells"].values())
    assert [c["metric"] for c in table["columns"][:3]] == ["Online", "Offline", "Offline-val"]
    assert table["completed_cells"] == 12
    assert table["appworld_metric"] == "appworld_tgc_v1"
    assert "AppWorld uses task goal completion (TGC)" in latex(table)


def test_environment_policy_cohorts_never_mix_into_the_original_results_table():
    source, queues = batch()
    source["appworld_policy_sha256"] = "teammate-policy"
    assert build_table(config(), [source], queues)["completed_cells"] == 0
    configured = {**config(), "appworld_policy_sha256": "teammate-policy"}
    assert build_table(configured, [source], queues)["completed_cells"] == 12


def test_explicit_benchmark_sources_replace_only_appworld_and_recompute_macro_average():
    previous, _ = batch(values=(0.6, 0.6, 0.6))
    original, _ = batch(values=(0.3, 0.3, 0.3))
    previous.update(id="previous", appworld_policy_sha256="team")
    original.update(id="original", appworld_policy_sha256="original")
    cfg = {**config(), "appworld_policy_sha256": "original", "explicit_sources_only": True,
           "sources": [{"batch": "previous", "appworld_policy_sha256": "team",
                        "benchmarks": ["bfcl", "terminalbench2"]},
                       {"batch": "original", "benchmarks": ["appworld"]}]}
    cells = row(build_table(cfg, [previous, original], []))["cells"]
    assert cells["bfcl.offline"]["mean"] == 60
    assert cells["appworld.offline"]["mean"] == 30
    assert cells["terminalbench2.offline"]["mean"] == 60
    assert cells["avg.offline"]["mean"] == 50
    assert all(s["batch"] == "original" for s in cells["appworld.offline"]["samples"])
    original["appworld_policy_sha256"] = "changed-policy"
    cells = row(build_table(cfg, [previous, original], []))["cells"]
    assert cells["appworld.offline"]["mean"] is None
    assert cells["avg.offline"]["mean"] is None


def test_partial_measurements_and_missing_seeds_stay_blank_then_fill_automatically():
    source, queues = batch(values=(0, 0, 0))
    original = source["runs"][-1]["scores"]["test"]
    original.update(count=4, complete=False)
    cells = row(build_table(config(), [source], queues))["cells"]
    assert cells["terminalbench2.offline"]["mean"] is None
    assert cells["terminalbench2.offline"]["ready"] == 2
    assert cells["avg.offline"]["mean"] is None
    assert cells["bfcl.offline"]["mean"] == 0
    original.update(count=5, complete=True)
    cells = row(build_table(config(), [source], queues))["cells"]
    assert cells["terminalbench2.offline"]["display"] == "0.00 (0.00)"
    assert cells["avg.offline"]["display"] == "0.00"


def test_reported_react_offline_is_retained_during_live_updates_without_inventing_seed_scores():
    cfg = config()
    cfg["reported_react_offline"] = {"source": "Original user-supplied table", "benchmarks": {
        "bfcl": {"mean": 58, "std": 2}, "appworld": {"mean": 11.33, "std": 6.11},
        "terminalbench2": {"mean": 26.44, "std": 1.99}}}
    source, queues = batch()
    for values in ([], [source]):
        table = build_table(cfg, values, queues)
        cells = row(table, "react")["cells"]
        assert [cells[f"{benchmark}.offline"]["display"] for benchmark, _ in BENCHMARKS] == [
            "58.00 (2.00)", "11.33 (6.11)", "26.44 (1.99)"]
        assert cells["avg.offline"]["display"] == "31.92" and cells["avg.offline"]["std"] is None
        assert all(c["samples"] == [] and c["historical"] for key, c in cells.items() if key.endswith(".offline"))
        assert all(c["mean"] is None for key, c in cells.items() if not key.endswith(".offline"))
        assert all(c["mean"] is None for r in table["rows"] if r["setting"] != "isolated"
                   for c in r["cells"].values())
        assert "ReAct Offline retains the previously reported baseline scores" in latex(table)
    assert cells["bfcl.offline"]["rank"] == 1
    assert row(table)["cells"]["bfcl.offline"]["rank"] == 2


@pytest.mark.parametrize("field,value", [("point", 10), ("policy_sha256", "another-policy")])
def test_validation_does_not_use_candidate_scores_or_another_checkpoint(field, value):
    source, queues = batch()
    queues[0]["rows"][0][field] = value
    cells = row(build_table(config(), [source], queues))["cells"]
    assert cells["bfcl.offline_val"]["mean"] is None
    assert cells["bfcl.offline_val"]["ready"] == 2
    assert cells["bfcl.offline"]["mean"] == 50


@pytest.mark.parametrize("field,value", [("model", "other"), ("temperature", 0.8),
                                         ("playbook_edit_mode", "add_update_delete")])
def test_cohort_rejects_other_models_temperatures_and_curator_variants(field, value):
    source, queues = batch()
    source[field] = value
    assert build_table(config(), [source], queues)["completed_cells"] == 0


def test_invalidated_mismatched_split_duplicate_sources_and_infrastructure_errors_stay_blank():
    source, queues = batch()
    for run in source["runs"]:
        if run["seed"] == 42:
            run["invalidated"] = True
        elif run["seed"] == 43:
            run["split_manifest_sha256"] = "other"
        else:
            run["scores"]["test"]["errors"] = 1
    table = build_table(config(), [source], queues)
    assert row(table)["cells"]["bfcl.offline"]["mean"] is None
    original, _ = batch()
    assert build_table(config(), [original, copy.deepcopy(original)], queues)["completed_cells"] == 0


def test_future_compatible_sequential_rerun_fills_but_historical_results_remain_blank():
    source, queues = batch()
    source.update(id="future", setting="sequential", created_at="2026-10-24T00:00:00+00:00")
    assert row(build_table(config(), [source], queues), setting="sequential")["cells"]["bfcl.online"]["mean"] == 50
    source["created_at"] = "2026-09-24T00:00:00+00:00"
    assert build_table(config(), [source], queues)["completed_cells"] == 0


def test_auto_discovery_compares_actual_time_across_kst_and_utc():
    source, queues = batch()
    source.update(id="new", setting="sequential", created_at="2026-10-23T08:59:59+09:00")
    assert build_table(config(), [source], queues)["completed_cells"] == 0
    source["created_at"] = "2026-10-23T09:00:00+09:00"
    assert row(build_table(config(), [source], queues), setting="sequential")["cells"]["bfcl.online"]["mean"] == 50


def test_column_ranks_ignore_sd_and_blank_cells_and_share_ties_after_rounding():
    first, queues = batch("ace", (0.5, 0.5, 0.5))
    second, _ = batch("gepa", (0.0, 0.5, 1.0))
    third, _ = batch("skillopt", (0.400001, 0.4, 0.4))
    first["runs"] += second["runs"] + third["runs"]
    table = build_table(config(), [first], queues)
    assert row(table, "ace")["cells"]["bfcl.online"]["rank"] == 1
    assert row(table, "gepa")["cells"]["bfcl.online"]["rank"] == 1
    assert row(table, "skillopt")["cells"]["bfcl.online"]["rank"] == 2
    assert row(table, "skillopt")["cells"]["bfcl.online"]["shade"] == "scorered!35!white"
    assert row(table, "ace")["cells"]["bfcl.online"]["shade"] == "scoreblue!35!white"
    text = latex(table)
    assert "Offline-val" in text and r"\multirow{3}" in text and r"\underline{" in text
    assert "15 of 16" not in text and "Sequential & --" in text
    assert r"\documentclass" in text


def test_live_manifest_table_updates_without_importing_agents(tmp_path):
    from scripts.research_journal.live_progress import LiveProgress
    from scripts.research_journal.store import JournalStore

    out = tmp_path / "outputs/result_table"
    out.mkdir(parents=True)
    (out / "results_table_manifest.json").write_text(json.dumps(config()))
    collector = LiveProgress(JournalStore(tmp_path / "journal", [tmp_path]), base_dir=tmp_path, ttl=0)
    tables = collector.get()["result_tables"]
    assert len(tables) == 1 and tables[0]["completed_cells"] == 0


def variant_config(*variants):
    return {**config(), "show_partial_seeds": True, "explicit_sources_only": True,
            "rows": [{"id": variant, "method": method, "label": method.upper(), "setting": "isolated",
                      "setting_label": label, "variant": variant} for method, variant, label in variants]}


def test_variant_rows_keep_val_and_train_as_val_separate_in_the_same_batch():
    source, queues = batch("gepa")
    other, _ = batch("gepa", (0.2, 0.4, 0.6))
    for run in source["runs"]:
        run["variant"] = "gepa_val"
    for run in other["runs"]:
        run["variant"] = "gepa_train"
    source["runs"] += other["runs"]
    cfg = variant_config(("gepa", "gepa_val", "Val"), ("gepa", "gepa_train", "Train-as-val"))
    table = build_table(cfg, [source], queues)
    assert len(table["rows"]) == 2 and table["row_layout"] == "variants"
    assert [r["cells"]["bfcl.offline"]["mean"] for r in table["rows"]] == [50, 40]
    assert "Train-as-val" in latex(table) and r"\multirow{3}" not in latex(table)


def test_finished_seeds_appear_immediately_and_star_disappears_only_after_third_seed():
    source, queues = batch()
    for run in source["runs"]:
        run["variant"] = "ace_aud"
    cfg = variant_config(("ace", "ace_aud", "AUD"))
    runs = {r["seed"]: r for r in source["runs"] if r["benchmark"] == "bfcl"}
    runs[44]["scores"]["test"].update(count=4, complete=False)
    table = build_table(cfg, [source], queues)
    value = table["rows"][0]["cells"]["bfcl.offline"]
    assert value["display"] == "25.00 (35.36)*" and value["ready"] == 2
    assert r"\textsuperscript{*}" in latex(table)
    runs[43]["scores"]["test"].update(count=4, complete=False)
    value = build_table(cfg, [source], queues)["rows"][0]["cells"]["bfcl.offline"]
    assert value["display"] == "0.00*" and value["std"] is None and value["ready"] == 1
    for seed in (43, 44):
        runs[seed]["scores"]["test"].update(count=5, complete=True)
    table = build_table(cfg, [source], queues)
    value = table["rows"][0]["cells"]["bfcl.offline"]
    assert value["display"] == "50.00 (50.00)" and not value["provisional"]
    assert table["provisional_cells"] == 0


def test_partial_macro_average_requires_the_same_finished_seed_across_three_benchmarks():
    source, queues = batch()
    for run in source["runs"]:
        run["variant"] = "ace_aud"
        keep = {"bfcl": 42, "appworld": 43, "terminalbench2": 44}[run["benchmark"]]
        if run["seed"] != keep:
            run["scores"]["test"].update(count=4, complete=False)
    cfg = variant_config(("ace", "ace_aud", "AUD"))
    table = build_table(cfg, [source], queues)
    assert all(table["rows"][0]["cells"][f"{b}.offline"]["provisional"] for b, _ in BENCHMARKS)
    assert table["rows"][0]["cells"]["avg.offline"]["mean"] is None
    for run in source["runs"]:
        if run["seed"] == 42:
            run["scores"]["test"].update(count=5, complete=True)
    value = build_table(cfg, [source], queues)["rows"][0]["cells"]["avg.offline"]
    assert value["display"] == "0.00*" and value["ready"] == 1


def test_variant_table_never_auto_discovers_an_unlisted_batch_with_the_same_split():
    source, queues = batch()
    source.update(id="unlisted", created_at="2026-10-24T00:00:00+00:00")
    for run in source["runs"]:
        run["variant"] = "ace_aud"
    table = build_table(variant_config(("ace", "ace_aud", "AUD")), [source], queues)
    assert table["displayed_cells"] == 0


def test_appworld_only_policy_table_has_no_unrequested_benchmarks_or_macro_average():
    source, queues = batch()
    for run in source["runs"]:
        run["variant"] = "ace_aud"
    cfg = {**variant_config(("ace", "ace_aud", "AUD")), "benchmarks": ["appworld"], "include_average": False}
    table = build_table(cfg, [source], queues)
    assert [c["id"] for c in table["columns"]] == ["appworld.online", "appworld.offline", "appworld.offline_val"]
    assert set(table["rows"][0]["cells"]) == {c["id"] for c in table["columns"]}
    assert table["rows"][0]["cells"]["appworld.offline"]["mean"] == 50
    text = latex(table)
    assert r"ll*{3}{c}" in text and "AppWorld" in text
    assert "BFCL" not in text and "Terminal-Bench" not in text and "Avg." not in text


@pytest.mark.parametrize("benchmarks", [[], ["unknown"], ["appworld", "appworld"]])
def test_result_table_rejects_invalid_benchmark_selection(benchmarks):
    with pytest.raises(ValueError, match="benchmarks"):
        build_table({**config(), "benchmarks": benchmarks}, [], [])


def early_fixture():
    source, _ = batch(values=(0.6, 0.6, 0.6))
    for run in source["runs"]:
        run["variant"] = "ace_aud"
        run["train"]["total"] = 10
    cfg = {**variant_config(("ace", "ace_aud", "AUD")), "offline_early": {
        "checkpoint_queue_id": "cp10", "points": {b: [0, 5, 10] for b, _ in BENCHMARKS}}}
    checkpoints = []
    for run in source["runs"]:
        for point, val, test in [(0, 0.5, 1.0), (5, 0.8, 0.2), (10, 0.8, 0.9)]:
            checkpoints.append({"id": f"{run['id']}_cp{point}", "stream": run["id"], "point": point,
                                "variant": "current", "run_variant": "ace_aud", "method": run["method"],
                                "benchmark": run["benchmark"], "seed": run["seed"], "selected": True,
                                "splits": {name: {"score": value, "done": 5, "total": 5, "errors": 0}
                                           for name, value in (("val", val), ("test", test))}})
    return cfg, source, [{"id": "cp10", "rows": checkpoints}]


def test_early_uses_validation_only_breaks_ties_by_earlier_cp_and_exports_four_columns():
    cfg, source, queues = early_fixture()
    table = build_table(cfg, [source], queues)
    value = table["rows"][0]["cells"]["bfcl.offline_early"]
    assert value["display"] == "20.00 (0.00)"
    assert all(s["checkpoint"] == 5 and s["val_score"] == 0.8 for s in value["samples"])
    assert table["rows"][0]["cells"]["bfcl.offline"]["display"] == "60.00 (0.00)"
    assert table["rows"][0]["cells"]["avg.offline_early"]["mean"] == 20
    assert [c["metric"] for c in table["columns"][:4]] == ["Online", "Offline", "Offline (early)", "Offline-val"]
    text = latex(table)
    assert r"\multicolumn{4}{c}{\textbf{BFCL}}" in text and r"\cmidrule(lr){3-6}" in text
    assert r"ll*{16}{c}" in text and "Test scores never affect selection" in text


def test_early_keeps_initial_shared_alias_and_compares_unrounded_validation_scores():
    cfg, source, queues = early_fixture()
    for cp in queues[0]["rows"]:
        if cp["seed"] == 42:
            cp["splits"]["val"]["score"] = 0.9 if cp["point"] == 0 else 0.8
            if cp["point"] == 0:
                cp["alias"] = "shared_cp0"
        elif cp["seed"] == 43 and cp["point"] == 10:
            cp["splits"]["val"]["score"] = 0.800001
    samples = build_table(cfg, [source], queues)["rows"][0]["cells"]["bfcl.offline_early"]["samples"]
    assert [s["checkpoint"] for s in samples] == [0, 10, 5]
    assert samples[0]["checkpoint_alias"] == "shared_cp0"


def test_missing_selected_test_does_not_select_another_cp_and_macro_uses_same_finished_seeds():
    cfg, source, queues = early_fixture()
    selected = next(cp for cp in queues[0]["rows"] if
                    (cp["benchmark"], cp["seed"], cp["point"]) == ("terminalbench2", 42, 5))
    selected["splits"]["test"].update(done=4, errors=1)
    table = build_table(cfg, [source], queues)
    cells = table["rows"][0]["cells"]
    missing = cells["terminalbench2.offline_early"]["samples"][0]
    assert missing["checkpoint"] == 5 and missing["value"] is None
    assert missing["reason"] == "selected_test_incomplete" and missing["errors"] == 1
    assert cells["terminalbench2.offline_early"]["display"] == "20.00 (0.00)*"
    assert [s["seed"] for s in cells["avg.offline_early"]["samples"]] == [43, 44]


@pytest.mark.parametrize("problem", ["wrong_queue", "wrong_variant", "duplicate", "missing", "partial_val"])
def test_early_requires_unique_cohort_checkpoints_and_complete_validation(problem):
    cfg, source, queues = early_fixture()
    selected = next(cp for cp in queues[0]["rows"] if
                    (cp["benchmark"], cp["seed"], cp["point"]) == ("bfcl", 42, 5))
    if problem == "wrong_queue":
        queues[0]["id"] = "old_queue"
    elif problem == "wrong_variant":
        selected["run_variant"] = "other_variant"
    elif problem == "duplicate":
        queues[0]["rows"].append(copy.deepcopy(selected))
    elif problem == "missing":
        queues[0]["rows"].remove(selected)
    else:
        selected["splits"]["val"].update(done=4, errors=1)
    value = build_table(cfg, [source], queues)["rows"][0]["cells"]["bfcl.offline_early"]
    assert value["samples"][0]["value"] is None
    assert value["ready"] == (0 if problem == "wrong_queue" else 2)
    assert value["provisional"] == (problem != "wrong_queue")


@pytest.mark.parametrize("method", ["gepa", "skillopt"])
def test_training_val_best_early_reuses_exact_displayed_final_test(method):
    cfg, source, queues = early_fixture()
    variant = method + "_val"
    cfg["rows"][0].update(id=variant, variant=variant, method=method)
    cfg["offline_early"]["reuse_training_val_best_variants"] = [variant]
    for run in source["runs"]:
        run.update(method=method, variant=variant, validation_source="val")
    for cp in queues[0]["rows"]:
        cp.update(method=method, run_variant=variant)
        if cp["point"] == 10:
            cp.update(policy_sha256="policy", selected_val_score=.9)
    table = build_table(cfg, [source], queues)
    cells = table["rows"][0]["cells"]
    for benchmark in [b for b, _ in BENCHMARKS] + ["avg"]:
        assert cells[benchmark + ".offline_early"]["display"] == cells[benchmark + ".offline"]["display"]
    sample = cells["bfcl.offline_early"]["samples"][0]
    assert sample["checkpoint"] == 10 and sample["value"] == .6
    assert sample["test_result_reused"] and sample["val_score"] == .9
    assert "training-time validation" in latex(table)
    source["runs"][0]["validation_source"] = "train"
    sample = build_table(cfg, [source], queues)["rows"][0]["cells"]["bfcl.offline_early"]["samples"][0]
    assert sample["value"] is None and sample["reason"] == "training_val_selection_unverified"


def offline_split_fixture():
    cfg, source, queues = early_fixture()
    cfg.update(offline_splits=True, show_star_markers=False)
    for cp in queues[0]["rows"]:
        cp["splits"]["train"] = {"score": .3, "done": 5, "total": 5, "errors": 0}
        if cp["point"] == 10:
            cp["policy_sha256"] = "policy"
    return cfg, source, queues


def test_offline_split_table_uses_only_pinned_queue_and_final_policy_instead_of_training_records():
    cfg, source, queues = offline_split_fixture()
    queues.append({**copy.deepcopy(queues[0]), "id": "unrelated_queue"})
    table = build_table(cfg, [source], queues)
    cells = table["rows"][0]["cells"]
    assert [c["metric"] for c in table["columns"][:4]] == ["Train", "Val", "Test", "Test (early)"]
    assert len(table["columns"]) == 16 and not any(c["id"].endswith(".online") for c in table["columns"])
    for benchmark in [b for b, _ in BENCHMARKS] + ["avg"]:
        assert cells[benchmark + ".offline_train"]["mean"] == 30
        assert cells[benchmark + ".offline_val"]["mean"] == 80
        assert cells[benchmark + ".offline"]["mean"] == 90
        assert cells[benchmark + ".offline_early"]["mean"] == 20
    sample = cells["bfcl.offline"]["samples"][0]
    assert sample["checkpoint_queue"] == "cp10" and sample["checkpoint"] == 10
    assert sample["evaluation_split"] == "test" and sample["count"] == sample["total"] == 5
    assert "Offline Train, Val, Test and Test (early)" in latex(table)
    assert "Online training performance" not in latex(table)


@pytest.mark.parametrize("problem", ["wrong_queue", "wrong_hash", "duplicate", "missing", "incomplete_training"])
def test_offline_split_final_policy_provenance_never_falls_back_to_separate_test(problem):
    cfg, source, queues = offline_split_fixture()
    final = next(cp for cp in queues[0]["rows"] if
                 (cp["benchmark"], cp["seed"], cp["point"]) == ("bfcl", 42, 10))
    if problem == "wrong_queue":
        queues[0]["id"] = "other"
    elif problem == "wrong_hash":
        final["policy_sha256"] = "different"
    elif problem == "duplicate":
        queues[0]["rows"].append(copy.deepcopy(final))
    elif problem == "missing":
        queues[0]["rows"].remove(final)
    else:
        source["runs"][0]["train"]["complete"] = False
    cells = build_table(cfg, [source], queues)["rows"][0]["cells"]
    for metric in ("offline_train", "offline_val", "offline"):
        assert cells["bfcl." + metric]["samples"][0]["value"] is None


def test_offline_split_incomplete_seed_is_excluded_without_stars_and_macro_matches_seed_membership():
    cfg, source, queues = offline_split_fixture()
    final = next(cp for cp in queues[0]["rows"] if
                 (cp["benchmark"], cp["seed"], cp["point"]) == ("terminalbench2", 42, 10))
    final["splits"]["train"].update(done=4, errors=1)
    table = build_table(cfg, [source], queues)
    cells = table["rows"][0]["cells"]
    value = cells["terminalbench2.offline_train"]
    assert value["provisional"] and value["ready"] == 2 and value["expected"] == 3
    assert value["display"] == "30.00 (0.00)"
    assert value["samples"][0]["errors"] == 1 and value["samples"][0]["value"] is None
    assert [s["seed"] for s in cells["avg.offline_train"]["samples"]] == [43, 44]
    assert all("*" not in c["display"] for c in cells.values())
    assert r"\textsuperscript{*}" not in latex(table)
    assert "missing seeds are excluded, with no star markers" in latex(table)


@pytest.mark.parametrize("method", ["gepa", "skillopt"])
def test_offline_split_native_val_reuses_cp_final_test_rather_than_a_separate_test_run(method):
    cfg, source, queues = offline_split_fixture()
    variant = method + "_val"
    cfg["rows"][0].update(id=variant, variant=variant, method=method)
    cfg["offline_early"]["reuse_training_val_best_variants"] = [variant]
    for run in source["runs"]:
        run.update(method=method, variant=variant, validation_source="val")
    for cp in queues[0]["rows"]:
        cp.update(method=method, run_variant=variant)
    cells = build_table(cfg, [source], queues)["rows"][0]["cells"]
    for benchmark in [b for b, _ in BENCHMARKS] + ["avg"]:
        assert cells[benchmark + ".offline_early"]["display"] == cells[benchmark + ".offline"]["display"]
        assert cells[benchmark + ".offline_early"]["mean"] == 90
    assert all(s["checkpoint"] == 10 and s["test_result_reused"]
               for s in cells["bfcl.offline_early"]["samples"])


def test_offline_split_columns_require_explicit_queue_configuration():
    with pytest.raises(ValueError, match="explicit checkpoint queue"):
        build_table({**config(), "offline_splits": True}, [], [])


def test_test_only_table_shows_current_cohort_final_tests_and_same_seed_macro_without_stars():
    source, queues = batch()
    source["inference_profile_sha256"] = "current-profile"
    cfg = {**config(), "test_only": True, "show_partial_seeds": True, "show_star_markers": False,
           "inference_profile_sha256": "current-profile"}
    source["runs"][-1]["scores"]["test"].update(count=4, complete=False)
    table = build_table(cfg, [source], queues)
    assert table["test_only"] and [c["metric"] for c in table["columns"]] == ["Test"] * 4
    assert [c["id"] for c in table["columns"]] == [b + ".offline" for b in [
        "bfcl", "appworld", "terminalbench2", "avg"]]
    cells = row(table)["cells"]
    assert cells["bfcl.offline"]["display"] == "50.00 (50.00)"
    assert cells["terminalbench2.offline"]["display"] == "25.00 (35.36)"
    assert cells["avg.offline"]["display"] == "25.00"
    assert [s["seed"] for s in cells["avg.offline"]["samples"]] == [42, 43]
    text = latex(table)
    assert "Final frozen-policy Test performance" in text and "Online training performance" not in text
    assert r"\textsuperscript{*}" not in text and r"ll*{4}{c}" in text
    source["inference_profile_sha256"] = "another-profile"
    assert build_table(cfg, [source], queues)["displayed_cells"] == 0


def test_test_only_table_does_not_mix_checkpoint_split_layouts():
    cfg, source, queues = early_fixture()
    with pytest.raises(ValueError, match="Test-only tables"):
        build_table({**cfg, "test_only": True}, [source], queues)
