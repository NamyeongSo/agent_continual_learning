# SPDX-License-Identifier: Apache-2.0
"""Score best/random/worst independent contexts against the unchanged saved rule-reference corpus."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace.analyze_rule_generality import (  # noqa: E402
    MODEL_PATH,
    MODEL_REVISION,
    MODELS,
    MODES,
    Progress,
    read,
    sha,
    write,
    write_csv,
)
from scripts.ace.reflection_review import independent_rules  # noqa: E402

STRATEGIES = ("best", "random", "worst")
SEEDS = (42, 43, 44)


def accumulation_runs():
    runs = {
        ("qwen38", "thinking"):
            ROOT / "outputs/ace_appworld_reflection_accumulation_qwen38_teammate_s42_44_20261116",
        ("qwen38", "nonthinking"):
            ROOT / "outputs/ace_appworld_nonthinking_replication_qwen38_20261117/accumulation",
    }
    for model, name in (("qwen36", "ace_appworld_qwen36_add_only_20261118"),
                        ("qwen35", "ace_appworld_qwen35_9b_add_only_20261119"),
                        ("gemma26", "ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124")):
        for mode in MODES:
            runs[model, mode] = ROOT / "outputs" / name / mode / "accumulation"
    return runs


def collect_queries(runs=None):
    contexts, selections, hashes = {}, [], {}
    runs = accumulation_runs() if runs is None else runs

    def load(path):
        hashes[str(path)] = sha(path.read_bytes())
        return read(path)

    for (model, mode), directory in runs.items():
        protocol = load(directory / "protocol.json")
        assert protocol["playbook_edit_mode"] == "add_only"
        bank = load(directory / "reflection_bank.json")["without_gt"]
        for seed in SEEDS:
            split = protocol["task_orders"][str(seed)]
            tasks = set(split["train"])
            assert len(tasks) == 50 and tasks.isdisjoint(set(split["val"]) | set(split["test"]))
            assert set(bank) == tasks
            for strategy in STRATEGIES:
                path = directory / f"conditions/s{seed}_without_gt_{strategy}/selections.json"
                choices = load(path)
                assert len(choices) == 50 and {r["candidate"]["task_id"] for r in choices} == tasks
                for choice in choices:
                    candidate = choice["candidate"]
                    task = candidate["task_id"]
                    assert choice["mode"] == strategy
                    candidates = bank[task]
                    assert len(candidates) == 5
                    rewards = [Fraction(c["score_numerator"], c["score_denominator"]) for c in candidates]
                    target = max(rewards) if strategy == "best" else min(rewards)
                    eligible = list(range(5)) if strategy == "random" else [i for i, r in enumerate(rewards)
                                                                           if r == target]
                    assert choice["tie_indices"] == eligible and choice["candidate_index"] in eligible
                    assert candidate == candidates[choice["candidate_index"]]
                    assert sha(candidate["reflection"]) == candidate["reflection_sha256"]
                    original = Path(candidate["source_reflection_path"])
                    assert original.parents[2].resolve() == Path(protocol["source_reflection_run"]).resolve()
                    context_path = str(original.parent / "updated_context.txt")
                    membership = {"model": model, "mode": mode, "arm": "without_gt", "seed": seed,
                                  "strategy": strategy, "task_id": task, "context_id": sha(context_path),
                                  "source_repeat": candidate["source_repeat"], "selection_path": str(path)}
                    selections.append(membership)
                    if context_path not in contexts:
                        update, sources = independent_rules(original, candidate, "without_gt")
                        assert Path(update["independent_update_path"]).resolve() == Path(context_path).resolve()
                        hashes.update(sources)
                        contexts[context_path] = {"id": sha(context_path), "model": model, "mode": mode,
                                                  "arm": "without_gt", "task_id": task,
                                                  "source_repeat": candidate["source_repeat"],
                                                  "source_path": context_path,
                                                  "source_sha256": update["independent_update_sha256"],
                                                  "rules": update["rules"], "selection_usages": []}
                    context = contexts[context_path]
                    assert (context["model"], context["mode"], context["task_id"]) == (model, mode, task)
                    context["selection_usages"].append(membership)
    assert len(selections) == len(runs) * 3 * 3 * 50
    return sorted(contexts.values(), key=lambda c: (MODES.index(c["mode"]), MODELS.index(c["model"]), c["id"])), \
        selections, hashes


def query_context_maxima(queries, targets, target_sizes, own_target_context, block_size=128):
    """An external source uses all reference contexts; a reference-member source excludes its whole context."""
    import numpy as np

    sizes = np.asarray(target_sizes)
    own = np.asarray(own_target_context)
    if len(own) != len(queries) or not len(sizes) or np.any(sizes <= 0) or sum(sizes) != len(targets):
        raise ValueError("Invalid rule/context partitions")
    if np.any(own < -1) or np.any(own >= len(sizes)):
        raise ValueError("Invalid source-context index")
    if len(sizes) == 1 and np.any(own == 0):
        raise ValueError("No other target context exists")
    if not all(np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5) for v in (queries, targets)):
        raise ValueError("Vectors must be normalized")
    starts = np.r_[0, np.cumsum(sizes)]
    maxima = np.full((len(queries), len(sizes)), np.nan, dtype=np.float32)
    nearest = np.full(maxima.shape, -1, dtype=np.int32)
    for first in range(0, len(queries), block_size):
        last = min(first + block_size, len(queries))
        sim = queries[first:last] @ targets.T
        np.clip(sim, -1, 1, out=sim)
        for column in range(len(sizes)):
            a, b = starts[column:column + 2]
            offset = sim[:, a:b].argmax(axis=1)
            valid = own[first:last] != column
            maxima[first:last, column][valid] = sim[np.arange(last - first), a + offset][valid]
            nearest[first:last, column][valid] = (a + offset)[valid]
    assert np.array_equal(np.isfinite(maxima).sum(axis=1), len(sizes) - (own >= 0))
    return maxima, nearest


def encode_queries(mode, contexts, reference, output, model, progress, batch_size):
    import numpy as np
    import torch

    directory = output / mode
    directory.mkdir(exist_ok=True)
    texts = {sha(r["content"]): r["content"] for c in contexts for r in c["rules"]}
    reference_protocol = read(reference / mode / "embedding_protocol.json")
    assert reference_protocol["revision"] == MODEL_REVISION and reference_protocol["dimension"] == 1024
    assert reference_protocol["normalized"] and reference_protocol["pooling"] == "CLS"
    assert reference_protocol["device"] == "cpu" and reference_protocol["dtype"] == "float32"
    cached_keys = read(reference / mode / "embedding_keys.json")
    cached = np.load(reference / mode / "embeddings.npy", mmap_mode="r")
    assert cached.shape == (len(cached_keys), 1024)
    cached_positions = {k: i for i, k in enumerate(cached_keys)}
    keys = sorted(texts)
    positions = {k: i for i, k in enumerate(keys)}
    vectors = np.lib.format.open_memmap(directory / "embeddings.npy", mode="w+", dtype=np.float32,
                                       shape=(len(keys), 1024))
    reused = [k for k in keys if k in cached_positions]
    if reused:
        vectors[[positions[k] for k in reused]] = cached[[cached_positions[k] for k in reused]]
    lengths = {}
    for start in range(0, len(keys), 128):
        group = keys[start:start + 128]
        tokens = model.tokenizer([texts[k] for k in group], truncation=False, add_special_tokens=True)["input_ids"]
        lengths.update({key: len(ids) for key, ids in zip(group, tokens, strict=True)})
    assert lengths and max(lengths.values()) <= model.max_seq_length
    missing = sorted((k for k in keys if k not in cached_positions), key=lambda k: (lengths[k], k))
    write(directory / "embedding_keys.json", keys)
    write(directory / "token_lengths.json", lengths)
    progress(f"{mode}_embedding", reused=len(reused), completed=0, total=len(missing))
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(missing), batch_size):
            group = missing[start:start + batch_size]
            encoded = model.encode([texts[k] for k in group], batch_size=len(group), convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
            vectors[[positions[k] for k in group]] = encoded
            vectors.flush()
            completed = start + len(group)
            if completed % (10 * batch_size) == 0 or completed == len(missing):
                progress(f"{mode}_embedding", reused=len(reused), completed=completed, total=len(missing),
                         strings_per_second=round(completed / max(time.monotonic() - started, .001), 2))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    vectors.flush()
    write(directory / "embedding_protocol.json", {**reference_protocol, "unique_texts": len(keys),
          "reused_embeddings": len(reused), "new_embeddings": len(missing), "max_tokens": max(lengths.values()),
          "reference_total_tokens": reference_protocol["total_tokens"], "total_tokens": sum(lengths.values()),
          "newly_encoded_total_tokens": sum(lengths[k] for k in missing),
          "cached_query_total_tokens": sum(lengths[k] for k in reused),
          "reference_cache": str(reference / mode), "truncated_inputs": 0,
          "encoding_seconds_this_invocation": round(time.monotonic() - started, 3)})
    return vectors, positions


def score_queries(mode, model_id, contexts, embeddings, positions, reference, output, progress):
    import numpy as np

    directory = output / mode / model_id
    directory.mkdir(exist_ok=True)
    reference_directory = reference / mode / model_id
    targets = [c for c in read(reference_directory / "contexts.json") if c["rules"]]
    target_rules = read(reference_directory / "rules.json")
    target_keys = read(reference / mode / "embedding_keys.json")
    target_positions = {k: i for i, k in enumerate(target_keys)}
    target_all_vectors = np.load(reference / mode / "embeddings.npy", mmap_mode="r")
    target_vectors = np.asarray(target_all_vectors[[target_positions[r["embedding_key"]] for r in target_rules]])
    target_ids = {c["id"]: i for i, c in enumerate(targets)}
    sources = [c for c in contexts if c["model"] == model_id]
    rules = []
    for c in sources:
        for r in c["rules"]:
            rules.append({"rule_id": sha(c["id"] + "|" + r["id"]), "context_id": c["id"],
                          "task_id": c["task_id"], "playbook_rule_id": r["id"], "text": r["content"],
                          "embedding_key": sha(r["content"]), "own_target_index": target_ids.get(c["id"], -1)})
    vectors = np.asarray(embeddings[[positions[r["embedding_key"]] for r in rules]])
    progress(f"{mode}_similarity", model=model_id, query_contexts=len(sources), reference_contexts=len(targets),
             query_rules=len(rules))
    maxima, nearest = query_context_maxima(vectors, target_vectors, [len(c["rules"]) for c in targets],
                                          [r["own_target_index"] for r in rules])
    np.save(directory / "context_max_cosine.npy", maxima)
    np.save(directory / "nearest_target_rule_index.npy", nearest)
    write(directory / "query_rules.json", rules)
    write(directory / "query_contexts.json", sources)
    write(directory / "reference_context_ids.json", [c["id"] for c in targets])
    rule_scores = []
    per_context = defaultdict(list)
    for i, r in enumerate(rules):
        values = maxima[i][np.isfinite(maxima[i])].astype(np.float64)
        mean = float(values.mean())
        rule_scores.append({"model": model_id, "mode": mode, "rule_id": r["rule_id"],
                            "context_id": r["context_id"], "task_id": r["task_id"],
                            "mean_context_max_cosine": mean, "target_contexts": len(values), "rule_text": r["text"]})
        per_context[r["context_id"]].append(mean)
    with (reference_directory / "rule_scores.csv").open() as stream:
        previous = {r["rule_id"]: float(r["mean_context_max_cosine"]) for r in csv.DictReader(stream)}
    shared = [r for r in rule_scores if r["rule_id"] in previous]
    assert all(abs(r["mean_context_max_cosine"] - previous[r["rule_id"]]) < 2e-6 for r in shared)
    context_scores = []
    for c in sources:
        values = per_context[c["id"]]
        context_scores.append({"model": model_id, "mode": mode, "arm": "without_gt", "context_id": c["id"],
                               "task_id": c["task_id"], "source_repeat": c["source_repeat"], "rules": len(values),
                               "context_mean_cosine": float(np.mean(values)) if values else None,
                               "reference_member": c["id"] in target_ids,
                               "target_contexts": len(targets) - (c["id"] in target_ids)})
    write_csv(directory / "rule_scores.csv", rule_scores)
    write_csv(directory / "context_scores.csv", context_scores)
    write(directory / "verification.json", {"shared_rule_scores_verified_against_previous_run": len(shared),
          "max_original_score_difference": max((abs(r["mean_context_max_cosine"] - previous[r["rule_id"]])
                                                 for r in shared), default=0),
          "query_contexts": len(sources), "ruleless_query_contexts": sum(c["rules"] == 0 for c in context_scores),
          "query_rules": len(rules), "reference_contexts": len(targets),
          "context_maxima": int(np.isfinite(maxima).sum())})
    return context_scores


def aggregate_selections(context_scores, selections):
    """Mean rules within context, then equal context weight within seed, then mean +/- sample SD of seed means."""
    import numpy as np

    by_id = {r["context_id"]: r for r in context_scores}
    per_selection = []
    for selected in selections:
        context = by_id[selected["context_id"]]
        assert all(context[k] == selected[k] for k in ("model", "mode", "task_id"))
        per_selection.append({**selected, "rules": context["rules"],
                              "context_mean_cosine": context["context_mean_cosine"]})
    seeds, summaries = [], []
    groups = sorted({(r["model"], r["mode"], r["strategy"]) for r in per_selection})
    for model, mode, strategy in groups:
        selected = [r for r in per_selection if (r["model"], r["mode"], r["strategy"]) == (model, mode, strategy)]
        seed_means = []
        for seed in SEEDS:
            group = [r for r in selected if r["seed"] == seed]
            if len(group) != 50 or len({r["task_id"] for r in group}) != 50:
                raise ValueError("Each regular strategy must retain all50 selected training tasks per seed")
            valid = [r["context_mean_cosine"] for r in group if r["context_mean_cosine"] is not None]
            mean = float(np.mean(valid)) if valid else None
            seeds.append({"model": model, "mode": mode, "arm": "without_gt", "strategy": strategy,
                          "seed": seed, "valid_contexts": len(valid), "empty_contexts": 50 - len(valid),
                          "mean_context_cosine": mean})
            if mean is not None:
                seed_means.append(mean)
        summaries.append({"model": model, "mode": mode, "arm": "without_gt", "strategy": strategy,
                          "mean_context_cosine": float(np.mean(seed_means)) if seed_means else None,
                          "sample_std_across_selection_seeds": float(np.std(seed_means, ddof=1))
                          if len(seed_means) > 1 else None,
                          "seed_count": len(seed_means), "selected_context_instances": len(selected),
                          "valid_context_instances": sum(r["context_mean_cosine"] is not None for r in selected),
                          "empty_context_instances": sum(r["context_mean_cosine"] is None for r in selected),
                          "unique_selected_contexts": len({r["context_id"] for r in selected})})
    return per_selection, seeds, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = Progress(args.output)
        try:
            progress("freeze_queries")
            assert (args.reference / "completed.json").exists() and read(args.reference / "verification.json")["passed"]
            contexts, selections, source_hashes = collect_queries()
            reference_contexts = {c["id"]: c for c in read(args.reference / "inputs.json")["contexts"]}
            for context in contexts:
                if context["id"] in reference_contexts:
                    previous = reference_contexts[context["id"]]
                    assert all(context[k] == previous[k] for k in
                               ("model", "mode", "task_id", "rules", "source_sha256"))
            write(args.output / "inputs.json", {"query_contexts": contexts, "selections": selections,
                                               "source_sha256": source_hashes})
            write(args.output / "protocol.json", {"reference": str(args.reference.resolve()),
                  "reference_inputs_sha256": sha((args.reference / "inputs.json").read_bytes()),
                  "inputs_sha256": sha((args.output / "inputs.json").read_bytes()),
                  "code_sha256": sha(Path(__file__).read_bytes()),
                  "base_code_sha256": sha((ROOT / "scripts/ace/analyze_rule_generality.py").read_bytes()),
                  "embedding_model": "BAAI/bge-m3", "revision": MODEL_REVISION, "device": "cpu", "threads": 8,
                  "batch_size": 16, "strategies": list(STRATEGIES), "models": list(MODELS), "modes": list(MODES),
                  "arm": "without_gt", "reference_pool": "Unchanged previous best_advantage/worst_advantage "
                  "selected contexts; separate by model/mode. No new context added to reference pool.",
                  "metric": "Per rule average of per-reference-context maximum cosine, excluding own context if "
                  "present; average rules equally within context, then selected contexts equally within seed, "
                  "then mean and sample SD across three selection seeds. Ruleless context undefined, not zero.",
                  "new_actor_or_curator_calls": 0})
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            progress("load_embedding_model")
            model = SentenceTransformer(str(MODEL_PATH), device="cpu", local_files_only=True).float()
            assert model.device.type == "cpu" and model.max_seq_length == 8192
            context_scores = []
            for mode in MODES:
                selected = [c for c in contexts if c["mode"] == mode]
                vectors, positions = encode_queries(mode, selected, args.reference, args.output, model, progress, 16)
                for model_id in MODELS:
                    context_scores.extend(score_queries(mode, model_id, selected, vectors, positions,
                                                       args.reference, args.output, progress))
                progress(f"{mode}_completed")
            per_selection, seeds, summaries = aggregate_selections(context_scores, selections)
            write_csv(args.output / "all_context_scores.csv", context_scores)
            write_csv(args.output / "selected_context_scores.csv", per_selection)
            write_csv(args.output / "strategy_seed_scores.csv", seeds)
            write_csv(args.output / "strategy_summary.csv", summaries)
            write(args.output / "strategy_summary.json", summaries)
            progress("completed", state="completed", summaries=summaries)
            write(args.output / "completed.json", {"contexts": len(context_scores), "selections": len(selections),
                                                   "summary_rows": len(summaries)})
        except Exception as exc:
            progress("failed", state="failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
