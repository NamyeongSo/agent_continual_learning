# SPDX-License-Identifier: Apache-2.0
"""CPU-only BGE-M3 rule specificity audit: mean of per-other-context maximum cosines."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26")
LABELS = {"qwen38": "Qwen3.8 27B", "qwen36": "Qwen3.6 35B-A3B", "qwen35": "Qwen3.5 9B",
          "gemma26": "Gemma4 26B-A4B"}
MODES = ("thinking", "nonthinking")
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots" / MODEL_REVISION


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_bytes())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Progress:
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()
        self.phase_started = self.started
        self.last_phase = None

    def __call__(self, phase, **values):
        if phase != self.last_phase:
            self.phase_started = time.monotonic()
            self.last_phase = phase
        row = {"at": datetime.now(UTC).isoformat(), "state": "running", "pid": os.getpid(), "phase": phase,
               "elapsed_seconds": round(time.monotonic() - self.started, 2),
               "phase_elapsed_seconds": round(time.monotonic() - self.phase_started, 2), **values}
        write(self.output / "status.json", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


def collect_contexts(database):
    """Freeze the active rule-review corpus and verify every used context against its original stage2 output."""
    from scripts.ace.reflection_review import added_rules

    contexts, source_hashes = {}, {}

    def source(path):
        path = Path(path)
        raw = path.read_bytes()
        source_hashes[str(path)] = sha(raw)
        return json.loads(raw)

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT snapshot FROM pairs WHERE active=1 ORDER BY id").fetchall()
    for raw, in rows:
        pair = json.loads(raw)
        if pair["model"] not in MODELS or pair["mode"] not in MODES or pair["arm"] != "without_gt":
            continue
        if pair.get("content_kind") != "independent_rules":
            raise ValueError("Expected the independent-rule comparison dataset")
        for strategy, candidate in pair["candidates"].items():
            path = candidate["independent_update_path"]
            usage = {"seed": pair["seed"], "strategy": strategy, "pair_id": pair["id"]}
            if path in contexts:
                current = contexts[path]
                assert (current["model"], current["mode"], current["task_id"]) == (
                    pair["model"], pair["mode"], pair["task_id"])
                assert current["rules"] == candidate["rules"]
                if usage not in current["selection_usages"]:
                    current["selection_usages"].append(usage)
                continue
            result = source(candidate["independent_result_path"])
            assert (result["stage"], result["task_id"], result["arm"], result["source_repeat"]) == (
                2, pair["task_id"], "without_gt", candidate["source_repeat"])
            reflection = source(candidate["source_path"])
            assert sha(reflection["reflection_for_curator"]) == candidate["reflection_sha256"]
            protocol = source(Path(candidate["source_path"]).parents[2] / "protocol.json")
            assert protocol["playbook_edit_mode"] == "add_only"
            capture = source(result["source_capture_path"])
            assert source_hashes[result["source_capture_path"]] == reflection["source_capture_sha256"]
            assert capture["playbook"] == reflection["playbook_after_tags"] == protocol["initial_playbook"]
            data = Path(path).read_bytes()
            source_hashes[path] = sha(data)
            assert sha(data) == candidate["independent_update_sha256"] == result["updated_context_sha256"]
            rules = added_rules(protocol["initial_playbook"], data.decode())
            assert rules == candidate["rules"]
            contexts[path] = {"id": sha(path), "model": pair["model"], "mode": pair["mode"],
                              "arm": "without_gt", "task_id": pair["task_id"],
                              "source_repeat": candidate["source_repeat"], "source_path": path,
                              "source_sha256": sha(data), "rules": rules, "selection_usages": [usage]}
    return sorted(contexts.values(), key=lambda c: (MODES.index(c["mode"]), MODELS.index(c["model"]), c["id"])), \
        source_hashes


def context_maxima(vectors, context_sizes, block_size=128, on_block=None):
    """Each source rule receives one maximum per OTHER context, regardless of its number of target rules."""
    import numpy as np

    sizes = np.asarray(context_sizes, dtype=np.int64)
    if len(sizes) < 2 or np.any(sizes <= 0) or sum(sizes) != len(vectors):
        raise ValueError("At least two nonempty contexts and a complete contiguous rule partition are required")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Embedding vectors must be normalized")
    starts = np.r_[0, np.cumsum(sizes)]
    source_contexts = np.repeat(np.arange(len(sizes)), sizes)
    maxima = np.full((len(vectors), len(sizes)), np.nan, dtype=np.float32)
    nearest = np.full(maxima.shape, -1, dtype=np.int32)
    for first in range(0, len(vectors), block_size):
        last = min(first + block_size, len(vectors))
        similarities = vectors[first:last] @ vectors.T
        np.clip(similarities, -1, 1, out=similarities)
        for target in range(len(sizes)):
            a, b = starts[target:target + 2]
            offset = similarities[:, a:b].argmax(axis=1)
            values = similarities[np.arange(last - first), a + offset]
            valid = source_contexts[first:last] != target
            maxima[first:last, target][valid] = values[valid]
            nearest[first:last, target][valid] = (a + offset)[valid]
        if on_block:
            on_block(last, len(vectors))
    assert np.all(np.isnan(maxima[np.arange(len(vectors)), source_contexts]))
    assert np.all(np.sum(np.isfinite(maxima), axis=1) == len(sizes) - 1)
    assert np.isfinite(maxima).sum() == len(vectors) * (len(sizes) - 1)
    return maxima, nearest


def encode_mode(mode, contexts, output, model, batch_size, progress, threads):
    import numpy as np
    import torch

    directory = output / mode
    directory.mkdir(exist_ok=True)
    texts = {sha(rule["content"]): rule["content"] for c in contexts for rule in c["rules"]}
    progress(f"{mode}_tokenizing", total=len(texts))
    lengths = {}
    for first in range(0, len(texts), 128):
        keys = list(texts)[first:first + 128]
        encoded = model.tokenizer([texts[k] for k in keys], truncation=False, add_special_tokens=True)["input_ids"]
        lengths.update({k: len(ids) for k, ids in zip(keys, encoded, strict=True)})
    if not lengths or max(lengths.values()) > model.max_seq_length:
        raise ValueError("Empty embedding corpus or overlong rule; refuse to truncate")
    keys = sorted(texts, key=lambda k: (lengths[k], k))
    write(directory / "embedding_keys.json", keys)
    write(directory / "token_lengths.json", lengths)
    fingerprint = sha(json.dumps([keys, MODEL_REVISION, 1024, "CLS", "L2", "float32"]))
    marker = directory / "embedding_progress.json"
    receipt = read(marker) if marker.exists() else {"completed": 0, "fingerprint": fingerprint}
    if receipt["fingerprint"] != fingerprint:
        raise ValueError("Embedding inputs changed; use a separate output directory")
    completed = receipt["completed"]
    path = directory / "embeddings.npy"
    vectors = np.lib.format.open_memmap(path, mode="r+" if completed else "w+", dtype=np.float32,
                                       shape=(len(keys), 1024))
    if vectors.shape != (len(keys), 1024) or vectors.dtype != np.float32:
        raise ValueError("Cached embedding shape or dtype mismatch")
    if completed and not np.allclose(np.linalg.norm(vectors[:completed], axis=1), 1, atol=1e-5):
        raise ValueError("Invalid saved embedding prefix")
    started = time.monotonic()
    initial_completed = completed
    progress(f"{mode}_embedding", completed=completed, total=len(keys), cpu_threads=threads)
    with torch.inference_mode():
        for first in range(completed, len(keys), batch_size):
            selected = keys[first:first + batch_size]
            vectors[first:first + len(selected)] = model.encode(
                [texts[k] for k in selected], batch_size=len(selected), normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False)
            completed = first + len(selected)
            vectors.flush()
            write(marker, {"completed": completed, "total": len(keys), "fingerprint": fingerprint})
            if completed == len(keys) or completed % (batch_size * 10) == 0:
                elapsed = time.monotonic() - started
                progress(f"{mode}_embedding", completed=completed, total=len(keys),
                         strings_per_second=round((completed - initial_completed) / max(elapsed, 1e-6), 2))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    write(directory / "embedding_protocol.json", {"model": "BAAI/bge-m3", "revision": MODEL_REVISION,
          "device": "cpu", "dtype": "float32", "pooling": "CLS", "normalized": True,
          "dimension": 1024, "max_sequence_length": model.max_seq_length, "truncated_inputs": 0,
          "unique_texts": len(keys), "max_tokens": max(lengths.values()), "total_tokens": sum(lengths.values()),
          "threads": threads, "batch_size": batch_size,
          "encoding_seconds_this_invocation": round(time.monotonic() - started, 3)})
    return vectors, {key: i for i, key in enumerate(keys)}


def score_model(mode, model_id, contexts, vectors, positions, output, block_size, progress):
    import numpy as np

    all_contexts = [c for c in contexts if c["model"] == model_id]
    nonempty = [c for c in all_contexts if c["rules"]]
    rules = []
    for ci, context in enumerate(nonempty):
        for rule in context["rules"]:
            rules.append({"rule_id": sha(context["id"] + "|" + rule["id"]), "context_id": context["id"],
                          "context_index": ci, "task_id": context["task_id"], "source_repeat": context["source_repeat"],
                          "playbook_rule_id": rule["id"], "text": rule["content"],
                          "embedding_key": sha(rule["content"]), "selection_usages": context["selection_usages"]})
    subset = np.asarray(vectors[[positions[r["embedding_key"]] for r in rules]])
    sizes = [len(c["rules"]) for c in nonempty]
    directory = output / mode / model_id
    directory.mkdir(exist_ok=True)
    write(directory / "rules.json", rules)
    write(directory / "contexts.json", all_contexts)
    progress(f"{mode}_similarity", model=model_id, completed=0, total=len(rules))
    maxima, nearest = context_maxima(subset, sizes, block_size,
                                    lambda n, total: progress(f"{mode}_similarity", model=model_id,
                                                              completed=n, total=total))
    np.save(directory / "context_max_cosine.npy", maxima)
    np.save(directory / "context_nearest_rule_index.npy", nearest)
    scores = []
    for index, rule in enumerate(rules):
        values = maxima[index][np.isfinite(maxima[index])].astype(np.float64)
        usages = rule["selection_usages"]
        scores.append({"model": model_id, "mode": mode, "arm": "without_gt", "rule_id": rule["rule_id"],
                       "context_id": rule["context_id"], "task_id": rule["task_id"],
                       "source_repeat": rule["source_repeat"], "playbook_rule_id": rule["playbook_rule_id"],
                       "mean_context_max_cosine": float(values.mean()),
                       "population_std_context_max_cosine": float(values.std()),
                       "min_context_max_cosine": float(values.min()),
                       "median_context_max_cosine": float(np.median(values)),
                       "max_context_max_cosine": float(values.max()), "other_contexts": len(values),
                       "selected_as": ";".join(sorted({u["strategy"] for u in usages})),
                       "selection_seeds": ";".join(map(str, sorted({u["seed"] for u in usages}))),
                       "rule_text": rule["text"]})
    write_csv(directory / "rule_scores.csv", scores)
    with gzip.open(directory / "rule_context_maxima.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rule_id", "other_context_id", "max_cosine", "nearest_rule_id"])
        for row, rule in enumerate(rules):
            for column, context in enumerate(nonempty):
                if np.isfinite(maxima[row, column]):
                    writer.writerow([rule["rule_id"], context["id"], float(maxima[row, column]),
                                     rules[nearest[row, column]]["rule_id"]])
    means = np.array([row["mean_context_max_cosine"] for row in scores])
    context_means = [means[np.array([r["context_index"] for r in rules]) == ci].mean()
                     for ci in range(len(nonempty))]
    summary = {"model": model_id, "mode": mode, "arm": "without_gt", "contexts": len(all_contexts),
               "nonempty_contexts": len(nonempty), "empty_contexts": len(all_contexts) - len(nonempty),
               "rules": len(rules), "unique_cosine_pairs": (len(rules) ** 2 - sum(n ** 2 for n in sizes)) // 2,
               "rule_context_maxima": int(np.isfinite(maxima).sum()),
               "rule_mean_cosine": float(means.mean()), "rule_population_std_cosine": float(means.std()),
               "rule_median_cosine": float(np.median(means)), "rule_p10_cosine": float(np.quantile(means, .1)),
               "rule_p90_cosine": float(np.quantile(means, .9)),
               "equal_context_mean_cosine": float(np.mean(context_means))}
    write(directory / "summary.json", summary)
    write_csv(directory / "lowest_mean_rules.csv", sorted(scores, key=lambda x: x["mean_context_max_cosine"])[:20])
    write_csv(directory / "highest_mean_rules.csv", sorted(scores, key=lambda x: -x["mean_context_max_cosine"])[:20])
    return summary


def plot(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = ("#3f719b", "#ba7945", "#548568", "#8c6ea5")
    for ax, mode in zip(axes, MODES, strict=True):
        data = []
        for model in MODELS:
            with (output / mode / model / "rule_scores.csv").open() as stream:
                data.append(np.array([float(r["mean_context_max_cosine"]) for r in csv.DictReader(stream)]))
        violin = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, color in zip(violin["bodies"], colors, strict=True):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.45)
        violin["cmeans"].set_color("#243344")
        ax.set_xticks(range(1, 5), [LABELS[m].replace(" ", "\n", 1) for m in MODELS])
        ax.set_title(f"{'Thinking' if mode == 'thinking' else 'Non-thinking'} · w/o GT")
        ax.grid(axis="y", alpha=.2)
        for i, values in enumerate(data, 1):
            ax.text(i, .03, f"n={len(values):,}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Per-rule mean of maximum cosine in each other context")
    fig.suptitle("Rule commonality across independent contexts", fontsize=17)
    fig.text(.5, .015, "BGE-M3 · Same model and mode only · Own context excluded · Equal weight per target context",
             ha="center", fontsize=9, color="#606b76")
    fig.tight_layout(rect=(0, .045, 1, .94))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"rule_context_generality_thinking_nonthinking.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      OPENBLAS_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = Progress(args.output)
        try:
            progress("freeze_inputs")
            contexts, hashes = collect_contexts(args.source_db)
            frozen = {"contexts": contexts, "source_sha256": hashes}
            inputs = args.output / "inputs.json"
            if inputs.exists() and read(inputs) != frozen:
                raise ValueError("Source context data changed; preserve this run and use a separate output path")
            write(inputs, frozen)
            write(args.output / "protocol.json", {
                "embedding_model": "BAAI/bge-m3", "revision": MODEL_REVISION, "device": "cpu",
                "models": list(MODELS), "modes_in_order": list(MODES), "arm": "without_gt",
                "source_database": str(args.source_db.resolve()), "inputs_sha256": sha(inputs.read_bytes()),
                "source_script_sha256": sha(Path(__file__).read_bytes()), "threads": args.threads,
                "batch_size": args.batch_size, "block_size": args.block_size,
                "metric": "For each rule, mean(max cosine to target-context rules) over every other nonempty "
                          "context of the same model and mode; source context excluded.",
                "selection_scope": "Actual best_advantage/worst_advantage selected independent contexts; "
                                   "deduplicate reused source context, retain selection memberships.",
                "empty_context_policy": "Exclude from target contexts; never impute zero cosine",
                "new_actor_or_curator_calls": 0})
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            progress("load_embedding_model", cpu_threads=args.threads)
            model = SentenceTransformer(str(MODEL_PATH), device="cpu", local_files_only=True).float()
            assert model.device.type == "cpu" and model.get_embedding_dimension() == 1024
            assert model.max_seq_length == 8192
            summaries = []
            for mode in MODES:
                selected = [c for c in contexts if c["mode"] == mode]
                vectors, positions = encode_mode(mode, selected, args.output, model, args.batch_size,
                                                 progress, args.threads)
                mode_summaries = [score_model(mode, m, selected, vectors, positions, args.output,
                                              args.block_size, progress) for m in MODELS]
                summaries.extend(mode_summaries)
                write_csv(args.output / mode / "summary.csv", mode_summaries)
                write(args.output / mode / "completed.json", {"mode": mode, "summary": mode_summaries,
                                                             "finished_at": datetime.now(UTC).isoformat()})
                progress(f"{mode}_completed", summary=mode_summaries)
            write_csv(args.output / "summary.csv", summaries)
            write(args.output / "summary.json", summaries)
            plot(args.output, summaries)
            progress("completed", state="completed", summary=summaries)
            write(args.output / "completed.json", {"finished_at": datetime.now(UTC).isoformat(), "models": 4,
                                                    "modes": list(MODES), "total_rule_scores": sum(
                                                        r["rules"] for r in summaries)})
        except Exception as exc:
            progress("failed", state="failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
