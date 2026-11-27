# SPDX-License-Identifier: Apache-2.0
"""Compare each selected reflection with the text actually inserted into its saved playbook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import runpy
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from exgentic.agents.ace.playbook_utils import parse_playbook_line  # noqa: E402

SPECS = runpy.run_path(str(OUT.parent / "build_report.py"))
MODEL_PATH = Path("/home/thskadud/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
                  "5617a9f61b028005a4858fdac845db406aefb181")
MODEL_NAME = "BAAI/bge-m3"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def inserted_content(before, after):
    """Recover inserted lines from the ADD-only checkpoint pair and remove ID/counter prefixes."""
    old = before.splitlines()
    new = after.splitlines()
    remaining = Counter(old)
    inserted = []
    for line in new:
        if remaining[line] > 0:
            remaining[line] -= 1
        else:
            inserted.append(line)
    assert not any(remaining.values()), "Existing checkpoint text was removed or rewritten"
    assert Counter(old) + Counter(inserted) == Counter(new)
    content = []
    for line in inserted:
        parsed = parse_playbook_line(line)
        content.append(parsed["content"] if parsed else line)
    return "\n".join(content).strip()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    rows, sources = [], {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for model, modes in SPECS["MODEL_RUNS"].items():
        if model not in SPECS["ADV_RUNS"]:
            continue
        jobs = []
        for mode, run in modes.items():
            for directory in sorted((run / "conditions").iterdir()):
                jobs.append((mode, directory))
        advantage = SPECS["ADV_RUNS"][model]
        for job in read(advantage / "plan.json"):
            run = Path(job["path"])
            jobs.append((run.parent.name, run / "conditions" / job["condition"]["id"]))
        for mode, directory in jobs:
            choices = read(directory / "selections.json")
            cp = directory / "checkpoints"
            before = read(cp / "000.json")["playbook"]
            for step, choice in enumerate(choices, 1):
                path = cp / f"{step:03d}.json"
                row = read(path)
                candidate = choice["candidate"]
                reflection = candidate["reflection"]
                assert sha(reflection) == row["reflection_sha256"] == candidate["reflection_sha256"]
                assert row["task_id"] == candidate["task_id"] and row["step"] == step
                assert row["session_count"] == step
                assert row["context_sha256"] == sha(row["playbook"])
                assert row["parent_context_sha256"] == sha(before)
                added = inserted_content(before, row["playbook"])
                changed = row["playbook"] != before
                assert changed == bool(added)
                condition = row["condition"]
                rows.append({"model": model, "mode": mode, "arm": condition["arm"], "seed": condition["seed"],
                             "strategy": condition["mode"], "step": step, "task_id": row["task_id"],
                             "source_repeat": candidate["source_repeat"], "checkpoint_path": str(path),
                             "reflection": reflection, "added_text": added, "context_changed": changed,
                             "reflection_sha256": sha(reflection), "added_text_sha256": sha(added),
                             "reflection_chars": len(reflection), "added_chars": len(added)})
                before = row["playbook"]
    assert len(rows) == 11016 and sum(r["context_changed"] for r in rows) == 10894
    write_csv(OUT / "pairs.csv", rows)
    (OUT / "pairs.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    (OUT / "source_sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"EXTRACTED {len(rows)} steps, {sum(r['context_changed'] for r in rows)} nonempty update pairs", flush=True)
    return rows


def embed(rows, *, threads, batch_size):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(MODEL_PATH), device="cpu", local_files_only=True)
    assert model.max_seq_length == 8192 and model.get_embedding_dimension() == 1024
    texts = {}
    for row in rows:
        if row["context_changed"]:
            texts[row["reflection_sha256"]] = row["reflection"]
            texts[row["added_text_sha256"]] = row["added_text"]
    keys = list(texts)
    token_lengths = {}
    for start in range(0, len(keys), 128):
        group = keys[start:start + 128]
        tokens = model.tokenizer([texts[key] for key in group], truncation=False, add_special_tokens=True)["input_ids"]
        token_lengths.update({key: len(ids) for key, ids in zip(group, tokens, strict=True)})
    max_length = max(token_lengths.values())
    too_long = {key: n for key, n in token_lengths.items() if n > model.max_seq_length}
    (OUT / "text_lengths.json").write_text(json.dumps(token_lengths) + "\n")
    print(f"TOKENIZED {len(keys)} unique strings, max={max_length}, over8192={len(too_long)}", flush=True)
    if too_long:
        raise ValueError(f"Inputs exceed 8192 tokens; review text_lengths.json before encoding {len(too_long)} strings")
    keys.sort(key=lambda key: (token_lengths[key], key))
    vectors = np.lib.format.open_memmap(OUT / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(keys), 1024))
    (OUT / "embedding_keys.json").write_text(json.dumps(keys) + "\n")
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            selected = keys[start:start + batch_size]
            vectors[start:start + len(selected)] = model.encode([texts[key] for key in selected], batch_size=len(selected),
                                                               normalize_embeddings=True, show_progress_bar=False)
            done = start + len(selected)
            if start == 0 or done % (batch_size * 10) == 0 or done == len(keys):
                vectors.flush()
                elapsed = time.monotonic() - started
                status = {"phase": "embedding", "completed": done, "total": len(keys),
                          "elapsed_seconds": round(elapsed, 1), "strings_per_second": round(done / elapsed, 2),
                          "max_current_tokens": max(token_lengths[key] for key in selected),
                          "updated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")}
                (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
                print(json.dumps(status), flush=True)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    positions = {key: i for i, key in enumerate(keys)}
    for row in rows:
        row["reflection_tokens"] = token_lengths.get(row["reflection_sha256"])
        row["added_tokens"] = token_lengths.get(row["added_text_sha256"])
        row["cosine_similarity"] = (
            float(vectors[positions[row["reflection_sha256"]]] @ vectors[positions[row["added_text_sha256"]]])
            if row["context_changed"] else None
        )
    metadata = {"model": MODEL_NAME, "revision": MODEL_PATH.name, "device": "cpu", "dtype": "float32",
                "max_seq_length": 8192, "pooling": "CLS", "l2_normalized": True, "dimensions": 1024,
                "unique_strings": len(keys), "max_observed_tokens": max_length, "truncated_inputs": 0,
                "batch_size": batch_size, "cpu_threads": threads,
                "source": "https://huggingface.co/BAAI/bge-m3",
                "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")}
    (OUT / "embedding_protocol.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUT / "pair_scores.csv", rows)
    return rows


def summarize(rows):
    summaries = []
    for fields in [("model",), ("model", "mode", "arm", "strategy")]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in fields)].append(row)
        records = []
        for key, samples in grouped.items():
            values = [r["cosine_similarity"] for r in samples if r["cosine_similarity"] is not None]
            record = {**dict(zip(fields, key, strict=True)), "steps": len(samples), "measured_pairs": len(values),
                      "no_update_steps": len(samples) - len(values), "mean_cosine": mean(values),
                      "std_cosine": stdev(values), "p10_cosine": float(np.quantile(values, 0.1)),
                      "median_cosine": float(np.median(values)), "min_cosine": min(values),
                      "share_below_0_3": mean(v < 0.3 for v in values),
                      "share_below_0_5": mean(v < 0.5 for v in values),
                      "share_below_0_7": mean(v < 0.7 for v in values)}
            records.append(record)
        write_csv(OUT / ("model_summary.csv" if len(fields) == 1 else "strategy_summary.csv"), records)
        if len(fields) == 1:
            summaries = records
    lowest = sorted((r for r in rows if r["cosine_similarity"] is not None), key=lambda r: r["cosine_similarity"])
    write_csv(OUT / "lowest_similarity_pairs.csv", lowest[:100])
    empty_reflections = [r for r in rows if r["reflection"].strip() in {"{}", "[]", "null", "None", ""}]
    write_csv(OUT / "empty_reflection_steps.csv", empty_reflections)
    lines = ["# Reflection와 실제 Playbook 추가 내용의 임베딩 유사도", "",
             "BAAI/bge-m3의 CLS dense 벡터를 L2 정규화한 뒤 cosine similarity를 계산했습니다. "
             "각 문자열 전체를 한 번에 인코딩했고 입력 절단은 0건입니다.", "",
             "| 모델 | 전체 step | 비교 pair | 추가 없음 | 평균 ± SD | 중앙값 | 하위 10% | cos < 0.5 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        lines.append(f"| {SPECS['LABELS'][s['model']]} | {s['steps']} | {s['measured_pairs']} | {s['no_update_steps']} | "
                     f"{s['mean_cosine']:.3f} ± {s['std_cosine']:.3f} | {s['median_cosine']:.3f} | "
                     f"{s['p10_cosine']:.3f} | {s['share_below_0_5']:.1%} |")
    lines += ["", "표의 SD는 학습 step별 cosine similarity의 표준편차입니다. "
              "0.5는 분포를 기술하기 위한 절단값입니다. 낮은 점수는 원문을 대조할 검토 후보입니다.",
              "", f"입력 Reflection이 빈 JSON 등인 step은 {len(empty_reflections)}건이며, "
              f"이 중 {sum(r['context_changed'] for r in empty_reflections)}건에서 Playbook이 추가됐습니다. "
              "실제 Curator 입력을 보존해 비교했으며 해당 step은 empty_reflection_steps.csv에 모았습니다.",
              "", "GPT-OSS-120B는 누적 학습 체크포인트가 생성되기 전입니다."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    plot(rows)
    (OUT / "status.json").write_text(json.dumps({"phase": "completed", "steps": len(rows),
        "measured_pairs": len(lowest), "updated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")}, indent=2))
    print("\n".join(lines), flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {"best": "#17765c", "random": "#77818e", "worst": "#b55643",
              "best_advantage": "#269ba4", "worst_advantage": "#ca903c"}
    for model in SPECS["ADV_RUNS"]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=True)
        for i, mode in enumerate(SPECS["MODEL_RUNS"][model]):
            for j, arm in enumerate(SPECS["ARMS"]):
                ax = axes[i, j]
                values = [[r["cosine_similarity"] for r in rows if r["model"] == model and r["mode"] == mode and
                           r["arm"] == arm and r["strategy"] == strategy and r["cosine_similarity"] is not None]
                          for strategy in SPECS["STRATEGIES"]]
                parts = ax.violinplot(values, showextrema=False, widths=0.7)
                for body, strategy in zip(parts["bodies"], SPECS["STRATEGIES"], strict=True):
                    body.set(facecolor=colors[strategy], edgecolor=colors[strategy], alpha=0.3)
                ax.scatter(range(1, 6), [mean(v) for v in values], color=[colors[s] for s in SPECS["STRATEGIES"]],
                           marker="_", s=150, linewidth=2, zorder=3)
                ax.set(title=f"{SPECS['mode_label'](mode)} · {SPECS['ARMS'][arm]}", ylim=(0, 1.02),
                       xticks=range(1, 6), xticklabels=[s.replace("_advantage", "\nadvantage") for s in SPECS["STRATEGIES"]])
                ax.grid(axis="y", color="#e6ebf1", linewidth=0.8)
                ax.spines[["top", "right"]].set_visible(False)
                if j == 0:
                    ax.set_ylabel("Reflection–added text cosine similarity")
        fig.suptitle(f"{SPECS['LABELS'][model]} · Reflection and inserted knowledge", fontsize=17, weight="bold", y=0.98)
        fig.text(0.5, 0.935, "BGE-M3 · Full-text dense embeddings · Mean marked by horizontal bars", ha="center")
        fig.subplots_adjust(top=0.87, bottom=0.09, left=0.075, right=0.98, hspace=0.34, wspace=0.15)
        fig.savefig(OUT / f"{model}_reflection_update_cosine_violin.png", dpi=180, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    rows = collect()
    if not args.prepare_only:
        summarize(embed(rows, threads=args.threads, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
