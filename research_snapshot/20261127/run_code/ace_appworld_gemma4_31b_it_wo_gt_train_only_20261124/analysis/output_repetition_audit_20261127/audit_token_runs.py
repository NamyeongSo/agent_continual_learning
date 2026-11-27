# SPDX-License-Identifier: Apache-2.0
"""Audit saved decoded output components on CPU without new model calls."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "2"

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
TOKENIZER = Path.home() / (
    ".cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/"
    "842da3794eaa0b77d5f08bae87a17459d91ff475/tokenizer.json"
)


def repetitive_runs(values, max_period=256, minimum_repeats=8, minimum_span=64):
    values = np.asarray(values, dtype=np.int32)
    for period in range(1, min(max_period, len(values) // minimum_repeats) + 1):
        equal = values[period:] == values[:-period]
        edges = np.flatnonzero(np.diff(np.r_[False, equal, False]))
        starts, ends = edges[::2], edges[1::2]
        lengths = ends - starts
        selected = (lengths >= (minimum_repeats - 1) * period) & (lengths + period >= minimum_span)
        for start, length in zip(starts[selected], lengths[selected], strict=True):
            yield int(start), int(length + period), period


def parts(message):
    seen = set()
    for key in ("reasoning_content", "reasoning", "content"):
        value = message.get(key)
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            yield key, value
    for index, call in enumerate(message.get("tool_calls") or []):
        value = (call.get("function") or {}).get("arguments")
        if isinstance(value, str) and value:
            yield f"tool_arguments_{index}", value


def main():
    assert any(period == 7 and span == 70 for _, span, period in repetitive_runs(list(range(7)) * 10))
    assert not list(repetitive_runs(list(range(1000))))
    assert not list(repetitive_runs([1, 2] * 15 + [-1] + [1, 2] * 15))
    hashes = json.loads((ROOT / "source_hashes.json").read_text())
    files = json.loads((ROOT / "calls.json").read_text())
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    records, texts = [], []
    for file in files:
        path = Path(file["path"])
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == hashes[str(path)]
        data = json.loads(raw)
        for choice in data["response"]["choices"]:
            for kind, text in parts(choice.get("message") or {}):
                records.append({**file, "kind": kind})
                texts.append(text)
    encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
    arrays, offsets, offset = [], [], 0
    for index, encoding in enumerate(encoded):
        offsets.append(offset)
        array = np.array([-index - 1, *encoding.ids], dtype=np.int32)
        arrays.append(array)
        offset += len(array)
    flat = np.concatenate(arrays)
    offsets = np.array(offsets)
    found = {}
    for start, span, period in repetitive_runs(flat):
        index = int(np.searchsorted(offsets, start, side="right") - 1)
        local = int(start - offsets[index] - 1)
        assert 0 <= local and local + span <= len(encoded[index].ids)
        previous = found.get(index)
        if previous and previous["repeated_span_tokens"] >= span:
            continue
        unit = encoded[index].ids[local:local + period]
        found[index] = {
            **records[index], "repeated_span_tokens": span, "period_tokens": period,
            "full_repetitions": span // period, "local_start_token": local,
            "repeated_unit": tokenizer.decode(unit),
            "whole_component": texts[index],
        }
    result = sorted(found.values(), key=lambda item: item["repeated_span_tokens"], reverse=True)
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "saved_responses": len(files), "output_components": len(records),
        "retokenized_output_tokens": sum(len(item.ids) for item in encoded),
        "tokenizer_path": str(TOKENIZER),
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "criterion": "Exact contiguous periodic block of 1..256 tokens repeated >=8 times over >=64 tokens",
        "flagged_components": len(result), "flagged_responses": len({item["path"] for item in result}),
        "scope": "Saved decoded reasoning, final content, and tool arguments; each component tested separately",
        "limitations": "Raw generated token IDs and timed-out partial outputs were not saved; decoded fields retokenized",
        "self_checks_passed": True, "source_hashes_verified": True,
    }
    (ROOT / "token_repetition_candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "token_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps([{k: v for k, v in item.items() if k != "whole_component"} for item in result[:10]],
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
