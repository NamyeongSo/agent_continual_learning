# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Pinned, local CPU text comparison; no actor/judge API or benchmark execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
import threading
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
DEFAULT_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"  # pragma: allowlist secret (public HF commit)
TEXT_KEYS = frozenset({"content", "message", "body", "description", "subject"})


def protected_facts(text: str) -> dict:
    """Guards are deliberately conservative; disagreement means unknown, never equal."""
    numbers = re.findall(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?(?!\w)", text)
    normalized = []
    from decimal import Decimal, InvalidOperation

    for value in numbers:
        try:
            normalized.append(str(Decimal(value.replace(",", "")).normalize()))
        except InvalidOperation:
            normalized.append(value)
    return {
        "numbers": sorted(Counter(normalized).items()),
        "identifiers": sorted(set(re.findall(r"\b(?:[A-Za-z]+[_-]?\d+[\w-]*|https?://\S+|[\w.+-]+@[\w.-]+)\b",
                                             text))),
        "negation": sorted(re.findall(r"\b(?:not|never|without|cannot)\b|n't\b", text.casefold())),
    }


class LocalTextEquivalence:
    """High-confidence bidirectional NLI estimates, with explicit unknown outcomes.

    CPU is intentional: loading this comparator must never allocate on GPU0 or
    change a shared serving process. NLI estimates are not equivalence proofs.
    """

    def __init__(self, *, model=DEFAULT_MODEL, revision=DEFAULT_REVISION, threshold=.95, max_tokens=512, cpu_threads=4):
        if not .5 < threshold <= 1 or not isinstance(max_tokens, int) or max_tokens < 8:
            raise ValueError("Invalid NLI confidence or token bound")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Text comparator requires a pinned model commit")
        self.model_name, self.revision = model, revision
        self.threshold, self.max_tokens = threshold, max_tokens
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("NLI CPU thread count must be positive")
        self.cpu_threads = cpu_threads
        self._model = None
        self._tokenizer = None
        self._manifest = None
        self._lock = threading.RLock()
        self._cache = {}

    @property
    def identity(self):
        try:
            versions = {name: importlib.metadata.version(name) for name in (
                "transformers", "torch", "tokenizers", "sentencepiece", "safetensors", "huggingface-hub",
            )}
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError("Install Exgentic's action-agreement optional dependencies for canonical metrics") from exc
        return {"kind": "bidirectional_nli_with_fact_guards", "model": self.model_name, "revision": self.revision,
                "threshold": self.threshold, "max_tokens": self.max_tokens, "device": "cpu",
                "cpu_threads": self.cpu_threads, **versions}

    def prepare(self):
        with self._lock:
            if self._model is not None:
                return self._manifest
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            path = Path(snapshot_download(
                self.model_name, revision=self.revision, token=False,
                allow_patterns=["config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json",
                                "added_tokens.json", "spm.model"],
            ))
            torch.set_num_threads(self.cpu_threads)
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=False, local_files_only=True,
            ).to("cpu")
            labels = {int(k): str(v).casefold() for k, v in model.config.id2label.items()}
            if labels != {0: "contradiction", 1: "entailment", 2: "neutral"}:
                raise ValueError(f"Unexpected NLI label mapping: {labels}")
            model.eval()
            assets = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())
                      if p.is_file() and (p.name in {"config.json", "model.safetensors", "special_tokens_map.json",
                                                    "added_tokens.json", "spm.model"}
                                          or p.name.startswith("tokenizer"))}
            self._model, self._tokenizer = model, tokenizer
            self._manifest = {**self.identity, "assets_sha256": assets}
            return self._manifest

    def compare(self, a: str, b: str) -> dict:
        if a == b:
            return {"lower": 1., "upper": 1., "reason": "exact_text", "approximate": False}
        if protected_facts(a) != protected_facts(b):
            return {"lower": 0., "upper": 1., "reason": "text_fact_guard_disagrees", "approximate": False}
        # Symmetric cache; both directions are always evaluated, with no truncation.
        key = tuple(sorted((a, b)))
        with self._lock:
            if key in self._cache:
                return dict(self._cache[key])
            self.prepare()
            if any(len(self._tokenizer(x, y, truncation=False)["input_ids"]) > self.max_tokens
                   for x, y in ((a, b), (b, a))):
                result = {"lower": 0., "upper": 1., "reason": "text_exceeds_nli_context", "approximate": False}
            else:
                import torch

                with torch.inference_mode():
                    inputs = self._tokenizer([a, b], [b, a], padding=True, truncation=False, return_tensors="pt")
                    logits = self._model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1).tolist()
                if not all(math.isfinite(v) and 0 <= v <= 1 for row in probabilities for v in row):
                    raise ValueError("Invalid NLI probabilities")
                if min(p[1] for p in probabilities) >= self.threshold:
                    lower = upper = 1.
                    reason = "bidirectional_entailment_estimate"
                elif min(p[0] for p in probabilities) >= self.threshold:
                    lower = upper = 0.
                    reason = "bidirectional_contradiction_estimate"
                else:
                    lower, upper, reason = 0., 1., "nli_inconclusive"
                result = {"lower": lower, "upper": upper, "reason": reason, "approximate": True,
                          "probabilities_ab_ba": probabilities}
            self._cache[key] = result
            return dict(result)
