# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Pinned, explicit lexical scoring for regenerated action sequences."""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
LEXICAL_METRICS = ("bleu1", "bleu4", "meteor")


def value_tokens(value):
    """Keep unescaped code/commands, identifiers and punctuation; sort object keys."""
    if isinstance(value, dict):
        yield "{"
        for key in sorted(value):
            yield str(key)
            yield from value_tokens(value[key])
        yield "}"
    elif isinstance(value, list):
        yield "["
        for item in value:
            yield from value_tokens(item)
        yield "]"
    else:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        yield from TOKEN_PATTERN.findall(text)


def action_tokens(actions, mode):
    """Strict/full includes payloads; loose is the ordered sequence of action names.

    These are input representations, not new standard BLEU/METEOR definitions.
    No conversation history, analysis, plan or JSON name/arguments wrapper is scored.
    """
    if mode not in ("strict", "loose"):
        raise ValueError("Expected strict or loose action representation")
    tokens = []
    for index, action in enumerate(actions):
        if index and mode == "strict":
            tokens.append("<ACTION>")
        tokens.append(action["name"])
        if mode == "strict":
            for key in sorted(action.get("arguments", {})):
                tokens.append(key)
                tokens.extend(value_tokens(action["arguments"][key]))
    return tuple(tokens)


class LexicalScorer:
    """NLTK 3.9.2 sentence BLEU and default English METEOR, cached deterministically."""

    def __init__(self, dependencies: Path, nltk_data: Path):
        sys.path.insert(0, str(dependencies))
        import nltk
        from nltk.corpus import wordnet
        from nltk.stem import PorterStemmer
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from nltk.translate.meteor_score import single_meteor_score

        if nltk.__version__ != "3.9.2":
            raise ValueError(f"Expected pinned NLTK 3.9.2, found {nltk.__version__}")
        nltk.data.path.insert(0, str(nltk_data))
        if wordnet.get_version() != "3.0":
            raise ValueError("Expected pinned WordNet 3.0")
        self.sentence_bleu = sentence_bleu
        self.single_meteor_score = single_meteor_score
        self.smoothing = SmoothingFunction(epsilon=.1).method1
        stemmer = PorterStemmer()

        class CachedStemmer:
            @staticmethod
            @lru_cache(maxsize=100000)
            def stem(word):
                return stemmer.stem(word)

        class CachedWordNet:
            @staticmethod
            @lru_cache(maxsize=100000)
            def synsets(word):
                return wordnet.synsets(word)

        self.stemmer, self.wordnet = CachedStemmer(), CachedWordNet()
        self.score_tokens = lru_cache(maxsize=20000)(self._score_tokens)

    def _score_tokens(self, hypothesis, reference):
        if not hypothesis or not reference:
            # Explicit extension for empty action sequences, matching the PRF convention.
            return (float(hypothesis == reference),) * 3
        bleu1, bleu4 = self.sentence_bleu(
            [reference], hypothesis, weights=[(1.0,), (.25, .25, .25, .25)],
            smoothing_function=self.smoothing, auto_reweigh=True,
        )
        meteor = self.single_meteor_score(
            reference, hypothesis, stemmer=self.stemmer, wordnet=self.wordnet,
            alpha=.9, beta=3.0, gamma=.5,
        )
        return bleu1, bleu4, meteor

    def score(self, prediction, reference, mode):
        scores = self.score_tokens(action_tokens(prediction, mode), action_tokens(reference, mode))
        return dict(zip(LEXICAL_METRICS, scores, strict=True))


def definitions():
    return {
        "implementation": "NLTK 3.9.2 + WordNet 3.0; separately installed analysis dependencies",
        "representation": "Strict/full: action names + sorted argument keys + lexical value tokens, including "
                          "unescaped Python/shell strings and punctuation. Identifiers retain underscores; "
                          "nested containers have brace/bracket markers; <ACTION> separates calls. Loose: "
                          "one atomic function/action name per call, preserving order and duplicate counts.",
        "bleu1": "Sentence BLEU with unigram weights (1), including brevity penalty; case-sensitive.",
        "bleu4": "Sentence BLEU with uniform 1..4-gram weights, method1 smoothing epsilon0.1 and "
                 "auto_reweigh=True (effective order for hypotheses shorter than four tokens).",
        "meteor": "NLTK English METEOR: lowercase preprocessing, Porter stemming, WordNet synonym matches, "
                  "alpha0.9/beta3/gamma0.5. This is an NLP lexical surrogate, not code semantic equality. "
                  "A one-token exact match scores0.5 due to the standard fragmentation penalty.",
        "empty_sequences": "Both empty score1; one empty score0 (explicit action-agreement extension).",
        "aggregation": "Macro means of sentence/trajectory scores with equal task weights, not corpus BLEU. "
                       "Scores range0..1; no truncation or token subsampling.",
        "sources": ["https://www.nltk.org/api/nltk.translate.bleu_score.html",
                    "https://www.nltk.org/api/nltk.translate.meteor_score.html"],
    }
