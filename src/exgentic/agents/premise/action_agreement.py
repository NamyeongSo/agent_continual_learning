# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Cached canonical action agreement with explicit unresolved score intervals."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import threading
from pathlib import Path

from .action_nli import TEXT_KEYS, LocalTextEquivalence
from .action_normalization import (
    ShellDescriptions,
    python_description,
    request_namespace,
    shell_prompt_visible,
)
from .span_policy_shift import canonical

VERSION = "canonical-action-agreement-v1"
METRICS = ("canonical_f1", "canonical_accuracy", "strict_f1", "strict_accuracy")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bound(lower, upper=None, *, reason, approximate=False, **details):
    upper = lower if upper is None else upper
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1):
        raise ValueError("Invalid action agreement interval")
    return {"lower": float(lower), "upper": float(upper), "reason": reason,
            "approximate": approximate, **details}


def conjunction(parts, reason):
    if not parts:
        return bound(1, reason=reason)
    return bound(min(p["lower"] for p in parts), min(p["upper"] for p in parts), reason=reason,
                 approximate=any(p.get("approximate") for p in parts), components=parts)


def ordered_f1(matrix, n, m):
    """Bounds of order-preserving weighted LCS F1, never cross-prefix matching."""
    if not n or not m:
        return bound(float(n == m), reason="empty_action_lists")
    scores = {}
    for kind in ("lower", "upper"):
        previous = [0.] * (m + 1)
        for i in range(n):
            current = [0.]
            for j in range(m):
                current.append(max(previous[j + 1], current[-1], previous[j] + matrix[i][j][kind]))
            previous = current
        scores[kind] = min(1., 2 * previous[-1] / (n + m))
    return bound(scores["lower"], scores["upper"], reason="ordered_canonical_f1",
                 approximate=any(p.get("approximate") for row in matrix for p in row))


class CanonicalActionAgreement:
    """Same-prefix behavioral abstraction, not program equivalence or a likelihood ratio."""

    def __init__(self, cache_dir: Path, *, text=None):
        self.text = text if text is not None else LocalTextEquivalence()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shell = ShellDescriptions()
        self._lock = threading.RLock()
        self._contexts, self._python, self._shell, self._pairs = {}, {}, {}, {}
        self._fingerprint = digest(self.identity)

    @property
    def identity(self):
        if hasattr(self, "_identity"):
            return self._identity
        dependencies = {}
        for name in ("tree-sitter", "tree-sitter-languages"):
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                dependencies[name] = "not_installed"
        files = [Path(__file__), Path(__file__).with_name("action_normalization.py"),
                 Path(__file__).with_name("action_nli.py")]
        self._identity = {
            "version": VERSION, "text": self.text.identity, "dependencies": dependencies,
            "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            "unknown": "interval_0_1_no_missing_as_zero", "sequence": "ordered_weighted_lcs",
            "environment_execution": False,
        }
        return self._identity

    def prepare(self, benchmarks):
        runtime = {"comparator": self.identity}
        if "terminalbench2" in benchmarks:
            self.shell.prepare()
        if benchmarks:
            runtime["text_assets"] = self.text.prepare()
        return runtime

    def _values(self, a, b, *, textual=False):
        if canonical(a) == canonical(b):
            return bound(1, reason="exact_value")
        if isinstance(a, str) and isinstance(b, str) and textual:
            return self.text.compare(a, b)
        if type(a) is not type(b):
            return bound(0, reason="argument_type_changed")
        if isinstance(a, dict):
            if a.keys() != b.keys():
                # Optional defaults/selector priority cannot be inferred from name changes.
                return bound(0, 1, reason="argument_keys_or_defaults_unresolved")
            return conjunction([self._values(a[k], b[k], textual=textual or k in TEXT_KEYS) for k in a],
                               "argument_fields")
        if isinstance(a, list):
            if len(a) != len(b):
                return bound(0, reason="argument_list_length_changed")
            if a[:2] == b[:2] == ["literal", "str"] and textual:
                return self.text.compare(a[2], b[2])
            return conjunction([self._values(x, y, textual=textual) for x, y in zip(a, b, strict=True)],
                               "ordered_argument_values")
        return bound(0, reason="argument_value_changed")

    def _program(self, a, b, namespace, dynamic):
        pa = self._python.setdefault(digest([namespace, dynamic, a]), None)
        if pa is None:
            pa = python_description(a, namespace, dynamic)
            self._python[digest([namespace, dynamic, a])] = pa
        pb = self._python.setdefault(digest([namespace, dynamic, b]), None)
        if pb is None:
            pb = python_description(b, namespace, dynamic)
            self._python[digest([namespace, dynamic, b])] = pb
        if pa.get("ast") is not None and pa.get("ast") == pb.get("ast"):
            return bound(1, reason="same_python_ast")
        if not pa["complete"] or not pb["complete"]:
            return bound(0, 1, reason="python_static_plan_unresolved",
                         unsupported=[p.get("reason") for p in (pa, pb) if not p["complete"]])
        events_a, events_b = pa["events"], pb["events"]
        event_matrix = [[self._api_event(x, y) for y in events_b] for x in events_a]
        events = ordered_f1(event_matrix, len(events_a), len(events_b))
        # Persistent module bindings are observable by subsequent actions. Equal API
        # requests alone cannot certify a rename/overwrite as behavior preserving.
        exports = (bound(1, reason="same_persistent_exports") if canonical(pa["exports"]) == canonical(pb["exports"])
                   else bound(0, 1, reason="persistent_exports_differ"))
        outputs = bound(0, reason="print_output_count_changed")
        if len(pa["outputs"]) == len(pb["outputs"]):
            outputs = conjunction([
                self._api_event({"name": "print", "arguments": x}, {"name": "print", "arguments": y}, textual=True)
                for x, y in zip(pa["outputs"], pb["outputs"], strict=True)
            ], "print_outputs")
        result = conjunction([events, exports, outputs], "python_api_plan_and_state")
        result["components"] = {"events": events, "exports": exports, "outputs": outputs}
        return result

    def _symbolic_value(self, a, b, *, textual=False):
        if canonical(a) == canonical(b):
            return bound(1, reason="same_symbolic_value")
        if a[0] == b[0] == "literal":
            if a[1] != b[1]:
                return bound(0, reason="literal_type_changed")
            return self._values(a[2], b[2], textual=textual)
        if a[0] == b[0] and a[0] in ("list", "tuple"):
            if len(a[1]) != len(b[1]):
                return bound(0, reason="literal_container_length_changed")
            return conjunction([self._symbolic_value(x, y, textual=textual)
                                for x, y in zip(a[1], b[1], strict=True)], "symbolic_container")
        if a[0] == b[0] == "dict":
            if len(a[1]) != len(b[1]):
                return bound(0, 1, reason="dictionary_defaults_unresolved")
            return conjunction([conjunction([
                self._symbolic_value(x[0], y[0]),
                self._symbolic_value(x[1], y[1], textual=textual or
                                     (x[0][:2] == ["literal", "str"] and x[0][2] in TEXT_KEYS)),
            ], "symbolic_dictionary_item") for x, y in zip(a[1], b[1], strict=True)], "symbolic_dictionary")
        # Different historical names/subscripts/results may denote the same runtime
        # value. Their identity alone cannot certify a behavioral difference.
        return bound(0, 1, reason="runtime_value_equivalence_unresolved")

    def _api_event(self, a, b, *, textual=False):
        if a["name"] != b["name"]:
            return bound(0, reason="operation_name_changed")
        aa, bb = a["arguments"], b["arguments"]
        if len(aa["args"]) != len(bb["args"]) or aa["kwargs"].keys() != bb["kwargs"].keys():
            return bound(0, 1, reason="api_signature_defaults_unresolved")
        return conjunction([
            *[self._symbolic_value(x, y, textual=textual) for x, y in zip(aa["args"], bb["args"], strict=True)],
            *[self._symbolic_value(aa["kwargs"][k], bb["kwargs"][k], textual=textual or k in TEXT_KEYS)
              for k in aa["kwargs"]],
        ], "api_arguments")

    def _structured(self, a, b):
        if a["name"] != b["name"]:
            return bound(0, reason="operation_name_changed")
        return self._values(a.get("arguments", {}), b.get("arguments", {}))

    def _atom(self, a, b, benchmark, request, context):
        if canonical(a) == canonical(b):
            return bound(1, reason="exact_action")
        if a["name"] != b["name"]:
            return bound(0, reason="operation_name_changed")
        aa, bb = a.get("arguments", {}), b.get("arguments", {})
        code_key = next((key for key in ("code", "python_code")
                         if isinstance(aa.get(key), str) and isinstance(bb.get(key), str)), None)
        if benchmark == "appworld" and code_key:
            remaining = self._values({k: v for k, v in aa.items() if k != code_key},
                                     {k: v for k, v in bb.items() if k != code_key})
            return conjunction([remaining, self._program(aa[code_key], bb[code_key], *context)], "python_action")
        if benchmark == "terminalbench2" and a["name"] == "terminal_command":
            other = self._values({k: v for k, v in aa.items() if k not in ("duration", "keystrokes")},
                                 {k: v for k, v in bb.items() if k not in ("duration", "keystrokes")})
            if aa.get("duration") != bb.get("duration"):
                return bound(0, reason="terminal_timing_changed")
            if not shell_prompt_visible(request):
                return bound(0, 1, reason="terminal_not_confirmed_at_shell_prompt")
            descriptions = []
            for value in (aa.get("keystrokes", ""), bb.get("keystrokes", "")):
                if value not in self._shell:
                    self._shell[value] = self.shell.describe(value)
                descriptions.append(self._shell[value])
            sa, sb = descriptions
            if not sa["complete"] or not sb["complete"]:
                return bound(0, 1, reason="shell_parse_unresolved")
            if sa["syntax"] == sb["syntax"]:
                return conjunction([other, bound(1, reason="same_bash_syntax")], "terminal_action")
            if sa["simple"] and sb["simple"]:
                return bound(0, reason="literal_shell_request_changed")
            return bound(0, 1, reason="shell_effect_equivalence_unresolved")
        return self._structured(a, b)

    def compare(self, predicted, reference, *, benchmark, request, metric="canonical_f1"):
        if metric not in ("canonical_f1", "canonical_accuracy"):
            raise ValueError("Canonical comparator requires canonical_f1 or canonical_accuracy")
        context_key = digest(request)
        key = digest([self._fingerprint, benchmark, context_key, predicted, reference, metric])
        with self._lock:
            if key in self._pairs:
                return dict(self._pairs[key])
            path = self.cache_dir / f"{key}.json"
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if path.exists():
                    cached = json.loads(path.read_text())
                    if cached.get("key") != key or cached.get("result_sha256") != digest(cached["result"]):
                        raise ValueError("Canonical action comparison cache checksum differs")
                    self._pairs[key] = cached["result"]
                    return dict(cached["result"])
                if canonical(predicted) == canonical(reference):
                    result = bound(1, reason="exact_action_list")
                    matrix = []
                elif len(predicted) > 64 or len(reference) > 64:
                    result = bound(0, 1, reason="action_list_size_limit")
                    matrix = []
                else:
                    if context_key not in self._contexts:
                        self._contexts[context_key] = request_namespace(request)
                    context = self._contexts[context_key]
                    matrix = [[self._atom(a, b, benchmark, request, context) for b in reference] for a in predicted]
                    if metric == "canonical_f1":
                        result = ordered_f1(matrix, len(predicted), len(reference))
                    elif len(predicted) != len(reference):
                        result = bound(0, reason="action_list_length_changed")
                    else:
                        diagonal = [matrix[i][i] for i in range(len(predicted))]
                        result = bound(float(all(p["lower"] == 1 for p in diagonal)),
                                       float(all(p["upper"] == 1 for p in diagonal)),
                                       reason="ordered_canonical_accuracy",
                                       approximate=any(p.get("approximate") for p in diagonal))
                result.update(metric=metric, version=VERSION, cache_key=key, environment_execution=False,
                              resolved=result["lower"] == result["upper"],
                              atom_comparisons=matrix)
                payload = {"key": key, "result": result, "result_sha256": digest(result)}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
                temporary.replace(path)
                self._pairs[key] = result
                return dict(result)
