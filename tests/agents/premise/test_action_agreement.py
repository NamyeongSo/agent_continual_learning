# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Offline regressions for representation invariance and conservative uncertainty."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from exgentic.agents.premise.action_agreement import CanonicalActionAgreement
from exgentic.agents.premise.action_nli import LocalTextEquivalence, protected_facts
from exgentic.agents.premise.action_normalization import python_description, request_namespace


class FakeText:
    identity: ClassVar[dict] = {"kind": "test_text", "revision": "test-pinned"}

    def __init__(self):
        self.calls = []

    def prepare(self):
        return self.identity

    def compare(self, a, b):
        self.calls.append((a, b))
        return {"lower": 1., "upper": 1., "reason": "test_paraphrase", "approximate": True}


def request(code=None, *, prompt="root@host:/work# "):
    messages = []
    if code is not None:
        messages.append({"role": "assistant", "tool_calls": [{"function": {
            "name": "execute", "arguments": json.dumps({"code": code}),
        }}]})
    messages.append({"role": "tool", "content": prompt})
    return {"messages": messages}


def action(code, *, name="execute", key="code", **kwargs):
    return {"name": name, "arguments": {key: code, **kwargs}}


@pytest.fixture
def comparator(tmp_path):
    return CanonicalActionAgreement(tmp_path / "cache", text=FakeText())


def compare(comparator, a, b, *, benchmark="appworld", context=None, metric="canonical_f1"):
    return comparator.compare(a, b, benchmark=benchmark, request=context or request(), metric=metric)


@pytest.mark.parametrize(("a", "b"), [
    ("apis.mail.send(to='USR001') # explanation", 'apis.mail.send( to = "USR001" )'),
    ("n = 2\napis.cart.add(quantity=n)", "n=2\napis.cart.add(quantity=2)"),
    ("for i in [1,2]:\n apis.cart.add(quantity=i)",
     "apis.cart.add(quantity=1)\napis.cart.add(quantity=2)\ni=2"),
])
def test_python_surface_and_resolved_dataflow_agree(comparator, a, b):
    result = compare(comparator, [action(a)], [action(b)])
    assert result["lower"] == result["upper"] == 1
    assert not result["environment_execution"]


@pytest.mark.parametrize(("a", "b"), [
    ("apis.mail.send(to='USR001')", "apis.mail.delete(to='USR001')"),
    ("apis.mail.send(to='USR001')", "apis.mail.send(to='USR002')"),
    ("print(apis.mail.send(to='USR001'))", "print(apis.mail.delete(to='USR001'))"),
    ("apis.cart.add(quantity=1)", "apis.cart.add(quantity=True)"),
])
def test_python_actual_request_changes_disagree(comparator, a, b):
    result = compare(comparator, [action(a)], [action(b)])
    assert result["upper"] == 0


def test_api_order_and_duplicates_remain_visible(comparator):
    a, b = "apis.mail.send(to='USR001')", "apis.mail.delete(to='USR002')"
    assert compare(comparator, [action(a + '\n' + b)], [action(b + '\n' + a)])["upper"] == .5
    assert compare(comparator, [action(a + '\n' + b)], [action(b + '\n' + a)],
                   metric="canonical_accuracy")["upper"] == 0
    assert compare(comparator, [action(a + '\n' + a)], [action(a)])["upper"] == pytest.approx(2 / 3)
    atoms = [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
    assert compare(comparator, atoms, atoms[::-1], benchmark="bfcl")["upper"] == .5
    assert compare(comparator, atoms, atoms[::-1], benchmark="bfcl", metric="canonical_accuracy")["upper"] == 0


@pytest.mark.parametrize(("a", "b", "history"), [
    ("x=apis.mail.list()", "y=apis.mail.list()", None),
    ("apis.mail.send(to=x)", "apis.mail.send(to=y)", "x=apis.a()\ny=apis.b()"),
    ("apis.mail.send(to=x)", "apis.mail.send(to='old')", "x='old'\ndel x"),
    ("apis.mail.send(to=x)", "apis.mail.send(to='old')", "x='old'\nexec('x=1')"),
    ("apis.mail.send(to='USR001')", "apis.mail.send(to='USR001', option=True)", None),
    ("import os\nos.remove('/tmp/example')", "pass", None),
])
def test_unproven_bindings_defaults_and_code_remain_unknown(comparator, a, b, history):
    result = compare(comparator, [action(a)], [action(b)], context=request(history))
    assert (result["lower"], result["upper"]) == (0, 1)


def test_historical_aliases_keep_assignment_versions(comparator):
    history = "x='old'\ny=x\nx='new'"
    ns, _ = request_namespace(request(history))
    assert ns["x"][-1] == "new" and ns["y"][-1] == "old"
    result = compare(comparator, [action("apis.mail.send(to=y)")], [action("apis.mail.send(to='old')")],
                     context=request(history))
    assert result["lower"] == 1
    result = compare(comparator, [action("apis.mail.send(to=y)")], [action("apis.mail.send(to=x)")],
                     context=request(history))
    assert result["upper"] == 0
    ns, _ = request_namespace(request("xs=[1]\nxs.append(2)"))
    assert ns["xs"][0] == "history_opaque"


def test_shadowed_builtins_and_api_names_are_unresolved():
    assert not python_description("print('a')", {"print": ["history_write", 1]})["complete"]
    assert not python_description("apis.mail.send()", {"apis": ["history_write", 1]})["complete"]


@pytest.mark.parametrize(("a", "b", "expected"), [
    ("echo 'hello'\n", 'echo "hello" # comment\n', (1, 1)),
    ("echo a; echo b\n", "echo a\necho b\n", (1, 1)),
    ("echo a > out\n", "echo a >> out\n", (0, 0)),
    ("echo a && echo b\n", "echo a || echo b\n", (0, 0)),
    ("echo a; echo b\n", "echo b; echo a\n", (0, 0)),
    ("echo a", "echo a\n", (0, 0)),
    ("echo\ta\n", "echo a\n", (0, 1)),
    ("C-c", "C-c\n", (0, 0)),
    ("echo '$HOME'\n", 'echo "$HOME"\n', (0, 1)),
    ("echo '*.py'\n", "echo *.py\n", (0, 1)),
    ("cat <<'EOF'\n$HOME\nEOF\n", "cat <<EOF\n$HOME\nEOF\n", (0, 1)),
    ("echo 'unterminated", "echo ok\n", (0, 1)),
])
def test_shell_syntax_keeps_operators_quotes_and_order(comparator, a, b, expected):
    pytest.importorskip("tree_sitter_languages")
    result = compare(comparator, [action(a, name="terminal_command", key="keystrokes", duration=1)],
                     [action(b, name="terminal_command", key="keystrokes", duration=1)], benchmark="terminalbench2")
    assert (result["lower"], result["upper"]) == expected


def test_terminal_timing_and_interactive_state_are_retained(comparator):
    a = action("echo 'a'\n", name="terminal_command", key="keystrokes", duration=1)
    b = action('echo "a"\n', name="terminal_command", key="keystrokes", duration=2)
    assert compare(comparator, [a], [b], benchmark="terminalbench2")["upper"] == 0
    b["arguments"]["duration"] = 1
    result = compare(comparator, [a], [b], benchmark="terminalbench2", context=request(prompt=">>> "))
    assert (result["lower"], result["upper"]) == (0, 1)


def test_text_applies_only_to_text_fields_and_preserves_approximation(comparator):
    a = {"name": "send", "arguments": {"content": "Booking confirmed", "to": "USR001"}}
    b = {"name": "send", "arguments": {"content": "Reservation confirmed", "to": "USR001"}}
    result = compare(comparator, [a], [b], benchmark="bfcl")
    assert result["lower"] == 1 and result["approximate"]
    assert len(comparator.text.calls) == 1
    b["arguments"]["to"] = "USR002"
    assert compare(comparator, [a], [b], benchmark="bfcl")["upper"] == 0


def test_cache_binds_original_context_and_pinned_comparator_and_checks_integrity(tmp_path):
    first = CanonicalActionAgreement(tmp_path / "cache", text=FakeText())
    a, b = [action("apis.mail.send(to='USR001') # a")], [action("apis.mail.send(to='USR001') # b")]
    result = compare(first, a, b)
    second = CanonicalActionAgreement(tmp_path / "cache", text=FakeText())
    assert compare(second, a, b) == result
    assert compare(second, a, b, context=request("x=1"))["cache_key"] != result["cache_key"]
    changed_text = FakeText()
    changed_text.identity = {"kind": "test_text", "revision": "different"}
    third = CanonicalActionAgreement(tmp_path / "cache", text=changed_text)
    assert compare(third, a, b)["cache_key"] != result["cache_key"]
    path = tmp_path / "cache" / f"{result['cache_key']}.json"
    path.write_text(path.read_text().replace('"lower": 1.0', '"lower": 0.0'))
    fourth = CanonicalActionAgreement(tmp_path / "cache", text=FakeText())
    with pytest.raises(ValueError, match="checksum"):
        compare(fourth, a, b)


def test_exact_list_fast_path_needs_no_parser_or_text_inference(comparator, monkeypatch):
    monkeypatch.setattr(comparator.text, "compare", lambda *a: pytest.fail("No text inference for exact lists"))
    monkeypatch.setattr(comparator.shell, "prepare", lambda: pytest.fail("No parser for exact lists"))
    atoms = [action("opaque invalid command", name="terminal_command", key="keystrokes")]
    assert compare(comparator, atoms, atoms, benchmark="terminalbench2")["lower"] == 1


def test_text_fact_guards_do_not_turn_numeric_or_negation_changes_into_equivalence():
    assert protected_facts("Pay 1,000.00 to USR001") == protected_facts("Pay 1000 to USR001")
    model = LocalTextEquivalence()
    for a, b in [("Pay $100 to USR001", "Pay $200 to USR001"), ("Do book", "Do not book")]:
        result = model.compare(a, b)
        assert (result["lower"], result["upper"]) == (0, 1)
        assert model._model is None


@pytest.mark.parametrize(("scores", "expected"), [
    ([[0., 6., 0.], [0., 6., 0.]], (1, 1)),
    ([[6., 0., 0.], [6., 0., 0.]], (0, 0)),
    ([[0., 6., 0.], [0., 0., 6.]], (0, 1)),
])
def test_nli_requires_both_directions_and_never_truncates(scores, expected):
    torch = pytest.importorskip("torch")
    model = LocalTextEquivalence(max_tokens=8)

    class Tokenizer:
        def __call__(self, a, b, **kwargs):
            assert kwargs.get("truncation") is False
            return {"input_ids": torch.tensor([[1, 2], [1, 2]])} if isinstance(a, list) else {"input_ids": [1, 2]}

    class Model:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor(scores))

    model._model, model._tokenizer = Model(), Tokenizer()
    result = model.compare("The task finished", "The job completed")
    assert (result["lower"], result["upper"]) == expected
    model._tokenizer = lambda *a, **kw: {"input_ids": list(range(9))}
    result = model.compare("The task has finished", "The job has completed")
    assert result["reason"] == "text_exceeds_nli_context"
