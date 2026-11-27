# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Bounded symbolic action descriptions; generated Python/shell is never executed."""

from __future__ import annotations

import ast
import json
import re
import shlex
import warnings

MAX_CHARS = 100_000
MAX_EVENTS = 512
DYNAMIC_NAMES = {"eval", "exec", "globals", "locals", "vars", "compile", "getattr", "setattr", "delattr"}


class UnsupportedProgramError(ValueError):
    pass


def literal(value):
    if isinstance(value, dict):
        return ["dict", [[literal(k), literal(v)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [literal(v) for v in value]]
    if value is None or type(value) in (str, int, float, bool):
        return ["literal", type(value).__name__, value]
    raise UnsupportedProgramError("Unsupported literal type")


def unliteral(value):
    if value[0] == "literal":
        return value[2]
    if value[0] in ("list", "tuple"):
        return [unliteral(v) for v in value[1]]
    raise UnsupportedProgramError("Iteration/condition requires literal values")


def request_namespace(request):
    """Preserve historical bindings and only infer direct, side-effect-free aliases."""
    env, dynamic = {}, False
    turn = 0
    for message in request.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            turn += 1
            args = call.get("function", {}).get("arguments", {})
            try:
                args = json.loads(args) if isinstance(args, str) else args
                if not isinstance(args, dict):
                    dynamic = True
                    continue
                code = args.get("code", args.get("python_code", ""))
                tree = ast.parse(code)
            except (SyntaxError, ValueError, TypeError):
                dynamic = True
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in DYNAMIC_NAMES:
                    dynamic = True
                if isinstance(node, ast.Delete):
                    dynamic = True
            for position, statement in enumerate(tree.body):
                old_env = dict(env)
                writes = set()
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        writes.add(node.id)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        writes.add(node.name)
                    if isinstance(node, ast.alias):
                        writes.add(node.asname or node.name.split(".")[0])
                for name in writes:
                    env[name] = ["history_write", turn, position, name]
                opaque = any(isinstance(n, ast.Call) and dotted_name(n.func) != "print" and
                             not (dotted_name(n.func) or "").startswith("apis.") for n in ast.walk(statement))
                if opaque:
                    env = {name: ["history_opaque", turn, position, name] for name in env}
                    continue
                if isinstance(statement, ast.Assign) and all(isinstance(t, ast.Name) for t in statement.targets):
                    if isinstance(statement.value, ast.Name) and statement.value.id in old_env:
                        value = old_env[statement.value.id]
                    else:
                        try:
                            scalar = ast.literal_eval(statement.value)
                            if scalar is not None and type(scalar) not in (str, int, float, bool):
                                continue  # Mutable containers may have changed in a later original action.
                            value = literal(scalar)
                        except (ValueError, TypeError, UnsupportedProgramError):
                            continue
                    for target in statement.targets:
                        env[target.id] = value
    return env, dynamic


def shell_prompt_visible(request):
    for message in reversed(request.get("messages", [])):
        if message.get("role") in ("user", "tool") and isinstance(message.get("content"), str):
            text = message["content"].rstrip()
            return bool(re.search(r"(?:^|\n)[^\n]*(?:[\w@:/~.-]+[#$])\s*$", text))
    return False


class PythonPlan:
    """Small partial interpreter over AST values, not Python eval/exec.

    Only API requests, print, literal containers, aliases and bounded literal loops
    are described. Unsupported flow remains unknown. Global exports keep names:
    renaming a persistent variable is not silently declared equivalent.
    """

    def __init__(self, namespace):
        self.env = dict(namespace)
        self.events, self.outputs, self.exports = [], [], {}
        self.steps = 0

    def value(self, node):
        self.steps += 1
        if self.steps > MAX_EVENTS * 8:
            raise UnsupportedProgramError("Symbolic step budget exceeded")
        try:
            return literal(ast.literal_eval(node))
        except (ValueError, TypeError, UnsupportedProgramError):
            pass
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise UnsupportedProgramError(f"Unresolved name: {node.id}")
            return self.env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return ["list" if isinstance(node, ast.List) else "tuple", [self.value(v) for v in node.elts]]
        if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
            return ["dict", [[self.value(k), self.value(v)] for k, v in zip(node.keys, node.values, strict=True)]]
        if isinstance(node, ast.Subscript):
            base, key = self.value(node.value), self.value(node.slice)
            if base[0] == "dict":
                for k, v in base[1]:
                    if k == key:
                        return v
            if base[0] in ("list", "tuple") and key[0:2] == ["literal", "int"]:
                try:
                    return base[1][key[2]]
                except IndexError as exc:
                    raise UnsupportedProgramError("Literal index out of bounds") from exc
            return ["subscript", base, key]
        if isinstance(node, ast.Call):
            return self.call(node)
        raise UnsupportedProgramError(f"Unsupported expression: {type(node).__name__}")

    def call(self, node):
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise UnsupportedProgramError("Dynamic argument expansion")
        name = dotted_name(node.func)
        if name and name.split(".")[0] in self.env:
            raise UnsupportedProgramError("Callable or API namespace may be shadowed")
        args = [self.value(a) for a in node.args]
        kwargs = {k.arg: self.value(k.value) for k in node.keywords}
        if name == "print":
            self.outputs.append({"args": args, "kwargs": kwargs})
            return literal(None)
        if name and name.startswith("apis."):
            if len(self.events) >= MAX_EVENTS:
                raise UnsupportedProgramError("Symbolic event budget exceeded")
            self.events.append({"name": name, "arguments": {"args": args, "kwargs": kwargs}})
            return ["api_result", len(self.events) - 1]
        if name == "range" and not kwargs:
            values = [unliteral(v) for v in args]
            if all(type(v) is int and abs(v) < 100_000 for v in values):
                sequence = range(*values)
                if len(sequence) <= 64:
                    return literal(list(sequence))
        raise UnsupportedProgramError(f"Unsupported or effectful call: {name}")

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            self.exports[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and value[0] in ("list", "tuple"):
            if len(target.elts) != len(value[1]):
                raise UnsupportedProgramError("Unpacking length mismatch")
            for node, item in zip(target.elts, value[1], strict=True):
                self.assign(node, item)
        else:
            raise UnsupportedProgramError("Attribute/container assignment")

    def statements(self, nodes):
        for node in nodes:
            if isinstance(node, ast.Expr):
                self.value(node.value)
            elif isinstance(node, ast.Assign):
                value = self.value(node.value)
                for target in node.targets:
                    self.assign(target, value)
            elif isinstance(node, ast.For) and not node.orelse:
                sequence = self.value(node.iter)
                if sequence[0] not in ("list", "tuple") or len(sequence[1]) > 64:
                    raise UnsupportedProgramError("Loop iteration is not bounded literal data")
                for item in sequence[1]:
                    self.assign(node.target, item)
                    self.statements(node.body)
            elif isinstance(node, ast.If):
                condition = unliteral(self.value(node.test))
                self.statements(node.body if condition else node.orelse)
            elif isinstance(node, ast.Pass):
                pass
            else:
                raise UnsupportedProgramError(f"Unsupported statement: {type(node).__name__}")


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def python_description(code, namespace, dynamic=False):
    if len(code) > MAX_CHARS:
        return {"kind": "python", "complete": False, "reason": "python_size_limit"}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return {"kind": "python", "complete": False, "reason": "python_parse_error"}
    result = {"kind": "python", "ast": ast.dump(tree, include_attributes=False), "complete": False}
    if dynamic or any(isinstance(n, ast.Name) and n.id in DYNAMIC_NAMES for n in ast.walk(tree)):
        return {**result, "reason": "dynamic_namespace_access"}
    plan = PythonPlan(namespace)
    try:
        plan.statements(tree.body)
    except (UnsupportedProgramError, ValueError, TypeError, IndexError, RecursionError) as exc:
        return {**result, "reason": str(exc)}
    return {**result, "complete": True, "events": plan.events, "outputs": plan.outputs,
            "exports": plan.exports, "reason": "bounded_api_plan"}


class ShellDescriptions:
    def __init__(self):
        self._parser = None

    def prepare(self):
        if self._parser is None:
            try:
                from tree_sitter_languages import get_parser
            except ImportError as exc:
                raise RuntimeError(
                    "Install Exgentic's action-agreement optional dependencies for Bash parsing"
                ) from exc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self._parser = get_parser("bash")

    def describe(self, code):
        if len(code) > MAX_CHARS:
            return {"kind": "shell", "complete": False, "reason": "shell_size_limit"}
        if re.fullmatch(r"C-[A-Za-z]", code):
            return {"kind": "shell", "complete": True, "syntax": ["control_key", code], "simple": True}
        if any(ord(char) < 32 and char != "\n" for char in code):
            return {"kind": "shell", "complete": False, "reason": "interactive_control_character"}
        self.prepare()
        source = code.encode()
        root = self._parser.parse(source).root_node
        if root.has_error:
            return {"kind": "shell", "complete": False, "reason": "shell_parse_error"}
        simple = True

        def visit(node):
            nonlocal simple
            if node.type == "comment":
                return None
            text = source[node.start_byte:node.end_byte].decode()
            if node.type in ("heredoc_start", "heredoc_end"):
                simple = False
                return [node.type, bool(re.search(r"['\"\\]", text))] if node.type == "heredoc_start" else [node.type]
            if node.type in {"expansion", "simple_expansion", "command_substitution", "process_substitution",
                             "function_definition", "for_statement", "while_statement", "if_statement", "heredoc_body"}:
                simple = False
            if node.type in ("word", "raw_string", "string"):
                # Never erase quoting around expansions or unquoted globbing.
                unsafe = (re.search(r"[$`\\]", text) if node.type == "string" else
                          re.search(r"[$`*?\[\\~]", text) if node.type == "word" else False)
                if unsafe:
                    simple = False
                    return [node.type, text]
                try:
                    values = shlex.split(text, posix=True)
                    if len(values) == 1:
                        return ["literal_word", values[0]]
                except ValueError:
                    pass
            if not node.children:
                return [node.type, text]
            children = []
            for child in node.children:
                if child.type == ";" and node.type in ("program", "compound_statement", "do_group"):
                    continue
                value = visit(child)
                if value is not None:
                    children.append(value)
            return [node.type, children]

        # Terminal keystrokes without a final Enter only type a pending command.
        # A parser erases this distinction, so retain it outside the Bash AST.
        syntax = ["terminal_input", {"submitted": code.endswith("\n")}, visit(root)]
        return {"kind": "shell", "complete": True, "syntax": syntax, "simple": simple,
                "reason": "bash_syntax_projection"}
