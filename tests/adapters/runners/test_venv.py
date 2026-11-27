# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Venv runner tests (skipped when uv is unavailable)."""

from __future__ import annotations

import os
import shutil

import pytest

from exgentic.adapters.runners import with_runner

from .conftest import Calculator

# Skip entire module if uv is not available.
_uv_available = shutil.which("uv") is not None

pytestmark = [
    pytest.mark.skipif(not _uv_available, reason="uv not available"),
]


@pytest.fixture(scope="module")
def venv_calc():
    """Shared venv-backed Calculator (venv creation is slow)."""
    proxy = with_runner(
        Calculator,
        runner="venv",
        env_name="tests/calculator",
        module_path="exgentic.testing.calculator",
        value=10,
    )
    yield proxy
    try:
        proxy.close()
    except Exception:
        pass


def test_call_method(venv_calc):
    assert venv_calc.add(2, 3) == 5


def test_accumulate(venv_calc):
    assert venv_calc.accumulate(5) == 15


def test_get_attribute(venv_calc):
    assert venv_calc.value == 15


def test_set_attribute(venv_calc):
    venv_calc.value = 42
    assert venv_calc.value == 42


def test_error_propagation(venv_calc):
    with pytest.raises(ZeroDivisionError):
        venv_calc.divide(1, 0)


def test_echo(venv_calc):
    assert venv_calc.echo({"key": [1, 2, 3]}) == {"key": [1, 2, 3]}


def test_different_pid(venv_calc):
    """Venv runner should run in a separate process."""
    assert venv_calc.pid() != os.getpid()


def test_occupied_port_is_not_mistaken_for_our_service(venv_calc):
    """A child that cannot bind must retry, not connect to another service."""
    from exgentic.adapters.runners.venv import VenvRunner

    occupied_port = int(venv_calc._transport._base_url.rsplit(":", 1)[1])
    runner = VenvRunner(
        Calculator,
        env_name="tests/calculator",
        module_path="exgentic.testing.calculator",
        port=occupied_port,
        value=99,
    )
    proxy = runner.start()
    try:
        assert proxy.value == 99
        assert proxy.pid() != venv_calc.pid()
    finally:
        proxy.close()


@pytest.mark.parametrize("transport_timeout", [12600, None])
def test_custom_rpc_timeout_is_not_forwarded_to_remote_constructor(transport_timeout):
    proxy = with_runner(
        Calculator, runner="venv", env_name="tests/calculator",
        module_path="exgentic.testing.calculator", value=7, transport_timeout=transport_timeout,
    )
    try:
        assert proxy.add(2, 3) == 5
        assert proxy._transport._client.timeout.read == transport_timeout
        if transport_timeout is None:
            assert proxy._transport._client.timeout.connect == 30
    finally:
        proxy.close()


def test_active_checkout_budget_policy_is_available_in_isolated_runner():
    from exgentic.core.experiment_limits import ActorTurnBudget, budget_error, experiment_limits

    proxy = with_runner(
        ActorTurnBudget, runner="venv", env_name="tests/calculator",
        limits=experiment_limits("bfcl"), prefix_turns=5,
    )
    try:
        proxy.begin(1)
        assert proxy.turn == 6
        with pytest.raises(Exception) as caught:
            proxy.record({"prompt_tokens": 100_001, "completion_tokens": 10})
        failure = budget_error(caught.value)
        assert failure is not None
        assert failure.details["reason"] == "input_tokens"
        assert failure.details["turn"] == 6
    finally:
        proxy.close()
