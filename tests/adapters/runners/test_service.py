# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""HTTP dispatch must preserve signal-based execution timeouts when requested."""

import signal
import threading

import httpx
import pytest

from exgentic.adapters.runners.service import _decode, _encode, create_app
from exgentic.adapters.runners.transport import ObjectHost


class SignalTarget:
    requires_main_thread = True

    def __init__(self):
        self._value = 0

    def install_timeout_handler(self):
        signal.signal(signal.SIGALRM, signal.getsignal(signal.SIGALRM))
        return threading.get_ident()

    @property
    def value(self):
        self.install_timeout_handler()
        return self._value

    @value.setter
    def value(self, value):
        self.install_timeout_handler()
        self._value = value


@pytest.mark.asyncio
async def test_signal_using_calls_getters_and_setters_run_on_main_thread():
    target = SignalTarget()
    transport = httpx.ASGITransport(app=create_app(ObjectHost(target)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/call", json={
            "method": "install_timeout_handler", "args": _encode(()), "kwargs": _encode({}),
        })
        assert response.json()["status"] == "ok"
        assert _decode(response.json()["result"]) == threading.main_thread().ident

        response = await client.post("/set", json={"name": "value", "value": _encode(7)})
        assert response.json()["status"] == "ok"
        response = await client.post("/get", json={"name": "value"})
        assert response.json()["status"] == "ok"
        assert _decode(response.json()["result"]) == 7


@pytest.mark.asyncio
async def test_regular_objects_keep_worker_thread_dispatch():
    target = SignalTarget()
    target.requires_main_thread = False
    transport = httpx.ASGITransport(app=create_app(ObjectHost(target)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/call", json={
            "method": "install_timeout_handler", "args": _encode(()), "kwargs": _encode({}),
        })
    # The same signal call fails off the main thread; opting in is necessary.
    assert response.json()["status"] == "error"
    assert "signal only works in main thread" in response.json()["error_msg"]
