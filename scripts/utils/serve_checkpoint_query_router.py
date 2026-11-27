# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Forward unchanged evaluation queries with an independent in-flight limit per GPU."""

from __future__ import annotations

import argparse
import heapq
import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class QuerySlots:
    def __init__(self, endpoints, limit=100):
        self.endpoints = endpoints
        self.limit = limit
        self.limits = {key: limit[key] if isinstance(limit, dict) else limit for key in endpoints}
        if any(type(value) is not int or value < 1 for value in self.limits.values()):
            raise ValueError("Every endpoint needs a positive concurrency limit")
        self.condition = threading.Condition()
        self.waiting = []
        self.ticket_serial = 0
        self.active = dict.fromkeys(endpoints, 0)
        self.maximum = dict.fromkeys(endpoints, 0)
        self.completed = dict.fromkeys(endpoints, 0)
        self.errors = dict.fromkeys(endpoints, 0)
        self.last_used = dict.fromkeys(endpoints, 0)
        self.serial = 0

    @contextmanager
    def claim(self, priority=1):
        with self.condition:
            self.ticket_serial += 1
            ticket = (priority, self.ticket_serial, object())
            heapq.heappush(self.waiting, ticket)
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.waiting[0] is ticket and any(
                n < self.limits[key] for key, n in self.active.items()))
            heapq.heappop(self.waiting)
            gpu = min((k for k in self.endpoints if self.active[k] < self.limits[k]),
                      key=lambda k: (self.active[k] / self.limits[k], self.last_used[k]))
            self.serial += 1
            self.last_used[gpu] = self.serial
            self.active[gpu] += 1
            self.maximum[gpu] = max(self.maximum[gpu], self.active[gpu])
            self.condition.notify_all()
        try:
            yield gpu, self.endpoints[gpu]
        finally:
            with self.condition:
                self.active[gpu] -= 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"per_gpu_query_limit": self.limit if isinstance(self.limit, int) else None,
                    "per_endpoint_query_limits": dict(self.limits), "waiting": len(self.waiting),
                    "queue": "single_priority", "health_priority": True,
                    "active": dict(self.active), "maximum": dict(self.maximum),
                    "completed": dict(self.completed), "errors": dict(self.errors), "endpoints": self.endpoints}


def is_health_probe(body):
    return (body.get("messages") == [{"role": "user", "content": "hi"}]
            and (body.get("max_completion_tokens") or body.get("max_tokens")) == 10
            and not body.get("tools") and "exgentic_reasoning_budget" not in body)


def handler(slots, output):
    log_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def reply(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, json.dumps(slots.snapshot()).encode())
            if self.path != "/v1/models":
                return self.reply(404, b'{}')
            for endpoint in slots.endpoints.values():
                try:
                    with urllib.request.urlopen(endpoint + "/models", timeout=5) as response:
                        return self.reply(response.status, response.read())
                except OSError:
                    pass
            return self.reply(503, b'{"error":"model endpoints unavailable"}')

        def do_POST(self):
            if self.path not in {"/v1/chat/completions", "/v1/completions"}:
                return self.reply(404, b'{}')
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            started = time.time()
            try:
                health = is_health_probe(json.loads(body))
            except (ValueError, TypeError, AttributeError):
                health = False
            with slots.claim(priority=0 if health else 1) as (gpu, endpoint):
                try:
                    request = urllib.request.Request(endpoint + self.path.removeprefix("/v1"), data=body,
                                                     headers={"Content-Type": "application/json",
                                                              "Authorization": "Bearer EMPTY"})
                    with urllib.request.urlopen(request, timeout=1800) as response:
                        status, result = response.status, response.read()
                        content_type = response.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as error:
                    status, result = error.code, error.read()
                    content_type = error.headers.get("Content-Type", "application/json")
                except OSError as error:
                    status, result = 502, json.dumps({"error": {"message": str(error)}}).encode()
                    content_type = "application/json"
                with slots.condition:
                    slots.completed[gpu] += int(status == 200)
                    slots.errors[gpu] += int(status != 200)
                with log_lock, (output / "query_routes.jsonl").open("a") as stream:
                    stream.write(json.dumps({"gpu": gpu, "started": started, "seconds": time.time() - started,
                                             "status": status, "health_probe": health}) + "\n")
                return self.reply(status, result, content_type)

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18008)
    parser.add_argument("--per-gpu-limit", type=int, default=100)
    parser.add_argument("--endpoint", action="append", default=[], metavar="NAME=URL")
    parser.add_argument("--endpoint-limit", action="append", default=[], metavar="NAME=N")
    args = parser.parse_args()
    if args.per_gpu_limit < 1:
        parser.error("--per-gpu-limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    endpoints = dict(item.split("=", 1) for item in args.endpoint) if args.endpoint else {
        "0": "http://127.0.0.1:18007/v1", "1": "http://127.0.0.1:18006/v1"}
    limits = dict.fromkeys(endpoints, args.per_gpu_limit)
    for item in args.endpoint_limit:
        key, value = item.split("=", 1)
        if key not in endpoints:
            parser.error("Endpoint limit names must match configured endpoints")
        limits[key] = int(value)
    slots = QuerySlots(endpoints, limits if args.endpoint_limit else args.per_gpu_limit)

    class Server(ThreadingHTTPServer):
        request_queue_size = 1024
        daemon_threads = True

    server = Server(("127.0.0.1", args.port), handler(slots, args.output))
    server.serve_forever()
