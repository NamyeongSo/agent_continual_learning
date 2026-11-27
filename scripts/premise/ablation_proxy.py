# SPDX-License-Identifier: Apache-2.0
"""Forward unmodified study requests through one bounded queue and retain usage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from scripts.utils.serve_checkpoint_query_router import QuerySlots, is_health_probe


def route(path):
    match = re.fullmatch(r"/jobs/([A-Za-z0-9_-]{1,120})(/.*)", urlsplit(path).path)
    if match is None:
        raise ValueError("A study job ID is required in the URL")
    return match.group(1), match.group(2)


def make_handler(upstream, output, limit=100):
    slots = QuerySlots({"8006": upstream.rstrip("/")}, limit=limit)
    lock = threading.Lock()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, data, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, json.dumps(slots.snapshot()).encode())
            try:
                _, path = route(self.path)
                with urllib.request.urlopen(upstream + path, timeout=10) as response:
                    return self.reply(response.status, response.read())
            except (ValueError, OSError) as error:
                return self.reply(503, json.dumps({"error": str(error)}).encode())

        def do_POST(self):
            started = time.monotonic()
            try:
                job, path = route(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                request = json.loads(body)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
            except (ValueError, TypeError) as error:
                return self.reply(400, json.dumps({"error": str(error)}).encode())
            health = is_health_probe(request)
            usage, response_id, error_text = None, None, None
            with slots.claim(priority=0 if health else 1):
                request_started = time.monotonic()
                query = urllib.request.Request(
                    upstream + path, data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": self.headers.get("Authorization", "Bearer EMPTY")},
                )
                try:
                    with urllib.request.urlopen(query, timeout=1800) as response:
                        status, result = response.status, response.read()
                        content_type = response.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as error:
                    status, result = error.code, error.read()
                    content_type = error.headers.get("Content-Type", "application/json")
                except OSError as error:
                    error_text = f"{type(error).__name__}: {error}"
                    status, result = 502, json.dumps({"error": error_text}).encode()
                    content_type = "application/json"
                try:
                    value = json.loads(result)
                    usage, response_id = value.get("usage"), value.get("id")
                except (ValueError, AttributeError):
                    pass
                record = {
                    "job": job, "path": path, "status": status, "health_probe": health,
                    "request_body_sha256": hashlib.sha256(body).hexdigest(),
                    "model": request.get("model"), "temperature": request.get("temperature"),
                    "max_tokens": request.get("max_tokens", request.get("max_completion_tokens")),
                    "extra_body_chat_template_kwargs": request.get("chat_template_kwargs"),
                    "response_id": response_id, "usage": usage, "error": error_text,
                    "request_seconds": time.monotonic() - request_started,
                    "queue_seconds": request_started - started,
                    "completed_at": time.time(),
                }
                with lock:
                    fd = os.open(output / "usage.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                    with os.fdopen(fd, "a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                with slots.condition:
                    slots.completed["8006"] += int(status == 200)
                    slots.errors["8006"] += int(status != 200)
            try:
                return self.reply(status, result, content_type)
            except (BrokenPipeError, ConnectionResetError):
                return None

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8006")
    parser.add_argument("--port", type=int, default=18118)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    class StudyServer(ThreadingHTTPServer):
        request_queue_size = 1024

    server = StudyServer(("127.0.0.1", args.port), make_handler(args.upstream, args.output, args.limit))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
