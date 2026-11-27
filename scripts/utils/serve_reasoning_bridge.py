# SPDX-License-Identifier: Apache-2.0
"""Measure reasoning separately and enforce unchanged visible completion budgets."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.utils.serve_vllm_tool_parser_bridge import parse_response


def split_tokens(ids, end_id, tool_ids):
    if end_id in ids:
        return ids.index(end_id) + 1
    for index in range(len(ids)):
        if ids[index : index + len(tool_ids)] == tool_ids:
            return index
    return len(ids)


def repeated(ids, width=32, repeats=12):
    return len(ids) >= width * repeats and all(
        ids[-width:] == ids[-width * (i + 1) : -width * i] for i in range(1, repeats)
    )


def is_health_probe(body):
    return (
        body.get("messages") == [{"role": "user", "content": "hi"}]
        and (body.get("max_completion_tokens") or body.get("max_tokens")) == 10
        and not body.get("tools")
        and "exgentic_reasoning_budget" not in body
    )


def make_handler(upstream, tokenizer, directory):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    lock = threading.Lock()
    end_id = tokenizer.convert_tokens_to_ids("</think>")
    tool_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/healthz":
                return self.reply(
                    {
                        "ok": True,
                        "parser": "qwen3_coder",
                        "upstream": upstream,
                        "usage": "original completion_tokens; measured reasoning_tokens in details",
                    }
                )
            try:
                with urllib.request.urlopen(upstream + self.path, timeout=10) as response:
                    return self.reply(json.load(response))
            except urllib.error.HTTPError as error:
                return self.reply({"error": error.read().decode()}, error.code)

        def do_POST(self):
            started = time.monotonic()
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/v1/chat/completions" or body.get("stream"):
                    return self.reply({"error": "This cohort requires nonstreaming chat completions"}, 400)
                probe = is_health_probe(body)
                if probe:
                    reasoning_limit, role, profile_sha = None, "health", None
                else:
                    reasoning_limit = int(body.pop("exgentic_reasoning_budget"))
                    role = body.pop("exgentic_role")
                    profile_sha = body.pop("exgentic_profile_sha256")
                visible_limit = int(body.pop("max_completion_tokens", None) or body.get("max_tokens"))
                if not 0 < visible_limit <= (2048 if role == "actor" else 8192):
                    raise ValueError("Invalid visible completion limit")
                body["max_tokens"] = visible_limit if probe else visible_limit + reasoning_limit + 16
                if not probe:
                    body["thinking_token_budget"] = reasoning_limit
                body["return_token_ids"] = True
                if body.get("tools") and body.get("tool_choice", "auto") == "auto":
                    tools = ChatCompletionRequest(**body)
                    body["tool_choice"] = "none"
                else:
                    tools = None
                query = urllib.request.Request(
                    upstream + self.path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(query, timeout=1800) as response:
                    value = json.load(response)
                counts = []
                for choice in value["choices"]:
                    ids = choice.pop("token_ids")
                    thinking = split_tokens(ids, end_id, tool_ids)
                    visible = len(ids) - thinking
                    repetition = repeated(ids[:thinking]) or repeated(ids[thinking:])
                    if visible > visible_limit or repetition:
                        choice["finish_reason"] = "length"
                    counts.append(
                        {
                            "reasoning_tokens": thinking,
                            "visible_tokens": visible,
                            "repetition_detected": repetition,
                            "finish_reason": choice["finish_reason"],
                        }
                    )
                    if choice["message"].get("reasoning") is not None:
                        choice["message"]["reasoning_content"] = choice["message"]["reasoning"]
                usage = value["usage"]
                usage["completion_tokens_details"] = {
                    **(usage.get("completion_tokens_details") or {}),
                    "reasoning_tokens": sum(c["reasoning_tokens"] for c in counts),
                }
                value = parse_response(value, tools, "qwen3_coder")
                record = {
                    "id": value["id"],
                    "at": time.time(),
                    "role": role,
                    "profile_sha256": profile_sha,
                    "temperature": body.get("temperature"),
                    "reasoning_effort": body.get("reasoning_effort"),
                    "enable_thinking": body.get("chat_template_kwargs", {}).get("enable_thinking"),
                    "visible_limit": visible_limit,
                    "reasoning_limit": reasoning_limit,
                    "usage": usage,
                    "choices": counts,
                    "seconds": time.monotonic() - started,
                }
                with lock, (directory / "requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.reply(value)
            except urllib.error.HTTPError as error:
                self.reply({"error": error.read().decode()}, error.code)
            except Exception as error:
                self.reply({"error": f"{type(error).__name__}: {error}"}, 502)

    return Handler


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8006")
    parser.add_argument("--port", type=int, default=18007)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", local_files_only=True)

    class Server(ThreadingHTTPServer):
        request_queue_size = 1024
        daemon_threads = True

    server = Server(("127.0.0.1", args.port), make_handler(args.upstream, tokenizer, args.output))
    print(json.dumps({"pid": os.getpid(), "port": args.port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
