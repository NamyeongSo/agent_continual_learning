# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Restore automatic tool parsing for experiment clients without restarting vLLM.

Run with the vLLM environment's Python. The upstream chat template must retain
tools with tool_choice=none and render the same prompt for auto and none.
Verify this with /tokenize before routing experiments through the bridge.
Native text requests, including streaming requests, pass through unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ParsingTokenizer:
    """Nonstreaming XML extraction requires sentinels but does not tokenize text."""

    def get_vocab(self):
        return {"<tool_call>": 1, "</tool_call>": 2, "<|tool_call>": 3, "<tool_call|>": 4}


def tool_parser(name: str, tools):
    if name == "gemma4":
        from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

        return Gemma4ToolParser(ParsingTokenizer(), tools=tools)
    if name == "qwen3_coder":
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

        return Qwen3CoderToolParser(ParsingTokenizer(), tools=tools)
    raise ValueError(f"Unsupported tool parser: {name}")


def harmony_parser(tools):
    from vllm.parser.harmony import HarmonyParser
    from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser

    parser = HarmonyParser(ParsingTokenizer(), tools)
    parser.tool_parser = GptOssToolParser(ParsingTokenizer(), tools)
    return parser


def parse_response(response: dict, request, parser_name: str = "qwen3_coder"):
    if request is None:
        return response
    if parser_name == "gpt_oss":
        for choice in response.get("choices", []):
            if choice.get("finish_reason") != "stop":
                continue
            message = choice["message"]
            tokens = choice.get("token_ids")
            if not message.get("tool_calls") and tokens:
                _, content, calls = harmony_parser(request.tools).parse(
                    "", request, enable_auto_tools=True, model_output_token_ids=tokens,
                )
                if calls:
                    message["tool_calls"] = [{"id": "chatcmpl-tool-" + uuid.uuid4().hex,
                                              "type": "function", "function": call.model_dump()} for call in calls]
                    message["content"] = content
                    choice["finish_reason"] = "tool_calls"
            if not request.return_token_ids:
                choice.pop("token_ids", None)
        return response
    parser = tool_parser(parser_name, request.tools)
    for choice in response.get("choices", []):
        # Keep length visible to the existing output-limit retry policy.
        if choice.get("finish_reason") != "stop":
            continue
        message = choice["message"]
        parsed = parser.extract_tool_calls(message.get("content") or "", request)
        if parsed.tools_called:
            message["tool_calls"] = [tool.model_dump() for tool in parsed.tool_calls]
            message["content"] = parsed.content
            choice["finish_reason"] = "tool_calls"
    return response


def make_handler(upstream: str, timeout: float, parser_name: str = "qwen3_coder"):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status: int, data: bytes):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                self.reply(200, json.dumps({"status": "ok", "parser": parser_name, "upstream": upstream}).encode())
                return
            try:
                with urllib.request.urlopen(upstream + self.path, timeout=20) as result:
                    self.reply(result.status, result.read())
            except urllib.error.HTTPError as exc:
                self.reply(exc.code, exc.read())

        def do_POST(self):
            started = time.monotonic()
            response_started = False
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                request = None
                streaming = bool(body.get("stream"))
                if self.path == "/v1/chat/completions" and body.get("tools"):
                    if body.get("tool_choice", "auto") == "auto":
                        if streaming:
                            self.reply(400, b'{"error":{"message":"Automatic tool parsing requires nonstreaming"}}')
                            return
                        request = ChatCompletionRequest(**body)
                        if parser_name == "gemma4":
                            # Match native Gemma4ToolParser.adjust_request: its
                            # delimiters must survive decoding for extraction.
                            body["skip_special_tokens"] = False
                        if parser_name == "gpt_oss":
                            # Keep Harmony input and generation unchanged; recover
                            # tool calls hidden by a server without auto-tools enabled.
                            body["return_token_ids"] = True
                        else:
                            body["tool_choice"] = "none"
                headers = {"Content-Type": "application/json"}
                if self.headers.get("Authorization"):
                    headers["Authorization"] = self.headers["Authorization"]
                query = urllib.request.Request(upstream + self.path, data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(query, timeout=timeout) as result:
                    if streaming:
                        self.send_response(result.status)
                        self.send_header("Content-Type", result.headers.get("Content-Type", "text/event-stream"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        response_started = True
                        self.close_connection = True
                        while chunk := result.readline():
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        metadata = {"stream": True}
                    else:
                        response = parse_response(json.load(result), request, parser_name)
                        self.reply(result.status, json.dumps(response).encode())
                        metadata = {
                            "usage": response.get("usage"),
                            "finish_reasons": [c.get("finish_reason") for c in response.get("choices", [])],
                            "tool_calls": sum(len(c["message"].get("tool_calls") or [])
                                              for c in response.get("choices", [])),
                        }
                print(json.dumps({"status": 200, "elapsed": round(time.monotonic() - started, 3),
                                  **metadata}), flush=True)
            except urllib.error.HTTPError as exc:
                self.reply(exc.code, exc.read())
                print(json.dumps({"status": exc.code, "elapsed": round(time.monotonic() - started, 3)}), flush=True)
            except (BrokenPipeError, ConnectionResetError):
                print(json.dumps({"status": "client_disconnected"}), flush=True)
            except Exception as exc:
                if not response_started:
                    self.reply(502, json.dumps({"error": {"message": f"{type(exc).__name__}: {exc}"}}).encode())
                print(json.dumps({"status": 502, "error_type": type(exc).__name__}), flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8006")
    parser.add_argument("--port", type=int, default=18006)
    parser.add_argument("--read-timeout", type=float, default=610)
    parser.add_argument("--tool-call-parser", choices=("qwen3_coder", "gemma4", "gpt_oss"), default="qwen3_coder")
    args = parser.parse_args()
    handler = make_handler(args.upstream.rstrip("/"), args.read_timeout, args.tool_call_parser)

    class BridgeServer(ThreadingHTTPServer):
        request_queue_size = 1024

    server = BridgeServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
