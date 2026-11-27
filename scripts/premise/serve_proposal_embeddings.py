# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Local CPU MiniLM encoder shared by proposal-retrieval experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def model_identity(model):
    model = Path(model).resolve()
    files = {
        str(path.relative_to(model)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(model.rglob("*"))
        if path.is_file() and path.suffix != ".md"
    }
    return {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": model.name,
        "files": files,
        "device": "cpu",
        "pooling": "attention_mask_mean",
        "normalized": True,
        "max_length": 256,
        "dimensions": 384,
    }


class Encoder:
    def __init__(self, model):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModel.from_pretrained(model, local_files_only=True).to("cpu").eval()
        self.identity = model_identity(model)
        self.lock, self.cache, self.full_cache = threading.Lock(), {}, {}

    def encode(self, texts):
        if not isinstance(texts, list) or len(texts) > 256 or any(not isinstance(text, str) for text in texts):
            raise ValueError("Expected at most 256 text strings")
        with self.lock, self.torch.inference_mode():
            missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
            for start in range(0, len(missing), 32):
                batch = missing[start : start + 32]
                lengths = [len(ids) for ids in self.tokenizer(batch, truncation=False)["input_ids"]]
                tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1).tolist()
                for text, vector, length in zip(batch, vectors, lengths, strict=True):
                    self.cache[text] = {"vector": vector, "input_wordpieces": length, "truncated": length > 256}
            return {"identity": self.identity, "items": [self.cache[text] for text in texts]}

    def encode_full(self, texts):
        """Preserve all input tokens; average normalized chunk vectors and normalize again."""
        if not isinstance(texts, list) or len(texts) > 64 or any(not isinstance(text, str) for text in texts):
            raise ValueError("Expected at most 64 full text strings")
        with self.lock, self.torch.inference_mode():
            for text in dict.fromkeys(text for text in texts if text not in self.full_cache):
                ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                special = self.tokenizer.num_special_tokens_to_add(pair=False)
                length = 256 - special
                chunks = [ids[start:start + length] for start in range(0, len(ids), length)] or [[]]
                vectors = []
                for start in range(0, len(chunks), 32):
                    if special != 2 or self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
                        raise ValueError("The pinned MiniLM encoder requires BERT CLS/SEP framing")
                    batch = [[self.tokenizer.cls_token_id, *chunk, self.tokenizer.sep_token_id]
                             for chunk in chunks[start:start + 32]]
                    width = max(map(len, batch))
                    tokens = {
                        "input_ids": self.torch.tensor(
                            [row + [self.tokenizer.pad_token_id] * (width - len(row)) for row in batch]),
                        "attention_mask": self.torch.tensor(
                            [[1] * len(row) + [0] * (width - len(row)) for row in batch]),
                    }
                    hidden = self.model(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                    vectors.append(self.torch.nn.functional.normalize(pooled, p=2, dim=1))
                pooled = self.torch.cat(vectors).mean(dim=0)
                vector = self.torch.nn.functional.normalize(pooled, p=2, dim=0).tolist()
                self.full_cache[text] = {"vector": vector, "input_wordpieces": len(ids) + special,
                                         "truncated": False, "chunks": len(chunks),
                                         "pooling": "full-wordpiece-chunks-mean-l2-v1"}
            return {"identity": self.identity, "items": [self.full_cache[text] for text in texts]}


def serve(model, port):
    encoder = Encoder(model)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def reply(self, status, value):
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"identity": encoder.identity, "capabilities": ["full_chunks_v1"]})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in {"/embed", "/embed-full"}:
                self.reply(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4_000_000:
                    raise ValueError("Invalid request size")
                encode = encoder.encode_full if self.path == "/embed-full" else encoder.encode
                self.reply(200, encode(json.loads(self.rfile.read(size))["texts"]))
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})

    print(json.dumps({"port": port, "identity": encoder.identity}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18886)
    args = parser.parse_args()
    serve(str(args.model.resolve()), args.port)


if __name__ == "__main__":
    main()
