# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""CPU-only embedding worker; run in an environment with torch and transformers.

Pooling follows the cached sentence-transformers configuration (BGE-M3 uses CLS).
This worker intentionally does not import the core Exgentic environment.
"""

import argparse
import json
from pathlib import Path


def pooling_mode(model: Path) -> str:
    config = json.loads((model / "1_Pooling/config.json").read_text())
    modes = {
        "cls": config.get("pooling_mode_cls_token", False),
        "mean": config.get("pooling_mode_mean_tokens", False),
    }
    enabled = [name for name, value in modes.items() if value]
    unsupported = any(
        value
        for key, value in config.items()
        if key.startswith("pooling_mode_") and key not in {"pooling_mode_cls_token", "pooling_mode_mean_tokens"}
    )
    if len(enabled) != 1 or unsupported:
        raise ValueError("Expected exactly one supported pooling mode: CLS or masked mean")
    return enabled[0]


def pool_hidden(hidden, mask, mode):
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        mask = mask.unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError(f"Unsupported pooling mode: {mode}")


def main():
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to("cpu").eval()
    mode = pooling_mode(Path(args.model))
    texts = json.loads(args.input.read_text())
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            tokens = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            hidden = model(**tokens).last_hidden_state
            pooled = pool_hidden(hidden, tokens["attention_mask"], mode)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
    np.save(args.output, np.concatenate(vectors) if vectors else np.empty((0, model.config.hidden_size)))


if __name__ == "__main__":
    main()
