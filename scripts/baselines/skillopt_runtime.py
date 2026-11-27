# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Use the pinned ReflACTTrainer unchanged with the Exgentic EnvAdapter API."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from scripts.baselines.runtime import PROMPTS, digest, final_test, initial_instruction, prompt, write


def train(runtime, upstream_dir):
    # Install transport and prompt hooks before importing trainer modules, whose
    # `from skillopt.model import chat_optimizer` binds at import time.
    import skillopt.model as model
    import skillopt.prompts as prompts

    def chat_optimizer(system, user, max_completion_tokens=8192, retries=3, stage="optimizer", **kwargs):
        return runtime.chat(system, user, stage)

    original_load = prompts.load_prompt

    def load_prompt(name, *args, **kwargs):
        path = PROMPTS / "skillopt" / "optimizer" / f"{name}.md"
        return path.read_text(encoding="utf-8") if path.exists() else original_load(name, *args, **kwargs)

    model.chat_optimizer = chat_optimizer
    model.get_token_summary = runtime.token_summary
    prompts.load_prompt = load_prompt

    from skillopt.config import flatten_config, load_config
    from skillopt.datasets.base import BaseDataLoader, BatchSpec
    from skillopt.engine.trainer import ReflACTTrainer
    from skillopt.envs.base import EnvAdapter

    config = runtime.config

    class Loader(BaseDataLoader):
        def get_train_size(self):
            return len(config["train"])

        def build_train_batch(self, batch_size, seed, **kwargs):
            raise RuntimeError("Use the single-pass epoch plan")

        def plan_train_epoch(self, *, epoch, steps_per_epoch, accumulation, batch_size, seed, **kwargs):
            if epoch != 1 or accumulation != 1:
                raise ValueError("Only the accepted single-pass accumulation=1 protocol is implemented")
            # Payload is an opaque cursor. Materialize IDs only when the trainer
            # requests the current batch through build_env_from_batch().
            return [BatchSpec(phase="train", split="train", seed=seed + index + 1,
                              batch_size=min(batch_size, len(config["train"]) - index * batch_size),
                              payload=index) for index in range(steps_per_epoch)]

        def build_eval_batch(self, env_num, split, seed, **kwargs):
            if split != "valid_seen":
                raise ValueError("Test data is available only after training freezes the best skill")
            return BatchSpec(phase="eval", split="val", seed=seed, batch_size=len(config["validation"]),
                             payload=config["validation"])

    class Adapter(EnvAdapter):
        analyst_workers = runtime.workers
        failure_only = False
        minibatch_size = 5
        edit_budget = 4

        def get_dataloader(self):
            return Loader()

        def build_env_from_batch(self, batch, **kwargs):
            if batch.phase == "train":
                start = batch.payload * 5
                return config["train"][start:start + batch.batch_size]
            return batch.payload

        def build_train_env(self, batch_size, seed, **kwargs):
            raise RuntimeError("Use the ordered stream dataloader")

        def build_eval_env(self, env_num, split, seed, **kwargs):
            raise RuntimeError("Use the fixed validation dataloader")

        def get_task_types(self):
            return [config["benchmark"]]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            match = re.search(r"step_(\d+)", out_dir)
            split = env_manager[0]["split"]
            label = f"train_batch_{int(match.group(1)) - 1}" if split == "train" else "validation"
            results = runtime.batch(env_manager, skill_content, label)
            rows = []
            for result in results:
                if split != "train":
                    rows.append({"id": result["id"], "hard": result["hard"], "soft": result["soft"],
                                 "task_type": config["benchmark"]})
                    continue
                trace = result["trace"]
                row = {**result, "task_type": config["benchmark"], "task_description": str(trace["task"]),
                       "n_turns": len(trace["actions"]), "fail_reason": "" if result["hard"] else str(trace["details"])}
                rows.append(row)
                if split == "train":
                    conversation = [{"role": "user", "content": str(trace["task"])},
                                    {"role": "system", "content": str(trace["public_tool_contract"])}]
                    conversation.extend({"step": index, "action": action["action"],
                                         "env_feedback": action.get("observation")}
                                        for index, action in enumerate(trace["actions"]))
                    write(Path(out_dir) / "predictions" / result["id"] / "conversation.json", conversation)
            if split == "train":
                runtime.online(env_manager, results, skill_content, int(match.group(1)) - 1)
            write(Path(out_dir) / "results.json", rows)
            return rows

        def get_error_minibatch_prompt(self):
            return prompt("skillopt", "optimizer/analyst_error.md")

        def get_success_minibatch_prompt(self):
            return prompt("skillopt", "optimizer/analyst_success.md")

    root = Path(inspect.getfile(ReflACTTrainer)).resolve().parents[2]
    cfg = flatten_config(load_config(str(root / "configs/_base_/default.yaml")))
    skill_init = PROMPTS / "skillopt/base/seed_skill.md"
    if config.get("actor_context", "legacy") != "legacy":
        skill_init = Path(upstream_dir) / "initial_skill.md"
        skill_init.write_text(initial_instruction("skillopt", config["actor_context"]), encoding="utf-8")
    cfg.update(
        env=config["benchmark"], out_root=str(upstream_dir), seed=config.get("optimizer_seed", config["seed"]),
        num_epochs=1, batch_size=5, accumulation=1, minibatch_size=5, analyst_workers=runtime.workers,
        edit_budget=4, min_edit_budget=2, lr_scheduler="cosine", lr_control_mode="fixed",
        skill_update_mode="patch", failure_only=False, use_gate=True, gate_metric="hard",
        use_semantic_density=False, use_slow_update=False, use_meta_skill=True,
        skill_init=str(skill_init), eval_test=False,
        model_backend="openai_compatible", optimizer_backend="openai_compatible",
        target_backend="openai_compatible", optimizer_model=config["model"].removeprefix("openai/"),
        target_model=config["model"].removeprefix("openai/"), reasoning_effort="",
    )
    result = ReflACTTrainer(cfg, Adapter()).train()
    learned = (upstream_dir / "best_skill.md").read_text(encoding="utf-8")
    write(runtime.output / "training_result.json", {
        "entrypoint": "skillopt.engine.trainer.ReflACTTrainer.train", "upstream": result,
        "prompt_sha256": digest(learned), "total_training_limit": None,
        "unique_train_encounters": len(list((runtime.output / "online").glob("*.json"))),
        "token_accounting": "analysis_calls.jsonl and actor session cost reports",
    })
    return final_test(runtime, learned)
