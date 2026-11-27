# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Optional rubric audits and measured reliability-constrained text repair.

Audits use task-disjoint calibration/development partitions reserved from train.
Final validation labels never enter repair packets. Multi-model alpha requires
at least three distinct model/endpoint identities, matching the paper's ensemble.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from . import rubric_prompts as prompts
from .reward_data import BENCHMARKS, digest, write_json
from .rubric_pipeline import criterion_text

PROPERTIES = ("atomicity", "consistency", "observability", "operationalizability", "scope")


def bounded_failure_packet(packet: dict, tokenizer, limit: int) -> dict:
    tokens = tokenizer.encode(json.dumps(packet, ensure_ascii=False), add_special_tokens=False).ids
    if len(tokens) <= limit:
        return packet
    budget = max(0, limit - 128)
    while budget >= 0:
        clipped = {
            "truncated": True,
            "original_tokens": len(tokens),
            "record_excerpt": tokenizer.decode(tokens[:budget]),
            "omission": "Calibration packet truncated; do not infer omitted evidence.",
        }
        size = len(tokenizer.encode(json.dumps(clipped, ensure_ascii=False), add_special_tokens=False).ids)
        if size <= limit:
            return clipped
        budget -= max(1, size - limit)
    raise ValueError("Repair packet limit cannot fit omission metadata")


def ordinal_alpha(scores) -> float | None:
    """Krippendorff alpha with ordinal distances derived from pooled marginals."""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 2:
        return None
    if not np.isfinite(values).all() or (values < 0).any() or (values > 10).any():
        raise ValueError("Ordinal ratings must be complete and in [0,10]")
    categories, encoded = np.unique(values, return_inverse=True)
    encoded = encoded.reshape(values.shape)
    count = len(categories)
    coincidence = np.zeros((count, count))
    for item in encoded:
        frequencies = np.bincount(item, minlength=count)
        coincidence += (np.outer(frequencies, frequencies) - np.diag(frequencies)) / (len(item) - 1)
    marginals = coincidence.sum(axis=1)
    expected = (np.outer(marginals, marginals) - np.diag(marginals)) / (marginals.sum() - 1)
    distances = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            distance = (marginals[i : j + 1].sum() - (marginals[i] + marginals[j]) / 2) ** 2
            distances[i, j] = distances[j, i] = distance
    denominator = (expected * distances).sum()
    return float(1 - (coincidence * distances).sum() / denominator) if denominator else None


def repair_budget(value: dict) -> dict:
    text = criterion_text(value)
    gate, clauses = value.get("applicability_gate", ""), value.get("anti_gaming", [])
    if not isinstance(gate, str) or not isinstance(clauses, list) or len(clauses) > 2:
        raise ValueError("At most one applicability gate and two anti-gaming clauses")
    if len(text["description"].split()) > 60 or any(
        not isinstance(clause, str) or len(clause.split()) > 40 for clause in clauses
    ):
        raise ValueError("Repair definition/clause word budget exceeded")
    description = "\n".join(part for part in [gate, text["description"], *clauses] if part)
    if len(description.split()) > 150:
        raise ValueError("Repair total word budget exceeded")
    return {"name": text["name"], "description": description}


def accepts_repair(
    original: dict, candidate: dict, *, reliability_slack: float = 0, preference_slack: float = 0
) -> bool:
    keys = ("vfr", "alpha", "preference_fit")
    if any(original.get(key) is None or candidate.get(key) is None for key in keys):
        return False
    return (
        candidate["vfr"] < original["vfr"]
        and candidate["alpha"] >= original["alpha"] - reliability_slack
        and candidate["preference_fit"] >= original["preference_fit"] - preference_slack
    )


def pareto_repairs(candidates: list[dict]) -> list[dict]:
    return [
        row
        for row in candidates
        if not any(
            other["metrics"]["alpha"] >= row["metrics"]["alpha"]
            and other["metrics"]["vfr"] <= row["metrics"]["vfr"]
            and (other["metrics"]["alpha"] > row["metrics"]["alpha"] or other["metrics"]["vfr"] < row["metrics"]["vfr"])
            for other in candidates
        )
    ]


@dataclass(frozen=True)
class AuditConfig:
    pairs_per_benchmark: int = 6
    repetitions: int = 5
    paraphrases: int = 5
    repair_beam: int = 2
    repair_depth: int = 2
    reliability_slack: float = 0
    preference_slack: float = 0
    packet_max_tokens: int = 16384

    def __post_init__(self):
        if (
            min(self.pairs_per_benchmark, self.repetitions, self.paraphrases, self.repair_beam, self.packet_max_tokens)
            < 1
        ):
            raise ValueError("Audit counts must be positive")
        if not 1 <= self.repair_depth <= len(prompts.OPERATORS):
            raise ValueError("Invalid repair depth")
        if self.reliability_slack < 0 or self.preference_slack < 0:
            raise ValueError("Repair slacks must be nonnegative")
        if self.packet_max_tokens < 128:
            raise ValueError("Repair packet budget must leave room for omission metadata")


class RubricAuditor:
    def __init__(self, pipeline, judges: list, config: AuditConfig):
        self.pipeline, self.judges, self.config = pipeline, judges, config
        self.identities = [(judge.base_url, judge.model) for judge in judges]
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("Audit ensemble requires distinct model/endpoint identities")
        self.calibration, self.dev = [], []
        for benchmark in BENCHMARKS:
            rows = [r for r in pipeline.audit if r["benchmark"] == benchmark]
            tasks = sorted({r["task_id"] for r in rows})
            if len(tasks) < 2:
                raise ValueError("Auditing needs at least two reserved train tasks per benchmark")
            for destination, selected in ((self.calibration, set(tasks[::2])), (self.dev, set(tasks[1::2]))):
                records = sorted((r for r in rows if r["task_id"] in selected), key=lambda r: digest(r["id"]))
                destination.extend(records[: config.pairs_per_benchmark])

    def ordinal(self, criterion, record, llm, tag, temperature=0):
        def parse(value):
            score = value.get("score")
            if type(score) is not int or not 0 <= score <= 10:
                raise ValueError("Invalid ordinal rating")
            return score

        return self.pipeline.call(
            "ordinal",
            [criterion["id"], record["id"], tag],
            prompts.ORDINAL,
            {
                "criterion": self.pipeline.public_criteria([criterion])[0],
                "comparison": self.pipeline.visible[record["id"]],
            },
            parse,
            temperature=temperature,
            llm=llm,
        )

    def attack(self, criterion, record, index):
        visible = self.pipeline.visible[record["id"]]
        prompt = {key: visible[key] for key in ("task", "context", "public_tools", "common_prefix")}
        reference = visible["trajectory_a" if record["label"] == "A" else "trajectory_b"]

        def parse_response(value):
            if not isinstance(value.get("response"), str) or not value["response"].strip():
                raise ValueError("Missing synthetic attack response")
            return value["response"]

        response = self.pipeline.call(
            "attack",
            [criterion["id"], record["id"], index],
            prompts.ATTACK,
            {
                "criterion": self.pipeline.public_criteria([criterion])[0],
                "prompt": prompt,
                "reference": reference,
                "strategy": prompts.STRATEGIES[index % 6],
            },
            parse_response,
            temperature=0.7,
        )
        return {
            "pair_id": record["id"],
            "prompt": prompt,
            "reference": reference,
            "response": response,
            "synthetic": True,
            "environment_executed": False,
        }

    def inspect_attack(self, criterion, attack):
        def parse_score(value):
            if type(value.get("score")) is not int or not 0 <= value["score"] <= 10:
                raise ValueError("Invalid pointwise attack score")
            return value["score"]

        score = self.pipeline.call(
            "attack_score",
            [criterion["id"], digest(attack)],
            prompts.POINT_SCORE,
            {
                "criterion": self.pipeline.public_criteria([criterion])[0],
                "prompt": attack["prompt"],
                "response": attack["response"],
            },
            parse_score,
        )
        exploit = False
        if score >= 7:

            def parse_verdict(value):
                if value.get("verdict") not in ("EXPLOITING", "GENUINE"):
                    raise ValueError("Invalid exploit verification")
                return value["verdict"] == "EXPLOITING"

            exploit = self.pipeline.call(
                "verify_attack",
                digest(attack),
                prompts.VERIFY,
                {key: attack[key] for key in ("prompt", "reference", "response")},
                parse_verdict,
                llm=self.judges[-1],
            )
        return {**attack, "score": score, "verified_exploit": exploit}

    def measure(self, criterion, records, stage: str, *, attacks=None, extended=False):
        pipeline = self.pipeline
        ratings = pipeline.jobs(
            f"{stage}_ratings",
            records,
            lambda record: [self.ordinal(criterion, record, judge, "deterministic") for judge in self.judges],
        )
        alpha = ordinal_alpha(ratings) if len(self.judges) >= 3 else None
        fit_rows = pipeline.jobs(
            f"{stage}_fit", records, lambda record: pipeline.score_pair(record, [criterion], stage="audit_fit")[0]
        )
        from .rubric_pipeline import preference_statistics

        stats = preference_statistics(fit_rows, {r["id"]: r["label"] for r in records})
        if attacks is None:
            attacks = pipeline.jobs(
                f"{stage}_attacks", list(enumerate(records)), lambda item: self.attack(criterion, item[1], item[0])
            )
        inspected = pipeline.jobs(
            f"{stage}_verification", attacks, lambda attack: self.inspect_attack(criterion, attack)
        )
        result = {
            "alpha": alpha,
            **stats,
            "vfr": sum(r["verified_exploit"] for r in inspected) / len(inspected),
            "attack_count": len(inspected),
            "judge_count": len(self.judges),
            "natural_scores": fit_rows,
            "attacks": inspected,
        }
        if extended:
            stochastic = pipeline.jobs(
                f"{stage}_consistency",
                records,
                lambda record: [
                    self.ordinal(criterion, record, self.judges[0], ["stochastic", run], 0.7)
                    for run in range(self.config.repetitions)
                ],
            )
            signs = [[np.sign(score - 5) for score in row] for row in stochastic]
            result["flip_rate"] = sum(len(set(row)) > 1 for row in signs) / len(signs)
            paraphrases = []
            for index in range(self.config.paraphrases):
                text = pipeline.call(
                    "paraphrase",
                    [criterion["id"], index],
                    prompts.PARAPHRASE,
                    {"definition": criterion["description"]},
                    lambda value: criterion_text({"name": criterion["name"], **value})["description"],
                    temperature=0.8,
                )
                paraphrases.append({**criterion, "description": text})
            vectors = []
            for index, paraphrase in enumerate(paraphrases):
                vectors.append(
                    pipeline.jobs(
                        f"{stage}_paraphrase_{index}",
                        records,
                        lambda record, c=paraphrase, i=index: self.ordinal(
                            c, record, self.judges[0], ["paraphrase", i]
                        ),
                    )
                )
            correlations = []
            for a, b in itertools.combinations(vectors, 2):
                if len(set(a)) > 1 and len(set(b)) > 1:
                    correlations.append(float(spearmanr(a, b).statistic))
            result["paraphrase_rho"] = float(np.mean(correlations)) if correlations else None
        return result

    def structure(self, criterion):
        def parse(value):
            if any(type(value.get(key)) is not bool for key in PROPERTIES):
                raise ValueError("Expected five boolean structure checks")
            return {key: value[key] for key in PROPERTIES}

        votes = [
            self.pipeline.call(
                "structure",
                criterion["id"],
                prompts.STRUCTURE,
                {"criterion": self.pipeline.public_criteria([criterion])[0]},
                parse,
                llm=judge,
            )
            for judge in self.judges
        ]
        passed = {key: sum(vote[key] for vote in votes) > len(votes) / 2 for key in PROPERTIES}
        return {
            "votes": votes,
            "properties": passed,
            "score": sum(passed.values()),
            "fraction": sum(passed.values()) / 5,
            "judge_count": len(votes),
            "paper_three_judge_ensemble": len(votes) >= 3,
        }

    def run(self, rubric: list[dict]) -> dict:
        report = {
            "judges": self.identities,
            "calibration_ids": [r["id"] for r in self.calibration],
            "dev_ids": [r["id"] for r in self.dev],
            "validation_used": False,
            "criteria": [],
        }
        for criterion in rubric:
            metrics = self.measure(criterion, self.calibration, f"audit_{criterion['id']}", extended=True)
            report["criteria"].append(
                {"criterion": criterion, "structure": self.structure(criterion), "metrics": metrics}
            )
            write_json(self.pipeline.output / "audit.json", report)
        return report

    def repair(self, rubric: list[dict], audit: dict) -> list[dict]:
        """Measured singleton calibration, predictor beam, dev acceptance and Pareto retention."""
        if len(self.judges) < 3:
            raise ValueError("Reliability-constrained repair requires three distinct judge models/endpoints")
        by_id = {row["criterion"]["id"]: row["metrics"] for row in audit["criteria"]}
        result, decisions = [], []
        for criterion in rubric:
            calibration = by_id[criterion["id"]]
            original = self.measure(criterion, self.dev, f"dev_{criterion['id']}")
            packet = {
                "natural_pairs": [
                    {
                        "comparison": self.pipeline.visible[row["pair_id"]],
                        "winner": self.pipeline.by_id[row["pair_id"]]["label"],
                        "scores": row,
                    }
                    for row in calibration["natural_scores"]
                ],
                "attacks": calibration["attacks"],
            }
            # Full trajectories from many failure examples cannot all fit in one repair request.
            # Preserve the original calibration file; truncate only this explanatory packet.
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(self.pipeline.manifest["environment_feedback"]["tokenizer_path"])
            packet = bounded_failure_packet(packet, tokenizer, self.config.packet_max_tokens)
            proposals = []
            rejected = []
            singleton_metrics = []
            for size in range(1, self.config.repair_depth + 1):
                for operators in itertools.combinations(prompts.OPERATORS, size):
                    try:
                        text = self.pipeline.call(
                            "repair",
                            [criterion["id"], operators],
                            prompts.REPAIR,
                            {
                                "criterion": self.pipeline.public_criteria([criterion])[0],
                                "failure_packet": packet,
                                "operators": operators,
                                "singleton_calibration": singleton_metrics,
                            },
                            repair_budget,
                        )
                    except ValueError as exc:
                        rejected.append({"operators": operators, "reason": str(exc)})
                        continue
                    candidate = {**criterion, **text, "id": f"r{digest([criterion['id'], operators])[:12]}"}

                    def parse_genericity(value):
                        rows = value.get("criteria")
                        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("id") != "r1":
                            raise ValueError("Invalid repair genericity response")
                        if type(rows[0].get("reusable")) is not bool:
                            raise ValueError("Repair genericity must be boolean")
                        return rows[0]["reusable"]

                    public, _aliases = self.pipeline.aliased_criteria([candidate])
                    reusable = self.pipeline.call(
                        "repair_genericity", candidate["id"], prompts.GENERICITY, {"criteria": public}, parse_genericity
                    )
                    if not reusable:
                        rejected.append({"operators": operators, "reason": "Revision is not task-independent"})
                        continue
                    if size == 1:
                        metrics = self.measure(
                            candidate, self.calibration, f"cal_{candidate['id']}", attacks=calibration["attacks"]
                        )
                        singleton_metrics.append(
                            {
                                "operators": operators,
                                "criterion": text,
                                "metrics": {k: metrics[k] for k in ("vfr", "alpha", "preference_fit")},
                            }
                        )
                        predicted_vfr = metrics["vfr"]
                    else:
                        prediction = self.pipeline.call(
                            "predict_repair",
                            candidate["id"],
                            "Estimate this revision's VFR from singleton calibration. "
                            'Return JSON {"vfr": 0.0}; the number is a prediction only.',
                            {
                                "original": {k: calibration[k] for k in ("vfr", "alpha")},
                                "candidate": text,
                                "singletons": singleton_metrics,
                            },
                        )
                        predicted_vfr = prediction.get("vfr")
                        if type(predicted_vfr) not in (int, float) or not 0 <= predicted_vfr <= 1:
                            raise ValueError("Invalid predicted VFR")
                    if predicted_vfr < calibration["vfr"]:
                        proposals.append(
                            {"criterion": candidate, "predicted_vfr": predicted_vfr, "operators": operators}
                        )
            proposals.sort(key=lambda row: (row["predicted_vfr"], row["criterion"]["id"]))
            accepted = []
            for proposal in proposals[: self.config.repair_beam]:
                metrics = self.measure(
                    proposal["criterion"], self.dev, f"dev_{proposal['criterion']['id']}", attacks=original["attacks"]
                )
                if accepts_repair(
                    original,
                    metrics,
                    reliability_slack=self.config.reliability_slack,
                    preference_slack=self.config.preference_slack,
                ):
                    accepted.append({**proposal, "metrics": metrics})
            frontier = pareto_repairs(accepted)
            winner = (
                min(frontier, key=lambda row: (row["metrics"]["vfr"], -row["metrics"]["alpha"])) if frontier else None
            )
            result.append(winner["criterion"] if winner else criterion)
            decisions.append(
                {
                    "original": criterion,
                    "original_metrics": original,
                    "frontier": frontier,
                    "selected": result[-1],
                    "proposal_count": len(proposals),
                    "budget_rejections": rejected,
                }
            )
            write_json(self.pipeline.output / "repair_decisions.json", decisions)
        write_json(self.pipeline.output / "rubrics" / "robustness_repaired.json", result)
        return result
