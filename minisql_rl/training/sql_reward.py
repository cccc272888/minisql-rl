"""Deterministic execution rewards for SQL reinforcement learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from minisql_rl.database import SQLSandbox
from minisql_rl.evaluation.sql_eval import evaluate_sql_text


@dataclass(frozen=True)
class SQLRewardConfig:
    """Reward tiers ordered by how useful a generated response is."""

    no_sql: float = -0.5
    execution_error: float = -0.25
    executable_wrong: float = 0.0
    execution_correct: float = 1.0
    strict_format_bonus: float = 0.05

    def __post_init__(self) -> None:
        if not self.no_sql < self.execution_error < self.executable_wrong < self.execution_correct:
            raise ValueError("reward tiers must increase from no_sql to execution_correct")
        if self.strict_format_bonus < 0:
            raise ValueError("strict_format_bonus cannot be negative")


@dataclass(frozen=True)
class SQLRewardOutcome:
    """One scalar reward together with auditable execution components."""

    reward: float
    strict_format: bool
    extracted: bool
    executable: bool
    execution_correct: bool
    extracted_sql: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_sql_response(
    response: str,
    expected_result_hash: str,
    sandbox: SQLSandbox,
    config: SQLRewardConfig | None = None,
) -> SQLRewardOutcome:
    """Score a model response without comparing its text with reference SQL."""

    config = config or SQLRewardConfig()
    result = evaluate_sql_text(response, expected_result_hash, sandbox)
    if result["extracted_sql"] is None:
        reward = config.no_sql
    elif not result["executable"]:
        reward = config.execution_error
    elif result["execution_correct"]:
        reward = config.execution_correct
    else:
        reward = config.executable_wrong

    if result["strict_format"]:
        reward += config.strict_format_bonus

    return SQLRewardOutcome(
        reward=reward,
        strict_format=bool(result["strict_format"]),
        extracted=result["extracted_sql"] is not None,
        executable=bool(result["executable"]),
        execution_correct=bool(result["execution_correct"]),
        extracted_sql=result["extracted_sql"],
        error=result["error"],
    )


def summarize_reward_outcomes(
    outcomes: Iterable[SQLRewardOutcome],
    *,
    num_generations: int,
) -> dict[str, float | int]:
    """Summarize rollout quality and the amount of usable GRPO signal."""

    values = list(outcomes)
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2")
    if len(values) % num_generations:
        raise ValueError("outcome count must be divisible by num_generations")

    total = len(values)
    groups = total // num_generations
    active_groups = 0
    for start in range(0, total, num_generations):
        group_rewards = {item.reward for item in values[start:start + num_generations]}
        active_groups += int(len(group_rewards) > 1)

    def rate(attribute: str) -> float:
        if not total:
            return 0.0
        return round(sum(int(getattr(item, attribute)) for item in values) / total, 4)

    return {
        "responses": total,
        "groups": groups,
        "mean_reward": round(sum(item.reward for item in values) / total, 6) if total else 0.0,
        "strict_format_rate": rate("strict_format"),
        "sql_extraction_rate": rate("extracted"),
        "executable_rate": rate("executable"),
        "execution_accuracy": rate("execution_correct"),
        "active_groups": active_groups,
        "active_group_rate": round(active_groups / groups, 4) if groups else 0.0,
    }
