from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .costs import CostSimulator, baseline_models, model_rank
from .policy import Router
from .text import estimate_tokens
from .types import CallContext, Trajectory


@dataclass(slots=True)
class CallEvaluation:
    call_index: int
    routed_model: str
    actual_model: str
    failure_risk: float
    success_proxy: float
    downgrade: bool
    actual_action_items: int


@dataclass(slots=True)
class EvaluationResult:
    strategy: str
    trajectory_id: str
    call_evaluations: list[CallEvaluation] = field(default_factory=list)
    total_cost: float = 0.0
    cache_hit_rate: float = 0.0
    success_proxy: float = 0.0
    routed_calls: int = 0
    switched_models: int = 0


class Evaluator:
    def __init__(self, simulator: CostSimulator | None = None) -> None:
        self.simulator = simulator or CostSimulator()

    def failure_risk(self, call: CallContext, routed_model: str) -> float:
        actual_rank = model_rank(call.model)
        routed_rank = model_rank(routed_model)

        prompt_tokens = max(1, estimate_tokens(call.input_text))
        tool_count = len(call.tools)
        reasoning_items = sum(
            1
            for item in call.input_items
            if isinstance(item, dict) and str(item.get("type", "")).lower() == "reasoning"
        )
        action_items = len(call.actual_action.get("items", [])) if call.actual_action else 0

        complexity = min(
            1.0,
            0.18 * (prompt_tokens / 2000.0)
            + 0.12 * min(tool_count, 10)
            + 0.14 * reasoning_items
            + 0.08 * action_items,
        )

        downgrade_gap = max(0, routed_rank - actual_rank)
        downgrade_penalty = 0.0
        if downgrade_gap > 0:
            downgrade_penalty = min(0.65, 0.10 + 0.08 * downgrade_gap * (1.0 + complexity))

        base_risk = min(0.55, 0.15 * complexity)
        return min(0.99, base_risk + downgrade_penalty)

    def success_proxy(self, call: CallContext, routed_model: str) -> float:
        return max(0.0, 1.0 - self.failure_risk(call, routed_model))

    def evaluate_trajectory(self, trajectory: Trajectory, router: Router) -> EvaluationResult:
        routed_models = [router.route(call.input_items, call.tools) for call in trajectory.calls]
        simulation = self.simulator.simulate_trajectory(
            trajectory,
            routed_models=routed_models,
            strategy=router.strategy,
        )

        call_evaluations: list[CallEvaluation] = []
        success_values: list[float] = []
        for call, routed_model in zip(trajectory.calls, routed_models):
            success = self.success_proxy(call, routed_model)
            success_values.append(success)
            call_evaluations.append(
                CallEvaluation(
                    call_index=call.call_index,
                    routed_model=routed_model,
                    actual_model=call.model,
                    failure_risk=1.0 - success,
                    success_proxy=success,
                    downgrade=model_rank(routed_model) > model_rank(call.model),
                    actual_action_items=len(call.actual_action.get("items", [])) if call.actual_action else 0,
                )
            )

        return EvaluationResult(
            strategy=router.strategy,
            trajectory_id=trajectory.trajectory_id,
            call_evaluations=call_evaluations,
            total_cost=simulation.total_cost,
            cache_hit_rate=simulation.cache_hit_rate,
            success_proxy=sum(success_values) / len(success_values) if success_values else 0.0,
            routed_calls=len(routed_models),
            switched_models=simulation.switched_models,
        )

    def evaluate_dataset(
        self,
        trajectories: Iterable[Trajectory],
        router: Router,
    ) -> tuple[list[EvaluationResult], list[dict]]:
        results: list[EvaluationResult] = []
        rows: list[dict] = []
        for trajectory in trajectories:
            result = self.evaluate_trajectory(trajectory, router)
            results.append(result)
            rows.append(
                {
                    "strategy": result.strategy,
                    "trajectory_id": result.trajectory_id,
                    "total_cost": result.total_cost,
                    "cache_hit_rate": result.cache_hit_rate,
                    "success_proxy": result.success_proxy,
                    "routed_calls": result.routed_calls,
                    "switched_models": result.switched_models,
                }
            )
        return results, rows

    def evaluate_logged(self, trajectories: Iterable[Trajectory]) -> list[dict]:
        rows: list[dict] = []
        for trajectory in trajectories:
            simulation = self.simulator.simulate_trajectory(
                trajectory,
                routed_models=baseline_models(trajectory),
                strategy="logged",
            )
            success_values = [self.success_proxy(call, call.model) for call in trajectory.calls]
            rows.append(
                {
                    "strategy": "logged",
                    "trajectory_id": trajectory.trajectory_id,
                    "total_cost": simulation.total_cost,
                    "total_tokens": simulation.total_tokens,
                    "cache_hit_rate": simulation.cache_hit_rate,
                    "success_proxy": sum(success_values) / len(success_values) if success_values else 0.0,
                    "routed_calls": len(simulation.samples),
                    "switched_models": simulation.switched_models,
                }
            )
        return rows
