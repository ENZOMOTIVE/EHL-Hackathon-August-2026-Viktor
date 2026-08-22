from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .text import estimate_tokens, longest_shared_prefix
from .types import CallContext, CostEstimate, Trajectory


@dataclass(frozen=True, slots=True)
class PricingTier:
    model_id: str
    uncached_token_rate: float
    cached_token_rate: float


MODEL_TIERS: list[PricingTier] = [
    PricingTier("claude-opus-5", 1.00, 0.10),
    PricingTier("claude-sonnet-5", 0.82, 0.082),
    PricingTier("claude-fable-5", 0.68, 0.068),
    PricingTier("gpt-5.6-terra", 0.54, 0.054),
    PricingTier("gpt-5.6-pro", 0.42, 0.042),
    PricingTier("gpt-5.6-mini", 0.30, 0.030),
    PricingTier("gpt-5", 0.22, 0.022),
    PricingTier("gpt-5-mini", 0.14, 0.014),
    PricingTier("gpt-5-nano", 0.08, 0.008),
]


MODEL_PRICING = {tier.model_id: tier for tier in MODEL_TIERS}


def model_rank(model_id: str) -> int:
    for index, tier in enumerate(MODEL_TIERS):
        if tier.model_id == model_id:
            return index
    return len(MODEL_TIERS)


def cheapest_model() -> str:
    return MODEL_TIERS[-1].model_id


def expensive_model() -> str:
    return MODEL_TIERS[0].model_id


def estimate_input_tokens(text: str) -> int:
    return estimate_tokens(text)


def shared_prefix_tokens(previous: str, current: str) -> int:
    prefix = longest_shared_prefix(previous, current)
    return estimate_tokens(prefix) if prefix else 0


@dataclass(slots=True)
class CostSample:
    call_index: int
    model: str
    input_tokens: int
    cached_tokens: int
    uncached_tokens: int
    cost: float
    cache_hit: bool
    shared_prefix_tokens: int


@dataclass(slots=True)
class SimulationResult:
    strategy: str
    trajectory_id: str
    samples: list[CostSample] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(sample.cost for sample in self.samples)

    @property
    def total_tokens(self) -> float:
        return sum(sample.input_tokens for sample in self.samples)

    @property
    def cached_tokens(self) -> float:
        return sum(sample.cached_tokens for sample in self.samples)

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_tokens
        return 0.0 if total == 0 else self.cached_tokens / total

    @property
    def switched_models(self) -> int:
        if not self.samples:
            return 0
        previous = self.samples[0].model
        switches = 0
        for sample in self.samples[1:]:
            if sample.model != previous:
                switches += 1
            previous = sample.model
        return switches


class CostSimulator:
    def __init__(self, pricing: Sequence[PricingTier] | None = None) -> None:
        self.pricing = list(pricing or MODEL_TIERS)
        self.pricing_by_model = {tier.model_id: tier for tier in self.pricing}

    def simulate_trajectory(
        self,
        trajectory: Trajectory,
        routed_models: Sequence[str] | None = None,
        strategy: str = "logged",
    ) -> SimulationResult:
        models = list(routed_models or [call.model for call in trajectory.calls])
        if len(models) != len(trajectory.calls):
            raise ValueError("routed_models must match trajectory call count")

        samples: list[CostSample] = []
        previous_model: str | None = None
        previous_text: str | None = None

        for index, (call, model_id) in enumerate(zip(trajectory.calls, models)):
            tier = self.pricing_by_model.get(model_id, self.pricing_by_model[cheapest_model()])
            input_tokens = estimate_input_tokens(call.input_text)
            if previous_model is None or previous_text is None or previous_model != model_id:
                shared_tokens = 0
                cached_tokens = 0
                uncached_tokens = input_tokens
            else:
                shared_tokens = shared_prefix_tokens(previous_text, call.input_text)
                cached_tokens = min(input_tokens, shared_tokens)
                uncached_tokens = input_tokens - cached_tokens
            cost = (cached_tokens * tier.cached_token_rate) + (uncached_tokens * tier.uncached_token_rate)
            samples.append(
                CostSample(
                    call_index=index,
                    model=model_id,
                    input_tokens=input_tokens,
                    cached_tokens=cached_tokens,
                    uncached_tokens=uncached_tokens,
                    cost=cost,
                    cache_hit=cached_tokens > 0,
                    shared_prefix_tokens=shared_tokens,
                )
            )
            previous_model = model_id
            previous_text = call.input_text

        return SimulationResult(strategy=strategy, trajectory_id=trajectory.trajectory_id, samples=samples)

    def simulate_dataset(
        self,
        trajectories: Iterable[Trajectory],
        routed_models_by_trajectory: dict[str, Sequence[str]] | None = None,
        strategy: str = "logged",
    ) -> list[SimulationResult]:
        results: list[SimulationResult] = []
        for trajectory in trajectories:
            routed_models = None
            if routed_models_by_trajectory is not None:
                routed_models = routed_models_by_trajectory.get(trajectory.trajectory_id)
            results.append(self.simulate_trajectory(trajectory, routed_models=routed_models, strategy=strategy))
        return results


def result_to_row(result: SimulationResult, success_proxy: float) -> dict[str, float | str | int]:
    return {
        "strategy": result.strategy,
        "trajectory_id": result.trajectory_id,
        "total_cost": result.total_cost,
        "total_tokens": result.total_tokens,
        "cache_hit_rate": result.cache_hit_rate,
        "success_proxy": success_proxy,
        "routed_calls": len(result.samples),
        "switched_models": result.switched_models,
    }


def baseline_models(trajectory: Trajectory) -> list[str]:
    return [call.model for call in trajectory.calls]
