from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence


ModelId = str


@dataclass(slots=True)
class CallContext:
    request_index: int
    trajectory_id: str
    first_user_hash: str
    model: ModelId
    call_index: int
    model_history: Sequence[ModelId]
    input_items: Sequence[dict[str, Any]]
    tools: Sequence[dict[str, Any]]
    input_text: str
    input_item_count: int
    actual_action: dict[str, Any] | None = None
    shared_prefix_tokens: int = 0
    estimated_input_tokens: int = 0
    model_rank: int | None = None


@dataclass(slots=True)
class RouteDecision:
    model: ModelId
    rationale: str = ""


@dataclass(slots=True)
class CostEstimate:
    tokens: float
    dollars: float | None = None
    cached_tokens: int = 0
    uncached_tokens: int = 0


@dataclass(slots=True)
class Trajectory:
    trajectory_id: str
    first_user_hash: str
    model: ModelId
    calls: list[CallContext] = field(default_factory=list)


@dataclass(slots=True)
class SimulationRow:
    strategy: str
    trajectory_id: str
    total_cost: float
    total_tokens: float
    cache_hit_rate: float
    success_proxy: float
    routed_calls: int
    switched_models: int


Objective = Literal["min_cost", "max_quality", "frontier"]
