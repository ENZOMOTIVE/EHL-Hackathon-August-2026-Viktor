from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .costs import cheapest_model, expensive_model, estimate_input_tokens
from .text import canonicalize_input_items
from .types import CallContext, RouteDecision


COMPLEX_TOOL_NAMES = {"bash", "python", "python3", "shell", "terminal", "execute"}


@dataclass(slots=True)
class Router:
    strategy: str = "static"
    cheap_model_id: str = cheapest_model()
    expensive_model_id: str = expensive_model()
    length_threshold: int = 2000

    def route(self, input_items: Iterable[dict], tools: Iterable[dict]) -> str:
        input_text = canonicalize_input_items(input_items)
        estimated_tokens = estimate_input_tokens(input_text)
        tool_names = self._tool_names(tools)

        if self.strategy == "static":
            return self.cheap_model_id
        if self.strategy == "length-based":
            return self.expensive_model_id if estimated_tokens > self.length_threshold else self.cheap_model_id
        if self.strategy == "tool-triggered":
            if COMPLEX_TOOL_NAMES.intersection(tool_names):
                return self.expensive_model_id
            return self.cheap_model_id
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def route_call(self, call: CallContext) -> RouteDecision:
        return RouteDecision(model=self.route(call.input_items, call.tools), rationale=self.strategy)

    @staticmethod
    def _tool_names(tools: Iterable[dict]) -> set[str]:
        names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            function_name = function.get("name") if isinstance(function, dict) else None
            name = tool.get("name") or function_name
            if name:
                names.add(str(name).lower())
        return names


class HeuristicRouter(Router):
    def __init__(self) -> None:
        super().__init__(strategy="length-based")
