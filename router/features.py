from __future__ import annotations

from dataclasses import dataclass

from .types import CallContext


@dataclass(slots=True)
class RouteFeatures:
    prompt_length: int
    call_index: int
    tool_count: int
    shared_prefix_tokens: int
    estimated_input_tokens: int


def extract_features(call: CallContext) -> RouteFeatures:
    return RouteFeatures(
        prompt_length=call.estimated_input_tokens,
        call_index=call.call_index,
        tool_count=len(call.tools),
        shared_prefix_tokens=call.shared_prefix_tokens,
        estimated_input_tokens=call.estimated_input_tokens,
    )
