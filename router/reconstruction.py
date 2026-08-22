from __future__ import annotations

from collections import defaultdict

from .text import canonicalize_input_items, longest_shared_prefix
from .types import CallContext, Trajectory


def _model_rank_map(models: list[str]) -> dict[str, int]:
    return {model: rank for rank, model in enumerate(models)}


MODEL_TIER_ORDER = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "gpt-5.6-terra",
    "gpt-5.6-pro",
    "gpt-5.6-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]


def reconstruct_trajectories(requests: list[CallContext]) -> list[Trajectory]:
    grouped: dict[str, list[CallContext]] = defaultdict(list)
    for request in requests:
        grouped[request.first_user_hash].append(request)

    trajectories: list[Trajectory] = []
    tier_map = _model_rank_map(MODEL_TIER_ORDER)
    for first_user_hash, calls in grouped.items():
        ordered = sorted(calls, key=lambda call: (call.input_item_count, call.request_index))
        model_history: list[str] = []
        for index, call in enumerate(ordered):
            call.trajectory_id = first_user_hash
            call.call_index = index
            call.model_history = tuple(model_history)
            call.model_rank = tier_map.get(call.model)
            call.estimated_input_tokens = max(1, len(call.input_text) // 4)
            if index + 1 < len(ordered):
                next_call = ordered[index + 1]
                current_text = canonicalize_input_items(call.input_items)
                next_text = canonicalize_input_items(next_call.input_items)
                prefix = longest_shared_prefix(current_text, next_text)
                call.shared_prefix_tokens = max(1, len(prefix) // 4) if prefix else 0
                if next_call.input_item_count >= call.input_item_count and list(next_call.input_items[: call.input_item_count]) == list(call.input_items):
                    appended = list(next_call.input_items[call.input_item_count :])
                else:
                    appended = []
                call.actual_action = {
                    "type": "embedded_output",
                    "items": appended,
                    "shared_prefix_chars": len(prefix),
                }
            else:
                call.shared_prefix_tokens = 0
                call.actual_action = None
            model_history.append(call.model)

        trajectories.append(
            Trajectory(
                trajectory_id=first_user_hash,
                first_user_hash=first_user_hash,
                model=ordered[0].model if ordered else "",
                calls=ordered,
            )
        )

    trajectories.sort(key=lambda trajectory: (-len(trajectory.calls), trajectory.trajectory_id))
    return trajectories


def trajectory_to_dataframe_rows(trajectories: list[Trajectory]) -> list[dict]:
    rows: list[dict] = []
    for trajectory in trajectories:
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "first_user_hash": trajectory.first_user_hash,
                "call_count": len(trajectory.calls),
                "calls": trajectory.calls,
            }
        )
    return rows
