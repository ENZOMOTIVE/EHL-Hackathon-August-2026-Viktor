from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from .text import canonicalize_input_items, estimate_tokens, hash_first_user_message
from .types import CallContext, Trajectory


def discover_jsonl_files(export_dir: str | Path) -> list[Path]:
    root = Path(export_dir)
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def _load_request_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            record["_source_file"] = str(path)
            record["_source_line"] = line_number
            records.append(record)
    return records


def load_raw_requests(export_dir: str | Path) -> list[CallContext]:
    requests: list[CallContext] = []
    source_index = 0
    for path in discover_jsonl_files(export_dir):
        for record in _load_request_records(path):
            input_items = list(record.get("input", []))
            tools = list(record.get("tools", []))
            input_text = canonicalize_input_items(input_items)
            first_user_hash = hash_first_user_message(input_items)
            model = str(record.get("model", ""))
            requests.append(
                CallContext(
                    request_index=source_index,
                    trajectory_id=first_user_hash,
                    first_user_hash=first_user_hash,
                    model=model,
                    call_index=-1,
                    model_history=(),
                    input_items=input_items,
                    tools=tools,
                    input_text=input_text,
                    input_item_count=len(input_items),
                    estimated_input_tokens=estimate_tokens(input_text),
                    model_rank=None,
                )
            )
            source_index += 1
    return requests


def load_trajectories(export_dir: str | Path) -> list[Trajectory]:
    from .reconstruction import reconstruct_trajectories

    return reconstruct_trajectories(load_raw_requests(export_dir))


def build_trajectory_dataframe(export_dir: str | Path) -> pd.DataFrame:
    trajectories = load_trajectories(export_dir)
    rows: list[dict] = []
    for trajectory in trajectories:
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "first_user_hash": trajectory.first_user_hash,
                "call_count": len(trajectory.calls),
                "models": [call.model for call in trajectory.calls],
                "calls": [call_to_dict(call) for call in trajectory.calls],
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["call_count", "trajectory_id"], ascending=[False, True]).reset_index(drop=True)
    return frame


def call_to_dict(call: CallContext) -> dict:
    return {
        "request_index": call.request_index,
        "trajectory_id": call.trajectory_id,
        "first_user_hash": call.first_user_hash,
        "call_index": call.call_index,
        "model": call.model,
        "model_history": list(call.model_history),
        "input_item_count": call.input_item_count,
        "input_text": call.input_text,
        "tools": list(call.tools),
        "estimated_input_tokens": call.estimated_input_tokens,
        "shared_prefix_tokens": call.shared_prefix_tokens,
        "actual_action": call.actual_action,
        "model_rank": call.model_rank,
    }
