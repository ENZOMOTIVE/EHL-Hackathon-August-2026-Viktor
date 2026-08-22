from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "name", "role", "type", "arguments", "value"):
            if key in value:
                parts.append(_extract_text(value[key]))
        if parts:
            return " ".join(part for part in parts if part)
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Iterable):
        return " ".join(_extract_text(item) for item in value)
    return str(value)


def canonicalize_input_items(input_items: Iterable[Any]) -> str:
    normalized = []
    for item in input_items:
        normalized.append(_extract_text(item).strip())
    return "\n".join(part for part in normalized if part)


def first_user_message_text(input_items: Iterable[Any]) -> str:
    for item in input_items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        item_type = str(item.get("type", "")).lower()
        if role == "user" or item_type == "message" and role == "user":
            return _extract_text(item.get("content", item))
    return ""


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_first_user_message(input_items: Iterable[Any]) -> str:
    return hash_text(first_user_message_text(input_items))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def longest_shared_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]
