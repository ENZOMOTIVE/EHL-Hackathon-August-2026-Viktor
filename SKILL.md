---
name: viktor-router-memory
description: Offline routing memory for Viktor that upgrades to a stronger model for tool-heavy or long-context calls while avoiding cache-evicting switches.
---

<purpose>This skill keeps Viktor on a cheap default model for routine calls and upgrades to a stronger model when the prompt is long or complex tools are present, so cost stays low without breaking context.</purpose>

<trigger_conditions>
- If the tools list contains `bash`, `python`, `python3`, `shell`, `terminal`, or `execute`, upgrade to the stronger model.
- If the estimated input size exceeds 2000 tokens using character_count / 4, upgrade to the stronger model.
- Otherwise keep the cheapest model.
- Do not switch models inside a trajectory unless the current call clearly needs the upgrade, because switches reset the cache.
</trigger_conditions>

<cache_warning>Do not switch models when the shared prefix between consecutive calls is substantial. A model switch evicts the prefix cache, so the next call is billed at the uncached rate on the full input and can cost more than the cheaper model saves.</cache_warning>

<defense>This rule set is intentionally simple: it minimizes expected cost on short, routine calls while preserving quality on tool-heavy or long-context requests that are most likely to fail under aggressive downgrades. In the offline evaluation pipeline, that makes it a strong Pareto candidate because the policy avoids unnecessary expensive calls yet upgrades only when prompt complexity or tool usage pushes failure risk upward. The main weakness is that the success signal is still a proxy, so the policy can overfit to heuristics such as length and tool presence instead of true task completion.</defense>
