from __future__ import annotations

import os
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .costs import CostSimulator
from .evaluator import Evaluator
from .loader import build_trajectory_dataframe, load_trajectories
from .policy import Router

STRATEGY_ORDER = ["logged", "static", "length-based", "tool-triggered"]


@dataclass(slots=True)
class PipelineArtifacts:
    trajectory_frame: pd.DataFrame
    detailed_frame: pd.DataFrame
    summary_frame: pd.DataFrame
    best_strategy: str | None = None
    skill_path: Path | None = None


def _aggregate_metrics(detailed_frame: pd.DataFrame) -> pd.DataFrame:
    if detailed_frame.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "trajectories",
                "routed_calls",
                "total_cost",
                "avg_cost_per_call",
                "mean_cache_hit_rate",
                "mean_success_proxy",
                "total_switches",
            ]
        )

    summary = (
        detailed_frame.groupby("strategy", as_index=False)
        .agg(
            trajectories=("trajectory_id", "nunique"),
            routed_calls=("routed_calls", "sum"),
            total_cost=("total_cost", "sum"),
            mean_cache_hit_rate=("cache_hit_rate", "mean"),
            mean_success_proxy=("success_proxy", "mean"),
            total_switches=("switched_models", "sum"),
        )
        .reset_index(drop=True)
    )
    summary["strategy"] = pd.Categorical(summary["strategy"], categories=STRATEGY_ORDER, ordered=True)
    summary = summary.sort_values("strategy").reset_index(drop=True)
    summary["strategy"] = summary["strategy"].astype(str)
    summary["avg_cost_per_call"] = summary["total_cost"] / summary["routed_calls"].where(summary["routed_calls"] != 0, 1)
    return summary[
        [
            "strategy",
            "trajectories",
            "routed_calls",
            "total_cost",
            "avg_cost_per_call",
            "mean_cache_hit_rate",
            "mean_success_proxy",
            "total_switches",
        ]
    ]


def _is_dominated(candidate: pd.Series, others: pd.DataFrame) -> bool:
    for _, row in others.iterrows():
        if row["strategy"] == candidate["strategy"]:
            continue
        no_worse_cost = row["avg_cost_per_call"] <= candidate["avg_cost_per_call"]
        no_worse_quality = row["mean_success_proxy"] >= candidate["mean_success_proxy"]
        strictly_better = (
            row["avg_cost_per_call"] < candidate["avg_cost_per_call"]
            or row["mean_success_proxy"] > candidate["mean_success_proxy"]
        )
        if no_worse_cost and no_worse_quality and strictly_better:
            return True
    return False


def _pareto_frontier(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return summary_frame
    frontier_rows = []
    for _, row in summary_frame.iterrows():
        if not _is_dominated(row, summary_frame):
            frontier_rows.append(row)
    if not frontier_rows:
        return summary_frame.iloc[0:0]
    return pd.DataFrame(frontier_rows).reset_index(drop=True)


def _select_best_strategy(summary_frame: pd.DataFrame) -> str | None:
    routed = summary_frame[summary_frame["strategy"] != "logged"].copy()
    if routed.empty:
        return None
    frontier = _pareto_frontier(routed)
    if frontier.empty:
        frontier = routed
    frontier = frontier.sort_values(
        by=["mean_success_proxy", "avg_cost_per_call", "strategy"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return str(frontier.iloc[0]["strategy"])


def _best_row(summary_frame: pd.DataFrame, strategy: str | None) -> pd.Series | None:
    if strategy is None:
        return None
    matches = summary_frame[summary_frame["strategy"] == strategy]
    if matches.empty:
        return None
    return matches.iloc[0]


def _skill_content(best_strategy: str, best_row: pd.Series) -> str:
    cheap_model = "gpt-5-nano"
    expensive_model = "claude-opus-5"
    threshold = 2000
    if best_strategy == "static":
        purpose = f"This skill keeps Viktor on {cheap_model} by default to minimize cost without changing behavior across turns."
        trigger_logic = [
            f"- Always select {cheap_model} for every call.",
            f"- Never upgrade mid-trajectory unless a separate human override is added.",
        ]
        cache_warning = (
            "Do not introduce model switches for ordinary calls: any switch evicts the prefix cache, "
            "so the next request pays uncached rates on the full input and usually erases the savings."
        )
    elif best_strategy == "tool-triggered":
        purpose = (
            f"This skill keeps Viktor on {cheap_model} unless complex tools appear, then upgrades to {expensive_model} "
            "to protect tool-heavy reasoning quality."
        )
        trigger_logic = [
            f"- If the tools list contains `bash`, `python`, `python3`, `shell`, `terminal`, or `execute`, select {expensive_model}.",
            f"- Otherwise select {cheap_model}.",
            f"- Never downgrade a tool-heavy call if the prompt is already long and nested; keep the stronger model in place.",
        ]
        cache_warning = (
            "Do not switch models once a trajectory has a large shared prefix: the cache resets on a switch, "
            "so the entire next input is repriced at the uncached rate and can cost more than the cheaper model saves."
        )
    else:
        purpose = (
            f"This skill routes short calls to {cheap_model} and sends longer calls to {expensive_model} to balance cost "
            "against quality on the offline trajectory frontier."
        )
        trigger_logic = [
            f"- Estimate input size with character_count / 4.",
            f"- If the estimate exceeds {threshold} tokens, upgrade to {expensive_model}.",
            f"- Otherwise keep {cheap_model}.",
            f"- This rule is evaluated per call, not per trajectory, but the model switch policy still respects cache eviction.",
        ]
        cache_warning = (
            "Do not switch models when the shared prefix is substantial; if the router flips models, the cache is evicted "
            "and the following call is billed as fully uncached, which can dominate any token-level savings."
        )

    defense = (
        f"On the measured frontier, `{best_strategy}` achieved the best cost-quality compromise among the routed baselines: "
        f"mean success proxy {best_row['mean_success_proxy']:.4f} at average cost per call {best_row['avg_cost_per_call']:.4f}, "
        f"with cache-hit rate {best_row['mean_cache_hit_rate']:.4f}. The rule set wins because it preserves quality where the "
        f"offline traces need it while avoiding unnecessary expensive calls, so the frontier shifts down in cost without a "
        f"commensurate loss in proxy success. One known weakness is that the success signal is still a proxy, so the policy can "
        f"overfit to judge-model or heuristic complexity rather than true task completion."
    )

    return "\n".join(
        [
            "---",
            'name: viktor-router-memory',
            f'description: Offline routing memory exported from the evaluation on {best_strategy}.',
            "---",
            "",
            f"<purpose>{purpose}</purpose>",
            "",
            "<trigger_conditions>",
            *trigger_logic,
            "</trigger_conditions>",
            "",
            f"<cache_warning>{cache_warning}</cache_warning>",
            "",
            f"<defense>{defense}</defense>",
            "",
            f"<!-- generated_at: {datetime.utcnow().isoformat()}Z -->",
        ]
    )


def _write_skill_file(summary_frame: pd.DataFrame, output_dir: Path) -> tuple[str | None, Path | None]:
    best_strategy = _select_best_strategy(summary_frame)
    if best_strategy is None:
        return None, None
    best_row = _best_row(summary_frame, best_strategy)
    if best_row is None:
        return None, None
    skill_path = output_dir / "SKILL.md"
    skill_path.write_text(_skill_content(best_strategy, best_row), encoding="utf-8")
    return best_strategy, skill_path


def _plot_pareto(summary_frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    if summary_frame.empty:
        plt.text(0.5, 0.5, "No trajectories found", ha="center", va="center", fontsize=14)
        plt.axis("off")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        return

    sns.set_theme(style="whitegrid")
    palette = dict(zip(STRATEGY_ORDER, sns.color_palette("deep", n_colors=len(STRATEGY_ORDER))))
    ax = sns.scatterplot(
        data=summary_frame,
        x="avg_cost_per_call",
        y="mean_success_proxy",
        hue="strategy",
        s=160,
        palette="deep",
    )
    frontier = summary_frame.sort_values("avg_cost_per_call").reset_index(drop=True)
    ax.plot(frontier["avg_cost_per_call"], frontier["mean_success_proxy"], color="0.35", linewidth=1.25, alpha=0.7)
    for _, row in summary_frame.iterrows():
        ax.annotate(
            row["strategy"],
            (row["avg_cost_per_call"], row["mean_success_proxy"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )
    ax.set_title("Pareto Frontier: Cost vs Success Proxy")
    ax.set_xlabel("Estimated cost per call")
    ax.set_ylabel("Quality / success proxy")
    ax.legend(title="Strategy", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_overview(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if summary_frame.empty:
        fig.text(0.5, 0.5, "No trajectories found", ha="center", va="center", fontsize=14)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    sns.set_theme(style="whitegrid")
    palette = dict(zip(STRATEGY_ORDER, sns.color_palette("deep", n_colors=len(STRATEGY_ORDER))))
    ordered = summary_frame.copy()
    ordered["strategy"] = pd.Categorical(ordered["strategy"], categories=STRATEGY_ORDER, ordered=True)
    ordered = ordered.sort_values("strategy")
    sns.barplot(data=ordered, x="strategy", y="avg_cost_per_call", ax=axes[0], hue="strategy", palette=palette, legend=False)
    axes[0].set_title("Average cost per call")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    metrics = ordered.melt(
        id_vars="strategy",
        value_vars=["mean_success_proxy", "mean_cache_hit_rate"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=metrics, x="strategy", y="value", hue="metric", ax=axes[1], palette="Set2")
    axes[1].set_title("Proxy quality and cache hit rate")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(title="")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(export_dir: str | Path, output_dir: str | Path) -> PipelineArtifacts:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trajectory_frame = build_trajectory_dataframe(export_dir)
    trajectories = load_trajectories(export_dir)

    simulator = CostSimulator()
    evaluator = Evaluator(simulator)

    detailed_rows: list[dict] = []
    detailed_rows.extend(evaluator.evaluate_logged(trajectories))

    for strategy in ("static", "length-based", "tool-triggered"):
        router = Router(strategy=strategy)
        _, rows = evaluator.evaluate_dataset(trajectories, router)
        detailed_rows.extend(rows)

    detailed_frame = pd.DataFrame(detailed_rows)
    summary_frame = _aggregate_metrics(detailed_frame)
    best_strategy, skill_path = _write_skill_file(summary_frame, output_path)

    trajectory_frame.to_csv(output_path / "trajectories.csv", index=False)
    detailed_frame.to_csv(output_path / "trajectory_metrics.csv", index=False)
    summary_frame.to_csv(output_path / "summary.csv", index=False)

    _plot_pareto(summary_frame, output_path / "pareto_frontier.png")
    _plot_overview(summary_frame, output_path / "metrics_overview.png")

    return PipelineArtifacts(
        trajectory_frame=trajectory_frame,
        detailed_frame=detailed_frame,
        summary_frame=summary_frame,
        best_strategy=best_strategy,
        skill_path=skill_path,
    )
