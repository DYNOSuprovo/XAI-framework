"""Matplotlib rendering of explanations (optional dependency).

One chart form: a horizontal bar chart of the top-``k`` attributions, most important
at the top, positive in blue and negative in red. Consensus explanations overlay
one marker per component method so disagreement is visible at a glance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .explanation import Explanation

# Colour-blind-safe diverging pair + neutral ink tokens.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"
# Distinct hue *and* marker shape per component so identity never rests on colour alone.
COMPONENT_STYLES = [
    ("#eb6834", "o"),
    ("#1baf7a", "s"),
    ("#4a3aa7", "^"),
    ("#eda100", "D"),
]


def plot_explanation(
    exp: Explanation, k: int = 10, ax: Any = None, show: bool = False, title: str | None = None
) -> Any:
    """Draw ``exp`` as a horizontal bar chart and return the matplotlib ``Axes``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("plotting needs matplotlib: pip install 'xai-framework[plot]'") from e

    from .explanation import ConsensusExplanation

    order = np.argsort(-np.abs(exp.values), kind="stable")[:k][::-1]  # top at the top
    names = [exp.feature_names[i] for i in order]
    is_consensus = isinstance(exp, ConsensusExplanation)
    vals = exp.normalized()[order] if is_consensus else exp.values[order]
    if exp.feature_values is not None:
        names = [f"{n} = {_fmt(exp.feature_values[i])}" for n, i in zip(names, order, strict=True)]

    if ax is None:
        height = max(2.4, 0.42 * len(order) + 1.2)
        _, ax = plt.subplots(figsize=(7.5, height))

    y = np.arange(len(order))
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in vals]
    ax.barh(y, vals, color=colors, height=0.55, zorder=2)

    # Where each row's outermost mark ends, so direct labels never sit on a marker.
    extent = vals.copy()
    if is_consensus:
        for (color, marker), (method, comp) in zip(
            COMPONENT_STYLES, exp.components.items(), strict=False
        ):
            comp_vals = comp.normalized()[order]
            extent = np.where(
                vals >= 0, np.maximum(extent, comp_vals), np.minimum(extent, comp_vals)
            )
            ax.scatter(
                comp_vals,
                y,
                marker=marker,
                s=42,
                facecolor=color,
                edgecolor="white",
                linewidth=1,
                label=method,
                zorder=3,
            )
        ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_MUTED)

    # Direct-label only the three largest bars.
    for i in np.argsort(-np.abs(vals))[:3]:
        v = vals[i]
        ax.annotate(
            f"{v:+.3g}",
            (extent[i], y[i]),
            xytext=(6 if v >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=9,
            color=INK_MUTED,
        )

    ax.set_yticks(y, names)
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    for label in ax.get_yticklabels():
        label.set_color(INK)

    xlabel = "share of total attribution" if is_consensus else "attribution"
    if exp.scope == "global":
        xlabel = "importance" + (" (normalised)" if is_consensus else "")
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_title(title or _default_title(exp), loc="left", color=INK, fontsize=11)
    ax.figure.tight_layout()
    if show:
        plt.show()
    return ax


def _default_title(exp: Explanation) -> str:
    method = (
        "consensus of " + " + ".join(exp.components)
        if hasattr(exp, "components") and exp.components
        else exp.method
    )
    if exp.scope == "global":
        return f"Global feature importance ({method})"
    if exp.task == "classification":
        head = f"Why class {exp.target!r}"
        if exp.prediction is not None:
            head += f" (p = {exp.prediction:.2f})"
    else:
        head = "Why this prediction" + (
            f" ({exp.prediction:.4g})" if exp.prediction is not None else ""
        )
    return f"{head} - {method}"


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)
