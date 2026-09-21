"""Consensus explanations: run several explainers and combine what they agree on.

Attribution methods answer subtly different questions and are known to disagree on
real models. Instead of picking one, the consensus explainer runs all of them, puts their
attributions on a common scale, and reports both a combined ranking *and* how much
the methods agree - so a user can tell a robust explanation from a fragile one.

Combination strategies
----------------------
``"mean"``
    Weighted mean of each method's L1-normalised attributions. Keeps sign and
    relative magnitude; the default.
``"rank"``
    Borda count over each method's importance ranking, signed by the mean
    direction. More robust to one method producing outsized values.

Agreement is the mean pairwise Spearman correlation between the methods'
importance rankings (by ``|attribution|``).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..explanation import ConsensusExplanation, Explanation
from ..registry import get_explainer, register_explainer
from .base import BaseExplainer

Aggregation = Literal["mean", "rank"]


def spearman_matrix(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Spearman correlation between the magnitude rankings of ``vectors``."""
    names = list(vectors)
    ranks = np.vstack(
        [pd.Series(np.abs(v)).rank(method="average").to_numpy() for v in vectors.values()]
    )
    m = len(names)
    out = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = ranks[i], ranks[j]
            degenerate = a.std() == 0 or b.std() == 0
            rho = float("nan") if degenerate else float(np.corrcoef(a, b)[0, 1])
            out[i, j] = out[j, i] = rho
    return pd.DataFrame(out, index=names, columns=names)


def combine_explanations(
    components: dict[str, Explanation],
    *,
    weights: dict[str, float] | None = None,
    aggregation: Aggregation = "mean",
    top_k: int = 5,
) -> ConsensusExplanation:
    """Merge explanations from different methods for the same instance/dataset.

    All components must share the same feature names in the same order. Use this
    directly when you already have explanations; :class:`ConsensusExplainer` calls
    it for you.
    """
    if not components:
        raise ValueError("need at least one component explanation")
    if aggregation not in ("mean", "rank"):
        raise ValueError("aggregation must be 'mean' or 'rank'")

    names = list(components)
    first = components[names[0]]
    feature_names = first.feature_names
    for name, exp in components.items():
        if exp.feature_names != feature_names:
            raise ValueError(f"component {name!r} has different feature names")
        if exp.scope != first.scope:
            raise ValueError("cannot combine local and global explanations")

    w = np.asarray([(weights or {}).get(n, 1.0) for n in names], dtype=float)
    if (w < 0).any() or w.sum() == 0:
        raise ValueError("weights must be non-negative and not all zero")
    w = w / w.sum()

    normalised = {n: components[n].normalized() for n in names}
    N = np.vstack([normalised[n] for n in names])  # (methods, features)
    mean_direction = w @ N

    if aggregation == "mean":
        values = mean_direction
    else:
        n_feat = len(feature_names)
        R = np.vstack([components[n].ranks() for n in names])
        borda = (n_feat - R + 1) / n_feat  # 1 for the top feature, 1/n for the last
        sign = np.sign(mean_direction)
        sign[sign == 0] = 1.0
        values = sign * (w @ borda)

    matrix = spearman_matrix(normalised)
    if len(names) > 1:
        upper = matrix.to_numpy()[np.triu_indices(len(names), k=1)]
        agreement = float(np.nanmean(upper)) if not np.all(np.isnan(upper)) else float("nan")
    else:
        agreement = float("nan")

    # Do the methods agree on the *direction* of the most important features?
    top_idx = np.argsort(-np.abs(values), kind="stable")[:top_k]
    signs = np.sign(N[:, top_idx])
    sign_agree = [bool(len({s for s in col if s != 0}) <= 1) for col in signs.T]
    sign_agreement = float(np.mean(sign_agree)) if sign_agree else float("nan")

    prediction = next((e.prediction for e in components.values() if e.prediction is not None), None)
    return ConsensusExplanation(
        feature_names=feature_names,
        values=values,
        method="consensus",
        scope=first.scope,
        feature_values=first.feature_values,
        base_value=None,
        prediction=prediction,
        target=first.target,
        task=first.task,
        metadata={
            "aggregation": aggregation,
            "weights": dict(zip(names, w.tolist(), strict=True)),
            "top_k_sign_agreement": sign_agreement,
        },
        components=dict(components),
        agreement=agreement,
        agreement_matrix=matrix,
    )


@register_explainer("consensus")
class ConsensusExplainer(BaseExplainer):
    """Run several explainers and combine them (see module docs).

    Parameters
    ----------
    model, X_background
        See :class:`~xai_framework.explainers.base.BaseExplainer`.
    methods
        Explainer names from the registry (``"coalition"``, ``"surrogate"``, ``"permutation"``),
        explainer classes, or already-constructed explainer instances.
    weights
        Optional per-method weights, keyed by method name.
    aggregation
        ``"mean"`` or ``"rank"``.
    explainer_kwargs
        Extra keyword arguments per method, e.g.
        ``{"surrogate": {"n_samples": 2000}, "coalition": {"n_background": 100}}``.
    """

    name = "consensus"
    supports_local = True
    supports_global = True

    def __init__(
        self,
        model: Any,
        X_background: Any,
        *,
        methods: Iterable[str | type[BaseExplainer] | BaseExplainer] = ("coalition", "surrogate"),
        weights: dict[str, float] | None = None,
        aggregation: Aggregation = "mean",
        explainer_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, X_background, **kwargs)
        self.weights = weights
        self.aggregation = aggregation
        explainer_kwargs = explainer_kwargs or {}
        self.explainers: dict[str, BaseExplainer] = {}
        for m in methods:
            explainer = self._instantiate(m, explainer_kwargs)
            if explainer.name == self.name:
                raise ValueError("cannot nest a consensus explainer inside another")
            self.explainers[explainer.name] = explainer
        if not self.explainers:
            raise ValueError("methods must contain at least one explainer")

    def _instantiate(
        self,
        method: str | type[BaseExplainer] | BaseExplainer,
        explainer_kwargs: dict[str, dict[str, Any]],
    ) -> BaseExplainer:
        if isinstance(method, BaseExplainer):
            return method
        cls = get_explainer(method) if isinstance(method, str) else method
        ok, reason = cls.is_available()
        if not ok:
            raise ImportError(f"explainer {cls.name!r} is unavailable: {reason}")
        return cls(
            self.adapter,
            self.X_background,
            random_state=self.random_state,
            **explainer_kwargs.get(cls.name, {}),
        )

    @property
    def methods(self) -> list[str]:
        return list(self.explainers)

    def _explain_instance(self, x: np.ndarray, target: int | None) -> Explanation:
        label = self.adapter.class_label(target) if target is not None else None
        components = {
            name: exp.explain_instance(x, target=label)
            for name, exp in self.explainers.items()
            if exp.supports_local
        }
        if not components:
            raise NotImplementedError("none of the configured explainers is local")
        return combine_explanations(components, weights=self.weights, aggregation=self.aggregation)

    def _explain_global(
        self, X: np.ndarray, y: np.ndarray | None, target: int | None
    ) -> Explanation:
        label = self.adapter.class_label(target) if target is not None else None
        components: dict[str, Explanation] = {}
        for name, exp in self.explainers.items():
            if not exp.supports_global:
                continue
            if exp.requires_y and y is None:
                warnings.warn(
                    f"skipping {name!r}: it needs ground-truth labels (pass y=)",
                    stacklevel=3,
                )
                continue
            components[name] = exp.explain_global(X, y, target=label)
        if not components:
            raise NotImplementedError("none of the configured explainers can run globally here")
        return combine_explanations(components, weights=self.weights, aggregation=self.aggregation)
