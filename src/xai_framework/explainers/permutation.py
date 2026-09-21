"""Permutation feature importance (Breiman, 2001; Fisher, Rudin & Dominici, 2019).

A global, model-agnostic method: shuffle one column at a time and measure how much
the model's score drops. Large drops mean the model relies on that feature. It needs
ground-truth labels, so it is only used when ``y`` is supplied.

Use held-out data. On training rows of a fully-grown ensemble the score is already
perfect and barely moves, which makes every feature look unimportant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..explanation import Explanation
from ..registry import register_explainer
from .base import BaseExplainer

#: ``metric(y_true, model_output) -> score`` where higher is better. ``model_output`` is
#: what :meth:`ModelAdapter.predict` returns: probabilities for classifiers, values otherwise.
Metric = Callable[[np.ndarray, np.ndarray], float]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def neg_log_loss(y_true_idx: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(proba[np.arange(len(y_true_idx)), y_true_idx], eps, 1.0)
    return float(np.mean(np.log(p)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -float(np.mean((np.asarray(y_true, dtype=float) - y_pred) ** 2))


def neg_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -float(np.mean(np.abs(np.asarray(y_true, dtype=float) - y_pred)))


BUILTIN_METRICS: dict[str, tuple[str, Metric]] = {
    # name: (input kind, function)
    "accuracy": ("labels", accuracy),
    "log_loss": ("proba_idx", neg_log_loss),
    "r2": ("output", r2_score),
    "neg_mse": ("output", neg_mse),
    "neg_mae": ("output", neg_mae),
}


@register_explainer("permutation")
class PermutationExplainer(BaseExplainer):
    """Global importance = drop in score when a feature is shuffled.

    Parameters
    ----------
    model, X_background
        See :class:`~xai_framework.explainers.base.BaseExplainer`.
    metric
        One of ``"log_loss"`` (default for classifiers - smoother than accuracy, so
        small effects still register), ``"accuracy"``, ``"r2"`` (default for
        regressors), ``"neg_mse"``, ``"neg_mae"``, or a callable
        ``metric(y_true, model_output) -> float`` where higher is better and
        ``model_output`` is the adapter's prediction (probabilities for classifiers).
    n_repeats
        How many independent shuffles to average per feature.
    max_rows
        Subsample the dataset to at most this many rows before scoring.
    """

    name = "permutation"
    supports_local = False
    supports_global = True
    requires_y = True

    def __init__(
        self,
        model: Any,
        X_background: Any,
        *,
        metric: str | Metric | None = None,
        n_repeats: int = 5,
        max_rows: int | None = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, X_background, **kwargs)
        if metric is None:
            metric = "log_loss" if self.adapter.is_classifier else "r2"
        if isinstance(metric, str):
            if metric not in BUILTIN_METRICS:
                raise ValueError(
                    f"unknown metric {metric!r}; choose from {sorted(BUILTIN_METRICS)}"
                )
            self.metric_name = metric
            self._kind, self._metric = BUILTIN_METRICS[metric]
        else:
            self.metric_name = getattr(metric, "__name__", repr(metric))
            self._kind, self._metric = "output", metric
        if self._kind in ("labels", "proba_idx") and not self.adapter.is_classifier:
            raise ValueError(f"metric {metric!r} needs a classifier")
        self.n_repeats = int(n_repeats)
        self.max_rows = max_rows

    def _explain_instance(self, x: np.ndarray, target: int | None) -> Explanation:
        raise NotImplementedError("permutation importance is a global method")

    def _score(self, X: np.ndarray, y: np.ndarray, y_idx: np.ndarray | None) -> float:
        out = self.adapter.predict(X)
        if self._kind == "labels":
            labels = np.asarray([self.adapter.class_label(i) for i in out.argmax(axis=1)])
            return self._metric(y, labels)
        if self._kind == "proba_idx":
            return self._metric(y_idx, out)
        return self._metric(y, out)

    def _explain_global(
        self, X: np.ndarray, y: np.ndarray | None, target: int | None
    ) -> Explanation:
        if y is None:
            raise ValueError("permutation importance needs ground-truth labels: pass y=")
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows")
        if self.max_rows is not None and len(X) > self.max_rows:
            idx = self.rng.choice(len(X), size=self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        y_idx = (
            np.asarray([self.adapter.class_index(v) for v in y])
            if self._kind == "proba_idx"
            else None
        )

        baseline = self._score(X, y, y_idx)
        n_feat = X.shape[1]
        drops = np.empty((self.n_repeats, n_feat))
        for r in range(self.n_repeats):
            for j in range(n_feat):
                Xp = X.copy()
                Xp[:, j] = self.rng.permutation(Xp[:, j])
                drops[r, j] = baseline - self._score(Xp, y, y_idx)

        return self._make_explanation(
            drops.mean(axis=0),
            None,
            None,
            scope="global",
            baseline_score=float(baseline),
            importances_std=drops.std(axis=0),
            n_repeats=self.n_repeats,
            n_rows=len(X),
            metric=self.metric_name,
        )
