"""Coalition explainer: Shapley-value attributions from feature coalitions.

Treats a prediction as a cooperative game in which the features are players. A
feature's attribution is its Shapley value (Shapley, 1953): its average marginal
contribution over every way of adding it to a coalition of the other features. The
value of a coalition is the model's expected output when the coalition's features are
fixed to the instance's values and the remaining features are drawn from background
data.

Shapley values are the only attribution that is *efficient* (they sum to
``prediction - base_value``), *symmetric*, and assign zero to features the model
ignores. Computing them exactly needs ``2^n_features`` coalitions, so this explainer

* enumerates every coalition when that is affordable (exact result),
* otherwise estimates them with the weighted-regression formulation of Lundberg & Lee
  (2017), sampling coalitions in importance-weighted, complementary pairs,
* and uses the closed form for linear regressors.

The implementation is self-contained (numpy only) and works with any model through
:class:`~xai_framework.model.ModelAdapter`.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any, Literal

import numpy as np

from ..explanation import Explanation
from ..registry import register_explainer
from .base import BaseExplainer

Algorithm = Literal["auto", "exact", "sampling", "linear"]

# Upper bound on model-input rows evaluated per batch, to keep memory flat.
_MAX_ROWS_PER_BATCH = 250_000


def shapley_kernel_weights(n_features: int, sizes: np.ndarray) -> np.ndarray:
    """Weight of a coalition of size ``s`` in the Shapley regression formulation.

    ``pi(s) = (n - 1) / (C(n, s) * s * (n - s))`` - the weighting under which a
    weighted least-squares fit over all coalitions recovers the exact Shapley values.
    """
    n = n_features
    sizes = np.asarray(sizes)
    return np.array([(n - 1) / (comb(n, int(s)) * s * (n - s)) for s in sizes], dtype=float)


def enumerate_coalitions(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """All non-trivial coalitions as a binary matrix plus their kernel weights."""
    rows = []
    for s in range(1, n_features):
        for subset in combinations(range(n_features), s):
            z = np.zeros(n_features)
            z[list(subset)] = 1.0
            rows.append(z)
    Z = np.vstack(rows)
    return Z, shapley_kernel_weights(n_features, Z.sum(axis=1))


def sample_coalitions(
    n_features: int, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Coalitions for a limited budget: enumerate the cheap sizes, sample the rest.

    Coalition sizes are taken in pairs ``(s, n - s)`` from the ends inward - these
    carry the most kernel weight and, starting with the singletons, guarantee that
    every feature is identifiable. A size pair is fully enumerated while it fits the
    budget; the remaining budget is spent on random complementary pairs drawn from
    the sizes not yet enumerated, in proportion to their kernel mass.

    Enumerated rows carry their exact kernel weight; sampled rows share the leftover
    kernel mass equally, so both kinds can sit in one weighted regression.
    """
    n = n_features
    sizes = np.arange(1, n)
    mass = (n - 1) / (sizes * (n - sizes))  # total kernel weight of each size
    rows: list[np.ndarray] = []
    weights: list[float] = []
    enumerated = np.zeros(n - 1, dtype=bool)
    remaining = int(n_samples)

    for s in range(1, n // 2 + 1):
        partners = [s] if 2 * s == n else [s, n - s]
        count = sum(comb(n, t) for t in partners)
        if count > remaining:
            break
        for t in partners:
            weight = float(mass[t - 1] / comb(n, t))
            for subset in combinations(range(n), t):
                z = np.zeros(n)
                z[list(subset)] = 1.0
                rows.append(z)
                weights.append(weight)
            enumerated[t - 1] = True
        remaining -= count

    left = ~enumerated
    n_pairs = remaining // 2
    if left.any() and n_pairs > 0:
        p = mass[left] / mass[left].sum()
        drawn = rng.choice(sizes[left], size=n_pairs, p=p)
        sampled: list[np.ndarray] = []
        for t in drawn:
            z = np.zeros(n)
            z[rng.choice(n, size=int(t), replace=False)] = 1.0
            sampled.extend([z, 1.0 - z])  # the complement has the same kernel weight
        share = float(mass[left].sum() / len(sampled))
        rows.extend(sampled)
        weights.extend([share] * len(sampled))

    return np.vstack(rows), np.asarray(weights, dtype=float)


def solve_shapley_regression(
    Z: np.ndarray, w: np.ndarray, y: np.ndarray, fx: float, base: float
) -> np.ndarray:
    """Weighted least squares for ``phi`` subject to ``base + sum(phi) == fx``.

    The efficiency constraint is imposed by eliminating the last feature, exactly
    as in Lundberg & Lee (2017).
    """
    n = Z.shape[1]
    if n == 1:
        return np.array([fx - base])
    delta = fx - base
    y_adj = y - base - Z[:, -1] * delta
    A = Z[:, :-1] - Z[:, [-1]]
    sw = np.sqrt(w)[:, None]
    beta = np.linalg.lstsq(A * sw, y_adj * sw[:, 0], rcond=None)[0]
    return np.append(beta, delta - beta.sum())


@register_explainer("coalition")
class CoalitionExplainer(BaseExplainer):
    """Shapley-value attributions estimated from feature coalitions.

    Parameters
    ----------
    model, X_background
        See :class:`~xai_framework.explainers.base.BaseExplainer`.
    n_background
        Rows sampled from ``X_background`` to represent "feature absent". Cost grows
        linearly with this; 25-100 is usually plenty.
    n_coalitions
        Coalitions evaluated per local explanation. ``"auto"`` uses
        ``min(2**n_features - 2, 2 * n_features + 2048)``. When every coalition fits
        in the budget the result is exact.
    global_n_coalitions
        Coalitions per row for global explanations, where many rows are averaged so
        each can be rougher. ``"auto"`` uses ``min(2**n_features - 2, 2 * n_features + 256)``.
    n_global
        Rows sampled from the dataset for global explanations.
    algorithm
        ``"auto"`` picks ``"linear"`` for linear regressors, ``"exact"`` when
        affordable, else ``"sampling"``. Force one to override.

    Attributes
    ----------
    background_
        The rows used as the background distribution.
    """

    name = "coalition"
    supports_local = True
    supports_global = True

    def __init__(
        self,
        model: Any,
        X_background: Any,
        *,
        n_background: int = 50,
        n_coalitions: int | str = "auto",
        global_n_coalitions: int | str = "auto",
        n_global: int = 100,
        algorithm: Algorithm = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, X_background, **kwargs)
        if algorithm not in ("auto", "exact", "sampling", "linear"):
            raise ValueError(f"unknown algorithm {algorithm!r}")
        if algorithm == "linear" and not self._linear_ok():
            raise ValueError("algorithm='linear' needs a linear regressor with coef_/intercept_")
        self.algorithm = algorithm
        self.n_background = n_background
        self.n_coalitions = n_coalitions
        self.global_n_coalitions = global_n_coalitions
        self.n_global = n_global
        self.background_ = self._subsample(self.X_background, n_background)
        self._base_cache: dict[int | None, float] = {}

    # ------------------------------------------------------------------ helpers
    def _linear_ok(self) -> bool:
        m = self.adapter.model
        return (
            self.adapter.family == "linear"
            and not self.adapter.is_classifier
            and np.ndim(getattr(m, "coef_", None)) == 1
        )

    def _budget(self, n_features: int, setting: int | str, auto_extra: int) -> int:
        total = 2**n_features - 2
        if setting == "auto":
            return int(min(total, 2 * n_features + auto_extra))
        return int(max(2, min(total, int(setting))))

    def _resolve(self, n_features: int, budget: int) -> str:
        if self.algorithm != "auto":
            return self.algorithm
        if self._linear_ok():
            return "linear"
        return "exact" if 2**n_features - 2 <= budget else "sampling"

    def _base_value(self, target: int | None) -> float:
        if target not in self._base_cache:
            self._base_cache[target] = float(self._predict_scalar(self.background_, target).mean())
        return self._base_cache[target]

    def _coalition_values(self, x: np.ndarray, Z: np.ndarray, target: int | None) -> np.ndarray:
        """``E_background[f(h(z))]`` for each coalition row of ``Z``."""
        B, n = self.background_.shape
        per_batch = max(1, _MAX_ROWS_PER_BATCH // B)
        out = np.empty(len(Z))
        for start in range(0, len(Z), per_batch):
            Zb = Z[start : start + per_batch].astype(bool)
            X_eval = np.where(Zb[:, None, :], x[None, None, :], self.background_[None, :, :])
            preds = self._predict_scalar(X_eval.reshape(-1, n), target)
            out[start : start + len(Zb)] = preds.reshape(len(Zb), B).mean(axis=1)
        return out

    def _shapley(
        self, x: np.ndarray, target: int | None, budget: int
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        n = len(x)
        algorithm = self._resolve(n, budget)
        fx = float(self._predict_scalar(x.reshape(1, -1), target)[0])

        if algorithm == "linear":
            m = self.adapter.model
            mean = self.background_.mean(axis=0)
            phi = np.asarray(m.coef_, dtype=float) * (x - mean)
            base = fx - phi.sum()
            return phi, base, {"algorithm": "linear", "exact": True}

        base = self._base_value(target)
        if n == 1:
            return np.array([fx - base]), base, {"algorithm": "exact", "exact": True}
        if algorithm == "exact":
            Z, w = enumerate_coalitions(n)
        else:
            Z, w = sample_coalitions(n, budget, self.rng)
        y = self._coalition_values(x, Z, target)
        phi = solve_shapley_regression(Z, w, y, fx, base)
        exact = algorithm == "exact" or len(Z) == 2**n - 2
        info = {"algorithm": algorithm, "exact": exact, "n_coalitions": len(Z)}
        return phi, base, info

    # ------------------------------------------------------------------ explanations
    def _explain_instance(self, x: np.ndarray, target: int | None) -> Explanation:
        budget = self._budget(len(x), self.n_coalitions, auto_extra=2048)
        phi, base, info = self._shapley(x, target, budget)
        return self._make_explanation(
            phi, x, target, base_value=base, n_background=len(self.background_), **info
        )

    def _explain_global(
        self, X: np.ndarray, y: np.ndarray | None, target: int | None
    ) -> Explanation:
        Xs = self._subsample(X, self.n_global)
        budget = self._budget(X.shape[1], self.global_n_coalitions, auto_extra=256)
        if self.adapter.is_classifier and target is None:
            # explain each row's own predicted class
            targets = self.adapter.predict(Xs).argmax(axis=1).tolist()
        else:
            targets = [target] * len(Xs)
        phis = np.vstack(
            [self._shapley(row, t, budget)[0] for row, t in zip(Xs, targets, strict=True)]
        )
        return self._make_explanation(
            np.abs(phis).mean(axis=0),
            None,
            target,
            scope="global",
            aggregation="mean(|phi|)"
            + (" over predicted classes" if self.adapter.is_classifier and target is None else ""),
            mean_signed=phis.mean(axis=0),
            n_rows=len(Xs),
            n_coalitions_per_row=budget,
            n_background=len(self.background_),
        )
