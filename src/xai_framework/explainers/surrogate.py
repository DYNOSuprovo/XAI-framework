"""Local surrogate explainer: a weighted linear model fitted around one instance.

The idea, introduced by LIME (Ribeiro, Singh & Guestrin, 2016): a complex model may
be impossible to describe globally, but in a small neighbourhood around one input
it is usually close to linear. So we perturb the instance, ask the model for
predictions on the perturbations, weight each by its proximity to the instance, and
fit a sparse linear model to those weighted samples. The linear model's coefficients
say how the black box behaves *right here*.

This implementation is self-contained (numpy only). Defaults follow the published
method - Gaussian perturbations drawn from the training distribution, an exponential
proximity kernel of width ``0.75 * sqrt(n_features)``, a ridge surrogate with
``alpha=1`` - with one addition: attributions are reported as *contributions*
(``coefficient * z-scored value``) so that ``base_value + sum(values)`` equals the
surrogate's local prediction, making them directly comparable to the coalition
explainer's Shapley values. Pass ``mode="coefficient"`` for the raw surrogate weights.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..explanation import Explanation
from ..registry import register_explainer
from .base import BaseExplainer


def weighted_ridge(
    Z: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = 1.0
) -> tuple[float, np.ndarray]:
    """Closed-form weighted ridge regression with an unpenalised intercept.

    Minimises ``sum_i w_i (y_i - b - z_i . beta)^2 + alpha * ||beta||^2``.
    Returns ``(intercept, coefficients)``.
    """
    n, p = Z.shape
    A = np.column_stack([np.ones(n), Z])
    G = A.T @ (A * w[:, None])
    reg = alpha * np.eye(p + 1)
    reg[0, 0] = 0.0  # never shrink the intercept
    b = A.T @ (w * y)
    try:
        beta = np.linalg.solve(G + reg, b)
    except np.linalg.LinAlgError:  # pragma: no cover - singular in degenerate cases
        beta = np.linalg.lstsq(G + reg, b, rcond=None)[0]
    return float(beta[0]), beta[1:]


def weighted_r2(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    ybar = np.sum(w * y) / np.sum(w)
    ss_res = np.sum(w * (y - pred) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


@register_explainer("surrogate")
class LocalSurrogateExplainer(BaseExplainer):
    """Local linear surrogate attributions for tabular data.

    Parameters
    ----------
    model, X_background
        See :class:`~xai_framework.explainers.base.BaseExplainer`.
    n_samples
        Number of perturbed samples drawn per explanation.
    kernel_width
        Width of the exponential proximity kernel in z-scored space. Defaults to
        ``0.75 * sqrt(n_features)``.
    num_features
        Keep only this many features in the surrogate (selected by largest
        coefficient magnitude, then refit). ``None`` keeps all of them.
    alpha
        Ridge penalty of the surrogate.
    categorical_features
        Column indices or names to treat as categorical: they are perturbed by
        sampling from their empirical distribution and represented in the surrogate
        as ``1[value == instance value]``.
    sample_around_instance
        Draw perturbations from ``N(x, std)`` instead of the training distribution
        ``N(mean, std)``. Off by default; turning it on gives a tighter, more local fit.
    mode
        ``"contribution"`` (default) reports ``coef * z`` so that
        ``base_value + sum(values)`` equals the surrogate's local prediction;
        ``"coefficient"`` reports raw surrogate coefficients.
    """

    name = "surrogate"
    supports_local = True
    supports_global = False

    def __init__(
        self,
        model: Any,
        X_background: Any,
        *,
        n_samples: int = 5000,
        kernel_width: float | None = None,
        num_features: int | None = None,
        alpha: float = 1.0,
        categorical_features: list[int | str] | None = None,
        sample_around_instance: bool = False,
        mode: Literal["contribution", "coefficient"] = "contribution",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, X_background, **kwargs)
        if mode not in ("contribution", "coefficient"):
            raise ValueError("mode must be 'contribution' or 'coefficient'")
        self.n_samples = int(n_samples)
        self.num_features = num_features
        self.alpha = float(alpha)
        self.sample_around_instance = sample_around_instance
        self.mode = mode

        n_feat = self.X_background.shape[1]
        self.kernel_width = (
            float(kernel_width) if kernel_width is not None else 0.75 * np.sqrt(n_feat)
        )
        self.categorical_idx = sorted(
            {self._feature_index(f) for f in (categorical_features or [])}
        )
        self._categorical_mask = np.zeros(n_feat, dtype=bool)
        self._categorical_mask[self.categorical_idx] = True

        self.mean_ = self.X_background.mean(axis=0)
        self.std_ = self.X_background.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # constant columns: avoid division by zero
        self._cat_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for j in self.categorical_idx:
            vals, counts = np.unique(self.X_background[:, j], return_counts=True)
            self._cat_values[j] = (vals, counts / counts.sum())

    def _feature_index(self, f: int | str) -> int:
        if isinstance(f, (int, np.integer)):
            return int(f)
        return self.feature_names.index(str(f))

    # ------------------------------------------------------------------ core
    def _sample(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Z_raw, Z)``: perturbed inputs and their interpretable encoding."""
        n, p = self.n_samples, len(x)
        noise = self.rng.standard_normal((n, p))
        center = x if self.sample_around_instance else self.mean_
        Z_raw = noise * self.std_ + center
        Z_raw[0] = x  # the instance itself is always the first sample

        Z = (Z_raw - self.mean_) / self.std_
        for j, (vals, probs) in self._cat_values.items():
            Z_raw[:, j] = self.rng.choice(vals, size=n, p=probs)
            Z_raw[0, j] = x[j]
            Z[:, j] = (Z_raw[:, j] == x[j]).astype(float)
        return Z_raw, Z

    def _kernel(self, Z: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(Z - Z[0], axis=1)
        return np.sqrt(np.exp(-(d**2) / self.kernel_width**2))

    def _explain_instance(self, x: np.ndarray, target: int | None) -> Explanation:
        Z_raw, Z = self._sample(x)
        y = self._predict_scalar(Z_raw, target)
        w = self._kernel(Z)

        intercept, coef = weighted_ridge(Z, y, w, self.alpha)
        selected = np.arange(len(x))
        if self.num_features is not None and self.num_features < len(x):
            selected = np.argsort(-np.abs(coef), kind="stable")[: self.num_features]
            intercept, coef_sel = weighted_ridge(Z[:, selected], y, w, self.alpha)
            coef = np.zeros(len(x))
            coef[selected] = coef_sel

        local_pred = intercept + Z @ coef
        z_x = Z[0]
        contributions = coef * z_x
        values = contributions if self.mode == "contribution" else coef

        return self._make_explanation(
            values,
            x,
            target,
            base_value=intercept,
            local_r2=weighted_r2(y, local_pred, w),
            local_prediction=float(local_pred[0]),
            coefficients=coef,
            contributions=contributions,
            selected_features=[self.feature_names[i] for i in selected],
            n_samples=self.n_samples,
            kernel_width=self.kernel_width,
            mode=self.mode,
        )
