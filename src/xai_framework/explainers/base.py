"""Abstract base class shared by every explainer.

Subclasses implement :meth:`_explain_instance` (local) and/or :meth:`_explain_global`
and get input handling, target resolution and feature-name bookkeeping for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ..explanation import Explanation
from ..model import ModelAdapter, to_numpy


class BaseExplainer(ABC):
    """Common interface for all explainers.

    Parameters
    ----------
    model
        A fitted model or a :class:`~xai_framework.model.ModelAdapter`.
    X_background
        Representative data (training or validation rows). Used to fit perturbation
        distributions, as the coalition background, and as the default dataset for global
        explanations. A DataFrame also supplies feature names.
    feature_names, class_names, task
        Forwarded to :class:`ModelAdapter` when ``model`` is not one already.
    random_state
        Seed for any stochastic component.
    """

    #: registry key; set by :func:`~xai_framework.registry.register_explainer`
    name: str = "base"
    supports_local: bool = True
    supports_global: bool = False
    #: whether :meth:`explain_global` needs ground-truth labels (``y``)
    requires_y: bool = False

    def __init__(
        self,
        model: Any,
        X_background: Any,
        *,
        feature_names: list[str] | None = None,
        class_names: list[Any] | None = None,
        task: str | None = None,
        random_state: int | None = None,
    ) -> None:
        if isinstance(X_background, pd.DataFrame) and feature_names is None:
            feature_names = [str(c) for c in X_background.columns]
        if isinstance(model, ModelAdapter):
            self.adapter = model
            if feature_names is not None and self.adapter.feature_names is None:
                self.adapter.feature_names = list(feature_names)
        else:
            self.adapter = ModelAdapter(
                model, task=task, feature_names=feature_names, class_names=class_names
            )
        self.X_background = to_numpy(X_background)
        if self.X_background.ndim != 2:
            raise ValueError("X_background must be 2-D (n_samples, n_features)")
        self.feature_names = self.adapter.ensure_feature_names(self.X_background.shape[1])
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        # Probe once so callables get their task inferred and class names filled in.
        self.adapter.predict(self.X_background[:1])

    # ------------------------------------------------------------------ availability
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Whether this explainer's optional dependencies are importable."""
        return True, ""

    # ------------------------------------------------------------------ public API
    def explain_instance(self, x: Any, target: Any = None) -> Explanation:
        """Explain a single prediction.

        Parameters
        ----------
        x
            One row: array, list, Series, or single-row DataFrame.
        target
            For classifiers, the class label (or index) to explain. Defaults to the
            predicted class.
        """
        if not self.supports_local:
            raise NotImplementedError(f"{self.name} does not produce local explanations")
        x = self._prepare_instance(x)
        target_idx = self._resolve_target(x, target)
        return self._explain_instance(x, target_idx)

    def explain(self, X: Any, target: Any = None) -> list[Explanation]:
        """Explain every row of ``X`` (convenience wrapper around :meth:`explain_instance`)."""
        X = to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return [self.explain_instance(row, target=target) for row in X]

    def explain_global(self, X: Any = None, y: Any = None, target: Any = None) -> Explanation:
        """Explain the model's behaviour over a dataset (defaults to ``X_background``)."""
        if not self.supports_global:
            raise NotImplementedError(f"{self.name} does not produce global explanations")
        X = self.X_background if X is None else to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {X.shape[1]}")
        y = None if y is None else np.asarray(y)
        target_idx = None if target is None else self.adapter.class_index(target)
        return self._explain_global(X, y, target_idx)

    # ------------------------------------------------------------------ to implement
    @abstractmethod
    def _explain_instance(self, x: np.ndarray, target: int | None) -> Explanation:
        """``x`` is a 1-D float array; ``target`` is a resolved class index or None."""

    def _explain_global(
        self, X: np.ndarray, y: np.ndarray | None, target: int | None
    ) -> Explanation:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def _prepare_instance(self, x: Any) -> np.ndarray:
        arr = to_numpy(x).reshape(-1)
        if len(arr) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {len(arr)}")
        return arr

    def _resolve_target(self, x: np.ndarray, target: Any) -> int | None:
        if not self.adapter.is_classifier:
            return None
        if target is None:
            return int(self.adapter.predict(x)[0].argmax())
        return self.adapter.class_index(target)

    def _predict_scalar(self, X: np.ndarray, target: int | None) -> np.ndarray:
        """Vectorised ``f(X)`` returning the single number being explained per row."""
        return self.adapter.predict_target(X, target)

    def _make_explanation(
        self,
        values: np.ndarray,
        x: np.ndarray | None,
        target: int | None,
        *,
        scope: str = "local",
        base_value: float | None = None,
        prediction: float | None = None,
        **metadata: Any,
    ) -> Explanation:
        if prediction is None and x is not None:
            prediction = float(self._predict_scalar(x.reshape(1, -1), target)[0])
        return Explanation(
            feature_names=self.feature_names,
            values=values,
            method=self.name,
            scope=scope,  # type: ignore[arg-type]
            feature_values=x,
            base_value=base_value,
            prediction=prediction,
            target=self.adapter.class_label(target) if target is not None else None,
            task=self.adapter.task,
            metadata=metadata,
        )

    def _subsample(self, X: np.ndarray, n: int | None) -> np.ndarray:
        if n is None or len(X) <= n:
            return X
        idx = self.rng.choice(len(X), size=n, replace=False)
        return X[idx]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.adapter!r}, n_background={len(self.X_background)})"
        )
