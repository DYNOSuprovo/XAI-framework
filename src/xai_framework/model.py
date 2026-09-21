"""Uniform prediction interface over scikit-learn style estimators and plain callables.

Every explainer in the framework talks to the model through :class:`ModelAdapter`, so
the explainers never need to know whether they are looking at a scikit-learn
estimator, an XGBoost booster, a Pipeline, or a bare ``def predict(X)`` function.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

Task = Literal["classification", "regression"]
Family = Literal["tree", "linear", "other"]

# Class-name fragments that identify tree ensembles (reserved for a fast tree path).
_TREE_NAME_FRAGMENTS = (
    "DecisionTree",
    "RandomForest",
    "ExtraTree",
    "GradientBoosting",
    "HistGradientBoosting",
    "XGB",
    "LGBM",
    "CatBoost",
)
_TREE_MODULE_FRAGMENTS = ("xgboost", "lightgbm", "catboost")


def to_numpy(X: Any) -> np.ndarray:
    """Convert DataFrames / Series / lists to a float ndarray."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _py_scalar(value: Any) -> Any:
    """Turn numpy scalars into plain Python so labels print and compare cleanly."""
    return value.item() if isinstance(value, np.generic) else value


class ModelAdapter:
    """Wraps a fitted model and exposes a single :meth:`predict` method.

    ``predict`` always returns class probabilities with shape ``(n, n_classes)`` for
    classification and a 1-D array ``(n,)`` for regression, regardless of what the
    underlying object looks like.

    Parameters
    ----------
    model
        A fitted estimator with ``predict_proba`` and/or ``predict``, or a callable
        ``f(X) -> np.ndarray``.
    task
        ``"classification"`` or ``"regression"``. Inferred when omitted: objects with
        ``predict_proba`` are classifiers, everything else is treated as regression.
        Callables that return a 2-D array are re-classified on first call.
    feature_names
        Names of the input columns. Inferred from ``model.feature_names_in_`` when
        available, otherwise ``f0 .. f{n-1}`` on first use.
    class_names
        Labels for each output column of a classifier. Inferred from ``model.classes_``.
    """

    def __init__(
        self,
        model: Any,
        *,
        task: Task | None = None,
        feature_names: list[str] | None = None,
        class_names: list[Any] | None = None,
    ) -> None:
        self.model = model
        self._explicit_task = task is not None
        self.task: Task = task or self._infer_task(model)
        self.family: Family = self._infer_family(model)
        self.feature_names: list[str] | None = (
            [str(f) for f in feature_names]
            if feature_names is not None
            else self._infer_feature_names(model)
        )
        self.class_names: list[Any] | None = (
            [_py_scalar(c) for c in class_names]
            if class_names is not None
            else self._infer_class_names(model)
        )

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _infer_task(model: Any) -> Task:
        if hasattr(model, "predict_proba"):
            return "classification"
        if hasattr(model, "predict") or callable(model):
            return "regression"
        raise TypeError(
            "model must expose predict_proba/predict or be a callable f(X) -> predictions"
        )

    @staticmethod
    def _infer_family(model: Any) -> Family:
        cls = type(model)
        name, module = cls.__name__, (cls.__module__ or "")
        if any(frag in module for frag in _TREE_MODULE_FRAGMENTS):
            return "tree"
        if any(frag in name for frag in _TREE_NAME_FRAGMENTS):
            return "tree"
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            return "linear"
        return "other"

    @staticmethod
    def _infer_feature_names(model: Any) -> list[str] | None:
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            # xgboost / lightgbm sklearn wrappers expose the booster's names
            booster = getattr(model, "get_booster", None)
            if callable(booster):
                try:
                    names = booster().feature_names
                except Exception:  # pragma: no cover - defensive
                    names = None
        return [str(n) for n in names] if names is not None else None

    @staticmethod
    def _infer_class_names(model: Any) -> list[Any] | None:
        classes = getattr(model, "classes_", None)
        return [_py_scalar(c) for c in classes] if classes is not None else None

    # ------------------------------------------------------------------ helpers
    @property
    def is_classifier(self) -> bool:
        return self.task == "classification"

    @property
    def n_classes(self) -> int | None:
        return len(self.class_names) if self.class_names is not None else None

    def ensure_feature_names(self, n_features: int) -> list[str]:
        """Return feature names, generating ``f0..f{n-1}`` if none are known."""
        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(n_features)]
        if len(self.feature_names) != n_features:
            raise ValueError(
                f"model expects {len(self.feature_names)} features but data has {n_features}"
            )
        return self.feature_names

    def class_index(self, target: Any) -> int:
        """Map a class label *or* a positional index to an output column index.

        Labels are looked up first so that integer class labels such as ``[10, 20]``
        resolve correctly; anything not found is treated as a positional index.
        """
        target = _py_scalar(target)
        if self.class_names is not None:
            for i, label in enumerate(self.class_names):
                if label == target and type(label) is type(target):
                    return i
        if isinstance(target, (int, np.integer)):
            idx = int(target)
            if self.n_classes is not None and not 0 <= idx < self.n_classes:
                raise ValueError(f"class index {idx} out of range for {self.n_classes} classes")
            return idx
        if self.class_names is not None and target in self.class_names:
            return self.class_names.index(target)
        raise ValueError(f"unknown class {target!r}; known classes: {self.class_names}")

    def class_label(self, index: int) -> Any:
        return self.class_names[index] if self.class_names is not None else index

    def _frame_if_needed(self, X: np.ndarray) -> Any:
        # Estimators fitted on DataFrames warn when given raw arrays; hand them a frame.
        if getattr(self.model, "feature_names_in_", None) is not None and self.feature_names:
            return pd.DataFrame(X, columns=self.feature_names)
        return X

    # ------------------------------------------------------------------ prediction
    def predict(self, X: Any) -> np.ndarray:
        """Predict probabilities ``(n, n_classes)`` or regression outputs ``(n,)``."""
        X = to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.asarray(self._raw_predict(X), dtype=float)

        if not self._explicit_task and out.ndim == 2 and out.shape[1] > 1:
            self.task = "classification"  # a callable that returns probabilities

        if self.is_classifier:
            if out.ndim == 1:
                # A callable returning P(class 1); expand to two columns.
                out = np.column_stack([1.0 - out, out])
            if self.class_names is None:
                self.class_names = list(range(out.shape[1]))
            return out
        return out.reshape(-1) if out.ndim == 2 and out.shape[1] == 1 else out

    def _raw_predict(self, X: np.ndarray) -> Any:
        model = self.model
        if self.is_classifier and hasattr(model, "predict_proba"):
            return model.predict_proba(self._frame_if_needed(X))
        if hasattr(model, "predict"):
            return model.predict(self._frame_if_needed(X))
        return model(X)

    def predict_target(self, X: Any, target: int | None = None) -> np.ndarray:
        """Scalar prediction per row: ``P(target)`` for classifiers, output for regressors."""
        out = self.predict(X)
        if self.is_classifier:
            return out[:, 0 if target is None else target]
        return out

    def predict_labels(self, X: Any) -> np.ndarray:
        """Hard predictions (class labels or regression outputs) for scoring."""
        out = self.predict(X)
        if self.is_classifier:
            idx = out.argmax(axis=1)
            return np.asarray([self.class_label(i) for i in idx])
        return out

    def __repr__(self) -> str:
        return (
            f"ModelAdapter({type(self.model).__name__}, task={self.task!r}, family={self.family!r})"
        )
