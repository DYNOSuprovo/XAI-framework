import numpy as np
import pandas as pd
import pytest

from xai_framework import ModelAdapter


def test_infers_classifier_and_tree_family(iris_rf):
    adapter = ModelAdapter(iris_rf)
    assert adapter.task == "classification"
    assert adapter.family == "tree"
    assert adapter.class_names == ["setosa", "versicolor", "virginica"]
    assert adapter.feature_names == list(iris_rf.feature_names_in_)


def test_predict_returns_probabilities(iris_rf, iris):
    X, _ = iris
    proba = ModelAdapter(iris_rf).predict(X.head(5))
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predict_accepts_single_row(iris_rf, iris):
    X, _ = iris
    out = ModelAdapter(iris_rf).predict(X.iloc[0].to_numpy())
    assert out.shape == (1, 3)


def test_regression_linear_family(linreg, regression):
    X, _ = regression
    adapter = ModelAdapter(linreg)
    assert adapter.task == "regression"
    assert adapter.family == "linear"
    assert adapter.predict(X[:4]).shape == (4,)


def test_pipeline_is_other_family(binary_pipeline):
    adapter = ModelAdapter(binary_pipeline)
    assert adapter.family == "other"
    assert adapter.task == "classification"


def test_callable_regression():
    adapter = ModelAdapter(lambda X: X[:, 0] * 2)
    assert adapter.task == "regression"
    np.testing.assert_allclose(adapter.predict(np.array([[1.0, 5.0], [2.0, 5.0]])), [2.0, 4.0])


def test_callable_returning_probabilities_becomes_classifier():
    def f(X):
        p = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p, p])

    adapter = ModelAdapter(f)
    out = adapter.predict(np.zeros((3, 2)))
    assert adapter.task == "classification"
    assert out.shape == (3, 2)
    assert adapter.class_names == [0, 1]


def test_callable_1d_probability_expands_to_two_columns():
    adapter = ModelAdapter(lambda X: np.full(len(X), 0.25), task="classification")
    out = adapter.predict(np.zeros((2, 1)))
    np.testing.assert_allclose(out, [[0.75, 0.25], [0.75, 0.25]])


def test_class_index_prefers_labels_over_positions():
    adapter = ModelAdapter(lambda X: X, task="classification", class_names=[10, 20])
    assert adapter.class_index(20) == 1
    assert adapter.class_index(0) == 0  # positional fallback
    assert adapter.class_index(np.int64(10)) == 0
    with pytest.raises(ValueError):
        adapter.class_index("nope")


def test_class_index_by_string_label(iris_rf):
    adapter = ModelAdapter(iris_rf)
    assert adapter.class_index("virginica") == 2
    assert adapter.class_label(1) == "versicolor"


def test_dataframe_fitted_model_gets_dataframe(iris_rf, iris):
    """A model fitted on a DataFrame must not warn about feature names for raw arrays."""
    import warnings

    X, _ = iris
    adapter = ModelAdapter(iris_rf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapter.predict(X.to_numpy()[:2])


def test_feature_count_mismatch_raises():
    adapter = ModelAdapter(lambda X: X[:, 0], feature_names=["a", "b"])
    with pytest.raises(ValueError):
        adapter.ensure_feature_names(3)


def test_predict_labels_maps_to_class_names(iris_rf, iris):
    X, y = iris
    labels = ModelAdapter(iris_rf).predict_labels(X.head(10))
    assert set(labels) <= {"setosa", "versicolor", "virginica"}
    assert isinstance(pd.Series(labels).iloc[0], str)
