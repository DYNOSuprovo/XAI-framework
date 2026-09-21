import numpy as np
import pytest

from xai_framework import PermutationExplainer
from xai_framework.explainers.permutation import accuracy, neg_log_loss, r2_score


def test_metrics():
    assert accuracy(np.array([1, 0, 1]), np.array([1, 1, 1])) == pytest.approx(2 / 3)
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert r2_score(y, np.full(3, 2.0)) == pytest.approx(0.0)
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert neg_log_loss(np.array([0, 1]), proba) == pytest.approx(np.mean(np.log([0.9, 0.8])))


def test_regression_importance_ordering(linreg, regression):
    X, y = regression
    exp = PermutationExplainer(linreg, X, random_state=0, n_repeats=3).explain_global(X, y)
    assert exp.scope == "global"
    assert exp.metadata["metric"] == "r2"
    assert exp.metadata["baseline_score"] > 0.99
    assert exp.values[0] > exp.values[1] > exp.values[2]
    assert abs(exp.values[2]) < 0.01  # the noise feature does not matter


def test_classifier_defaults_to_log_loss(iris_rf, iris):
    X, y = iris
    exp = PermutationExplainer(iris_rf, X, random_state=0, n_repeats=2).explain_global(X, y)
    assert exp.metadata["metric"] == "log_loss"
    assert exp.top(1)[0][0].startswith("petal")
    assert (exp.values[:2] < exp.values[2:]).all()  # sepal features matter less


def test_accuracy_metric_with_string_labels(iris_rf, iris):
    X, y = iris
    explainer = PermutationExplainer(iris_rf, X, metric="accuracy", random_state=0, n_repeats=1)
    exp = explainer.explain_global(X, y)
    assert exp.metadata["baseline_score"] == pytest.approx(1.0)


def test_custom_metric_and_subsampling(linreg, regression):
    X, y = regression

    def neg_mae(y_true, y_pred):
        return -float(np.mean(np.abs(y_true - y_pred)))

    explainer = PermutationExplainer(linreg, X, metric=neg_mae, max_rows=50, random_state=0)
    exp = explainer.explain_global(X, y)
    assert exp.metadata["metric"] == "neg_mae"
    assert exp.metadata["n_rows"] == 50


def test_bad_metric_choices(linreg, regression):
    X, _ = regression
    with pytest.raises(ValueError, match="unknown metric"):
        PermutationExplainer(linreg, X, metric="f1")
    with pytest.raises(ValueError, match="classifier"):
        PermutationExplainer(linreg, X, metric="log_loss")


def test_requires_y(linreg, regression):
    X, _ = regression
    with pytest.raises(ValueError, match="y="):
        PermutationExplainer(linreg, X).explain_global(X)


def test_no_local(linreg, regression):
    X, _ = regression
    with pytest.raises(NotImplementedError):
        PermutationExplainer(linreg, X).explain_instance(X[0])
