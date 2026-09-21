import numpy as np
import pandas as pd
import pytest

from xai_framework import LocalSurrogateExplainer
from xai_framework.explainers.surrogate import weighted_r2, weighted_ridge


def test_weighted_ridge_recovers_linear_relationship():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 2))
    y = 1.5 + 2.0 * Z[:, 0] - 1.0 * Z[:, 1]
    intercept, coef = weighted_ridge(Z, y, np.ones(500), alpha=1e-6)
    assert intercept == pytest.approx(1.5, abs=1e-3)
    np.testing.assert_allclose(coef, [2.0, -1.0], atol=1e-3)
    assert weighted_r2(y, intercept + Z @ coef, np.ones(500)) == pytest.approx(1.0, abs=1e-6)


def test_surrogate_recovers_signs_and_ranking_on_linear_model(linreg, regression):
    X, _ = regression
    exp = LocalSurrogateExplainer(linreg, X, random_state=0).explain_instance(X[0])
    coef = exp.metadata["coefficients"]
    # coefficients live in z-scored space, but signs and ordering must match the model
    assert coef[0] > 0 and coef[1] < 0
    assert abs(coef[0]) > abs(coef[1]) > abs(coef[2])
    assert abs(coef[2]) < 0.05 * abs(coef[0])
    assert exp.metadata["local_r2"] > 0.99


def test_contribution_mode_is_additive(linreg, regression):
    X, _ = regression
    exp = LocalSurrogateExplainer(linreg, X, random_state=0).explain_instance(X[3])
    assert exp.base_value + exp.values.sum() == pytest.approx(exp.metadata["local_prediction"])
    # for a linear model the surrogate reproduces the prediction almost exactly
    assert exp.metadata["local_prediction"] == pytest.approx(exp.prediction, abs=0.05)


def test_coefficient_mode_returns_raw_coefficients(linreg, regression):
    X, _ = regression
    exp = LocalSurrogateExplainer(linreg, X, random_state=0, mode="coefficient").explain_instance(
        X[3]
    )
    np.testing.assert_array_equal(exp.values, exp.metadata["coefficients"])


def test_reproducible_with_random_state(linreg, regression):
    X, _ = regression
    a = LocalSurrogateExplainer(linreg, X, random_state=7, n_samples=500).explain_instance(X[1])
    b = LocalSurrogateExplainer(linreg, X, random_state=7, n_samples=500).explain_instance(X[1])
    np.testing.assert_array_equal(a.values, b.values)


def test_classification_target_defaults_to_predicted_class(iris_rf, iris):
    X, _ = iris
    exp = LocalSurrogateExplainer(iris_rf, X, random_state=0, n_samples=1000).explain_instance(
        X.iloc[0]
    )
    assert exp.target == "setosa"
    assert exp.prediction > 0.9
    assert exp.feature_names == list(X.columns)
    assert exp.top(1)[0][0].startswith("petal")


def test_explicit_target_by_label(iris_rf, iris):
    X, _ = iris
    explainer = LocalSurrogateExplainer(iris_rf, X, random_state=0, n_samples=1000)
    exp = explainer.explain_instance(X.iloc[0], target="virginica")
    assert exp.target == "virginica"
    assert exp.prediction < 0.1


def test_num_features_limits_surrogate(linreg, regression):
    X, _ = regression
    exp = LocalSurrogateExplainer(linreg, X, random_state=0, num_features=2).explain_instance(X[0])
    assert (exp.values != 0).sum() == 2
    assert set(exp.metadata["selected_features"]) == {"f0", "f1"}


def test_categorical_feature_handling():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 3, size=300)
    cont = rng.normal(size=300)
    X = pd.DataFrame({"cat": cat, "cont": cont})
    # model: category 2 adds a big offset
    model = lambda A: (A[:, 0] == 2) * 5.0 + A[:, 1]  # noqa: E731
    explainer = LocalSurrogateExplainer(model, X, categorical_features=["cat"], random_state=0)
    exp = explainer.explain_instance(np.array([2.0, 0.0]))
    assert exp.metadata["coefficients"][0] > 1.0  # being in category 2 matters


def test_sample_around_instance_option(linreg, regression):
    X, _ = regression
    exp = LocalSurrogateExplainer(
        linreg, X, random_state=0, sample_around_instance=True
    ).explain_instance(X[0])
    assert exp.metadata["coefficients"][0] > 0


def test_batch_explain(linreg, regression):
    X, _ = regression
    exps = LocalSurrogateExplainer(linreg, X, random_state=0, n_samples=300).explain(X[:3])
    assert len(exps) == 3


def test_bad_mode_raises(linreg, regression):
    X, _ = regression
    with pytest.raises(ValueError):
        LocalSurrogateExplainer(linreg, X, mode="nope")


def test_surrogate_has_no_global(linreg, regression):
    X, _ = regression
    with pytest.raises(NotImplementedError):
        LocalSurrogateExplainer(linreg, X).explain_global()
