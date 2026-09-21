from math import comb

import numpy as np
import pytest

from xai_framework import CoalitionExplainer
from xai_framework.explainers.coalition import (
    enumerate_coalitions,
    sample_coalitions,
    shapley_kernel_weights,
    solve_shapley_regression,
)

# --------------------------------------------------------------------------- building blocks


def test_kernel_weights_are_symmetric_and_largest_at_the_ends():
    w = shapley_kernel_weights(6, np.arange(1, 6))
    np.testing.assert_allclose(w, w[::-1])
    assert w[0] > w[2]


def test_enumerate_coalitions_covers_every_non_trivial_subset():
    Z, w = enumerate_coalitions(5)
    assert Z.shape == (2**5 - 2, 5)
    assert len({tuple(row) for row in Z}) == len(Z)
    assert (Z.sum(axis=1) > 0).all() and (Z.sum(axis=1) < 5).all()
    assert len(w) == len(Z)


def test_sample_coalitions_enumerates_cheap_sizes_then_samples_pairs():
    rng = np.random.default_rng(0)
    Z, w = sample_coalitions(8, 100, rng)
    assert Z.shape == (100, 8)
    sizes = Z.sum(axis=1)
    # sizes (1, 7) = 16 rows and (2, 6) = 56 rows fit the budget and are enumerated ...
    assert (sizes[:8] == 1).all() and (sizes[8:16] == 7).all()
    assert (sizes[16:44] == 2).all() and (sizes[44:72] == 6).all()
    assert len({tuple(r) for r in Z[:72]}) == 72
    # ... the remaining 28 rows are complementary pairs from sizes 3..5
    rest = Z[72:]
    np.testing.assert_array_equal(rest[0::2] + rest[1::2], 1.0)
    assert set(np.unique(rest.sum(axis=1))) <= {3, 4, 5}
    # weights: exact kernel weight per enumerated row, equal share for sampled rows,
    # and the total equals the full kernel mass
    full = shapley_kernel_weights(8, np.arange(1, 8)) * [comb(8, s) for s in range(1, 8)]
    assert w.sum() == pytest.approx(full.sum())
    np.testing.assert_allclose(w[:16], shapley_kernel_weights(8, [1])[0])
    np.testing.assert_allclose(w[16:72], shapley_kernel_weights(8, [2])[0])
    assert (w[72:] == w[72]).all()


def test_small_budget_still_identifies_every_feature():
    """Even 2*n coalitions must separate all features (the singletons are enumerated)."""
    rng = np.random.default_rng(0)
    n, c = 5, np.array([3.0, -2.0, 1.0, 0.5, -0.1])
    Z, w = sample_coalitions(n, 2 * n, rng)
    phi = solve_shapley_regression(Z, w, Z @ c, fx=c.sum(), base=0.0)
    np.testing.assert_allclose(phi, c, atol=1e-10)


def test_solve_regression_recovers_additive_game_exactly():
    """For an additive game v(S) = base + sum_{j in S} c_j the Shapley values are c."""
    rng = np.random.default_rng(0)
    n = 6
    c = rng.normal(size=n)
    base = 0.3
    Z, w = enumerate_coalitions(n)
    y = base + Z @ c
    phi = solve_shapley_regression(Z, w, y, fx=base + c.sum(), base=base)
    np.testing.assert_allclose(phi, c, atol=1e-10)
    # sampled coalitions give the same answer for an additive game
    Zs, ws = sample_coalitions(n, 60, rng)
    phi_s = solve_shapley_regression(Zs, ws, base + Zs @ c, fx=base + c.sum(), base=base)
    np.testing.assert_allclose(phi_s, c, atol=1e-10)


def test_single_feature_gets_everything():
    phi = solve_shapley_regression(np.ones((1, 1)), np.ones(1), np.array([0.7]), fx=0.7, base=0.2)
    np.testing.assert_allclose(phi, [0.5])


# --------------------------------------------------------------------------- explainer


def test_linear_regression_closed_form(linreg, regression):
    X, _ = regression
    explainer = CoalitionExplainer(linreg, X, random_state=0)
    exp = explainer.explain_instance(X[0])
    assert exp.metadata["algorithm"] == "linear"
    expected = linreg.coef_ * (X[0] - explainer.background_.mean(axis=0))
    np.testing.assert_allclose(exp.values, expected, atol=1e-10)
    assert exp.base_value + exp.values.sum() == pytest.approx(exp.prediction)


def test_exact_and_sampling_agree_with_closed_form_on_linear_model(linreg, regression):
    X, _ = regression
    lin = CoalitionExplainer(linreg, X, random_state=0).explain_instance(X[2])
    exact = CoalitionExplainer(linreg, X, random_state=0, algorithm="exact").explain_instance(X[2])
    samp = CoalitionExplainer(
        linreg, X, random_state=0, algorithm="sampling", n_coalitions=40
    ).explain_instance(X[2])
    assert exact.metadata["exact"] is True and exact.metadata["n_coalitions"] == 6
    np.testing.assert_allclose(exact.values, lin.values, atol=1e-8)
    np.testing.assert_allclose(samp.values, lin.values, atol=1e-8)


def test_classifier_is_exact_for_few_features_and_additive(iris_rf, iris):
    X, _ = iris
    explainer = CoalitionExplainer(iris_rf, X, random_state=0)
    exp = explainer.explain_instance(X.iloc[0])
    assert exp.metadata["algorithm"] == "exact"
    assert exp.target == "setosa"
    assert exp.base_value + exp.values.sum() == pytest.approx(exp.prediction, abs=1e-9)
    assert exp.top(1)[0][0].startswith("petal")
    other = explainer.explain_instance(X.iloc[0], target="virginica")
    assert other.target == "virginica" and other.prediction < 0.1


def test_binary_classes_mirror_each_other(binary_gb, binary):
    X, _ = binary
    e1 = CoalitionExplainer(binary_gb, X, random_state=0).explain_instance(X[0], target=1)
    e0 = CoalitionExplainer(binary_gb, X, random_state=0).explain_instance(X[0], target=0)
    np.testing.assert_allclose(e0.values, -e1.values, atol=1e-9)
    assert e0.base_value + e1.base_value == pytest.approx(1.0)


def test_sampling_path_ignores_noise_feature(binary_pipeline, binary):
    X, _ = binary
    explainer = CoalitionExplainer(binary_pipeline, X, random_state=0, algorithm="sampling")
    exp = explainer.explain_instance(X[0])
    assert exp.metadata["algorithm"] == "sampling"
    assert exp.ranks()[2] == 3
    assert exp.base_value + exp.values.sum() == pytest.approx(exp.prediction, abs=1e-9)


def test_reproducible_with_random_state(binary_pipeline, binary):
    X, _ = binary
    kw = dict(random_state=3, algorithm="sampling", n_coalitions=20)
    a = CoalitionExplainer(binary_pipeline, X, **kw).explain_instance(X[1])
    b = CoalitionExplainer(binary_pipeline, X, **kw).explain_instance(X[1])
    np.testing.assert_array_equal(a.values, b.values)


def test_budget_switches_from_exact_to_sampling(iris_rf, iris):
    X, _ = iris
    exp = CoalitionExplainer(iris_rf, X, random_state=0, n_coalitions=8).explain_instance(X.iloc[0])
    assert exp.metadata["algorithm"] == "sampling"
    assert exp.metadata["n_coalitions"] == 8


def test_global_mean_abs(iris_rf, iris):
    X, _ = iris
    explainer = CoalitionExplainer(iris_rf, X, random_state=0, n_global=40)
    glob = explainer.explain_global()
    assert glob.scope == "global"
    assert (glob.values >= 0).all()
    assert glob.metadata["n_rows"] == 40
    assert "predicted classes" in glob.metadata["aggregation"]
    assert glob.top(1)[0][0].startswith("petal")
    per_class = explainer.explain_global(target="setosa")
    assert per_class.target == "setosa"
    assert "predicted classes" not in per_class.metadata["aggregation"]


def test_bad_algorithm_choices(iris_rf, iris, linreg, regression):
    X, _ = iris
    with pytest.raises(ValueError, match="unknown algorithm"):
        CoalitionExplainer(iris_rf, X, algorithm="bogus")
    with pytest.raises(ValueError, match="linear"):
        CoalitionExplainer(iris_rf, X, algorithm="linear")  # a classifier
    Xr, _ = regression
    assert CoalitionExplainer(linreg, Xr, algorithm="linear").algorithm == "linear"
