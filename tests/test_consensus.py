import numpy as np
import pytest

from xai_framework import (
    ConsensusExplainer,
    ConsensusExplanation,
    Explanation,
    LocalSurrogateExplainer,
)
from xai_framework.explainers.consensus import combine_explanations, spearman_matrix


def make(method, values, **kw):
    names = [f"f{i}" for i in range(len(values))]
    return Explanation(
        feature_names=names, values=np.array(values, dtype=float), method=method, **kw
    )


def test_spearman_matrix_identity_and_reversal():
    m = spearman_matrix(
        {"a": np.array([3, 2, 1]), "b": np.array([3, 2, 1]), "c": np.array([1, 2, 3])}
    )
    assert m.loc["a", "b"] == pytest.approx(1.0)
    assert m.loc["a", "c"] == pytest.approx(-1.0)
    assert m.loc["a", "a"] == 1.0


def test_mean_aggregation_and_agreement():
    comps = {"a": make("a", [4.0, 2.0, 0.0]), "b": make("b", [2.0, 1.0, 0.0])}
    exp = combine_explanations(comps)
    assert isinstance(exp, ConsensusExplanation)
    np.testing.assert_allclose(exp.values, [2 / 3, 1 / 3, 0.0])
    assert exp.agreement == pytest.approx(1.0)
    assert exp.agreement_level() == "high"
    assert exp.metadata["top_k_sign_agreement"] == 1.0


def test_weights_change_the_mix():
    comps = {"a": make("a", [1.0, 0.0]), "b": make("b", [0.0, 1.0])}
    exp = combine_explanations(comps, weights={"a": 3, "b": 1})
    np.testing.assert_allclose(exp.values, [0.75, 0.25])
    with pytest.raises(ValueError):
        combine_explanations(comps, weights={"a": -1, "b": 1})


def test_rank_aggregation_uses_borda_scores():
    comps = {"a": make("a", [0.9, 0.1, -0.05]), "b": make("b", [0.1, 0.8, -0.05])}
    exp = combine_explanations(comps, aggregation="rank")
    # each method ranks a different feature first; both features end up tied on top
    assert exp.values[0] == pytest.approx(exp.values[1])
    assert exp.values[2] < 0  # sign follows the mean direction
    assert exp.metadata["aggregation"] == "rank"


def test_disagreement_is_detected():
    comps = {"a": make("a", [1.0, 0.5, 0.1]), "b": make("b", [0.1, 0.5, 1.0])}
    exp = combine_explanations(comps)
    assert exp.agreement == pytest.approx(-1.0)
    assert exp.agreement_level() == "low"
    assert "caution" in exp.to_text()


def test_single_component_has_nan_agreement():
    exp = combine_explanations({"a": make("a", [1.0, 0.5])})
    assert np.isnan(exp.agreement)
    assert exp.agreement_level() == "n/a"


def test_mismatched_components_raise():
    with pytest.raises(ValueError):
        combine_explanations({"a": make("a", [1.0]), "b": make("b", [1.0, 2.0])})
    with pytest.raises(ValueError):
        combine_explanations({"a": make("a", [1.0]), "b": make("b", [1.0], scope="global")})
    with pytest.raises(ValueError):
        combine_explanations({}, aggregation="mean")
    with pytest.raises(ValueError):
        combine_explanations({"a": make("a", [1.0])}, aggregation="median")


def test_consensus_explainer_local(iris_rf, iris):
    X, _ = iris
    explainer = ConsensusExplainer(
        iris_rf,
        X,
        methods=("coalition", "surrogate"),
        random_state=0,
        explainer_kwargs={"surrogate": {"n_samples": 1000}},
    )
    assert explainer.methods == ["coalition", "surrogate"]
    exp = explainer.explain_instance(X.iloc[0])
    assert set(exp.components) == {"coalition", "surrogate"}
    assert exp.target == "setosa"
    assert all(c.target == "setosa" for c in exp.components.values())
    assert exp.top(1)[0][0].startswith("petal")
    assert exp.agreement > 0.5
    df = exp.to_dataframe()
    assert {"consensus", "coalition", "surrogate"} <= set(df.columns)


def test_consensus_explainer_global_skips_permutation_without_y(iris_rf, iris):
    X, y = iris
    explainer = ConsensusExplainer(
        iris_rf,
        X,
        methods=("coalition", "permutation"),
        random_state=0,
        explainer_kwargs={"coalition": {"n_global": 30}, "permutation": {"n_repeats": 1}},
    )
    with pytest.warns(UserWarning, match="permutation"):
        exp = explainer.explain_global()
    assert list(exp.components) == ["coalition"]
    both = explainer.explain_global(X, y)
    assert set(both.components) == {"coalition", "permutation"}
    assert both.scope == "global"


def test_accepts_explainer_instances_and_classes(linreg, regression):
    X, _ = regression
    surrogate = LocalSurrogateExplainer(linreg, X, random_state=0, n_samples=300)
    explainer = ConsensusExplainer(
        linreg, X, methods=(surrogate, LocalSurrogateExplainer), random_state=0
    )
    # same name twice collapses to one entry
    assert explainer.methods == ["surrogate"]
    exp = explainer.explain_instance(X[0])
    assert np.isnan(exp.agreement)


def test_nested_consensus_rejected(linreg, regression):
    X, _ = regression
    with pytest.raises(ValueError):
        ConsensusExplainer(linreg, X, methods=("consensus",))
    with pytest.raises(ValueError):
        ConsensusExplainer(linreg, X, methods=())
