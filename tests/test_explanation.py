import json

import numpy as np
import pytest

from xai_framework import ConsensusExplanation, Explanation
from xai_framework.explanation import normalize_attributions, rank_by_magnitude


def make(values, **kw):
    names = [f"f{i}" for i in range(len(values))]
    return Explanation(feature_names=names, values=np.array(values, dtype=float), method="t", **kw)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        Explanation(feature_names=["a"], values=np.array([1.0, 2.0]), method="t")


def test_ranks_and_top():
    exp = make([0.1, -0.9, 0.5])
    np.testing.assert_array_equal(exp.ranks(), [3, 1, 2])
    assert exp.top(2) == [("f1", -0.9), ("f2", 0.5)]


def test_normalized_unit_l1():
    exp = make([2.0, -2.0, 4.0])
    np.testing.assert_allclose(np.abs(exp.normalized()).sum(), 1.0)
    np.testing.assert_allclose(normalize_attributions(np.zeros(3)), np.zeros(3))


def test_rank_by_magnitude_is_stable_for_ties():
    np.testing.assert_array_equal(rank_by_magnitude(np.array([1.0, 1.0, 2.0])), [2, 3, 1])


def test_dataframe_sorted_by_rank_with_feature_values():
    exp = make([0.1, -0.9, 0.5], feature_values=np.array([1, 2, 3]))
    df = exp.to_dataframe()
    assert list(df["feature"]) == ["f1", "f2", "f0"]
    assert list(df.columns) == ["feature", "feature_value", "value", "abs_value", "rank"]


def test_to_dict_is_json_serialisable():
    exp = make(
        [0.1, -0.9],
        feature_values=np.array([1.5, 2.5]),
        target=np.int64(1),
        prediction=np.float32(0.7),
        metadata={"arr": np.arange(2), "scalar": np.float64(1.0)},
    )
    payload = json.dumps(exp.to_dict())
    assert '"target": 1' in payload


def test_text_mentions_direction_and_values():
    exp = make([0.3, -0.2, 0.0], feature_values=np.array([1, 2, 3]), target="yes", prediction=0.8)
    text = exp.to_text()
    assert "Predicted class 'yes' with probability 0.800" in text
    assert "push towards 'yes'" in text
    assert "push away from 'yes'" in text
    assert "f0 = 1" in text


def test_text_for_regression_and_global():
    local = make([1.0, -1.0], task="regression", prediction=42.0)
    assert "Predicted value 42" in local.to_text()
    assert "increase the prediction" in local.to_text()
    glob = make([1.0, 0.5], scope="global")
    assert "Global feature importance" in glob.to_text()


def test_consensus_agreement_levels():
    comps = {"a": make([1.0, 0.5]), "b": make([0.9, 0.6])}
    for rho, level in [(0.9, "high"), (0.6, "moderate"), (0.1, "low"), (float("nan"), "n/a")]:
        exp = ConsensusExplanation(
            feature_names=["f0", "f1"],
            values=np.array([1.0, 0.5]),
            method="consensus",
            components=comps,
            agreement=rho,
        )
        assert exp.agreement_level() == level
        assert level in exp.to_text() or level == "n/a"


def test_consensus_dataframe_has_component_columns():
    comps = {"a": make([1.0, 0.5]), "b": make([0.9, 0.6])}
    exp = ConsensusExplanation(
        feature_names=["f0", "f1"],
        values=np.array([1.0, 0.5]),
        method="consensus",
        components=comps,
    )
    df = exp.to_dataframe()
    assert {"consensus", "a", "b"} <= set(df.columns)
    d = exp.to_dict()
    assert set(d["components"]) == {"a", "b"}
    assert d["agreement"] is None
