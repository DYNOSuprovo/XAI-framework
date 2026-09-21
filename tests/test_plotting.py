import numpy as np
import pytest

from xai_framework import Explanation
from xai_framework.explainers.consensus import combine_explanations

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def make(method, values, **kw):
    names = [f"f{i}" for i in range(len(values))]
    return Explanation(
        feature_names=names, values=np.array(values, dtype=float), method=method, **kw
    )


def test_plot_single_explanation_returns_axes():
    exp = make(
        "surrogate", [0.5, -0.3, 0.1], feature_values=np.array([1.0, 2.0, 3.0]), target="yes"
    )
    ax = exp.plot(k=2)
    assert len(ax.patches) == 2
    assert "yes" in ax.get_title(loc="left")
    matplotlib.pyplot.close(ax.figure)


def test_plot_consensus_overlays_markers_and_legend():
    exp = combine_explanations({"a": make("a", [1.0, 0.5]), "b": make("b", [0.6, 0.9])})
    ax = exp.plot()
    assert ax.get_legend() is not None
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"a", "b"}
    matplotlib.pyplot.close(ax.figure)


def test_plot_global():
    exp = make("coalition", [0.2, 0.7], scope="global")
    ax = exp.plot(title="custom")
    assert ax.get_title(loc="left") == "custom"
    assert "importance" in ax.get_xlabel()
    matplotlib.pyplot.close(ax.figure)
