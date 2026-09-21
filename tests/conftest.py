"""Shared fixtures: small fitted models on toy data so the suite stays fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


@pytest.fixture(scope="session")
def iris():
    data = load_iris(as_frame=True)
    X = data.data
    y = pd.Series(data.target_names[data.target], name="species")
    return X, y


@pytest.fixture(scope="session")
def iris_rf(iris):
    X, y = iris
    return RandomForestClassifier(n_estimators=30, random_state=SEED).fit(X, y)


@pytest.fixture(scope="session")
def binary():
    """Binary problem with a known structure: only x0 and x1 matter, x2 is noise."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(400, 3))
    logits = 3.0 * X[:, 0] - 2.0 * X[:, 1]
    y = (logits + rng.normal(scale=0.3, size=400) > 0).astype(int)
    return X, y


@pytest.fixture(scope="session")
def binary_gb(binary):
    X, y = binary
    return GradientBoostingClassifier(n_estimators=40, random_state=SEED).fit(X, y)


@pytest.fixture(scope="session")
def binary_pipeline(binary):
    X, y = binary
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)


@pytest.fixture(scope="session")
def regression():
    """Linear regression data: y = 3*x0 - 2*x1 + 0*x2 + noise."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 3))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.1, size=300)
    return X, y


@pytest.fixture(scope="session")
def linreg(regression):
    X, y = regression
    return LinearRegression().fit(X, y)
