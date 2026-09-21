# Contributing to XAI-Framework

Thanks for your interest! This project is small and friendly; the fastest way to help
is to pick an issue labelled `good first issue` or `help wanted` and open a PR.

## Development setup

```bash
git clone https://github.com/<your-fork>/XAI-Framework.git
cd XAI-Framework
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                           # ~2 s
ruff check src tests && ruff format --check src tests
```

Python 3.10+ is required. The test suite uses scikit-learn toy datasets only, so it
must stay fast (< 10 s). If your change needs a slow test, mark it with
`@pytest.mark.slow` and keep it off by default.

## Project layout

```
src/xai_framework/
  auto.py              explain() / explain_global() - the one-call entry point
  model.py             ModelAdapter: uniform predict() over sklearn / boosters / callables
  explanation.py       Explanation + ConsensusExplanation data classes
  registry.py          name -> explainer class, plus entry-point plugin loading
  plotting.py          matplotlib rendering (optional dependency)
  explainers/
    base.py            BaseExplainer - subclass this
    coalition.py       Shapley values from feature coalitions (exact / sampled / linear)
    surrogate.py       local weighted linear surrogate
    permutation.py     permutation importance
    consensus.py       combine_explanations() + ConsensusExplainer
tests/                 pytest suite, one file per module
examples/              runnable scripts
docs/issues/           drafts of the contributor issues (mirrored on GitHub)
```

## Adding an explainer

1. Create `src/xai_framework/explainers/<name>.py`.
2. Subclass `BaseExplainer`. Set `name`, `supports_local`, `supports_global`, and
   `requires_y` if the global method needs labels. Implement `_explain_instance`
   (receives a 1-D float array and a resolved class index) and/or `_explain_global`.
   Build the result with `self._make_explanation(...)` so metadata and naming stay
   consistent.
3. Decorate the class with `@register_explainer("<name>")` and import it in
   `explainers/__init__.py`.
4. If it needs an optional dependency, override `is_available()` and import the
   dependency lazily inside `__init__` (see how `plotting.py` guards matplotlib).
5. Add `tests/test_<name>.py`. At minimum: it runs on a classifier and a regressor,
   feature names and target propagate, and any invariant the method promises
   (e.g. additivity) holds.
6. Add a short section to the README's method table.

External packages can register explainers without touching this repo through the
`xai_framework.explainers` entry-point group:

```toml
[project.entry-points."xai_framework.explainers"]
anchors = "my_package.anchors:AnchorsExplainer"
```

## Conventions

* **Style**: `ruff` (config in `pyproject.toml`), line length 100, type hints on
  public functions, NumPy-style docstrings.
* **Determinism**: every stochastic step goes through `self.rng`
  (`np.random.default_rng(random_state)`); never call `np.random.*` globally.
* **Inputs**: accept arrays and DataFrames; convert with `xai_framework.model.to_numpy`.
  Talk to the model only through `self.adapter`.
* **Outputs**: always an `Explanation`. Attributions are one number per feature in
  model input order; positive means "pushes the prediction / target probability up".
* **Dependencies**: core deps are `numpy` and `pandas` only. The explainers are native
  implementations - we do not wrap third-party explanation packages. Anything else
  must be optional and guarded.
* **Commits**: small, focused, imperative subject line ("Add Anchors explainer").

## Pull requests

* Open an issue first for anything non-trivial so we can agree on the approach.
* One PR per change. Include tests and update docs/CHANGELOG.
* CI must pass (tests on 3.10-3.14, ruff).
* Fill in the PR template; a maintainer will review within a few days.

## Reporting bugs

Use the bug report template. Include the model type, `xai_framework.__version__`,
a minimal reproducer, and the full traceback.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind.
