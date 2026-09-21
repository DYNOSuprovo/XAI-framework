# Roadmap

The MVP (v0.1) ships native tabular explainers - coalition (Shapley values), local
surrogate and permutation importance - plus the consensus layer. Everything below is
open for contribution - each item has a fully written issue draft in [`docs/issues/`](docs/issues/) that is mirrored on the
GitHub issue tracker (`python scripts/create_issues.py --repo <owner>/XAI-Framework`).

## Good first issues

| # | Issue | Area |
|---|---|---|
| 01 | [`Explanation.to_markdown()`](docs/issues/01-to-markdown.md) | output |
| 02 | [Batch explanations + `summarize()`](docs/issues/02-batch-explain-summary.md) | API |
| 03 | [XGBoost / LightGBM / CatBoost integration tests](docs/issues/03-xgboost-lightgbm-catboost-tests.md) | testing |
| 04 | [Tutorial notebook](docs/issues/04-tutorial-notebook.md) | docs |

## New explainers

| # | Issue | Scope |
|---|---|---|
| 05 | [Anchors (rule-based)](docs/issues/05-anchors-explainer.md) | local |
| 06 | [Counterfactuals](docs/issues/06-counterfactual-explainer.md) | local |
| 07 | [Partial dependence & ALE](docs/issues/07-pdp-ale-global.md) | global |
| 10 | [Surrogate discretisation + reference parity tests](docs/issues/10-surrogate-discretization-and-reference-parity.md) | local |
| 21 | [Fast exact Shapley values for tree ensembles](docs/issues/21-tree-shapley-fast-path.md) | local + global |
| 11 | [Text support](docs/issues/11-text-support.md) | text |
| 12 | [Image support](docs/issues/12-image-support.md) | image |
| 20 | [Integrated Gradients for PyTorch / TensorFlow](docs/issues/20-deep-learning-explainers.md) | deep learning |

## Making explanations more trustworthy

| # | Issue |
|---|---|
| 08 | [Faithfulness & stability metrics; metric-weighted consensus](docs/issues/08-faithfulness-metrics.md) |
| 18 | [Disagreement diagnostics](docs/issues/18-disagreement-diagnostics.md) |
| 09 | [Pipelines in the original feature space](docs/issues/09-pipeline-feature-mapping.md) |
| 19 | [Conformance suite for third-party explainers](docs/issues/19-explainer-conformance-suite.md) |

## Tooling, docs and infrastructure

| # | Issue |
|---|---|
| 13 | [CLI](docs/issues/13-cli.md) |
| 14 | [Standalone HTML report](docs/issues/14-html-report.md) |
| 15 | [Performance: caching, parallelism, benchmarks](docs/issues/15-performance.md) |
| 16 | [Documentation site](docs/issues/16-docs-site.md) |
| 17 | [PyPI release workflow](docs/issues/17-release-workflow.md) |

## Milestones (proposed)

- **v0.2** - good first issues, fast tree path, Anchors, counterfactuals, PDP/ALE, CLI, release workflow.
- **v0.3** - faithfulness metrics and diagnostics, pipeline mapping, docs site.
- **v0.4** - text and image support, deep-learning explainers.

Have an idea that is not here? Open a feature request - the templates are in
`.github/ISSUE_TEMPLATE/`.
