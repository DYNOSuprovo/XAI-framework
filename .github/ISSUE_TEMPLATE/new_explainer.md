---
name: New explainer
about: Propose adding an explanation method
labels: enhancement, explainer
---

**Method**
Name and reference (paper / library):

**Scope**
- [ ] local (per instance)
- [ ] global (per dataset)
- needs ground-truth `y`? yes / no

**Data types**
tabular / text / image / other

**Dependencies**
Optional dependency needed? Which, and is it maintained?

**Output**
What does one attribution per feature mean for this method? Is it additive
(`base_value + sum(values) == prediction`)? If not, how should it be normalised
for the consensus?

**Plan**
- [ ] `src/xai_framework/explainers/<name>.py` subclassing `BaseExplainer`
- [ ] registered with `@register_explainer("<name>")`
- [ ] `tests/test_<name>.py`
- [ ] README method table + CHANGELOG entry
