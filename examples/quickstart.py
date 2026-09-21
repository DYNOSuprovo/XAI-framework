"""Quick start: explain a scikit-learn classifier in three lines.

Run with:  python examples/quickstart.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xai_framework import explain

# 1. Any fitted model will do - here a random forest on the breast-cancer dataset.
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=0
)
model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

# 2. Explain one prediction. "auto" runs the coalition (Shapley) and surrogate
#    explainers and combines them.
exp = explain(
    model, X_train, instance=X_test.iloc[0], class_names=["malignant", "benign"], random_state=0
)
print(exp.to_text())
print()
print(exp.to_dataframe().head(8).to_string(index=False))
print()
print("Pairwise agreement between methods:")
print(exp.agreement_matrix.round(2))

# 3. Explain the model as a whole (coalition mean |phi| + permutation importance).
#    Use held-out data here: permutation importance on training rows of a
#    fully-grown forest is ~zero for every feature.
glob = explain(model, X_test, y=y_test, class_names=["malignant", "benign"], random_state=0)
print()
print(glob.to_text())

# 4. Plot (needs matplotlib: pip install matplotlib).
try:
    import matplotlib

    matplotlib.use("Agg")
    exp.plot(k=10).figure.savefig("local_explanation.png", dpi=150)
    glob.plot(k=10).figure.savefig("global_explanation.png", dpi=150)
    print("\nSaved local_explanation.png and global_explanation.png")
except ImportError:
    print("\nInstall matplotlib to save the plots.")
