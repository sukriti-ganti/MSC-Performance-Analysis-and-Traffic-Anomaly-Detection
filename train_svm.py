"""
train_svm.py
────────────
Trains an SVM (RBF kernel) classifier on the 260-sample dataset.
Saves trained model + test split to results/model_svm.pkl
"""

import os
import joblib
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

from preprocess import load_data

os.makedirs("results", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
X, y, feature_names = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[SVM] Train: {len(y_train)}  Test: {len(y_test)}")

# ── Build pipeline ─────────────────────────────────────────────────────────────
# kernel="rbf" allows curved decision boundary in 22-dimensional space
# C=10        controls penalty for misclassification
# gamma=scale auto-scales to 1/(n_features * variance)
# probability=True lets SVM output confidence scores
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
        decision_function_shape="ovr"  # one-vs-rest for 3 classes
    ))
])

# ── 5-fold cross validation ────────────────────────────────────────────────────
cv = cross_val_score(svm, X_train, y_train, cv=5, scoring="f1_macro")
print(f"[SVM] CV F1 (macro): {cv.mean():.4f} ± {cv.std():.4f}")

# ── Train ──────────────────────────────────────────────────────────────────────
svm.fit(X_train, y_train)

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump({
    "pipeline":      svm,
    "X_test":        X_test,
    "y_test":        y_test,
    "feature_names": feature_names,
    "cv_f1_mean":    cv.mean(),
    "cv_f1_std":     cv.std(),
}, "results/model_svm.pkl")

print("[SVM] Saved → results/model_svm.pkl")