# train_svm.py
# Trains SVM (RBF kernel) classifier
# Saves model to results/model_svm.pkl

import joblib
import os
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

from preprocess import load_data

os.makedirs("results", exist_ok=True)

X, y, feature_names = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Build pipeline ─────────────────────────────────
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",     # RBF = radial basis function
        C=10,             # penalty for misclassification
        gamma="scale",    # auto-scales to 1/(n_features * variance)
        class_weight="balanced",
        probability=True, # needed to get probability scores
        random_state=42
    ))
])

# ── 5-fold cross validation ────────────────────────
cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring="f1")
print(f"[SVM] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Train on full training set ─────────────────────
svm.fit(X_train, y_train)

# ── Save ───────────────────────────────────────────
joblib.dump({
    "pipeline":      svm,
    "X_test":        X_test,
    "y_test":        y_test,
    "feature_names": feature_names,
    "cv_f1_mean":    cv_scores.mean(),
    "cv_f1_std":     cv_scores.std(),
}, "results/model_svm.pkl")

print("[SVM] Saved → results/model_svm.pkl")