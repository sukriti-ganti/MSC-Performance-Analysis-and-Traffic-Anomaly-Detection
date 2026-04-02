"""
train_knn.py
────────────
Trains a K-Nearest Neighbors classifier on the 260-sample dataset.
Uses GridSearchCV to automatically find the best K.
Saves trained model + test split to results/model_knn.pkl
"""

import os
import joblib
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from preprocess import load_data

os.makedirs("results", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
X, y, feature_names = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[KNN] Train: {len(y_train)}  Test: {len(y_test)}")

# ── Find best K using GridSearch ───────────────────────────────────────────────
# Tries K = 3, 5, 7, 9, 11 and picks whichever gives best F1 macro
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

param_grid = {"clf__n_neighbors": [3, 5, 7, 9, 11]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)

best_k = grid.best_params_["clf__n_neighbors"]
print(f"[KNN] Best K: {best_k}  |  Grid CV F1: {grid.best_score_:.4f}")

# ── Train final model with best K ─────────────────────────────────────────────
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=best_k))
])

cv = cross_val_score(knn, X_train, y_train, cv=5, scoring="f1_macro")
print(f"[KNN] CV F1 (macro): {cv.mean():.4f} ± {cv.std():.4f}")

knn.fit(X_train, y_train)

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump({
    "pipeline":      knn,
    "X_test":        X_test,
    "y_test":        y_test,
    "feature_names": feature_names,
    "cv_f1_mean":    cv.mean(),
    "cv_f1_std":     cv.std(),
    "best_k":        best_k,
}, "results/model_knn.pkl")

print(f"[KNN] Saved → results/model_knn.pkl  (K={best_k})")