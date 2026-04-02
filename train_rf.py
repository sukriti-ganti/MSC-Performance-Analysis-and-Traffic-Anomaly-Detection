# train_rf.py
# Trains Random Forest classifier
# Saves model to results/model_rf.pkl

import joblib
import os
import numpy as np
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

from preprocess import load_data

os.makedirs("results", exist_ok=True)

X, y, feature_names = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Build pipeline ─────────────────────────────────
rf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# ── 5-fold cross validation ────────────────────────
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
print(f"[RF] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Train on full training set ─────────────────────
rf.fit(X_train, y_train)

# ── Save model + test data ─────────────────────────
joblib.dump({
    "pipeline":      rf,
    "X_test":        X_test,
    "y_test":        y_test,
    "feature_names": feature_names,
    "cv_f1_mean":    cv_scores.mean(),
    "cv_f1_std":     cv_scores.std(),
}, "results/model_rf.pkl")

print("[RF] Saved → results/model_rf.pkl")