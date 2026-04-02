"""
train_rf.py
───────────
Trains a Random Forest classifier on the 260-sample dataset.
Saves trained model + test split to results/model_rf.pkl
"""

import os
import joblib
import numpy as np
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

from preprocess import load_data

os.makedirs("results", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
X, y, feature_names = load_data()

# 80% train, 20% test — stratify keeps class ratios the same in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[RF] Train: {len(y_train)}  Test: {len(y_test)}")

# ── Build pipeline ─────────────────────────────────────────────────────────────
# StandardScaler normalises all features to mean=0, std=1
# RandomForest builds 200 decision trees, each votes, majority wins
rf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,        # number of trees
        class_weight="balanced", # handles 70/130/60 imbalance
        random_state=42,
        n_jobs=-1                # use all CPU cores
    ))
])

# ── 5-fold cross validation on training set ────────────────────────────────────
cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1_macro")
print(f"[RF] CV F1 (macro): {cv.mean():.4f} ± {cv.std():.4f}")

# ── Train on full training set ─────────────────────────────────────────────────
rf.fit(X_train, y_train)

# ── Save model + test data together ────────────────────────────────────────────
joblib.dump({
    "pipeline":      rf,
    "X_test":        X_test,
    "y_test":        y_test,
    "feature_names": feature_names,
    "cv_f1_mean":    cv.mean(),
    "cv_f1_std":     cv.std(),
}, "results/model_rf.pkl")

print("[RF] Saved → results/model_rf.pkl")