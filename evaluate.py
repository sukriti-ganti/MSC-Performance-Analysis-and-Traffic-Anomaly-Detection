# evaluate.py
# Loads all 3 trained models and evaluates them
# Generates confusion matrix and per-class metrics for each

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

CLASS_NAMES = ["Normal", "Abnormal"]
MODELS = {
    "Random Forest": "results/model_rf.pkl",
    "SVM":           "results/model_svm.pkl",
    "KNN":           "results/model_knn.pkl",
}

summary = []

for model_name, pkl_path in MODELS.items():

    # ── Load model + test data ─────────────────────
    obj      = joblib.load(pkl_path)
    pipe     = obj["pipeline"]
    X_test   = obj["X_test"]
    y_test   = obj["y_test"]
    cv_f1    = obj["cv_f1_mean"]

    y_pred   = pipe.predict(X_test)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score   (y_test, y_pred, zero_division=0)
    f1   = f1_score       (y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  CV F1     : {cv_f1:.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=CLASS_NAMES))

    summary.append({
        "Model":     model_name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-Score":  round(f1,   4),
        "CV F1":     round(cv_f1,4),
    })

    # ── Confusion Matrix ───────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 14})
    ax.set_title(f"{model_name} — Confusion Matrix",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fname = model_name.replace(" ","_").lower()
    plt.savefig(f"results/confusion_{fname}.png", dpi=150)
    plt.close()
    print(f"  Saved → results/confusion_{fname}.png")

    # ── Per-class bar chart ────────────────────────
    report = classification_report(y_test, y_pred,
                                   target_names=CLASS_NAMES,
                                   output_dict=True)
    metrics_df = pd.DataFrame({
        cls: [report[cls]["precision"],
              report[cls]["recall"],
              report[cls]["f1-score"]]
        for cls in CLASS_NAMES
    }, index=["Precision","Recall","F1"])

    metrics_df.T.plot(kind="bar", figsize=(6,4),
                      ylim=(0,1.1), colormap="Set2",
                      title=f"{model_name} — Per-class Metrics",
                      edgecolor="white")
    plt.ylabel("Score"); plt.xticks(rotation=0)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"results/metrics_{fname}.png", dpi=150)
    plt.close()

# ── Feature Importance (RF only) ──────────────────
rf_obj   = joblib.load("results/model_rf.pkl")
rf_pipe  = rf_obj["pipeline"]
feat     = rf_obj["feature_names"]
imp      = rf_pipe.named_steps["clf"].feature_importances_
fi       = pd.Series(imp, index=feat).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
fi.head(15).plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.invert_yaxis()
ax.set_title("Random Forest — Feature Importances (Top 15)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
for i, v in enumerate(fi.head(15)):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
plt.close()
print("\n[evaluate] Feature importance saved.")
print("\nTop 10 features:")
print(fi.head(10).to_string())

# ── Save summary CSV ───────────────────────────────
pd.DataFrame(summary).to_csv("results/evaluation_summary.csv", index=False)
print("\n[evaluate] Summary saved → results/evaluation_summary.csv")