# compare.py
# Bar chart comparing all 3 models side by side
# Run after evaluate.py

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/evaluation_summary.csv")

metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
models    = df["Model"].tolist()
x         = np.arange(len(metrics))
width     = 0.25
colors    = ["#2E5496", "#C0392B", "#27AE60"]

fig, ax = plt.subplots(figsize=(10, 6))

for i, (model, color) in enumerate(zip(models, colors)):
    vals = df[df["Model"] == model][metrics].values.flatten()
    bars = ax.bar(x + i*width, vals, width,
                  label=model, color=color,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — Random Forest vs SVM vs KNN",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.close()
print("✅ Comparison chart saved → results/model_comparison.png")