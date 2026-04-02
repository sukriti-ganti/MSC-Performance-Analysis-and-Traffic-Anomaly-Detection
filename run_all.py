"""
run_all.py
──────────
Runs the entire ML pipeline in the correct order.
Just run: python run_all.py
"""

import subprocess
import sys

steps = [
    ("Preprocessing check",  "preprocess.py"),
    ("Train Random Forest",  "train_rf.py"),
    ("Train SVM",            "train_svm.py"),
    ("Train KNN",            "train_knn.py"),
    ("Evaluate all models",  "evaluate.py"),
    ("Compare models",       "compare.py"),
    ("Generate dashboard",   "dashboard.py"),
]

print("\n" + "="*55)
print("  MSC Anomaly Detection — Full ML Pipeline")
print("  3 Classes: Normal | Abnormal | Drone")
print("="*55)

for label, script in steps:
    print(f"\n{'─'*55}")
    print(f"  ▶  {label}")
    print(f"{'─'*55}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n❌  {script} failed. Fix the error above and re-run.")
        sys.exit(1)
    print(f"✅  {script} complete.")

print("\n" + "="*55)
print("  PIPELINE COMPLETE")
print("  All outputs saved in /results/")
print("="*55)
print("""
  Files generated:
  ├── results/model_rf.pkl
  ├── results/model_svm.pkl
  ├── results/model_knn.pkl
  ├── results/confusion_random_forest.png
  ├── results/confusion_svm.png
  ├── results/confusion_knn.png
  ├── results/metrics_random_forest.png
  ├── results/metrics_svm.png
  ├── results/metrics_knn.png
  ├── results/feature_importance.png
  ├── results/model_comparison.png
  ├── results/dashboard.png
  └── results/evaluation_summary.csv
""")