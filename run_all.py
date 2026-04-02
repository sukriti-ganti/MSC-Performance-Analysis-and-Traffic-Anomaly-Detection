# run_all.py
# Runs the entire pipeline in order
# Just run: python run_all.py

import subprocess
import sys

scripts = [
    ("Preprocessing check",  "preprocess.py"),
    ("Train Random Forest",  "train_rf.py"),
    ("Train SVM",            "train_svm.py"),
    ("Train KNN",            "train_knn.py"),
    ("Evaluate all models",  "evaluate.py"),
    ("Compare models",       "compare.py"),
    ("Generate dashboard",   "dashboard.py"),
]

for label, script in scripts:
    print(f"\n{'─'*50}")
    print(f"  ▶  {label}")
    print(f"{'─'*50}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"❌ {script} failed. Fix the error and re-run.")
        break
    print(f"✅ {script} done.")

print("\n\n🏁 Pipeline complete. Check /results/ folder.")
