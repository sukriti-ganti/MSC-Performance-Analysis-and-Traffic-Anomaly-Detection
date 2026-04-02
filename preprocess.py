"""
preprocess.py
─────────────
Loads PBL_Dataset_260.csv, encodes all categorical columns,
returns X, y, and feature names.

Called by all training scripts. Do not run directly.
Labels: Normal=0, Abnormal=1, Drone=2
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(csv_path="PBL_Dataset_260.csv"):

    df = pd.read_csv(csv_path)

    # ── Ordinal encoding (order matters) ──────────────────────────────────────
    df["traffic_intensity"] = df["traffic_intensity"].map(
        {"Low": 0, "Medium": 1, "High": 2, "Very High": 3})

    df["throughput"] = df["throughput"].map(
        {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3})

    df["modulation"] = df["modulation"].map(
        {"BPSK": 0, "QPSK": 1, "16-QAM": 2})

    # ── Nominal encoding (no order, just convert text to numbers) ─────────────
    for col in ["traffic_type", "access_technology"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # ── BER stored as "8.38e-02" string → actual float ────────────────────────
    df["ber"] = df["ber"].astype(float)

    # ── Target labels: Normal=0, Abnormal=1, Drone=2 ─────────────────────────
    label_map = {"Normal": 0, "Abnormal": 1, "Drone": 2}
    y = df["label"].map(label_map).values

    # ── Features: drop id and label columns ───────────────────────────────────
    X = df.drop(columns=["scenario_id", "label"]).values.astype(float)
    feature_names = list(df.drop(columns=["scenario_id", "label"]).columns)

    print(f"[preprocess] Shape : {X.shape}")
    print(f"[preprocess] Normal={sum(y==0)}  Abnormal={sum(y==1)}  Drone={sum(y==2)}")

    return X, y, feature_names


if __name__ == "__main__":
    X, y, features = load_data()
    print("Features:", features)