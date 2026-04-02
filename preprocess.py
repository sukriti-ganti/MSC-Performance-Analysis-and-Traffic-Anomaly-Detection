# preprocess.py
# Loads CSV, encodes all text columns, returns X and y
# Called by all 3 training scripts — never run this alone

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(csv_path="PBL_Combined_Dataset.csv"):

    df = pd.read_csv(csv_path)

    # ── Ordinal encoding (order matters) ──────────────
    df["traffic_intensity"] = df["traffic_intensity"].map(
        {"Low": 0, "Medium": 1, "High": 2, "Very High": 3})

    df["throughput"] = df["throughput"].map(
        {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3})

    df["modulation"] = df["modulation"].map(
        {"BPSK": 0, "QPSK": 1, "16-QAM": 2})

    # ── Nominal encoding (no order) ───────────────────
    for col in ["traffic_type", "access_technology"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # ── BER: stored as "8.38e-02" string → float ─────
    df["ber"] = df["ber"].astype(float)

    # ── Split into X (features) and y (target) ────────
    X = df.drop(columns=["scenario_id", "label"]).values.astype(float)
    y = (df["label"] == "Abnormal").astype(int).values
    feature_names = list(df.drop(columns=["scenario_id", "label"]).columns)

    print(f"[preprocess] Shape: {X.shape}")
    print(f"[preprocess] Normal: {sum(y==0)}  Abnormal: {sum(y==1)}")

    return X, y, feature_names