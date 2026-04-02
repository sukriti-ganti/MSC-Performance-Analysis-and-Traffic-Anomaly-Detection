"""
dashboard.py
────────────
9-panel dataset visualisation.
Covers NPS, EMTL, Wireless, and class distribution.
Run independently — does not need trained models.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Load and encode (same as preprocess.py) ────────────────────────────────────
df = pd.read_csv("PBL_Dataset_260.csv")

df["traffic_intensity"] = df["traffic_intensity"].map(
    {"Low": 0, "Medium": 1, "High": 2, "Very High": 3})
df["throughput"] = df["throughput"].map(
    {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3})
df["modulation"] = df["modulation"].map(
    {"BPSK": 0, "QPSK": 1, "16-QAM": 2})
for col in ["traffic_type", "access_technology"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df["ber"] = df["ber"].astype(float)

X      = df.drop(columns=["scenario_id", "label"]).values.astype(float)
labels = df["label"].map({"Normal": 0, "Abnormal": 1, "Drone": 2}).values

C = {0: "#2ecc71", 1: "#e74c3c",  2: "#3498db"}
N = {0: "Normal",  1: "Abnormal", 2: "Drone"}

fig = plt.figure(figsize=(20, 14))
fig.suptitle("MSC Anomaly Detection — Full Dataset Dashboard\n"
             "NPS + EMTL + Wireless  |  260 Samples  |  3 Classes",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: PCA scatter ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
Xp = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
for l, n in N.items():
    m = labels == l
    ax1.scatter(Xp[m, 0], Xp[m, 1], c=C[l], s=20, alpha=0.75,
                label=n, edgecolors="white", linewidths=0.3)
ax1.set_title("PCA — Feature Space (2D)")
ax1.legend(markerscale=1.5)
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")

# ── Panel 2: Avg delay histogram ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for l, n in N.items():
    ax2.hist(df[labels == l]["avg_delay"].astype(float), bins=22,
             alpha=0.70, color=C[l], label=n, density=True)
ax2.axvline(0.7, color="black", linestyle="--", linewidth=1.5, label="0.7s threshold")
ax2.set_title("NPS: Avg Delay Distribution")
ax2.set_xlabel("Seconds"); ax2.legend(fontsize=8)

# ── Panel 3: RSSI vs SNR scatter ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for l, n in N.items():
    m = labels == l
    ax3.scatter(df[m]["rssi_dbm"].astype(float),
                df[m]["snr_db"].astype(float),
                c=C[l], s=20, alpha=0.72, label=n,
                edgecolors="white", linewidths=0.3)
ax3.set_title("Wireless: RSSI vs SNR")
ax3.set_xlabel("RSSI (dBm)"); ax3.set_ylabel("SNR (dB)")
ax3.legend(fontsize=8)

# ── Panel 4: Doppler shift distribution ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for l, n in N.items():
    ax4.hist(df[labels == l]["doppler_shift_hz"].astype(float), bins=25,
             alpha=0.70, color=C[l], label=n, density=True)
ax4.set_title("EMTL: Doppler Shift Distribution\n(Drone = high Doppler)")
ax4.set_xlabel("Doppler Shift (Hz)"); ax4.legend(fontsize=8)

# ── Panel 5: Packet loss boxplot ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
bp = ax5.boxplot(
    [df[labels == l]["packet_loss"].astype(float) for l in [0, 1, 2]],
    labels=list(N.values()), patch_artist=True,
    widths=0.5, medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], C.values()):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax5.set_title("NPS: Packet Loss by Class")
ax5.set_ylabel("Packet Loss (fraction)")

# ── Panel 6: Velocity distribution ────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
for l, n in N.items():
    ax6.hist(df[labels == l]["velocity_mps"].astype(float), bins=22,
             alpha=0.70, color=C[l], label=n, density=True)
ax6.set_title("EMTL: Velocity Distribution\n(Drone = 5–30 m/s)")
ax6.set_xlabel("Velocity (m/s)"); ax6.legend(fontsize=8)

# ── Panel 7: Handovers boxplot ────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
bp2 = ax7.boxplot(
    [df[labels == l]["handovers"].astype(float) for l in [0, 1, 2]],
    labels=list(N.values()), patch_artist=True,
    widths=0.5, medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp2["boxes"], C.values()):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax7.set_title("Wireless: Handovers by Class\n(Drone = most handovers)")
ax7.set_ylabel("Handover Count")

# ── Panel 8: Interference vs Fading depth ─────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
for l, n in N.items():
    m = labels == l
    ax8.scatter(df[m]["interference_dbm"].astype(float),
                df[m]["fading_depth_db"].astype(float),
                c=C[l], s=20, alpha=0.72, label=n,
                edgecolors="white", linewidths=0.3)
ax8.set_title("EMTL: Interference vs Fading Depth")
ax8.set_xlabel("Interference (dBm)")
ax8.set_ylabel("Fading Depth (dB)")
ax8.legend(fontsize=8)

# ── Panel 9: Class distribution pie ───────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
counts = [sum(labels == l) for l in [0, 1, 2]]
ax9.pie(counts,
        labels=["Normal (70)", "Abnormal (130)", "Drone (60)"],
        colors=list(C.values()), autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax9.set_title("Class Distribution")

plt.savefig("results/dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("[dashboard] Saved → results/dashboard.png")