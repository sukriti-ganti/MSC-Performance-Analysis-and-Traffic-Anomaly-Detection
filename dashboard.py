# dashboard.py
# 6-panel dataset visualization
# Run independently, doesn't need trained models

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("PBL_Combined_Dataset.csv")

# Same encoding as preprocess.py
df["traffic_intensity"] = df["traffic_intensity"].map(
    {"Low":0,"Medium":1,"High":2,"Very High":3})
df["throughput"] = df["throughput"].map(
    {"Very Low":0,"Low":1,"Medium":2,"High":3})
df["modulation"] = df["modulation"].map({"BPSK":0,"QPSK":1,"16-QAM":2})
for col in ["traffic_type","access_technology"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df["ber"] = df["ber"].astype(float)

X      = df.drop(columns=["scenario_id","label"]).values.astype(float)
labels = (df["label"] == "Abnormal").astype(int).values
C      = {0:"#2ecc71", 1:"#e74c3c"}
N      = {0:"Normal",  1:"Abnormal"}

fig = plt.figure(figsize=(18, 11))
fig.suptitle("MSC Anomaly Detection — Combined Dataset Dashboard",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

# 1 — PCA
ax1 = fig.add_subplot(gs[0,0])
Xp = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
for l,n in N.items():
    m = labels==l
    ax1.scatter(Xp[m,0],Xp[m,1],c=C[l],s=20,alpha=0.75,
                label=n,edgecolors="white",linewidths=0.3)
ax1.set_title("PCA — Feature Space"); ax1.legend()
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")

# 2 — Delay histogram
ax2 = fig.add_subplot(gs[0,1])
for l,n in N.items():
    ax2.hist(df[labels==l]["avg_delay"],bins=22,alpha=0.72,
             color=C[l],label=n,density=True)
ax2.axvline(0.7,color="black",linestyle="--",linewidth=1.5,label="0.7s threshold")
ax2.set_title("NPS: Avg Delay Distribution")
ax2.set_xlabel("Seconds"); ax2.legend()

# 3 — RSSI vs SNR
ax3 = fig.add_subplot(gs[0,2])
for l,n in N.items():
    m = labels==l
    ax3.scatter(df[m]["rssi_dbm"].astype(float),
                df[m]["snr_db"].astype(float),
                c=C[l],s=20,alpha=0.72,label=n,
                edgecolors="white",linewidths=0.3)
ax3.set_title("Wireless: RSSI vs SNR")
ax3.set_xlabel("RSSI (dBm)"); ax3.set_ylabel("SNR (dB)"); ax3.legend()

# 4 — Packet loss boxplot
ax4 = fig.add_subplot(gs[1,0])
bp = ax4.boxplot(
    [df[labels==0]["packet_loss"].astype(float),
     df[labels==1]["packet_loss"].astype(float)],
    labels=["Normal","Abnormal"],patch_artist=True,
    widths=0.5,medianprops=dict(color="black",linewidth=2))
bp["boxes"][0].set_facecolor("#a8e6cf")
bp["boxes"][1].set_facecolor("#ffb3b3")
ax4.set_title("NPS: Packet Loss Distribution")
ax4.set_ylabel("Packet Loss")

# 5 — Interference vs Fading Depth
ax5 = fig.add_subplot(gs[1,1])
for l,n in N.items():
    m = labels==l
    ax5.scatter(df[m]["interference_dbm"].astype(float),
                df[m]["fading_depth_db"].astype(float),
                c=C[l],s=20,alpha=0.72,label=n,
                edgecolors="white",linewidths=0.3)
ax5.set_title("EMTL: Interference vs Fading Depth")
ax5.set_xlabel("Interference (dBm)")
ax5.set_ylabel("Fading Depth (dB)"); ax5.legend()

# 6 — Class pie
ax6 = fig.add_subplot(gs[1,2])
ax6.pie([70,130],labels=["Normal (70)","Abnormal (130)"],
        colors=["#2ecc71","#e74c3c"],autopct="%1.0f%%",
        startangle=90,wedgeprops={"edgecolor":"white","linewidth":2})
ax6.set_title("Class Distribution")

plt.savefig("results/dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → results/dashboard.png")