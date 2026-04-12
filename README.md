# Intelligent Detection of Abnormal Traffic in Mobile Switching Centres

Classification of MSC traffic into Normal, Abnormal, and Drone anomaly using NPS parameters as the primary layer and EMTL parameters as an extension.

---

## Overview

This project classifies traffic in Mobile Switching Centres into three categories:

- **Normal** — stable MSC operation, low interference, 1 to 5 active devices
- **Abnormal** — MSC overload caused by signalling storms, 6 to 15 active devices
- **Drone** — fast-moving wireless anomaly detected via electromagnetic parameters

NPS parameters (packet loss, average delay, throughput) handle the primary Normal vs Abnormal classification. NPS parameters alone cannot detect drones because drone scenarios produce metric values that overlap with Normal conditions. The EMTL extension adds Doppler shift, fading rate, and interference features that distinguish the Drone class.

---

## Dataset

- 260 samples collected manually from Cisco Packet Tracer
- 70 Normal / 130 Abnormal / 60 Drone
- 21 features across three layers: NPS, EMTL, and Wireless Communications
- 13 simulation scenario groups (N1 to N5, A1 to A5, D1 to D3)

### How the data was collected

Each run used `ping -n 50 <server_ip>` on each active device. Average RTT and lost packet count were recorded from the output.

```
avg_delay   = Average RTT / 2000        (ms to seconds, one-way)
packet_loss = Lost / 50
throughput  = from Router CLI: show interfaces fa0/0 -> Output rate
```

EMTL and Wireless features were derived mathematically from the NPS values using Doppler, FSPL, and BER formulas.

### Feature layers

| Layer | Features | Count |
|---|---|---|
| NPS | active_pcs, traffic_type, traffic_intensity, avg_delay, packet_loss, throughput | 6 |
| EMTL | velocity_mps, doppler_shift_hz, path_loss_db, fading_depth_db, interference_dbm, fading_rate, freq_variation_hz | 7 |
| Wireless | rssi_dbm, snr_db, ber, channel_variation, handovers, modulation, access_technology | 7 |
| Target | label (Normal / Abnormal / Drone) | 1 |

---

## Results

| Model | Accuracy | F1 (macro) | False Negatives |
|---|---|---|---|
| Random Forest | 100% | 1.0000 | 0 |
| SVM (RBF) | 100% | 1.0000 | 0 |
| KNN (K=5) | 98.08% | 0.9814 | 0 |

Top features by importance (Random Forest):

1. packet_loss — 0.1713 — NPS
2. active_pcs — 0.1404 — NPS
3. fading_rate — 0.1108 — EMTL
4. velocity_mps — 0.1046 — EMTL
5. channel_variation — 0.1012 — Wireless
6. avg_delay — 0.0742 — NPS

NPS features dominate MSC overload detection. EMTL and Wireless features dominate drone detection.

---

## Project Structure

```
.
├── PBL_Dataset_260.csv         # Full combined dataset
├── preprocess.py               # Load and encode all features
├── train_rf.py                 # Train Random Forest
├── train_svm.py                # Train SVM
├── train_knn.py                # Train KNN
├── evaluate.py                 # Confusion matrix, metrics, feature importance
├── compare.py                  # Side-by-side model comparison chart
├── dashboard.py                # 9-panel dataset visualisation
├── run_all.py                  # Run full pipeline in one command
└── results/
    ├── model_rf.pkl
    ├── model_svm.pkl
    ├── model_knn.pkl
    ├── confusion_random_forest.png
    ├── confusion_svm.png
    ├── confusion_knn.png
    ├── feature_importance.png
    ├── model_comparison.png
    ├── dashboard.png
    └── evaluation_summary.csv
```

---

## Run

Install dependencies:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

Run the full pipeline:

```bash
python run_all.py
```

Or run each step individually:

```bash
python preprocess.py    # verify encoding works
python train_rf.py
python train_svm.py
python train_knn.py
python evaluate.py
python compare.py
python dashboard.py
```

---

## Cisco Packet Tracer Topology

```
PC0 --+
PC1 --+
PC2 --+-- Switch0 -- Router0 (MSC) -- Switch1 -- Server0
PC3 --+                                        +-- Server1
PC4 --+
AP0 --- Laptop0, Smartphone0
AP1 --- Laptop1, Smartphone1
```

Router0 simulates the MSC. All traffic passes through it. Increasing active device count raises the M/M/1 traffic intensity rho = lambda / mu.

---

## Theory

**M/M/1 Queue (NPS)**
```
Wq = rho / (mu * (1 - rho))
```
When rho approaches 1.0, queuing delay grows without bound. This is why avg_delay jumps from under 0.5s at Normal load to over 3s at Abnormal load.

**Doppler Shift (EMTL)**
```
fd = v * cos(theta) / lambda     lambda = 0.125m at 2.4 GHz
```
A drone at 20 m/s produces fd of 40 to 160 Hz. A walking user at 1.5 m/s produces fd of 8 to 12 Hz. This is the primary drone discriminator.

**Free Space Path Loss (EMTL)**
```
FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
```

**BER (Wireless)**
```
BER = 0.5 * exp(-SNR_linear)     SNR_linear = 10^(SNR_dB / 10)
```

---

## ISAC Extension

If ISAC (Integrated Sensing and Communication) is implemented on existing base stations, the same signals used for communication can also be used for sensing. The Doppler shift and fading rate features identified in this project are exactly the signals an ISAC system would use for aerial node detection, so no additional hardware is required.

---

## Team

Sukriti | KavyaSree | Harini J

Under the guidance of Dr. Prasanna Lakshmi Akella

Department of Electronics and Communication Engineering
KL University Hyderabad, April 2026
