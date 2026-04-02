# Dataset Description

## 1. Overview

This dataset was generated through Network Performance Simulation (NPS) of a Mobile Switching Centre (MSC)-like environment using Cisco Packet Tracer. 

Multiple traffic scenarios were created, including normal operational conditions and abnormal traffic conditions such as traffic surges and overload situations. Network performance parameters were collected for each scenario and labeled accordingly.

The dataset is used to train and evaluate a supervised machine learning model for abnormal traffic detection.

---

## 2. Dataset Structure

Each row in the dataset represents a single traffic scenario observation.

### Total Features: 4
### Target Variable: 1

---

## 3. Feature Description

| Feature Name   | Description | Unit | Type |
|---------------|------------|------|------|
| avg_delay     | Average packet delay observed in the network | milliseconds (ms) | Numerical |
| packet_loss   | Percentage of packets lost during transmission | % | Numerical |
| throughput    | Rate of successful data transmission | Mbps | Numerical |
| traffic_rate  | Traffic intensity or load generated in the network | Relative scale / Mbps | Numerical |
| label         | Traffic condition classification (Normal / Abnormal) | Categorical | Target |

---

## 4. Label Definition

- **Normal**: Network operating within acceptable performance thresholds.
- **Abnormal**: Network experiencing overload, high delay, excessive packet loss, or unstable throughput conditions.

---

## 5. Data Collection Method

- Network topology simulated in Cisco Packet Tracer.
- Central router configured to emulate MSC behavior.
- Traffic intensity varied systematically across multiple scenarios.
- Performance parameters recorded manually and compiled into structured CSV format.

---

## 6. Usage

This dataset is used to:

- Train a Decision Tree classifier
- Evaluate abnormal traffic detection accuracy
- Analyze MSC overload behavior

---

## 7. File Format

The dataset is stored in CSV format and can be loaded using:

```python
import pandas as pd
data = pd.read_csv("train_dataset.csv")
