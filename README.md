# Asymteric-Fraud-Risk-Detection-System

## 📌 Overview

This project explores a **real-world limitation in traditional machine learning systems**:
they assume **symmetric costs for gains and losses**.

In reality, especially in financial systems:

* ✅ Gain → small positive impact
* ❌ Loss → disproportionately large negative impact

This system applies concepts from **Behavioral Finance** and **Prospect Theory** to build a **cost-sensitive fraud detection model** that reflects real-world risk asymmetry.

---

## ⚠️ Problem Statement

Traditional models assume:

```
Gain = +5  
Loss = -5
```

But in real-world financial systems:

```
Gain = +5  
Loss = -20 (or worse)
```

### ❗ Consequences:

* High **false positives** → poor user experience (accounts blocked unnecessarily)
* High **false negatives** → severe financial loss

---

## 🎯 Objectives

* Detect fraudulent transactions accurately
* Reduce false positives (better UX)
* Penalize false negatives more heavily
* Model **asymmetric risk behavior**
* Compare **symmetric vs asymmetric models**

---

## 🧠 Key Concepts

### 📉 Loss Aversion

Humans perceive losses more strongly than gains.

### 📊 Prospect Theory

Value is not linear — losses have higher psychological (and financial) weight than gains.

### ⚖️ Asymmetric Cost Function

We redefine model loss as:

* False Positive → small penalty
* False Negative → large penalty

---

## 🏗️ System Architecture

```
Data → Preprocessing → Model → Risk Scoring → API → Decision
```

### Components:

* Data Pipeline
* ML Model (Cost-Sensitive)
* Custom Loss Function
* Risk Scoring Engine
* API Layer (FastAPI/Flask)
* Explainability Module

---

## 📂 Project Structure

```
asymmetric-fraud-system/
│── data/                  # Dataset / synthetic data
│── models/                # Saved models
│── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── loss_functions.py
│   ├── train.py
│   ├── evaluate.py
│   ├── api.py
│── notebooks/             # Experiments & EDA
│── tests/                 # Unit tests
│── requirements.txt
│── README.md
│── documentation.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/asymmetric-fraud-system.git
cd asymmetric-fraud-system

pip install -r requirements.txt
```

---

## 🚀 Usage

### Train Model

```bash
python src/train.py
```

### Run Evaluation

```bash
python src/evaluate.py
```

### Start API

```bash
python src/api.py
```

---

## 🔍 Example API Request

```json
POST /predict

{
  "transaction_amount": 5000,
  "location": "unknown",
  "device": "new"
}
```

### Response:

```json
{
  "fraud_probability": 0.87,
  "risk_score": "HIGH",
  "decision": "BLOCK",
  "explanation": "Unusual location + high amount"
}
```

---

## 📊 Evaluation Metrics

We go beyond accuracy:

* Precision
* Recall
* F1 Score
* ROC-AUC
* **Cost-Based Evaluation (Key Focus)**

### 💡 Expected Loss Formula

\text{Expected Loss} = (FP \cdot C_{FP}) + (FN \cdot C_{FN})

---

## ⚖️ Symmetric vs Asymmetric Comparison

| Metric          | Symmetric Model | Asymmetric Model  |
| --------------- | --------------- | ----------------- |
| False Positives | Lower priority  | Controlled        |
| False Negatives | Underweighted   | Heavily penalized |
| Real-world risk | Poor modeling   | Realistic         |

---

## 📈 Features

* ✅ Custom asymmetric loss function
* ✅ Cost-sensitive learning
* ✅ Fraud probability + risk scoring
* ✅ Explainable predictions (SHAP / feature importance)
* ✅ Simulation of real-world scenarios
* ✅ Threshold tuning

---

## 🧪 Example Scenarios

### 1. Small anomaly

* Traditional model → BLOCK (False Positive ❌)
* Asymmetric model → ALLOW (Better UX ✅)

### 2. Large fraud

* Traditional model → MISSED ❌
* Asymmetric model → DETECTED ✅

---

## 🔮 Future Improvements

* Real-time streaming (Kafka)
* Deep learning-based anomaly detection
* Graph-based fraud detection
* Integration with banking systems
* Adaptive risk thresholds

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

MIT License

---

## ⭐ Final Thought

This project highlights a key gap between **academic ML assumptions** and **real-world financial risk**, and proposes a system that aligns models with how risk actually behaves.

---
