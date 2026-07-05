# 🛡️ FraudShield: MLOps Fraud Detection Engine

A production-grade fraud detection pipeline that integrates PyTorch, MLflow, and FastAPI. This project demonstrates end-to-end MLOps practices, including automated model registration, real-time feature scaling, and system telemetry.

## 🏗️ Project Structure

```text
.
├── data/               # (Empty: Place dataset here)
├── models/             # (Generated: Contains serialized scaler)
├── src/                # Core logic
│   ├── app.py          # FastAPI serving engine
│   ├── model.py        # PyTorch Neural Network architecture
│   ├── monitor.py      # Telemetry & Latency audit tool
│   ├── train.py        # Training pipeline & MLflow registration
│   └── __init__.py     # Package initialization
├── .gitignore          # Excludes data, venv, and cache
└── requirements.txt    # Project dependencies
```

## 🚀 Setup Instructions

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation

This project uses the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

1. Download the `creditcard.csv` file from Kaggle.
2. Create the folder structure: `mkdir -p data/raw`
3. Place the `creditcard.csv` file into: `data/raw/creditcard.csv`

### 3. Execution Pipeline

1. **Start MLflow Server:**
```bash
mlflow server --host 127.0.0.1 --port 5000
```

2. **Train & Register Model:**
```bash
python -m src.train
```

3. **Deploy API:**
```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

*Access the interactive Swagger UI at: `http://127.0.0.1:8000/docs`*

## 📊 Monitoring

To verify your production environment status, check feature distribution, and audit API response latency, run:

```bash
python -m src.monitor
```

## ⚙️ Key MLOps Features

* **Automated Scaling:** Uses `StandardScaler` serialized via `pickle` to ensure live inference matches training distribution.
* **Class Imbalance Handling:** Uses weighted `BCEWithLogitsLoss` to optimize for rare fraud events.
* **MLflow Tracking:** All training runs and model artifacts are versioned and registered.
* **Production Gateway:** FastAPI service with input validation and real-time scaling for low-latency inference.

---

*Developed as a high-performance MLOps demonstrator.*
