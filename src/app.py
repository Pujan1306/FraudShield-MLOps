import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pytorch
import torch
app = FastAPI(title="Fraud Detection Serving Engine", version="1.0")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

scaler_path = "models/scaler.pkl"
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("StandardScaler loaded successfully.")
else:
    print("Scaler not found! Make sure you ran train.py first.")
    scaler = None

try:
    model_uri = "models:/Production_Fraud_Net/3"
    print(f"Fetching registered production model: {model_uri}...")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    print("Serving engine linked to registered model assets.")
except Exception as e:
    print(f"Failed to load model from registry! Error detail: {e}")
    model = None

class TransactionData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: TransactionData):
    if model is None:
        raise HTTPException(status_code=500, detail="Serving model state is uninitialized.")

    if len(data.features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Feature shape mismatch. Expected 30 dimensions, received {len(data.features)}"
        )

    scaled_array = scaler.transform([data.features])
    input_tensor = torch.tensor(scaled_array, dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_tensor)

        probability = torch.sigmoid(logits).item()

        is_fraud = 1 if probability > 0.5 else 0

    return {
        "status": "Success",
        "prediction": is_fraud,
        "fraud_probability": round(probability, 6),
        "verdict": "FRAUDULENT_TRANSACTION" if is_fraud == 1 else "NORMAL_APPROVED"
    }