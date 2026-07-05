import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
from src.model import FraudClassificationModel

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fraud_Detection_Pipeline")

def load_and_preprocess_data(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    print("Load real creditcard dataset...")
    df = pd.read_csv(file_path)
    df.to_csv("data/raw/train_baseline.csv", index=False)

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("StandardScaler saved successfully to models/scaler.pkl")

    return X, y

def main():
    csv_path = "data/raw/creditcard.csv"
    X, y = load_and_preprocess_data(csv_path)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    prod_logs = pd.DataFrame(X_val)
    prod_logs["target"] = y_val.astype(int)
    os.makedirs("data/logs", exist_ok=True)
    prod_logs.to_csv("data/logs/production_logs.csv", index=False)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    num_features = X_train.shape[1]
    model = FraudClassificationModel(input_dim=num_features)

    pos_weight_value = torch.tensor([284315 / 492], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print(f"Dataset split complete. Training on {len(X_train)} rows. Validating on {len(X_val)} rows.")

    with mlflow.start_run() as run:
        mlflow.log_param("lr", 0.001)
        mlflow.log_param("batch_size", 1024)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("data_source", "Kaggle_CreditCard_ULB")

        print("Training neural network on real transaction logs...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

                pred = (outputs > 0).float()
                correct_train += (pred == batch_y).sum().item()

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = correct_train / len(train_dataset)

            model.eval()
            val_loss = 0.0
            correct_val = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)

                    pred = (outputs > 0).float()
                    correct_val += (pred == batch_y).sum().item()

                epoch_val_loss = val_loss / len(val_dataset)
                epoch_val_acc = correct_val / len(val_dataset)

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
            )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="fraud_model",
            registered_model_name="Production_Fraud_Net",
            serialization_format="pickle"
        )
        print(f"Model registered successfully! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()