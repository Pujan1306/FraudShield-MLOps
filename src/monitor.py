import os
import time
import pandas as pd
import requests


def analyze_production_logs():
    log_path = "data/logs/production_logs.csv"
    print("\n[MONITOR] Auditing Production Log Stream...")

    if not os.path.exists(log_path):
        print("No production logs found yet. Run an inference pass or train sequence first.")
        return

    df = pd.read_csv(log_path)
    total_records = len(df)

    if "target" in df.columns:
        fraud_count = df["target"].sum()
        fraud_ratio = (fraud_count / total_records) * 100
        print(f"-> Total transactions logged: {total_records}")
        print(f"-> Observed Fraud Rate in Stream: {fraud_ratio:.4f}%")
    else:
        print(f"-> System logged {total_records} features. Waiting for baseline target tags.")


def benchmark_api_latency():
    url = "http://127.0.0.1:8000/predict"
    print("\n[MONITOR] Benchmarking Endpoint Latency...")

    dummy_payload = {"features": [0.0] * 30}

    try:
        start_time = time.time()
        response = requests.post(url, json=dummy_payload, timeout=2)
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            print(f"Serving Engine Online | Latency: {latency:.2f}ms")
            if latency > 100:
                print("Warning: System latency exceeds production budget (>100ms).")
        else:
            print(f"API responded with anomalous status code: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("Serving Engine Offline! Start your Uvicorn app instance on port 8000.")


if __name__ == "__main__":
    print("Launching Local MLOps Telemetry Sweep...")
    analyze_production_logs()
    benchmark_api_latency()