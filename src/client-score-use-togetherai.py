import socket
import json
import pandas as pd
import joblib
import os
import time
import math
import csv
import logging
from datetime import datetime
from together import Together
from together.error import ServiceUnavailableError

# Config: server address, model path, API key, and model name
HOST = 'localhost'
PORT = 9999
MODEL_PATH = 'anomaly_model.joblib'
#---------------------------------we add
API_KEY = '490acbd83989f44751fa2f6ec07bf7b9f247e5f87fa5309983be9e6622aa43eb'
client = Together(api_key=API_KEY)
MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
#---------------------------------we add
LOG_PATH = 'logs/anomalies.csv'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load trained model
model = joblib.load(MODEL_PATH)


def pre_process_data(data):
    """
    One-hot encode 'protocol' and reindex to align with training features.

    Args:
        data (dict): Sensor data with keys ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol']
    Returns:
        pd.DataFrame: Single-row DataFrame matching model.feature_names_in_
    """
    df = pd.DataFrame([data])
    #TODO 2: Here you have to add code to pre-process the data as per your model requirements.
    df = pd.get_dummies(df, columns=['protocol'], drop_first=True)
    if 'protocol_UDP' not in df:
        df['protocol_UDP'] = 0
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df


# //compute_confidence score added changes
def compute_confidence(score: float) -> float:
    """
    Map the IsolationForest decision_function score to a [0–1] confidence of anomaly.

    Args:
        score (float): raw output of model.decision_function(X)
    Returns:
        float: anomaly confidence between 0.0 and 1.0
    """
    try:
        return 1 / (1 + math.exp(score))
    except OverflowError:
        return 0.0 if score > 0 else 1.0
# //-----score added


# //CSV logging added changes
def init_anomaly_log(path: str):
    """
    Ensure directory exists and CSV has header. No-op if file exists.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'src_port', 'dst_port', 'packet_size',
                'duration_ms', 'protocol', 'score', 'confidence',
                'label', 'reason'
            ])
        logging.info(f"Created anomaly log: {path}")


def log_anomaly(path: str, data: dict, score: float, confidence: float, label: str, reason: str):
    """
    Append a single anomaly record to the CSV.
    """
    try:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data['src_port'], data['dst_port'], data['packet_size'],
                data['duration_ms'], data['protocol'],
                f"{score:.3f}", f"{confidence:.4f}", label, reason
            ])
        logging.warning(f"Logged anomaly to {path} (score={score:.3f}, conf={confidence:.2%})")
    except Exception as e:
        logging.error(f"Failed to write anomaly to CSV: {e}")
# //-----score added



def alert_anomaly(label: str, reason: str):
    """
    Print a formatted anomaly alert.
    """
    print("\n🚨 Anomaly Detected!")
    print(f"Label: {label}")
    print(f"Reason: {reason}\n")


def main():
    # Initialize anomaly logging
    init_anomaly_log(LOG_PATH)

    # Connect to the anomaly data server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        data_buffer = ''  # renamed for clarity
        print('Client connected to server.\n')

        while True:
            chunk = sock.recv(1024).decode()
            if not chunk:
                break
            data_buffer += chunk

            while '\n' in data_buffer:
                line, data_buffer = data_buffer.split('\n', 1)
                try:
                    data = json.loads(line)
                    print(f"Data Received:\n{data}\n")

                    #TODO 3: process received data and detect anomalies
                    processed = pre_process_data(data)
                    pred = model.predict(processed)[0]         # 1: normal, -1: anomaly
                    score = model.decision_function(processed)[0]

                    # //compute_confidence score added changes
                    confidence = compute_confidence(score)
                    print(f"→ score = {score:.3f} | conf = {confidence:.2%}")
                    # //-----score added

                    if pred != -1:
                        print(f"✅ Normal data. Confidence: {confidence:.2%}\n")
                        continue

                    #TODO 4: connect to LLM and caption anomaly
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that labels sensor anomalies."},
                        {"role": "user", "content": (
                            f"Sensor reading: {json.dumps(data)}\n"
                            f"Anomaly score: {score:.3f}\n"
                            f"Confidence: {confidence:.2%}\n\n"
                            "Describe the type of anomaly and suggest a possible cause."
                        )}
                    ]

                    # Retry logic for 503 errors
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            resp = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                stream=False
                            )
                            break
                        except ServiceUnavailableError:
                            wait = 2 ** attempt
                            print(f"⚠️ Attempt {attempt} failed (503). Retrying in {wait}s...")
                            time.sleep(wait)
                    else:
                        print("❌ LLM unavailable after retries. Skipping anomaly alert.")
                        continue

                    # Extract and print alert
                    content = resp.choices[0].message.content
                    lines = content.strip().split("\n")
                    label = lines[0].split(":", 1)[-1].strip() if ":" in lines[0] else lines[0]
                    reason = (
                        lines[1].split(":", 1)[-1].strip()
                        if len(lines) > 1 and ":" in lines[1]
                        else "\n".join(lines[1:])
                    )
                    alert_anomaly(label, reason)

                    # //CSV logging added changes
                    log_anomaly(LOG_PATH, data, score, confidence, label, reason)
                    # //-----score added

                except json.JSONDecodeError:
                    print("Error decoding JSON.")


if __name__ == '__main__':
    main()
