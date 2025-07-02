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

# Use OpenAI SDK to call AvalAI endpoint
from langchain_openai import ChatOpenAI

# Config: server address, model path, API key, and model name
HOST = 'localhost'
PORT = 9999
MODEL_PATH = 'anomaly_model.joblib'
#---------------------------------use AvalAI via OpenAI SDK
llm = ChatOpenAI(
    model='gpt-4o-mini', 
    base_url="https://api.avalai.ir/v1", 
    api_key="aa-fm9PqtDoolbUZjjiVsKJI3Pv0jdp83LoD2jIJQoiBC30VsmR"
) 
#---------------------------------
LOG_PATH = 'logs/anomalies.csv'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load trained model
model = joblib.load(MODEL_PATH)


def pre_process_data(data):
    """
    One-hot encode 'protocol' and reindex to align with training features.
    """
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['protocol'], drop_first=True)
    if 'protocol_UDP' not in df:
        df['protocol_UDP'] = 0
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df


def compute_confidence(score: float) -> float:
    """Map decision_function score to [0,1] confidence."""
    try:
        return 1 / (1 + math.exp(score))
    except OverflowError:
        return 0.0 if score > 0 else 1.0


def init_anomaly_log(path: str):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp','src_port','dst_port','packet_size',
                'duration_ms','protocol','score','confidence','label','reason'
            ])
        logging.info(f"Created anomaly log: {path}")


def log_anomaly(path: str, data: dict, score: float, confidence: float, label: str, reason: str):
    """Append one anomaly record to CSV."""
    try:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data['src_port'], data['dst_port'], data['packet_size'],
                data['duration_ms'], data['protocol'],
                f"{score:.3f}", f"{confidence:.2%}", label, reason
            ])
        logging.warning(f"Logged anomaly: score={score:.3f}, conf={confidence:.2%}")
    except Exception as e:
        logging.error(f"Failed to write anomaly to CSV: {e}")


def alert_anomaly(label: str, reason: str):
    print("\n🚨 Anomaly Detected!")
    print(f"Label: {label}")
    print(f"Reason: {reason}\n")


def main():
    init_anomaly_log(LOG_PATH)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        buffer = ''
        print('Client connected to server.\n')
        while True:
            chunk = sock.recv(1024).decode()
            if not chunk:
                break
            buffer += chunk
            while '\n' in buffer:
                line, buffer = buffer.split('\n',1)
                try:
                    data = json.loads(line)
                    print(f"Data Received:\n{data}\n")
                    processed = pre_process_data(data)
                    pred = model.predict(processed)[0]
                    score = model.decision_function(processed)[0]

                    confidence = compute_confidence(score)
                    print(f"→ score={score:.3f} | conf={confidence:.2%}")

                    if pred != -1:
                        print(f"✅ Normal data. Confidence: {confidence:.2%}\n")
                        continue

                    messages = [
                        {"role":"system","content":"You are a helpful assistant that labels sensor anomalies."},
                        {"role":"user","content":(
                            f"Sensor reading: {json.dumps(data)}\n"
                            f"Score: {score:.3f}\n"
                            f"Confidence: {confidence:.2%}\n\n"
                            "Describe the anomaly type and possible cause"
                        )}
                    ]

                    max_retries=3
                    for attempt in range(1,max_retries+1):
                        try:
                            resp = llm.invoke(messages)
                            break
                        except ChatOpenAIError as e:
                            wait=2**attempt
                            print(f"⚠️ Attempt {attempt} failed ({e}). Retrying in {wait}s...")
                            time.sleep(wait)
                    else:
                        print("❌ AvalAI unavailable after retries. Skipping.")
                        continue

                    content = content = resp.content
                    lines = content.strip().split("\n")
                    label = lines[0].split(':',1)[-1].strip() if ':' in lines[0] else lines[0]
                    reason = lines[1].split(':',1)[-1].strip() if len(lines)>1 and ':' in lines[1] else '\n'.join(lines[1:])
                    alert_anomaly(label,reason)

                    log_anomaly(LOG_PATH,data,score,confidence,label,reason)

                except json.JSONDecodeError:
                    print("Error decoding JSON.")

if __name__=='__main__':
    main()

# PCA visualization added
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_pca(train_json: str, log_csv: str):
    import json as _json
    with open(train_json) as f:
        normal_df = pd.DataFrame(_json.load(f))
    if os.path.exists(log_csv):
        anom_df = pd.read_csv(log_csv)
    else:
        anom_df = pd.DataFrame(columns=normal_df.columns)
    def _prep(df):
        df = pd.get_dummies(df,columns=['protocol'],drop_first=True)
        if 'protocol_UDP' not in df:
            df['protocol_UDP']=0
        return df.reindex(columns=model.feature_names_in_,fill_value=0)
    normal_X=_prep(normal_df)
    anom_X=_prep(anom_df)
    all_X=pd.concat([normal_X,anom_X],ignore_index=True)
    X_norm=StandardScaler().fit_transform(all_X)
    coords=PCA(n_components=2,random_state=42).fit_transform(X_norm)
    labels=['Normal']*len(normal_X)+['Anomaly']*len(anom_X)
    plt.figure(figsize=(8,6))
    for cls,marker in [('Normal','o'),('Anomaly','x')]:
        idx=[i for i,l in enumerate(labels) if l==cls]
        plt.scatter(coords[idx,0],coords[idx,1],label=cls,marker=marker,alpha=0.7)
    plt.title('PCA Projection of Sensor Data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.show()
visualize_pca('dataset/training_data.json', LOG_PATH)
