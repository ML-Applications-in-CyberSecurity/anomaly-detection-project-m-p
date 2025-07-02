import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

model = joblib.load('anomaly_model.joblib')

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
visualize_pca('dataset/training_data.json', 'logs/anomalies.csv')