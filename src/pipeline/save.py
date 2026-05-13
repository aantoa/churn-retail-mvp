import os

def save_processed_data(df, path="data/processed/clientes_churn.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_scored_data(df, path="data/processed/clientes_churn_scored.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_cluster_data(df, path="data/processed/clientes_churn_final.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)