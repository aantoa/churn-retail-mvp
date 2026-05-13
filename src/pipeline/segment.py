import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================
# FUNCIÓN PARA ENCONTRAR LA RAÍZ DEL PROYECTO
# =========================================
def get_project_root():
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").exists() or (parent / ".env").exists():
            return parent
    return current

PROJECT_ROOT = get_project_root()
MODELS_PATH = PROJECT_ROOT / "src" / "models"

# =========================================
# FEATURES (DEL NOTEBOOK)
# =========================================
FEATURES_CLUSTER = [
    'frequency',
    'monetary',
    'avg_ticket',
    'active_month_ratio',
    'avg_days_between',
    'max_gap_days',
    'descuento_porc_promedio',
    'ratio_descuento_sobre_venta',
    'margen_pct',
    'dias_promedio_vencimiento',
    'total_galones',
    'tenure_days'
]


# =========================================
# COLUMNAS CON LOG
# =========================================
COLS_SKEWED = [
    'frequency',
    'monetary',
    'avg_ticket',
    'total_galones',
    'tenure_days'
]


# =========================================
# CARGAR MODELOS
# =========================================
def load_scaler():
    path = MODELS_PATH / "scaler_cluster.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el scaler en: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_kmeans():
    path = MODELS_PATH / "kmeans.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo KMeans en: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================
# ETIQUETAS
# =========================================
def label_cluster(cluster):
    mapping = {
        0: 'En riesgo',
        1: 'Promocionales',
        2: 'Recurrentes',
        3: 'Estratégicos'
    }
    return mapping.get(cluster, 'Sin etiqueta')


# =========================================
# SEGMENTACIÓN
# =========================================
def segment_clients(df: pd.DataFrame) -> pd.DataFrame:

    scaler = load_scaler()
    kmeans = load_kmeans()

    # Validación
    missing = [c for c in FEATURES_CLUSTER if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para clustering: {missing}")

    X = df[FEATURES_CLUSTER].copy()

    # -------------------------
    # Log transform (igual notebook)
    # -------------------------
    for col in COLS_SKEWED:
        X[col] = np.log1p(X[col])

    # -------------------------
    # Escalar
    # -------------------------
    X_scaled = scaler.transform(X)

    # -------------------------
    # Cluster
    # -------------------------
    df["cluster"] = kmeans.predict(X_scaled)

    # -------------------------
    # Etiqueta
    # -------------------------
    df["segmento"] = df["cluster"].map(label_cluster)

    return df