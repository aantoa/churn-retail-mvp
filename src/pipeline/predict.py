import pickle
import pandas as pd
from pathlib import Path

# =========================================
# FUNCIÓN PARA ENCONTRAR LA RAÍZ DEL PROYECTO
# =========================================
def get_project_root():
    """Busca la raíz del proyecto (donde está requirements.txt o .env)"""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").exists() or (parent / ".env").exists():
            return parent
    return current  # fallback

PROJECT_ROOT = get_project_root()
MODELS_PATH = PROJECT_ROOT / "src" / "models"

# =========================================
# FEATURES (DEL NOTEBOOK)
# =========================================
FEATURES = [
    'frequency',
    'monetary',
    'tenure_days',
    'active_month_ratio',
    'avg_ticket',
    'std_ticket',
    'total_galones',
    'avg_days_between',
    'max_gap_days',
    'descuento_porc_promedio',
    'ratio_descuento_sobre_venta',
    'margen_pct',
    'dias_promedio_vencimiento'
]


# =========================================
# CARGAR MODELO
# =========================================
def load_model():
    path = MODELS_PATH / "xgb_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_threshold():
    path = MODELS_PATH / "threshold.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el threshold en: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# =========================================
# VALIDACIÓN
# =========================================
def validate_input(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para el modelo: {missing}")


# =========================================
# PREDICCIÓN
# =========================================
def predict_churn(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    threshold = load_threshold()
    validate_input(df)

    X = df[FEATURES].copy()

    # Manejar valores nulos con medianas (como hacía el imputer)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    df["score_churn"] = model.predict_proba(X)[:, 1]
    df["pred_churn"] = (df["score_churn"] >= threshold).astype(int)

    return df