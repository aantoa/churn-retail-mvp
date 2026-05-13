import pandas as pd
import numpy as np


# =========================================
# 1. NORMALIZAR COLUMNAS
# =========================================
def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return df


# =========================================
# 2. FEATURE ENGINEERING CLIENTE (IGUAL AL NOTEBOOK)
# =========================================
def build_customer_features(group, snapshot_date):

    group = group.sort_values('fechaemision').copy()
    fechas = group['fechaemision']

    fecha_primera = fechas.min()
    fecha_ultima = fechas.max()

    # RFM
    recency = (snapshot_date - fecha_ultima).days
    frequency = len(group)
    monetary = group['total_con_impuesto'].sum()

    # Ciclo de vida
    tenure_days = (fecha_ultima - fecha_primera).days
    active_month_ratio = (
        fechas.dt.to_period('M').nunique() /
        max(
            1,
            (fecha_ultima.year - fecha_primera.year) * 12 +
            (fecha_ultima.month - fecha_primera.month) + 1
        )
    )

    # Ticket
    avg_ticket = group['total_con_impuesto'].mean()
    std_ticket = group['total_con_impuesto'].std()
    total_galones = group['galones'].sum()

    # Intervalos
    diffs = fechas.diff().dropna().dt.days
    avg_days_between = diffs.mean() if len(diffs) > 0 else np.nan
    max_gap_days = diffs.max() if len(diffs) > 0 else np.nan

    # 🔥 VARIABLES 70D (IMPORTANTES)
    last_70 = snapshot_date - pd.Timedelta(days=70)
    prev_70_start = snapshot_date - pd.Timedelta(days=140)
    prev_70_end = snapshot_date - pd.Timedelta(days=70)

    grp_last_70 = group[(group['fechaemision'] >= last_70) & (group['fechaemision'] < snapshot_date)]
    grp_prev_70 = group[(group['fechaemision'] >= prev_70_start) & (group['fechaemision'] < prev_70_end)]

    compras_70d = len(grp_last_70)
    gasto_70d = grp_last_70['total_con_impuesto'].sum()

    spend_last_70 = grp_last_70['total_con_impuesto'].sum()
    spend_prev_70 = grp_prev_70['total_con_impuesto'].sum()
    spend_trend_70 = (spend_last_70 - spend_prev_70) / (spend_prev_70 + 1)

    # Variables comerciales
    descuento_porc_promedio = group['descuento_porc'].mean()
    descuento_total = group['descuento'].sum()

    ratio_descuento_sobre_venta = (
        descuento_total / (group['total_sin_impuesto'].sum() + 1e-6)
    )

    margen_pct = (
        group['margen_bruto'].sum() / (group['total_sin_impuesto'].sum() + 1e-6)
    )

    dias_promedio_vencimiento = group['dias_vencimiento'].mean()
    flag_credito = group['flag_credito'].mean()

    return pd.Series({
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'tenure_days': tenure_days,
        'active_month_ratio': active_month_ratio,
        'avg_ticket': avg_ticket,
        'std_ticket': std_ticket,
        'total_galones': total_galones,
        'avg_days_between': avg_days_between,
        'max_gap_days': max_gap_days,
        'compras_70d': compras_70d,
        'gasto_70d': gasto_70d,
        'spend_trend_70': spend_trend_70,
        'descuento_porc_promedio': descuento_porc_promedio,
        'ratio_descuento_sobre_venta': ratio_descuento_sobre_venta,
        'margen_pct': margen_pct,
        'dias_promedio_vencimiento': dias_promedio_vencimiento,
        'flag_credito': flag_credito
    })


# =========================================
# 3. PIPELINE PRINCIPAL
# =========================================
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = normalize_columns(df_raw)

    # =========================
    # LIMPIEZA
    # =========================
    df['fechaemision'] = pd.to_datetime(df['fechaemision'], errors='coerce')
    df['fechavencimiento'] = pd.to_datetime(df['fechavencimiento'], errors='coerce')

    num_cols = [
        'descuento_porc', 'galones',
        'total_con_impuesto', 'total_sin_impuesto',
        'total_costo', 'descuento'
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[
        'cliente_id_new', 'documento_id_new',
        'fechaemision', 'total_con_impuesto'
    ])

    df = df[df['total_con_impuesto'] > 0]

    # =========================
    # COMPRAS
    # =========================
    keys = ['cliente_id_new', 'documento_id_new', 'fechaemision']

    compras = (
        df.groupby(keys, as_index=False)
        .agg({
            'total_con_impuesto': 'sum',
            'total_sin_impuesto': 'sum',
            'total_costo': 'sum',
            'descuento': 'sum',
            'descuento_porc': 'mean',
            'total_promocion': 'sum',
            'total_promocion_pv': 'sum',
            'cantidad': 'sum',
            'galones': 'sum',
            'terminopago_resumen': 'first',
            'fechavencimiento': 'max'
        })
    )

    # =========================
    # VARIABLES TRANSACCIONALES
    # =========================
    compras['margen_bruto'] = compras['total_sin_impuesto'] - compras['total_costo']

    compras['flag_credito'] = (
        compras['terminopago_resumen']
        .astype(str).str.lower().eq('credito')
        .astype(int)
    )

    compras['dias_vencimiento'] = (
        compras['fechavencimiento'] - compras['fechaemision']
    ).dt.days

    # =========================
    # SNAPSHOT
    # =========================
    snapshot_date = compras['fechaemision'].max() + pd.Timedelta(days=1)

    # =========================
    # AGREGACIÓN CLIENTE
    # =========================
    df_cliente = (
        compras
        .groupby('cliente_id_new')
        .apply(lambda g: build_customer_features(g, snapshot_date))
        .reset_index()
    )

    # =========================
    # CHURN
    # =========================
    df_cliente['churn'] = (df_cliente['recency'] > 70).astype(int)

    # =========================
    # LIMPIEZA FINAL
    # =========================
    df_cliente = df_cliente.replace([np.inf, -np.inf], np.nan)

    num_cols_cliente = df_cliente.select_dtypes(include=[np.number]).columns
    df_cliente[num_cols_cliente] = df_cliente[num_cols_cliente].fillna(0)

    return df_cliente
