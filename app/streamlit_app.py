import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Rutas
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.llm.strategy import (
    generar_estrategia,
    asignar_nivel_riesgo_score
)

from src.ui.helpers import (
    SEGMENT_COLORS,
    canal_ui,
    prioridad_ui,
    segmento_ui,
    riesgo_badge,
    score_bar,
    segmento_badge,
    toast_html
)

from src.pipeline.preprocess import preprocess_data
from src.pipeline.predict import predict_churn
from src.pipeline.segment import segment_clients
from src.pipeline.save import save_cluster_data

# -------------------------
# Configuración
# -------------------------
st.set_page_config(page_title="Churn Retail MVP", layout="wide")

# -------------------------
# CSS
# -------------------------
def load_css():
    with open(ROOT / "styles" / "styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
""", unsafe_allow_html=True)

# -------------------------
# Cargar datos
# -------------------------
@st.cache_data
def load_data():
    path = ROOT / "data" / "processed" / "clientes_churn_final.csv"
    return pd.read_csv(path)

df = load_data()

# -------------------------
# Parse estrategia (NUEVO)
# -------------------------
import json
import ast
import html
import textwrap
import streamlit.components.v1 as components

def parse_estrategia(estrategia_raw):
    import re

    texto = str(estrategia_raw).strip()
    texto = texto.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if match:
        texto = match.group(0)

    try:
        return json.loads(texto)
    except Exception:
        try:
            return ast.literal_eval(texto)
        except Exception:
            return {
                "analisis_breve": "No se pudo interpretar la estrategia generada.",
                "prioridad": "Media - Monitorear comportamiento",
                "canal_sugerido": "WhatsApp",
                "incentivo_recomendado": "Pendiente de revisión comercial.",
                "estrategias": {
                    "accion_1": "Reintentar la generación de estrategia para este cliente.",
                    "accion_2": "Validar manualmente la prioridad comercial del cliente.",
                    "accion_3": "Registrar el caso para seguimiento posterior.",
                    "accion_4": "Asignar una acción comercial temporal según su segmento."
                }
            }

def mostrar_metricas_cliente(cliente):
    frecuencia = int(cliente["frequency"])
    monto = float(cliente["monetary"])
    recency = int(cliente["recency"])
    segmento = str(cliente["segmento"])
    segmento_cfg = segmento_ui(segmento)

    st.markdown(
        f"""
    <div class="customer-kpi-band animated-card">

    <div class="customer-kpi-item">
        <div class="customer-kpi-icon ui-purple">
        <i class="bi bi-bag-check"></i>
        </div>
        <div>
        <div class="customer-kpi-label">Frecuencia</div>
        <div class="customer-kpi-value">{frecuencia}</div>
        <div class="customer-kpi-help">Compras totales</div>
        </div>
    </div>

    <div class="customer-kpi-separator"></div>

    <div class="customer-kpi-item">
        <div class="customer-kpi-icon ui-green">
        <i class="bi bi-cash-coin"></i>
        </div>
        <div>
        <div class="customer-kpi-label">Monto total</div>
        <div class="customer-kpi-value">S/ {monto:,.2f}</div>
        <div class="customer-kpi-help">Gasto acumulado</div>
        </div>
    </div>

    <div class="customer-kpi-separator"></div>

    <div class="customer-kpi-item">
        <div class="customer-kpi-icon ui-blue">
        <i class="bi bi-calendar-event"></i>
        </div>
        <div>
        <div class="customer-kpi-label">Recency</div>
        <div class="customer-kpi-value">{recency}</div>
        <div class="customer-kpi-help">Días desde última compra</div>
        </div>
    </div>

    <div class="customer-kpi-separator"></div>

    <div class="customer-kpi-item">
        <div class="customer-kpi-icon {segmento_cfg["class"]}">
            <i class="bi {segmento_cfg["icon"]}"></i>
        </div>
        <div>
        <div class="customer-kpi-label">Segmento</div>
        <div class="customer-kpi-value kpi-segment">{html.escape(segmento)}</div>
        <div class="customer-kpi-help">Cluster comercial</div>
        </div>
    </div>

    </div>
    """,
        unsafe_allow_html=True
    )

def mostrar_riesgo_y_segmento(cliente):
    score = cliente["score_churn"] if "score_churn" in cliente.index else cliente["churn"]
    score = float(score)
    score_pct = score * 100
    segmento = cliente["segmento"]

    if score >= 0.75:
        nivel = "Crítico"
        ui_class = "ui-red"
        icono = "bi-exclamation-triangle-fill"
    elif score >= 0.50:
        nivel = "Alto"
        ui_class = "ui-orange"
        icono = "bi-shield-check"
    elif score >= 0.30:
        nivel = "Medio"
        ui_class = "ui-blue"
        icono = "bi-activity"
    else:
        nivel = "Bajo"
        ui_class = "ui-green"
        icono = "bi-check-circle-fill"

    c1, c2 = st.columns([1.25, 2.0])

    risk_html = f"""
    <div class="risk-meter-card animated-card">
    <div class="risk-meter-header">
        <div>
        <div class="risk-card-title">Riesgo de churn</div>
        <div class="risk-card-subtitle">Probabilidad estimada de abandono</div>
        </div>
        <div class="risk-meter-score {ui_class}">{score_pct:.1f}%</div>
    </div>

    <div class="risk-meter-track">
        <div class="risk-meter-fill {ui_class}" style="width:{score_pct}%;"></div>
    </div>

    <div class="risk-meter-scale">
        <span>Bajo</span>
        <span>Medio</span>
        <span>Alto</span>
        <span>Crítico</span>
    </div>

    <div class="risk-meter-status {ui_class}">
        Riesgo {nivel}
    </div>
    </div>
    """

    with c1:
        st.markdown(risk_html, unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"""
            <div class="client-profile-card animated-card {ui_class}">
            <div class="client-profile-icon">
                <i class="bi {icono}"></i>
            </div>

            <div class="client-profile-content">
                <div class="client-profile-label">PERFIL DEL CLIENTE</div>
                <div class="client-profile-title">{html.escape(str(segmento))}</div>
                <div class="client-profile-risk">Riesgo {nivel}</div>
                <div class="client-profile-desc">
                Cliente priorizado según score de churn, segmento comercial y reglas internas de retención.
                Requiere acciones alineadas a su nivel de riesgo.
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def mostrar_panel_ejecutivo(estrategia_json):
    prioridad = estrategia_json.get("prioridad", "No disponible")
    canal = estrategia_json.get("canal_sugerido", "No disponible")
    incentivo = estrategia_json.get("incentivo_recomendado", "No disponible")
    acciones = estrategia_json.get("estrategias", {})

    accion_principal = acciones.get("accion_1", "No disponible")

    canal_cfg = canal_ui(canal)
    prioridad_cfg = prioridad_ui(prioridad)

    st.markdown(
        f"""
    <div class="strategy-flow-title">
    <span class="section-icon {prioridad_cfg["class"]}">
        <i class="bi {prioridad_cfg["icon"]}"></i>
    </span>
    Estrategia recomendada
    <span class="priority-pill {prioridad_cfg["class"]}">{html.escape(prioridad)}</span>
    </div>

    <div class="strategy-flow-card animated-card">

    <div class="flow-item">
        <div class="flow-icon {canal_cfg["class"]}">
        <i class="bi {canal_cfg["icon"]}"></i>
        </div>
        <div class="flow-content">
        <div class="flow-label">CANAL PRINCIPAL</div>
        <div class="flow-title">{html.escape(canal)}</div>
        <div class="flow-desc">{canal_cfg["desc"]}</div>
        </div>
    </div>

    <div class="flow-arrow">›</div>

    <div class="flow-item">
        <div class="flow-icon ui-green">
        <i class="bi bi-ticket-perforated"></i>
        </div>
        <div class="flow-content">
        <div class="flow-label">INCENTIVO RECOMENDADO</div>
        <div class="flow-title">{html.escape(incentivo)}</div>
        <div class="flow-desc">Beneficio comercial definido según riesgo y segmento.</div>
        </div>
    </div>

    <div class="flow-arrow">›</div>

    <div class="flow-item">
        <div class="flow-icon ui-orange">
        <i class="bi bi-person-check"></i>
        </div>
        <div class="flow-content">
        <div class="flow-label">ACCIÓN PRINCIPAL</div>
        <div class="flow-title">{html.escape(accion_principal)}</div>
        <div class="flow-desc">Primera acción del plan de retención.</div>
        </div>
    </div>

    </div>
    """,
            unsafe_allow_html=True
        )

    st.markdown(
            """
    <div class="plan-title">
    <span class="section-icon ui-purple"><i class="bi bi-calendar4-week"></i></span>
    Plan de acción de retención comercial de 30 días
    </div>
    """,
            unsafe_allow_html=True
        )

    labels_acciones = ["Semana 1", "Semana 2", "Semana 3", "Semana 4"]

    for i in range(1, 5):
        accion = acciones.get(f"accion_{i}", "No disponible")
        label = labels_acciones[i - 1]

        st.markdown(
                f"""
            <div class="action-row animated-card">
            <div class="timeline-dot"></div>
            <div class="action-chip">{label}</div>
            <div class="action-text">{html.escape(str(accion))}</div>
            </div>
            """,
                unsafe_allow_html=True
            )

def mostrar_estrategia_cliente(cliente):
    st.subheader("LLM + reglas de negocio")

    with st.spinner("Buscando estrategia existente o generando una nueva..."):
        estrategia_raw = generar_estrategia(cliente)

    estrategia_json = parse_estrategia(estrategia_raw)

    mostrar_panel_ejecutivo(estrategia_json)

# -------------------------
# Sidebar
# -------------------------
# =========================================
# Sidebar navigation
# =========================================

PAGES = {
    "dashboard": {
        "label": "Dashboard",
        "icon": ":material/dashboard:"
    },

    "buscar": {
        "label": "Buscar cliente",
        "icon": ":material/person:"
    },

    "upload": {
        "label": "Subir archivo",
        "icon": ":material/upload:"
    },

    "about": {
        "label": "Acerca de",
        "icon": ":material/info:"
    }
}


# estado inicial
if "page" not in st.session_state:
    st.session_state.page = "dashboard"


# =========================================
# Sidebar UI
# =========================================

with st.sidebar:

    # BRAND
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">
            <i class="bi bi-cart3"></i>
        </div>
        <div>
            <div class="sidebar-title">Churn Retail</div>
            <div class="sidebar-subtitle">MVP</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # MENU
    with st.sidebar:
        for key, item in PAGES.items():

            active = st.session_state.page == key

            if st.button(
                item["label"],
                icon=item["icon"],   # directo, sin mapear
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary"
            ):
                st.session_state.page = key
                st.rerun()

    st.sidebar.markdown(
        """
        <div class="sidebar-bottom">
            <div class="project-box">
                <div class="project-title">
                    Proyecto integrador
                </div>
                <div class="project-text">
                    Sistema inteligente de predicción de churn y generación automática de estrategias comerciales.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================
# Router
# =========================================

query_params = st.query_params

if "page" in query_params:
    st.session_state.page = query_params["page"]

page = st.session_state.page

# -------------------------
# Dashboard
# -------------------------
if page == "dashboard":

    st.markdown('<div class="main-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Resumen general del modelo de churn y segmentación de clientes.</div>',
        unsafe_allow_html=True
    )

    # =========================
    # Métricas principales
    # =========================
    total_clientes = len(df)
    clientes_riesgo = int((df["pred_churn"] == 1).sum())
    churn_rate = clientes_riesgo / total_clientes * 100
    score_promedio = df["score_churn"].mean() * 100
    n_segmentos = df["segmento"].nunique()

    # métricas modelo
    roc_auc = 0.869
    recall_modelo = 0.90   # ← ESTE ES EL HERO KPI
    threshold = 0.42       # usa tu valor real si lo tienes

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon purple"><i class="bi bi-people"></i></div>
            <div>
                <div class="kpi-label">Total clientes</div>
                <div class="kpi-value">{total_clientes:,}</div>
                <div class="kpi-help">Base analizada</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon red"><i class="bi bi-exclamation-triangle"></i></div>
            <div>
                <div class="kpi-label">Clientes en riesgo</div>
                <div class="kpi-value">{clientes_riesgo:,}</div>
                <div class="kpi-help">{churn_rate:.2f}% detectados</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card kpi-highlight">
            <div class="kpi-icon orange"><i class="bi bi-bullseye"></i></div>
            <div>
                <div class="kpi-label">Recall churn</div>
                <div class="kpi-value">{recall_modelo:.2%}</div>
                <div class="kpi-help">Clientes detectados</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon blue"><i class="bi bi-graph-up"></i></div>
            <div>
                <div class="kpi-label">ROC-AUC</div>
                <div class="kpi-value">{roc_auc:.3f}</div>
                <div class="kpi-help">Performance modelo</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon gray"><i class="bi bi-diagram-3"></i></div>
            <div>
                <div class="kpi-label">Segmentos</div>
                <div class="kpi-value">{n_segmentos}</div>
                <div class="kpi-help">Clusters</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # =========================
    # Gráfico 1: Donut churn
    # =========================
    churn_counts = df["pred_churn"].value_counts().sort_index()

    labels = ["Activos", "En riesgo"]
    values = [
        int(churn_counts.get(0, 0)),
        int(churn_counts.get(1, 0))
    ]

    fig1 = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                domain=dict(x=[0.08, 0.92], y=[0.16, 0.92]),
                marker=dict(
                    colors=["#22c55e", "#f43f5e"],
                    line=dict(color="white", width=3)
                ),
                textinfo="none",
                textfont=dict(
                    color="white",
                    size=14,
                    family="Arial Black"
                ),
                insidetextorientation="horizontal",
                showlegend=False
            )
        ]
    )

    fig1.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[
            dict(
                text=f"<b style='font-size:30px'>{churn_rate:.1f}%</b><br>"
                    "<span style='font-size:14px;color:#0f172a'>Churn</span>",
                x=0.5,
                y=0.5,
                showarrow=False
            )
        ],
        legend=dict(
            font=dict(size=12, color="#0f172a")
        )
    )

    # =========================
    # Gráfico 2: Clientes por segmento
    # =========================
    seg_counts = (
        df["segmento"]
        .value_counts()
        .reset_index()
    )

    seg_counts.columns = ["segmento", "clientes"]

    segment_colors = SEGMENT_COLORS

    fig2 = px.bar(
        seg_counts,
        x="segmento",
        y="clientes",
        text="clientes",
        color="segmento",
        color_discrete_map=segment_colors
    )

    fig2.update_traces(textposition="outside")

    fig2.update_layout(
        height=310,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title=""
    )

    # =========================
    # Gráfico 3: Churn por segmento (ordenado de menor a mayor)
    # =========================
    churn_segmento = (
        df.groupby("segmento")["churn"]
        .mean()
        .mul(100)
        .reset_index()
    )
    churn_segmento.columns = ["segmento", "churn_pct"]

    # 👇 Ordenar de menor a mayor porcentaje de churn
    churn_segmento = churn_segmento.sort_values("churn_pct", ascending=True)

    fig4 = px.bar(
        churn_segmento,
        x="segmento",
        y="churn_pct",
        text=churn_segmento["churn_pct"].round(1).astype(str) + "%",
        color="segmento",
        color_discrete_map=segment_colors,
        category_orders={"segmento": churn_segmento["segmento"].tolist()}  # 🔥 fuerza el orden
    )

    fig4.update_traces(textposition="outside")
    fig4.update_layout(
        height=310,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="% Churn",
        yaxis=dict(range=[0, 110])
    )

    

    # =========================
    # Gráfico 3: Bullet chart riesgo por segmento
    # =========================
    seg_risk = (
        df.groupby("segmento")["score_churn"]
        .mean()
        .reset_index()
    )

    seg_risk["churn_pct"] = seg_risk["score_churn"] * 100
    seg_risk = seg_risk.sort_values("churn_pct", ascending=True)
    

    def risk_color(x):
        if x >= 75:
            return "#dc2626"
        elif x >= 50:
            return "#f97316"
        elif x >= 30:
            return "#2563eb"
        else:
            return "#16a34a"

    seg_risk["color"] = seg_risk["churn_pct"].apply(risk_color)

    fig3 = go.Figure()

    # bandas de referencia
    for y in seg_risk["segmento"]:
        fig3.add_trace(go.Bar(
            x=[100],
            y=[y],
            orientation="h",
            marker=dict(color="#f1f5f9"),
            width=0.55,
            showlegend=False,
            hoverinfo="skip"
        ))

    # valor real
    fig3.add_trace(go.Bar(
        x=seg_risk["churn_pct"],
        y=seg_risk["segmento"],
        orientation="h",
        marker=dict(color=seg_risk["color"]),
        width=0.32,
        text=seg_risk["churn_pct"].round(1).astype(str) + "%",
        textposition="outside",
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>Score churn promedio: %{x:.1f}%<extra></extra>"
    ))


    fig3.update_layout(
        height=310,
        barmode="overlay",
        margin=dict(l=10, r=55, t=20, b=10),
        xaxis=dict(
            title="",
            range=[0, 105],
            gridcolor="#e5e7eb",
            zeroline=False,
            showgrid=False,
            showticklabels=False,
        ),
        yaxis=dict(
            title="",
            automargin=True
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # =========================
    # Mostrar gráficos en cards nativas
    # =========================
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.10, 1.05])

    with c1:
        with st.container(border=True):
            st.subheader("Distribución de churn")
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown(f"""
            <div class="chart-note">
                <i class="bi bi-info-circle"></i>
                <span>El {churn_rate:.1f}% de los clientes presenta riesgo de churn.</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        segmento_top = seg_counts.iloc[0]["segmento"]
        clientes_top = seg_counts.iloc[0]["clientes"]
        pct_top = clientes_top / total_clientes * 100

        with st.container(border=True):
            st.subheader("Clientes por segmento")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(f"""
            <div class="chart-note">
                <i class="bi bi-bar-chart"></i>
                <span>El segmento {segmento_top} concentra el {pct_top:.1f}% de la base.</span>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        # Obtener el segmento con mayor riesgo (último después de ordenar ascendente)
        segmento_riesgo_top = churn_segmento.iloc[-1]["segmento"]
        riesgo_top = churn_segmento.iloc[-1]["churn_pct"]

        with st.container(border=True):
            st.subheader("Churn por segmento")
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown(f"""
            <div class="chart-note chart-note-success">
                <i class="bi bi-stars"></i>
                <span>{segmento_riesgo_top} presenta el mayor porcentaje de churn ({riesgo_top:.1f}%). Prioridad alta para retención.</span>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # Vista general
    # =========================
    st.subheader("Vista general de clientes")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2.2, 1.2, 1.2, 0.8])

    with filter_col1:
        search = st.text_input(
            "Buscar cliente",
            placeholder="ID cliente..."
        )

    with filter_col2:
        segmento_filter = st.selectbox(
            "Segmento",
            ["Todos"] + sorted(df["segmento"].dropna().unique().tolist())
        )

    with filter_col3:
        riesgo_filter = st.selectbox(
            "Riesgo",
            ["Todos", "Crítico", "Alto", "Medio", "Bajo"]
        )

    df_view = df.copy()

    if search:
        df_view = df_view[
            df_view["cliente_id_new"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    if segmento_filter != "Todos":
        df_view = df_view[df_view["segmento"] == segmento_filter]

    if riesgo_filter != "Todos":
        df_view = df_view[
            df_view["score_churn"]
            .apply(asignar_nivel_riesgo_score)
            .eq(riesgo_filter)
        ]

    with filter_col4:
        st.markdown(
            f"""
            <div class="filter-count">
                <div>{len(df_view):,}</div>
                <span>clientes</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    cols_show = [
        "cliente_id_new",
        "recency",
        "frequency",
        "monetary",
        "churn",
        "score_churn",
        "pred_churn",
        "segmento"
    ]

    cols_show = [c for c in cols_show if c in df_view.columns]


    # =========================
    # Header tabla
    # =========================

    html_table = """
    <div class="customers-table-card">
        <div class="customers-table-header">
            <div>Cliente</div>
            <div>Segmento</div>
            <div>Frecuencia</div>
            <div>Monto total</div>
            <div>Recency</div>
            <div>Riesgo</div>
            <div>Score churn</div>
        </div>
    """
    # =========================
    # Rows
    # =========================

    for _, row in df_view.head(20).iterrows():
        cliente_id = html.escape(str(row["cliente_id_new"]))
        segmento = segmento_badge(row["segmento"])
        riesgo = riesgo_badge(row["score_churn"])
        score = score_bar(row["score_churn"])

        html_table += f"""
      <div class="customers-table-row">
        <div class="customer-id">{cliente_id}</div>
        <div>{segmento}</div>
        <div>{int(row["frequency"])}</div>
        <div>S/ {float(row["monetary"]):,.2f}</div>
        <div>{int(row["recency"])} días</div>
        <div>{riesgo}</div>
        <div>{score}</div>
      </div>
    """
    html_table += "</div>"

    st.markdown(html_table, unsafe_allow_html=True)

# -------------------------
# Buscar cliente
# -------------------------
elif page == "buscar":

    st.markdown(
        """
    <div class="search-hero animated-card">
    <div>
        <div class="search-kicker">CONSULTA INDIVIDUAL</div>
        <div class="search-title">Análisis de retención por cliente</div>
        <div class="search-subtitle">
        Visualizacion de perfil, riesgo de churn y estrategia personalizada.
        </div>
    </div>
    </div>
    """,
        unsafe_allow_html=True
    )

    cliente_id = st.text_input(
        label="Ingresa el ID del cliente",
        placeholder="Buscar cliente por ID. Ejemplo: 1027461"
    )

    if cliente_id:
        result = df[
            df["cliente_id_new"]
            .astype(str)
            .str.upper()
            .eq(cliente_id.strip().upper())
        ]

        if result.empty:
            st.warning("Cliente no encontrado")
        else:
            cliente = result.iloc[0]

            mostrar_metricas_cliente(cliente)
            mostrar_riesgo_y_segmento(cliente)
            mostrar_estrategia_cliente(cliente)
# -------------------------
# Subir archivo
# -------------------------
    
elif page == "upload":

    st.markdown('<div class="main-title">Carga masiva</div>', unsafe_allow_html=True)

    files = st.file_uploader(
        "Sube uno o más archivos CSV o Excel",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="uploader"
    )

    if files:
        st.markdown('<i class="bi bi-files"></i> Archivos cargados:', unsafe_allow_html=True)
        for f in files:
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;<i class="bi bi-file-earmark-spreadsheet"></i> {f.name}', unsafe_allow_html=True)

        # Detectar cambio de archivos (resetea estado)
        if "ultimos_archivos" not in st.session_state:
            st.session_state.ultimos_archivos = None
            st.session_state.procesando_upload = False

        nombres_archivos = tuple(f.name for f in files)
        if st.session_state.ultimos_archivos != nombres_archivos:
            st.session_state.ultimos_archivos = nombres_archivos
            st.session_state.procesando_upload = False

        # Botón con estado
        st.markdown('<div class="btn-upload">', unsafe_allow_html=True)
        boton_clicked = st.button(
            "Procesar datos",
            type="primary",
            icon=":material/play_arrow:",
            disabled=st.session_state.procesando_upload
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Cuando se hace clic, se inicia el procesamiento
        if boton_clicked and not st.session_state.procesando_upload:
            st.session_state.procesando_upload = True
            st.rerun()  # Solo este rerun es necesario para deshabilitar el botón

        # Procesar SOLO si está en estado "procesando"
        if st.session_state.procesando_upload:
            error_ocurrio = False
            try:
                # 1. Unir archivos
                with st.spinner("Uniendo archivos..."):
                    dfs = []
                    for file in files:
                        if file.name.endswith(".csv"):
                            df_temp = pd.read_csv(file)
                        else:
                            df_temp = pd.read_excel(file)
                        dfs.append(df_temp)
                    df_raw = pd.concat(dfs, ignore_index=True)
                toast_html("Datos unificados correctamente", "bi-files", icon_color="#22c55e")

                with st.expander("Vista previa de los datos unificados"):
                    st.dataframe(df_raw.head(10))

                # 2. Preprocesamiento
                with st.spinner("Preprocesando datos (RFM)..."):
                    df_proc = preprocess_data(df_raw)
                toast_html("Preprocesamiento completado", "bi-arrow-repeat", icon_color="#22c55e")

                # 3. Predicción
                with st.spinner("Prediciendo churn con XGBoost..."):
                    df_pred = predict_churn(df_proc)
                toast_html("Predicciones generadas", "bi-robot", icon_color="#22c55e")

                # 4. Segmentación
                with st.spinner("Segmentando clientes con KMeans..."):
                    df_seg = segment_clients(df_pred)
                toast_html("Segmentación completada", "bi-diagram-3", icon_color="#22c55e")

                # 5. Estrategias
                with st.spinner("Generando estrategias con IA (puede tardar)..."):
                    df_final = df_seg.copy()
                    df_final["estrategia"] = df_final.apply(generar_estrategia, axis=1)
                toast_html("Estrategias generadas", "bi-brain", icon_color="#22c55e")

                toast_html("Pipeline ejecutado correctamente", "bi-check-circle-fill", icon_color="#22c55e")
            
                st.subheader("Resultados")
                st.dataframe(df_final.head(30))

                st.download_button(
                    "Descargar resultados (CSV)",
                    df_final.to_csv(index=False),
                    "clientes_churn_final.csv",
                    mime="text/csv",
                    icon=":material/download:"
                )

            except Exception as e:
                error_ocurrio = True
                st.error(f"Error en el procesamiento: {e}")

            finally:
                if not error_ocurrio:
                    st.session_state.procesando_upload = False
                else:
                    st.session_state.procesando_upload = False
                    st.rerun()  # Solo si hubo error para limpiar

# -------------------------
# Acerca de
# -------------------------

elif page == "about":
    st.markdown('<div class="main-title">Acerca del proyecto</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2.2, 1.8])

    with col1:
        st.markdown("""
        ### <i class="bi bi-bullseye" style="color: #635bff;"></i> Objetivo

        Sistema híbrido que combina **Machine Learning** (XGBoost) e **IA Generativa** (Gemini + Hugging Face) para:

        - <i class="bi bi-graph-up"></i> Predecir clientes con riesgo de churn (abandono)
        - <i class="bi bi-diagram-3"></i> Segmentar automáticamente en 4 perfiles comerciales
        - <i class="bi bi-magic"></i> Generar estrategias personalizadas de retención

        ---

        ### <i class="bi bi-database" style="color: #635bff;"></i> Dataset

        - <i class="bi bi-file-earmark-excel"></i> **3 archivos Excel** de una empresa retail
        - <i class="bi bi-bar-chart-steps"></i> **+4 millones** de transacciones (2022-2025)
        - <i class="bi bi-people"></i> **80,355 clientes** únicos analizados

        ---

        ### <i class="bi bi-cpu" style="color: #635bff;"></i> Modelos utilizados

        | Componente | Tecnología | Métrica clave |
        |------------|------------|---------------|
        | <i class="bi bi-robot"></i> Predicción churn | XGBoost | ROC-AUC: **0.869** |
        | <i class="bi bi-funnel"></i> Segmentación | KMeans | **4 clusters** interpretables |
        | <i class="bi bi-stars"></i> Estrategias | Gemini + Hugging Face | Reglas de negocio + IA |

        ---

        ### <i class="bi bi-person-workspace" style="color: #635bff;"></i> Equipo

        **<i class="bi bi-person-plus-fill"></i> Trabajo en equipo:**
        - Amalia Anto Alzamora
        - Leticia Verano Custodio
        ---

        ### <i class="bi bi-tools" style="color: #635bff;"></i> Tecnologías

        <span class="badge-tech">Python</span>
        <span class="badge-tech">Pandas</span>
        <span class="badge-tech">Scikit-learn</span>
        <span class="badge-tech">XGBoost</span>
        <span class="badge-tech">KMeans</span>
        <span class="badge-tech">Streamlit</span>
        <span class="badge-tech">Plotly</span>
        <span class="badge-tech">Gemini API</span>
        <span class="badge-tech">Hugging Face</span>

        ---

        ### <i class="bi bi-github" style="color: #635bff;"></i> Repositorio

        [GitHub - Proyecto Integrador MDS](https://github.com/aantoa/churn-retail-mvp)
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="client-profile-card animated-card ui-purple-mvp" style="padding: 28px; text-align: center;">
            <div style="margin-bottom: 20px;">
                <i class="bi bi-trophy" style="font-size: 52px; color: #635bff;"></i>
            </div>
            <div style="font-size: 24px; font-weight: 900; color: #0f172a; margin-bottom: 8px;">MVP</div>
            <div style="display: inline-block; background: #f3f0ff; color: #635bff; padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 700; margin-bottom: 20px;">Proyecto Integrador</div>
            <div style="text-align: left; font-size: 14px; color: #334155; line-height: 1.6;">
                <strong><i class="bi bi-backpack"></i> Maestría en Data Science</strong><br>
                Universidad Nacional de Ingeniería<br><br>
                <strong><i class="bi bi-file-text"></i> Entregable:</strong><br>
                - Bitácora<br>
                - MVP funcional<br>
                - Pipeline completo<br>
                - IA Generativa aplicada
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Métricas adicionales
    st.markdown("---")
    st.subheader("Estadísticas del modelo")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("ROC-AUC", "0.869", delta="+2.3% vs baseline LR")
    with col_b:
        st.metric("Recall (churn)", "90%", delta="Alta sensibilidad")
    with col_c:
        st.metric("Segmentos", "4", delta="Interpretables")
    with col_d:
        st.metric("LLM usado", "Gemini + HF", delta="2 modelos")