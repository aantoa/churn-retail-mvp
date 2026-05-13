import html
from src.llm.strategy import asignar_nivel_riesgo_score
import streamlit as st
import time

# =========================================
# Colores globales segmentos
# =========================================

SEGMENT_COLORS = {
    "Estratégicos": "#635bff",
    "Recurrentes": "#2563eb",
    "Promocionales": "#f97316",
    "En riesgo": "#dc2626"
}

# =========================================
# Canal UI
# =========================================

def canal_ui(canal):

    canal = str(canal).strip()

    config = {
        "WhatsApp": {
            "icon": "bi-whatsapp",
            "class": "ui-green",
            "desc": "Contacto rápido y directo."
        },

        "Teléfono": {
            "icon": "bi-telephone-outbound",
            "class": "ui-orange",
            "desc": "Contacto personalizado de alta prioridad."
        },

        "Email": {
            "icon": "bi-envelope-paper",
            "class": "ui-blue",
            "desc": "Comunicación formal y trazable."
        },

        "Visita comercial": {
            "icon": "bi-person-walking",
            "class": "ui-purple",
            "desc": "Intervención presencial para clientes clave."
        }
    }

    return config.get(canal, {
        "icon": "bi-chat-dots",
        "class": "ui-slate",
        "desc": "Canal sugerido por reglas comerciales."
    })

# =========================================
# Prioridad UI
# =========================================

def prioridad_ui(prioridad):

    prioridad = str(prioridad)

    if "Crítica" in prioridad:
        return {
            "class": "ui-red",
            "label": "Prioridad crítica",
            "icon": "bi-exclamation-triangle-fill"
        }

    if "Alta" in prioridad:
        return {
            "class": "ui-orange",
            "label": "Prioridad alta",
            "icon": "bi-lightning-charge-fill"
        }

    if "Media" in prioridad:
        return {
            "class": "ui-blue",
            "label": "Prioridad media",
            "icon": "bi-activity"
        }

    return {
        "class": "ui-green",
        "label": "Prioridad baja",
        "icon": "bi-check-circle-fill"
    }

# =========================================
# Segmento UI
# =========================================

def segmento_ui(segmento):

    segmento = str(segmento).strip()

    config = {
        "Estratégicos": {
            "class": "ui-purple",
            "icon": "bi-gem"
        },

        "Recurrentes": {
            "class": "ui-blue",
            "icon": "bi-arrow-repeat"
        },

        "Promocionales": {
            "class": "ui-orange",
            "icon": "bi-ticket-perforated"
        },

        "En riesgo": {
            "class": "ui-red",
            "icon": "bi-exclamation-triangle-fill"
        }
    }

    return config.get(segmento, {
        "class": "ui-slate",
        "icon": "bi-diagram-3"
    })

# =========================================
# Riesgo badge
# =========================================

def riesgo_badge(score):

    nivel = asignar_nivel_riesgo_score(score)

    badge_map = {
        "Crítico": "badge-critical",
        "Alto": "badge-high",
        "Medio": "badge-medium",
        "Bajo": "badge-low"
    }

    css_class = badge_map.get(nivel, "badge-low")

    return f'<span class="{css_class}">{nivel}</span>'

# =========================================
# Score bar UI
# =========================================

def score_bar(score):

    score = float(score)
    pct = score * 100

    if score >= 0.75:
        color = "#dc2626"

    elif score >= 0.50:
        color = "#f97316"

    elif score >= 0.30:
        color = "#2563eb"

    else:
        color = "#16a34a"

    return f"""
    <div class="score-cell">
        <div class="score-track">
            <div class="score-fill"
                 style="width:{pct:.1f}%; background:{color};">
            </div>
        </div>
        <span class="score-label">{pct:.1f}%</span>
    </div>
    """

# =========================================
# Segment badge
# =========================================

def segmento_badge(segmento):

    cfg = segmento_ui(segmento)

    return f"""
    <span class="segment-badge {cfg["class"]}">
        <i class="bi {cfg["icon"]}"></i>
        {html.escape(str(segmento))}
    </span>
    """


# =========================================
# Mensaje con toast UI
# =========================================

def toast_html(message, icon_class, bg_color="#fcfcfc", icon_color="#22c55e"):
    """Muestra un mensaje temporal con icono de Bootstrap"""
    container = st.empty()
    container.markdown(f"""
        <div style="position: fixed; top: 80px; right: 20px; background-color: {bg_color};
                    color: #1e293b; padding: 16px 24px; border-radius: 16px;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.2); z-index: 9999;
                    font-family: system-ui, -apple-system, sans-serif; font-size: 16px;
                    font-weight: 500; backdrop-filter: blur(8px);">
            <i class="bi {icon_class}" style="color: {icon_color}; margin-right: 12px; font-size: 20px;"></i>
            {message}
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    container.empty()