from pathlib import Path
import os
import time
import json
import ast
import re
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
import google.generativeai as genai


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = ROOT / "data" / "processed" / "estrategias_clientes.csv"

load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


PRIORIDADES_VALIDAS = [
    "Baja - Mantener relación",
    "Media - Monitorear comportamiento",
    "Media - Incentivar recompra",
    "Alta - Contacto personalizado",
    "Crítica - Recuperación inmediata"
]

CANALES_VALIDOS = [
    "WhatsApp",
    "Teléfono",
    "Email",
    "Visita comercial"
]

SEGMENTOS_VALIDOS = [
    "Estratégicos",
    "Recurrentes",
    "Promocionales",
    "En riesgo"
]


client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    provider="featherless-ai",
    token=HF_TOKEN
)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None


def safe_value(row, col, default="No disponible"):
    return row[col] if col in row.index and pd.notna(row[col]) else default


def formato_monto(valor):
    if valor == "No disponible":
        return valor
    return f"S/ {float(valor):,.2f}"


def formato_pct(valor):
    if valor == "No disponible":
        return valor

    valor = float(valor)

    if abs(valor) <= 1:
        return f"{valor:.1%}"

    return f"{valor:.1f}%"


def asignar_nivel_riesgo_score(score):
    score = float(score)

    if score >= 0.75:
        return "Crítico"
    elif score >= 0.50:
        return "Alto"
    elif score >= 0.30:
        return "Medio"
    else:
        return "Bajo"


def asignar_prioridad_prompt(row):
    score = float(safe_value(row, "score_churn", safe_value(row, "churn", 0)))
    segmento = safe_value(row, "segmento", "")

    if score >= 0.75:
        return "Crítica - Recuperación inmediata"

    elif score >= 0.50:
        if segmento in ["Estratégicos", "En riesgo"]:
            return "Alta - Contacto personalizado"
        return "Media - Incentivar recompra"

    elif score >= 0.30:
        if segmento == "Promocionales":
            return "Media - Incentivar recompra"
        return "Media - Monitorear comportamiento"

    else:
        return "Baja - Mantener relación"


def sugerir_canal(prioridad):
    if prioridad == "Crítica - Recuperación inmediata":
        return "Teléfono"
    elif prioridad == "Alta - Contacto personalizado":
        return "WhatsApp"
    elif prioridad in ["Media - Monitorear comportamiento", "Media - Incentivar recompra"]:
        return "WhatsApp"
    else:
        return "Email"


def parse_estrategia(estrategia_raw):
    texto = str(estrategia_raw).strip()
    texto = texto.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        texto = match.group(0)

    try:
        return json.loads(texto)
    except Exception:
        try:
            return ast.literal_eval(texto)
        except Exception:
            return None


def estrategia_valida(data):
    required_keys = {
        "analisis_breve",
        "prioridad",
        "canal_sugerido",
        "incentivo_recomendado",
        "estrategias"
    }

    if not isinstance(data, dict):
        return False

    if set(data.keys()) != required_keys:
        return False

    if data["prioridad"] not in PRIORIDADES_VALIDAS:
        return False

    if data["canal_sugerido"] not in CANALES_VALIDOS:
        return False

    estrategias = data.get("estrategias")

    if not isinstance(estrategias, dict):
        return False

    acciones_requeridas = {
        "accion_1",
        "accion_2",
        "accion_3",
        "accion_4"
    }

    if set(estrategias.keys()) != acciones_requeridas:
        return False

    valores = [
        str(estrategias[k]).strip()
        for k in ["accion_1", "accion_2", "accion_3", "accion_4"]
    ]

    if any(not v for v in valores):
        return False

    if len(set(v.lower() for v in valores)) != 4:
        return False

    texto_total = json.dumps(data, ensure_ascii=False).lower()

    palabras_invalidas = [
        "another strategy",
        "could be effective",
        "customer",
        "shopping",
        "business",
        "pending",
        "pendiente",
        "<div",
        "</div>",
        "<br",
        "revisar el perfil del cliente"
    ]

    if any(p in texto_total for p in palabras_invalidas):
        return False

    return True


def generate_strategy_hf(prompt, max_retries=1, wait_seconds=3):
    last_error = None

    for intento in range(1, max_retries + 1):
        try:
            response = client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.4
            )

            return response.choices[0].message["content"]

        except HfHubHTTPError as e:
            last_error = e
            print(f"Hugging Face falló en intento {intento}: {e}")
            time.sleep(wait_seconds)

        except Exception as e:
            last_error = e
            print(f"Error inesperado en Hugging Face intento {intento}: {e}")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Hugging Face no respondió correctamente: {last_error}")


def generate_strategy_gemini(prompt):
    if gemini_model is None:
        raise ValueError("Gemini API Key no configurada.")

    response = gemini_model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 800,
            "response_mime_type": "application/json"
        }
    )

    return response.text


def generate_strategy_llm(prompt):
    try:
        return generate_strategy_hf(prompt)
    except Exception as e:
        print(f"Hugging Face no disponible. Se usará Gemini. Error: {e}")

    try:
        return generate_strategy_gemini(prompt)
    except Exception as e:
        print(f"Gemini también falló. Error: {e}")

    return None


def build_customer_context(row):
    score = safe_value(row, "score_churn", safe_value(row, "churn", 0))
    pred = safe_value(row, "pred_churn", safe_value(row, "churn", 0))

    prioridad = safe_value(row, "prioridad_prompt", None)
    canal = safe_value(row, "canal_prompt", None)
    nivel = safe_value(row, "nivel_riesgo_prompt", None)

    if prioridad is None:
        prioridad = asignar_prioridad_prompt(row)

    if canal is None:
        canal = sugerir_canal(prioridad)

    if nivel is None:
        nivel = asignar_nivel_riesgo_score(score)

    return f"""
Cliente con las siguientes características comerciales:
- ID cliente: {safe_value(row, "cliente_id_new")}
- Segmento comercial: {safe_value(row, "segmento")}
- Probabilidad estimada de churn: {formato_pct(score)}
- Predicción de churn del modelo: {"Sí" if int(pred) == 1 else "No"}
- Nivel de riesgo asignado: {nivel}
- Prioridad sugerida por reglas internas: {prioridad}
- Canal preliminar sugerido: {canal}
- Días desde la última compra: {safe_value(row, "recency")}
- Frecuencia total de compra: {safe_value(row, "frequency")}
- Monto total comprado: {formato_monto(safe_value(row, "monetary"))}
- Ticket promedio: {formato_monto(safe_value(row, "avg_ticket"))}
- Compras en los últimos 70 días: {safe_value(row, "compras_70d")}
- Gasto en los últimos 70 días: {formato_monto(safe_value(row, "gasto_70d"))}
- Tendencia reciente de gasto: {safe_value(row, "spend_trend_70")}
- Descuento promedio aplicado: {formato_pct(safe_value(row, "descuento_porc_promedio"))}
- Margen promedio: {formato_pct(safe_value(row, "margen_pct"))}
- Cliente con crédito: {"Sí" if int(safe_value(row, "flag_credito", 0)) == 1 else "No"}
""".strip()


def build_prompt(context):
    return f"""
Actúa como un Gerente Senior de Retención de Clientes en una empresa retail.

Tu objetivo es crear una estrategia personalizada de retención para un cliente, usando su comportamiento de compra, segmento y probabilidad de churn.

CONTEXTO DEL CLIENTE:
{context}

REGLAS DE NEGOCIO:

Los segmentos de cliente pueden ser:
1. Estratégicos: clientes de alto valor y relevancia para el negocio.
2. Recurrentes: clientes que compran con frecuencia pero pueden estar reduciendo su actividad.
3. Promocionales: clientes sensibles a descuentos o promociones.
4. En riesgo: clientes con alta probabilidad de abandono.

Los niveles de prioridad son:
1. Baja - Mantener relación
2. Media - Monitorear comportamiento
3. Media - Incentivar recompra
4. Alta - Contacto personalizado
5. Crítica - Recuperación inmediata

INTERPRETACIÓN OBLIGATORIA:

- Si la probabilidad de churn es mayor o igual a 0.75, usa prioridad "Crítica - Recuperación inmediata".
- Si la probabilidad de churn está entre 0.50 y 0.75, usa prioridad "Alta - Contacto personalizado".
- Si la probabilidad de churn está entre 0.30 y 0.50, usa prioridad "Media - Incentivar recompra".
- Si la probabilidad de churn es menor a 0.30, usa prioridad "Baja - Mantener relación".

- Si el cliente es Estratégico, prioriza retención personalizada y beneficios exclusivos.
- Si es Recurrente, enfócate en reactivar frecuencia de compra.
- Si es Promocional, usa incentivos económicos concretos.
- Si está En riesgo, prioriza acciones inmediatas de recuperación.

- El canal debe ser coherente con la urgencia:
  - Alta o Crítica: Teléfono o WhatsApp
  - Media: WhatsApp o Email
  - Baja: Email

SALIDA REQUERIDA:

Devuelve SOLAMENTE un objeto JSON válido, sin markdown, sin HTML y sin texto adicional.

El JSON debe tener EXACTAMENTE esta estructura:

{{
  "analisis_breve": "Una frase breve explicando el comportamiento del cliente sin repetir números literalmente.",
  "prioridad": "Baja - Mantener relación | Media - Monitorear comportamiento | Media - Incentivar recompra | Alta - Contacto personalizado | Crítica - Recuperación inmediata",
  "canal_sugerido": "WhatsApp | Teléfono | Email | Visita comercial",
  "incentivo_recomendado": "Ejemplo concreto de incentivo comercial",
  "estrategias": {{
    "accion_1": "Acción concreta para la semana 1.",
    "accion_2": "Acción concreta para la semana 2.",
    "accion_3": "Acción concreta para la semana 3.",
    "accion_4": "Acción concreta para la semana 4."
  }}
}}

REGLAS OBLIGATORIAS:

- Todo debe estar en español.
- No uses inglés.
- No uses HTML.
- No uses markdown.
- No generes texto fuera del JSON.
- Deben existir exactamente accion_1, accion_2, accion_3 y accion_4.
- Las acciones deben ser distintas entre sí.
- Evita frases genéricas como "revisar el perfil del cliente".
- No inventes datos que no estén en el contexto.
- Las cuatro acciones deben formar un plan de retención de 30 días.
- accion_1 corresponde a la semana 1.
- accion_2 corresponde a la semana 2.
- accion_3 corresponde a la semana 3.
- accion_4 corresponde a la semana 4.
"""


def generar_estrategia_validada(prompt, max_intentos=2):
    last_error = None

    for _ in range(max_intentos + 1):
        estrategia_raw = generate_strategy_llm(prompt)

        if estrategia_raw is None:
            raise ValueError("No se pudo generar respuesta con Hugging Face ni Gemini.")

        estrategia_json = parse_estrategia(estrategia_raw)

        if estrategia_valida(estrategia_json):
            return json.dumps(estrategia_json, ensure_ascii=False)

        last_error = estrategia_raw

        prompt = prompt + """

La respuesta anterior no cumplió las reglas.
Vuelve a responder SOLO con JSON válido.
Debe respetar exactamente esta estructura: analisis_breve, prioridad, canal_sugerido, incentivo_recomendado y estrategias.
El campo estrategias debe tener exactamente accion_1, accion_2, accion_3 y accion_4.
Todo debe estar en español.
No uses inglés.
No uses HTML.
No uses markdown.
No uses placeholders.
"""

    raise ValueError(f"No se pudo generar una estrategia válida. Última respuesta: {last_error}")


def cargar_estrategias():
    if STRATEGY_PATH.exists():
        return pd.read_csv(STRATEGY_PATH)

    return pd.DataFrame(columns=["cliente_id_new", "estrategia"])


def guardar_estrategia(cliente_id, estrategia):
    df_hist = cargar_estrategias()

    nueva = pd.DataFrame([
        {
            "cliente_id_new": str(cliente_id),
            "estrategia": estrategia
        }
    ])

    df_hist = pd.concat([df_hist, nueva], ignore_index=True)

    df_hist = df_hist.drop_duplicates(
        subset=["cliente_id_new"],
        keep="last"
    )

    df_hist.to_csv(STRATEGY_PATH, index=False)


def normalizar_estrategia_con_reglas(row, estrategia_json):
    prioridad_regla = asignar_prioridad_prompt(row)
    canal_regla = sugerir_canal(prioridad_regla)

    estrategia_json["prioridad"] = prioridad_regla
    estrategia_json["canal_sugerido"] = canal_regla

    return estrategia_json


def generar_estrategia(row):
    cliente_id = str(row["cliente_id_new"])

    df_hist = cargar_estrategias()

    existente = df_hist[
        df_hist["cliente_id_new"].astype(str) == cliente_id
    ]

    if not existente.empty:
        estrategia_guardada = existente.iloc[0]["estrategia"]
        estrategia_json = parse_estrategia(estrategia_guardada)

        if estrategia_valida(estrategia_json):
            estrategia_json = normalizar_estrategia_con_reglas(row, estrategia_json)
            return json.dumps(estrategia_json, ensure_ascii=False)

    context = build_customer_context(row)
    prompt = build_prompt(context)

    estrategia = generar_estrategia_validada(prompt)
    estrategia_json = parse_estrategia(estrategia)

    estrategia_json = normalizar_estrategia_con_reglas(row, estrategia_json)

    estrategia = json.dumps(estrategia_json, ensure_ascii=False)

    guardar_estrategia(cliente_id, estrategia)

    return estrategia