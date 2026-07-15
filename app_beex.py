"""
BEEX - Clasificador de Intenciones de Clientes
Demo funcional del modelo ML (APF3) integrado con interfaz Streamlit
Curso: Innovación y Transformación Digital — UTP
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import os

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
st.set_page_config(
    page_title="BEEX - Clasificador de Intenciones",
    page_icon="🎯",
    layout="centered"
)

# ==============================================================================
# CARGA DEL MODELO ENTRENADO
# ==============================================================================
@st.cache_resource
def cargar_modelo():
    modelo_path = os.path.join(os.path.dirname(__file__), "modelo_beex_produccion.pkl")
    return joblib.load(modelo_path)

modelo = cargar_modelo()

# ==============================================================================
# DICCIONARIOS DE INTERPRETACIÓN DE NEGOCIO
# ==============================================================================
INTERPRETACION = {
    "Soporte": {
        "icono": "🔧",
        "color": "#3498db",
        "descripcion": "El cliente presenta una incidencia técnica (conectividad, acceso, fallas de sistema).",
        "accion_recomendada": "Enrutar a Cola de Soporte Técnico — Nivel 1.",
        "sla_sugerido": "Primera respuesta en ≤ 2 minutos.",
        "prioridad": "Normal"
    },
    "Facturacion": {
        "icono": "💰",
        "color": "#f39c12",
        "descripcion": "El cliente tiene una consulta o reclamo relacionado con cobros, facturas o pagos.",
        "accion_recomendada": "Enrutar a Cola de Facturación — Verificar estado de cuenta antes de responder.",
        "sla_sugerido": "Primera respuesta en ≤ 3 minutos.",
        "prioridad": "Normal"
    },
    "Ventas": {
        "icono": "🛒",
        "color": "#2ecc71",
        "descripcion": "El cliente muestra interés comercial en contratar, renovar o adquirir un servicio.",
        "accion_recomendada": "Enrutar a Cola de Ventas — Preparar cotización y catálogo de planes.",
        "sla_sugerido": "Primera respuesta en ≤ 1 minuto (oportunidad de venta).",
        "prioridad": "Alta (oportunidad comercial)"
    },
    "Reclamos": {
        "icono": "⚠️",
        "color": "#e74c3c",
        "descripcion": "El cliente expresa insatisfacción severa, solicita baja, denuncia o escalamiento.",
        "accion_recomendada": "Enrutar a Cola de Reclamos — PRIORIDAD MÁXIMA. Notificar a supervisor.",
        "sla_sugerido": "Contención inmediata en ≤ 30 segundos.",
        "prioridad": "CRÍTICA"
    }
}

# ==============================================================================
# FUNCIONES DE PROCESAMIENTO
# ==============================================================================
def detectar_urgencia(texto):
    """Detecta palabras clave de urgencia en el mensaje del cliente."""
    palabras_urgencia = [
        'urgente', 'ya', 'harto', 'basura', 'cancelar', 'queja',
        'denuncia', 'robo', 'estafado', 'supervisor', 'baja',
        'exijo', 'denunciar', 'inmediato', 'ahora'
    ]
    return 1 if any(w in texto.lower() for w in palabras_urgencia) else 0

def predecir_intencion(texto):
    """Clasifica la intención del cliente usando el modelo SVM entrenado."""
    texto_limpio = texto.strip().lower()
    prediccion = modelo.predict([texto_limpio])[0]
    return prediccion

# ==============================================================================
# INTERFAZ DE USUARIO
# ==============================================================================
st.title("🎯 BEEX — Clasificador de Intenciones de Clientes")
st.markdown("*Motor de Inteligencia Conversacional Centralizada con IA Generativa*")
st.markdown("---")

# --- Panel lateral con información del modelo ---
with st.sidebar:
    st.header("📊 Información del Modelo")
    st.markdown("""
    **Algoritmo:** SVM (RBF Kernel)

    **Vectorizer:** TF-IDF (unigramas + bigramas)

    **F1-Score ponderado:** 94.60%

    **Clases:** Soporte, Facturación,
    Ventas, Reclamos

    **Dataset:** 1,500 interacciones
    sintéticas omnicanal
    """)
    st.markdown("---")
    st.markdown("**Fuente:** APF3 — Proyecto Final")
    st.markdown("**Empresa:** Beex Contact Center")

# --- Entrada de datos ---
st.subheader("💬 Ingrese la interacción del cliente")
st.markdown("*Escriba o pegue el mensaje/texto del cliente (chat de WhatsApp, transcripción de llamada, webchat, etc.)*")

# Ejemplos predefinidos
ejemplos = {
    "Seleccionar ejemplo...": "",
    "🔧 Soporte técnico": "mi internet no funciona pucha, se me cayó la red otra vez y no puedo ingresar al sistema",
    "💰 Consulta de facturación": "quiero pagar mi recibo pero me sale error, cuánto es mi deuda este mes",
    "🛒 Interés en ventas": "buenos días, me interesa contratar el servicio de fibra óptica, cuáles son los planes",
    "⚠️ Reclamo urgente": "EXIJO hablar con un supervisor, me cobraron doble y llevo 3 días sin solución, esto es una basura"
}

ejemplo_seleccionado = st.selectbox("O seleccione un ejemplo:", list(ejemplos.keys()))

texto_input = ""
if ejemplo_seleccionado != "Seleccionar ejemplo...":
    texto_input = ejemplos[ejemplo_seleccionado]

texto_usuario = st.text_area(
    "Mensaje del cliente:",
    value=texto_input,
    height=120,
    placeholder="Ej: 'mi internet no funciona, necesito ayuda urgente'"
)

# --- Botón de clasificación ---
if st.button("🔍 Clasificar Intención", type="primary", use_container_width=True):
    if texto_usuario.strip():
        # Predecir
        intencion = predecir_intencion(texto_usuario)
        info = INTERPRETACION[intencion]
        urgencia = detectar_urgencia(texto_usuario)

        st.markdown("---")
        st.subheader("📋 Resultado de la Clasificación")

        # Resultado principal
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"### {info['icono']} {intencion}")
        with col2:
            st.markdown(f"**Descripción:** {info['descripcion']}")

        # Detalles de enrutamiento
        st.markdown("#### 🎯 Acción de Enrutamiento Recomendada")
        st.info(info["accion_recomendada"])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("⏱️ SLA Sugerido", info["sla_sugerido"])
        with col_b:
            st.metric("🚦 Prioridad", info["prioridad"])
        with col_c:
            st.metric("🚨 Bandera de Urgencia", "ACTIVADA" if urgencia else "Normal")

        if urgencia:
            st.warning("⚡ **ALERTA DE URGENCIA:** Se detectaron términos de alta fricción en el mensaje. "
                       "Se recomienda escalamiento inmediato o activación de protocolo de retención (Fast-Track).")

        # Métricas del modelo
        st.markdown("---")
        st.subheader("📈 Métricas del Modelo (APF3)")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Accuracy", "94.56%")
        with col_m2:
            st.metric("Precision (W)", "94.77%")
        with col_m3:
            st.metric("Recall (W)", "94.56%")
        with col_m4:
            st.metric("F1-Score (W)", "94.60%")

    else:
        st.warning("⚠️ Por favor, ingrese un mensaje del cliente para clasificar.")

# --- Pie de página ---
st.markdown("---")
st.markdown(
    "*Proyecto Final — Curso: Innovación y Transformación Digital | "
    "Universidad Tecnológica del Perú | BEEX Contact Center*",
    help="Modelo SVM entrenado con TF-IDF (unigramas + bigramas). "
         "Dataset sintético de 1,500 interacciones simuladas."
)
