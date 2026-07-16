"""
BEEX — Suite de Inteligencia Conversacional para Contact Center
Producto Final (PROY) — Implementación de HU-01 a HU-08
Curso: Innovación y Transformación Digital — UTP
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import os
import random
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="BEEX — Suite de Inteligencia Conversacional",
    page_icon="beex",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS PERSONALIZADO
# ==============================================================================
st.markdown("""
<style>
    /* Paleta corporativa y temas dinámicos */
    :root {
        --beex-blue:   #1a56db;
        --beex-green:  #0e9f6e;
        --beex-orange: #f59e0b;
        --beex-red:    #e02424;
        --beex-gray:   #6b7280;

        /* Light Mode (Sketch) */
        --c-bg: #ffffff;
        --c-border: #e5e7eb;
        --c-text: #1a1a1a;
        --c-text-muted: #6b7280;
        --chat-bg: #f9f8f5;
        --chat-border: #c8c7bc;
        --bubble-in: #f0efe9;
        --bubble-out: #e0dfd8;
        --input-bg: #f9f8f5;
        --btn-sec-bg: #ffffff;
        --btn-sec-border: #999999;
        
        /* Cajas de Información */
        --copilot-bg: rgba(14, 159, 110, 0.08);
        --copilot-border: #0e9f6e;
        --faq-bg: rgba(26, 86, 219, 0.08);
        --faq-border: #1a56db;
        --urgency-bg: linear-gradient(90deg, #fee2e2, #fef2f2);
        --urgency-border: #e02424;
        --urgency-text: #7f1d1d;
        --checklist-text: #374151;
        --checklist-border: #f3f4f6;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            /* Dark Mode */
            --c-bg: #262730;
            --c-border: #444444;
            --c-text: #e0e0e0;
            --c-text-muted: #a3a8b8;
            --chat-bg: #1a1c23;
            --chat-border: #333333;
            --bubble-in: #333333;
            --bubble-out: #1a56db;
            --input-bg: #1a1c23;
            --btn-sec-bg: #262730;
            --btn-sec-border: #666666;

            /* Cajas de Información Oscuro */
            --copilot-bg: rgba(14, 159, 110, 0.15);
            --copilot-border: #059669;
            --faq-bg: rgba(26, 86, 219, 0.15);
            --faq-border: #3b82f6;
            --urgency-bg: linear-gradient(90deg, #450a0a, #7f1d1d);
            --urgency-border: #ef4444;
            --urgency-text: #fca5a5;
            --checklist-text: #d1d5db;
            --checklist-border: #444444;
        }
    }

    /* Sobrescribir Botones Principales Nativos */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: var(--beex-blue) !important;
        border-color: var(--beex-blue) !important;
        color: white !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #1e40af !important;
        border-color: #1e40af !important;
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: var(--beex-blue) !important;
        color: var(--beex-blue) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: 1.5px solid var(--chat-border); padding-bottom: 0px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        padding: 0 8px;
        font-weight: 500;
        color: var(--c-text-muted);
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--c-text) !important;
        border-bottom-color: var(--c-text) !important;
    }

    /* Cards genéricas */
    .beex-card {
        background: var(--c-bg);
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid var(--c-border);
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
        margin-bottom: 16px;
    }

    /* Badges de intención */
    .badge-soporte     { background:#dbeafe; color:#1e40af; padding:4px 12px; border-radius:20px; font-weight:700; font-size:13px; }
    .badge-facturacion { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-weight:700; font-size:13px; }
    .badge-ventas      { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:20px; font-weight:700; font-size:13px; }
    .badge-reclamos    { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-weight:700; font-size:13px; }

    /* Banner urgencia */
    .urgencia-banner {
        background: var(--urgency-bg);
        border-left: 4px solid var(--urgency-border);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 12px;
        font-weight: 600;
        color: var(--urgency-text);
    }

    /* Copilot & FAQ boxes — colores consistentes */
    .copilot-box {
        background: var(--copilot-bg);
        border-left: 4px solid var(--copilot-border);
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        color: var(--c-text);
        line-height: 1.6;
    }
    .faq-box {
        background: var(--faq-bg);
        border-left: 4px solid var(--faq-border);
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        color: var(--c-text);
        line-height: 1.6;
    }

    /* Checklist — sin bullets de lista, solo texto plano con ícono */
    .checklist-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 8px 0;
        font-size: 14px;
        color: var(--checklist-text);
        border-bottom: 1px solid var(--checklist-border);
    }
    .checklist-item:last-child { border-bottom: none; }
    .checklist-icon-ok   { color: #0e9f6e; font-weight: 700; flex-shrink: 0; }
    .checklist-icon-warn { color: #e02424; font-weight: 700; flex-shrink: 0; }

    /* ── CHAT SIMULADO ──────────────────────────────── */
    .chat-wrapper {
        background: #f9f8f5;
        border: 1.5px solid #c8c7bc;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Header del chat */
    .chat-header-sim {
        background: #fff;
        border-bottom: 1.5px solid #c8c7bc;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .chat-avatar {
        width: 38px; height: 38px;
        border-radius: 50%;
        background: #e0dfd8;
        border: 1.5px solid #c8c7bc;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
        font-weight: 600;
        color: #555;
    }
    .chat-user-info { flex: 1; }
    .chat-user-name { font-weight: 600; font-size: 15px; color: #1a1a1a; }
    .chat-user-meta { font-size: 12px; color: #888; margin-top: 2px; }
    .chat-status-pill {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Área de mensajes */
    .messages-area-sim {
        padding: 16px;
        min-height: 200px;
        max-height: 520px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
        scroll-behavior: smooth;
    }

    /* Burbujas */
    .msg-in {
        align-self: flex-start;
        max-width: 70%;
    }
    .msg-out {
        align-self: flex-end;
        max-width: 70%;
    }
    .bubble-in {
        background: #f0efe9;
        border: 1.5px solid #c8c7bc;
        border-radius: 4px 12px 12px 12px;
        padding: 10px 14px;
        font-size: 13px;
        color: #1a1a1a;
        line-height: 1.5;
    }
    .bubble-out {
        background: #e0dfd8;
        border: 1.5px solid #c8c7bc;
        border-radius: 12px 4px 12px 12px;
        padding: 10px 14px;
        font-size: 13px;
        color: #1a1a1a;
        line-height: 1.5;
    }
    .msg-time-sim {
        font-size: 11px;
        color: #888;
        margin-top: 3px;
    }
    .msg-time-sim.right { text-align: right; }

    /* Badge de intención inline en el chat */
    .intent-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 4px;
    }
    .intent-Soporte     { background:#dbeafe; color:#1e40af; }
    .intent-Facturacion { background:#fef3c7; color:#92400e; }
    .intent-Ventas      { background:#d1fae5; color:#065f46; }
    .intent-Reclamos    { background:#fee2e2; color:#991b1b; }

    /* Sugerencia Copilot inline */
    .copilot-inline {
        background: #fafaf7;
        border: 1.5px solid #c8c7bc;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 6px;
        font-size: 12px;
        color: #555;
        line-height: 1.5;
        max-width: 80%;
    }
    .copilot-inline-header {
        font-size: 10px;
        font-weight: 700;
        color: #1a56db;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .copilot-urgencia-inline {
        background: #fee2e2;
        border: 1.5px solid #fca5a5;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        color: #991b1b;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Tabla historial zebra */
    .dataframe tbody tr:nth-child(even) { background-color: #f9fafb; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CARGA DEL MODELO (HU-05)
# ==============================================================================
@st.cache_resource
def cargar_modelo():
    modelo_path = os.path.join(os.path.dirname(__file__), "modelo_beex_produccion.pkl")
    modelo = joblib.load(modelo_path)
    if hasattr(modelo, "steps") and len(modelo.steps) > 0:
        svm_step = modelo.steps[-1][1]
        if hasattr(svm_step, "probability") and svm_step.probability == "deprecated":
            svm_step.probability = False
    return modelo

modelo = cargar_modelo()

# ==============================================================================
# DICCIONARIOS DE NEGOCIO
# ==============================================================================
INTERPRETACION = {
    "Soporte": {
        "icono": "🔧", "badge": "badge-soporte", "color": "#1a56db",
        "descripcion": "El cliente presenta una incidencia técnica (conectividad, acceso, fallas de sistema).",
        "accion_recomendada": "Enrutar a Cola de Soporte Técnico — Nivel 1.",
        "sla_sugerido": "≤ 2 min", "prioridad": "Normal", "emoji_prioridad": "🟡"
    },
    "Facturacion": {
        "icono": "💰", "badge": "badge-facturacion", "color": "#f59e0b",
        "descripcion": "El cliente tiene una consulta o reclamo relacionado con cobros, facturas o pagos.",
        "accion_recomendada": "Enrutar a Cola de Facturación — Verificar estado de cuenta antes de responder.",
        "sla_sugerido": "≤ 3 min", "prioridad": "Normal", "emoji_prioridad": "🟡"
    },
    "Ventas": {
        "icono": "🛒", "badge": "badge-ventas", "color": "#0e9f6e",
        "descripcion": "El cliente muestra interés comercial en contratar, renovar o adquirir un servicio.",
        "accion_recomendada": "Enrutar a Cola de Ventas — Preparar cotización y catálogo de planes.",
        "sla_sugerido": "≤ 1 min", "prioridad": "Alta", "emoji_prioridad": "🟠"
    },
    "Reclamos": {
        "icono": "⚠️", "badge": "badge-reclamos", "color": "#e02424",
        "descripcion": "El cliente expresa insatisfacción severa, solicita baja, denuncia o escalamiento.",
        "accion_recomendada": "Enrutar a Cola de Reclamos — PRIORIDAD MÁXIMA. Notificar a supervisor.",
        "sla_sugerido": "≤ 30 seg", "prioridad": "CRÍTICA", "emoji_prioridad": "🔴"
    }
}

# ==============================================================================
# MENSAJES SIMULADOS PARA EL CHAT (por categoría)
# ==============================================================================
MENSAJES_SIMULADOS = {
    "Soporte": [
        "mi internet no funciona, se cayó hace una hora y no puedo trabajar",
        "no puedo entrar al sistema, me dice contraseña incorrecta pero estoy seguro que es la correcta",
        "la señal está muy lenta, pago el plan de 300 megas y apenas llega a 10",
        "se me cayó la red otra vez, esto pasa todos los días desde hace semanas",
        "el servicio no funciona desde esta mañana, reinicié el router y nada",
        "tengo problemas para acceder a la plataforma, me da error 503",
        "el técnico vino ayer y sigue sin funcionar, necesito ayuda urgente",
    ],
    "Facturacion": [
        "quiero saber cuánto debo este mes, no me llegó el recibo",
        "me cobraron un monto diferente al que acordamos, necesito revisarlo",
        "quiero pagar mi factura pero me sale error en la página de pagos",
        "cuándo vence mi próximo pago y por cuánto es exactamente",
        "aparece un cobro extra que no reconozco en mi estado de cuenta",
        "me llegó una factura duplicada, me cobraron dos veces el mismo mes",
        "quiero cambiar mi método de pago a débito automático",
    ],
    "Ventas": [
        "buenos días, me interesa contratar el servicio de fibra óptica, cuáles son los planes",
        "tienen alguna promoción para nuevos clientes? me recomendaron Beex",
        "quiero mejorar mi plan actual, qué opciones tienen disponibles",
        "quiero contratar el servicio para mi negocio, tienen planes empresariales",
        "me interesa el plan doble de internet y televisión, cuánto sale",
        "están disponibles en mi zona? vivo en Miraflores",
        "quiero renovar mi contrato, tienen descuento por fidelidad",
    ],
    "Reclamos": [
        "EXIJO hablar con un supervisor, llevo 3 días sin solución y nadie me ayuda",
        "esto es inaceptable, me prometieron resolver en 24 horas y ya pasó una semana",
        "quiero darme de baja del servicio, estoy completamente insatisfecho",
        "voy a presentar una queja formal en INDECOPI si no me resuelven hoy",
        "me cobraron doble y nadie me da respuesta, esto es un robo",
        "el técnico nunca llegó a la cita que tenía programada, perdí el día",
        "nunca más voy a recomendar este servicio, es una basura total",
    ]
}

NOMBRES_CLIENTES = ["Juan Pérez", "María García", "Carlos López", "Ana Torres",
                    "Luis Mendoza", "Rosa Sánchez", "Pedro Díaz", "Carmen Vega"]
CANALES_CHAT = ["WhatsApp", "Webchat", "Instagram", "Teléfono"]

# ==============================================================================
# BASE DE CONOCIMIENTO FAQ LOCAL (HU-07)
# ==============================================================================
FAQ_BEEX = {
    "Soporte": [
        {
            "pregunta": "No tengo conexión a internet / la red se cayó",
            "keywords": ["internet", "red", "conexión", "conectividad", "caído", "no funciona", "sin señal", "cayó"],
            "respuesta": "Le pedimos disculpas por el inconveniente. Por favor verifique: (1) reinicie su router desconectándolo 30 segundos, (2) compruebe que los cables estén bien conectados. Si el problema persiste, crearemos un ticket de soporte con prioridad N1 y un técnico lo contactará en menos de 2 horas."
        },
        {
            "pregunta": "No puedo acceder al sistema / error de acceso",
            "keywords": ["sistema", "acceso", "contraseña", "usuario", "login", "ingresar", "plataforma", "error"],
            "respuesta": "Para restablecer su acceso: (1) visite nuestro portal en beex.pe/reset, (2) ingrese su correo registrado y recibirá un enlace en 5 minutos. Si el problema es de plataforma, escale al área técnica con el código de error que aparece en pantalla."
        },
        {
            "pregunta": "El servicio es lento / velocidad baja",
            "keywords": ["lento", "velocidad", "lentitud", "tarda", "demora", "cargando", "megas", "señal"],
            "respuesta": "Para diagnóstico de velocidad: (1) ejecute una prueba en fast.com, (2) compare con el plan contratado. Si la velocidad es inferior al 80% del plan, gestionaremos una revisión técnica gratuita en las próximas 4 horas hábiles."
        }
    ],
    "Facturacion": [
        {
            "pregunta": "Cuánto debo / cuál es mi deuda",
            "keywords": ["deuda", "saldo", "cuánto debo", "monto", "recibo", "factura", "cobro", "debo", "vence"],
            "respuesta": "Puede consultar su saldo en tiempo real en: (1) App Beex → Mi Cuenta → Estado de cuenta, (2) enviando 'SALDO' al WhatsApp corporativo. Su próximo vencimiento y el monto exacto aparecerán de forma inmediata."
        },
        {
            "pregunta": "Me cobraron de más / cargo incorrecto",
            "keywords": ["cobraron", "cargo", "incorrecto", "doble", "cobro", "error en factura", "mal cobrado", "duplicada"],
            "respuesta": "Lamentamos el inconveniente. Solicitaremos una revisión de su cuenta en las próximas 24 horas hábiles. Si el cargo es incorrecto, se realizará la devolución o nota de crédito en el siguiente ciclo de facturación."
        },
        {
            "pregunta": "No puedo pagar / error al pagar",
            "keywords": ["pagar", "error", "pago", "no me deja", "problemas de pago", "método"],
            "respuesta": "Los métodos de pago disponibles son: (1) Pago en línea en beex.pe/pago, (2) depósito bancario (BCP o Interbank — código de empresa 12345), (3) agentes Kasnet o Western Union con su DNI."
        }
    ],
    "Ventas": [
        {
            "pregunta": "Quiero contratar / cuáles son los planes",
            "keywords": ["contratar", "planes", "precio", "cotización", "cuánto cuesta", "servicio", "fibra", "internet", "disponibles"],
            "respuesta": "¡Excelente decisión! Nuestros planes: 📶 Basic 100 Mbps — S/ 59.90/mes | ⚡ Plus 300 Mbps — S/ 89.90/mes | 🚀 Pro 600 Mbps — S/ 129.90/mes. Todos incluyen instalación gratuita. ¿Le gustaría que un asesor lo contacte?"
        },
        {
            "pregunta": "Quiero renovar / actualizar mi plan",
            "keywords": ["renovar", "actualizar", "upgrade", "cambiar plan", "mejorar", "fidelidad"],
            "respuesta": "Puede renovar o actualizar su plan sin costo de penalización en los últimos 30 días de su contrato. Para iniciar: (1) acceda a beex.pe/renovar o (2) indíqueme su número de cliente para gestionar el cambio."
        },
        {
            "pregunta": "Tienen promociones o descuentos",
            "keywords": ["promoción", "descuento", "oferta", "especial", "promo", "gratis", "recomendaron"],
            "respuesta": "Actualmente: 🎉 2 meses con 30% descuento para nuevos clientes | 📦 Planes doble play desde S/ 99.90 | 🤝 S/ 20 de descuento por referir un amigo. Válido hasta fin de mes."
        }
    ],
    "Reclamos": [
        {
            "pregunta": "Quiero hablar con un supervisor / escalar mi reclamo",
            "keywords": ["supervisor", "escalamiento", "escalar", "jefe", "encargado", "quiero hablar", "nadie"],
            "respuesta": "Entiendo su malestar y lo tomamos muy en serio. Lo conectaré de inmediato con nuestro supervisor de turno. Estoy registrando su reclamo con código prioritario para que tenga todo el contexto. Gracias por su paciencia."
        },
        {
            "pregunta": "Quiero dar de baja el servicio",
            "keywords": ["baja", "cancelar", "cancelación", "terminar", "no quiero", "darme de baja", "insatisfecho"],
            "respuesta": "Lamentamos que esté considerando darse de baja. Antes de proceder, nos gustaría ofrecerle: (1) revisión técnica gratuita, (2) ajuste de plan que se adapte mejor a su presupuesto. ¿Podría contarnos el motivo principal?"
        },
        {
            "pregunta": "Voy a denunciar / presentar queja formal",
            "keywords": ["denunciar", "queja", "indecopi", "denuncia", "formal", "libro de reclamaciones", "robo"],
            "respuesta": "Tiene todo el derecho. Nuestro Libro de Reclamaciones está en beex.pe/reclamos (Ley N.° 29571). También puede acudir a INDECOPI. Escalaré su caso al equipo de Atención Especializada para una solución hoy mismo."
        }
    ]
}

# ==============================================================================
# SCRIPTS COPILOT PARA AGENTES (HU-06)
# ==============================================================================
COPILOT_SCRIPTS = {
    "Soporte": {
        "apertura": "Buenos días/tardes, le habla [NOMBRE AGENTE] de Beex Soporte Técnico. Veo que presenta un inconveniente con su servicio, ¿me podría confirmar su número de cliente o DNI para revisar su caso?",
        "checklist": [
            ("ok", "Verificar si el problema es solo para el cliente o incidencia masiva en la zona"),
            ("ok", "Solicitar número de teléfono de contacto alternativo"),
            ("ok", "Revisar el historial de tickets anteriores del cliente"),
            ("ok", "Registrar descripción exacta del problema y hora de inicio"),
            ("ok", "Si no se resuelve en < 5 min → escalar a Soporte Nivel 2"),
        ],
        "cierre": "Hemos registrado su caso con el número [TICKET-XXXX]. Nuestro equipo técnico lo contactará en un máximo de 2 horas. ¿Hay algo más en lo que pueda ayudarle hoy?"
    },
    "Facturacion": {
        "apertura": "Buenos días/tardes, le habla [NOMBRE AGENTE] del área de Facturación de Beex. Para validar su identidad y acceder a su cuenta, ¿me podría indicar su número de DNI o número de contrato?",
        "checklist": [
            ("ok", "Validar identidad del cliente (DNI o número de contrato)"),
            ("ok", "Abrir el sistema de facturación y revisar el estado de cuenta"),
            ("ok", "Identificar el cargo en disputa con fecha y monto exacto"),
            ("ok", "Verificar si el cliente tiene un plan especial o descuento activo"),
            ("ok", "Si el cobro es incorrecto → iniciar proceso de devolución o nota de crédito"),
        ],
        "cierre": "He registrado su solicitud. Recibirá un correo de confirmación en los próximos 10 minutos. La resolución tomará máximo 24 horas hábiles. ¿Tiene alguna otra consulta?"
    },
    "Ventas": {
        "apertura": "¡Bienvenido a Beex! Le habla [NOMBRE AGENTE] del equipo comercial. ¡Qué gusto atenderle! Cuénteme, ¿qué tipo de servicio está buscando hoy?",
        "checklist": [
            ("ok", "Identificar necesidades específicas del cliente (velocidad, uso, presupuesto)"),
            ("ok", "Consultar si ya es cliente de Beex (posible upgrade)"),
            ("ok", "Presentar máximo 3 opciones de planes"),
            ("ok", "Mencionar la promoción del mes antes de cerrar"),
            ("ok", "Si el cliente duda → ofrecer período de prueba o demo del servicio"),
        ],
        "cierre": "¡Excelente elección! He registrado su solicitud de contratación del plan [PLAN]. En las próximas 2 horas recibirá el contrato digital. ¿Desea coordinar la fecha de instalación ahora?"
    },
    "Reclamos": {
        "apertura": "Buenos días/tardes, le habla [NOMBRE AGENTE], supervisor de Atención al Cliente de Beex. Entiendo que tiene una situación urgente y tiene toda mi atención. ¿Me puede contar en detalle qué ocurrió?",
        "checklist": [
            ("warn", "ESCUCHAR sin interrumpir — dejar que el cliente exprese su frustración completa"),
            ("warn", "Validar emocionalmente: 'Entiendo su malestar y es completamente válido'"),
            ("warn", "Revisar historial completo del cliente antes de proponer solución"),
            ("warn", "NO prometer tiempos que no se pueden cumplir"),
            ("warn", "Ofrecer compensación según protocolo de retención"),
            ("warn", "Documentar TODO con detalle para el informe de calidad"),
        ],
        "cierre": "Le doy mi palabra de que su caso quedará resuelto. He escalado con prioridad máxima y recibirá una llamada de seguimiento pronto. El número de su reclamo es [RECLAMO-XXXX]."
    }
}

# ==============================================================================
# FUNCIONES DE PROCESAMIENTO
# ==============================================================================
def detectar_urgencia(texto: str) -> bool:
    palabras_urgencia = [
        'urgente', 'ya', 'harto', 'basura', 'cancelar', 'queja',
        'denuncia', 'robo', 'estafado', 'supervisor', 'baja',
        'exijo', 'denunciar', 'inmediato', 'ahora', 'inaceptable',
        'indecopi', 'abogado', 'demanda', 'nunca más', 'pésimo'
    ]
    return any(w in texto.lower() for w in palabras_urgencia)

def predecir_intencion(texto: str) -> str:
    texto_limpio = texto.strip().lower()
    return modelo.predict([texto_limpio])[0]

def buscar_faq(texto: str, intencion: str) -> list:
    texto_lower = texto.lower()
    resultados = [item for item in FAQ_BEEX.get(intencion, [])
                  if any(kw in texto_lower for kw in item["keywords"])]
    return resultados if resultados else FAQ_BEEX.get(intencion, [])

import json
import os

def inicializar_session_state():
    if "historial" not in st.session_state:
        st.session_state.historial = []
        if os.path.exists("historial_persistente.json"):
            try:
                with open("historial_persistente.json", "r", encoding="utf-8") as f:
                    st.session_state.historial = json.load(f)
            except Exception:
                pass
    if "chat_mensajes" not in st.session_state:
        st.session_state.chat_mensajes = []
        if os.path.exists("chat_persistente.json"):
            try:
                with open("chat_persistente.json", "r", encoding="utf-8") as f:
                    st.session_state.chat_mensajes = json.load(f)
            except Exception:
                pass
    if "ultima_clasificacion" not in st.session_state:
        st.session_state.ultima_clasificacion = None
    if "total_sesion" not in st.session_state:
        st.session_state.total_sesion = len(st.session_state.historial)
    if "chat_cliente" not in st.session_state:
        st.session_state.chat_cliente = random.choice(NOMBRES_CLIENTES)
    if "chat_canal" not in st.session_state:
        st.session_state.chat_canal = random.choice(CANALES_CHAT)

inicializar_session_state()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## BEEX Suite")
    st.markdown("*Inteligencia Conversacional para Contact Center*")
    st.markdown("---")

    st.markdown("### Modelo en Producción")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Accuracy",  "94.56%")
        st.metric("Precision", "94.77%")
    with col_s2:
        st.metric("Recall",    "94.56%")
        st.metric("F1-Score",  "94.60%")

    st.markdown("---")
    st.markdown("**Algoritmo:** SVM · Kernel RBF")
    st.markdown("**Vectorizer:** TF-IDF (1-2 gramas)")
    st.markdown("**Dataset:** 1,500 interacciones sintéticas")
    st.markdown("**Inferencia:** ~0.029 seg/mensaje")
    st.markdown("---")
    st.markdown("**Sesión actual**")
    st.metric("Clasificaciones realizadas", st.session_state.total_sesion)
    if st.session_state.historial:
        urgentes = sum(1 for h in st.session_state.historial if h["urgencia"])
        st.metric("⚡ Alertas de urgencia", urgentes)
    st.markdown("---")
    if st.button("Limpiar sesión", use_container_width=True):
        for k in ["historial", "ultima_clasificacion", "total_sesion",
                  "chat_mensajes", "chat_cliente", "chat_canal"]:
            if k in st.session_state:
                del st.session_state[k]
        if os.path.exists("historial_persistente.json"):
            os.remove("historial_persistente.json")
        if os.path.exists("chat_persistente.json"):
            os.remove("chat_persistente.json")
        st.rerun()
    st.markdown("---")
    st.caption("Proyecto Final · Innovación y TDI · UTP")
    st.caption("Beex Contact Center · Julio 2026")

# ==============================================================================
# HEADER PRINCIPAL
# ==============================================================================
st.markdown("# BEEX — Suite de Inteligencia Conversacional")
st.markdown("*Motor centralizado de clasificación de intenciones, historial omnicanal y asistencia al agente con IA*")
st.markdown("---")

# ==============================================================================
# PESTAÑAS PRINCIPALES
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Clasificador",
    "Chat Simulado",
    "Historial",
    "Dashboard",
    "Copilot & FAQ"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLASIFICADOR (diseño original restaurado)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ingrese la interacción del cliente")
    st.markdown("*Escriba o pegue el mensaje del cliente (WhatsApp, transcripción, webchat, etc.)*")

    col_input, col_config = st.columns([2, 1])

    with col_config:
        st.markdown("#### Configuración del canal")
        canal = st.selectbox(
            "📡 Canal de origen",
            options=["💬 WhatsApp", "🌐 Webchat", "📞 Teléfono", "📧 Email"],
            help="Canal de origen de la interacción. (HU-04)"
        )
        agente = st.text_input("👤 ID / Nombre del agente", placeholder="Ej: AG-042 o Juan Pérez")

        if "Teléfono" in canal:
            st.markdown("---")
            st.markdown("#### Speech-to-Text (HU-08)")
            st.info("Suba un audio o escriba la transcripción manual de la llamada.")
            audio_file = st.file_uploader("Subir audio de llamada", type=["wav", "mp3", "ogg", "m4a"])
            if audio_file:
                st.audio(audio_file)
                st.warning("Transcripción automática no disponible en demo. Escriba el texto manualmente.", icon="🎙️")

    with col_input:
        ejemplos = {
            "Seleccionar ejemplo...": "",
            "🔧 Soporte técnico": "mi internet no funciona pucha, se me cayó la red otra vez y no puedo ingresar al sistema",
            "💰 Consulta de facturación": "quiero pagar mi recibo pero me sale error, cuánto es mi deuda este mes",
            "🛒 Interés en ventas": "buenos días, me interesa contratar el servicio de fibra óptica, cuáles son los planes disponibles",
            "⚠️ Reclamo urgente": "EXIJO hablar con un supervisor, me cobraron doble y llevo 3 días sin solución, esto es una basura",
            "📞 Renovación de plan": "quiero mejorar mi plan actual, tienen alguna promoción para clientes antiguos",
            "🔧 Error de acceso": "no puedo entrar a la plataforma, me dice contraseña incorrecta pero es la correcta"
        }
        ejemplo_sel = st.selectbox("O seleccione un ejemplo:", list(ejemplos.keys()))
        valor_inicial = ejemplos[ejemplo_sel] if ejemplo_sel != "Seleccionar ejemplo..." else ""

        texto_usuario = st.text_area(
            "Mensaje del cliente:",
            value=valor_inicial,
            height=120,
            placeholder="Ej: 'mi internet no funciona, necesito ayuda urgente'"
        )
        clasificar = st.button("Clasificar Intención", type="primary",
                               use_container_width=True, key="btn_clasificar")

    # ── Resultado de clasificación — DISEÑO ORIGINAL ────────────────────────
    if clasificar:
        if texto_usuario.strip():
            intencion  = predecir_intencion(texto_usuario)
            info       = INTERPRETACION[intencion]
            urgencia   = detectar_urgencia(texto_usuario)
            timestamp  = datetime.now().strftime("%H:%M:%S")
            canal_limpio = canal.split(" ", 1)[-1]

            registro = {
                "timestamp": timestamp, "canal": canal_limpio,
                "agente": agente if agente else "N/A",
                "mensaje": texto_usuario[:100] + ("..." if len(texto_usuario) > 100 else ""),
                "intencion": intencion, "prioridad": info["prioridad"],
                "urgencia": urgencia, "sla": info["sla_sugerido"]
            }
            st.session_state.historial.append(registro)
            st.session_state.ultima_clasificacion = {
                "intencion": intencion, "info": info,
                "urgencia": urgencia, "texto": texto_usuario, "canal": canal_limpio
            }
            st.session_state.total_sesion += 1

            # ── Resultado visual (diseño original del APF3) ──
            st.markdown("---")
            st.subheader("📋 Resultado de la Clasificación")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### {info['icono']} {intencion}")
            with col2:
                st.markdown(f"**Descripción:** {info['descripcion']}")

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
                st.warning("⚡ **ALERTA DE URGENCIA:** Se detectaron términos de alta fricción. "
                           "Se recomienda escalamiento inmediato o activación de protocolo Fast-Track.")

            st.markdown("---")
            st.subheader("📈 Métricas del Modelo (APF3)")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: st.metric("Accuracy",    "94.56%")
            with col_m2: st.metric("Precision (W)", "94.77%")
            with col_m3: st.metric("Recall (W)",    "94.56%")
            with col_m4: st.metric("F1-Score (W)",  "94.60%")

            st.info("💡 Vaya a la pestaña **🤖 Copilot & FAQ** para ver el script y las respuestas automáticas.")
        else:
            st.warning("⚠️ Por favor, ingrese un mensaje del cliente para clasificar.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CHAT SIMULADO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Chat Simulado — Copilot en tiempo real")
    st.caption(
        "Simule una conversación de cliente en tiempo real. Presione **'Nuevo mensaje del cliente'** "
        "para recibir un mensaje aleatorio. El modelo lo clasificará automáticamente y el Copilot "
        "mostrará sugerencias de respuesta inline."
    )

    # ── Controles del chat ───────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        st.markdown(
            f"**Cliente activo:** {st.session_state.chat_cliente} &nbsp;·&nbsp; "
            f"**Canal:** {st.session_state.chat_canal}"
        )
    with ctrl_col2:
        if st.button("Cambiar cliente", use_container_width=True):
            st.session_state.chat_cliente = random.choice(NOMBRES_CLIENTES)
            st.session_state.chat_canal   = random.choice(CANALES_CHAT)
            st.session_state.chat_mensajes = []
            if os.path.exists("chat_persistente.json"):
                os.remove("chat_persistente.json")
            st.rerun()
    with ctrl_col3:
        if st.button("Limpiar chat", use_container_width=True):
            st.session_state.chat_mensajes = []
            if os.path.exists("chat_persistente.json"):
                os.remove("chat_persistente.json")
            st.rerun()

    st.markdown("---")

    # ── Layout: chat a la izquierda, panel copilot a la derecha ─────────────
    chat_col, panel_col = st.columns([3, 2])

    with chat_col:
        # Build the ENTIRE chat as a single HTML string so everything renders inside the wrapper
        canal_emoji = {
            "WhatsApp": "💬", "Webchat": "🌐",
            "Instagram": "📸", "Teléfono": "📞"
        }.get(st.session_state.chat_canal, "💬")

        client_id = f"CL-{hash(st.session_state.chat_cliente) % 9000 + 1000}"

        # Inline CSS for the component (rendered in iframe, needs its own styles)
        chat_css = """<style>
          :root {
              --chat-bg: #f9f8f5; --chat-border: #c8c7bc; --c-bg: #ffffff;
              --c-text: #1a1a1a; --c-text-muted: #555555;
              --bubble-in: #f0efe9; --bubble-out: #e0dfd8;
              --bubble-out-text: #1a1a1a; --pill-bg: #e0dfd8;
              --urgency-bg: rgba(224, 36, 36, 0.1); --urgency-border: #e02424; --urgency-text: #e02424;
          }
          @media (prefers-color-scheme: dark) {
              :root {
                  --chat-bg: #1a1c23; --chat-border: #333333; --c-bg: #262730;
                  --c-text: #e0e0e0; --c-text-muted: #a3a8b8;
                  --bubble-in: #333333; --bubble-out: #1a56db;
                  --bubble-out-text: #ffffff; --pill-bg: #333333;
                  --urgency-bg: rgba(239, 68, 68, 0.15); --urgency-border: #ef4444; --urgency-text: #fca5a5;
              }
          }
          * { box-sizing: border-box; margin: 0; padding: 0; }
          html, body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: transparent; height: 100%; color: var(--c-text); }
          .chat-wrapper {
              background: var(--chat-bg); border: 1.5px solid var(--chat-border);
              border-bottom: none; border-radius: 12px 12px 0 0; overflow: hidden;
              height: 100%; display: flex; flex-direction: column;
          }
          .chat-header-sim {
              background: var(--c-bg); border-bottom: 1.5px solid var(--chat-border);
              padding: 12px 16px; display: flex; align-items: center; gap: 12px;
          }
          .chat-avatar {
              width: 38px; height: 38px; border-radius: 50%; background: var(--pill-bg);
              border: 1.5px solid var(--chat-border); display: flex; align-items: center;
              justify-content: center; font-size: 16px; font-weight: 600; color: var(--c-text-muted);
              flex-shrink: 0;
          }
          .chat-user-info { flex: 1; }
          .chat-user-name { font-weight: 600; font-size: 15px; color: var(--c-text); }
          .chat-user-meta { font-size: 12px; color: var(--c-text-muted); margin-top: 2px; }
          .chat-status-pill {
              background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7;
              border-radius: 20px; padding: 3px 10px; font-size: 11px; font-weight: 600;
          }
          .messages-area {
              padding: 16px; display: flex; flex-direction: column; gap: 12px;
              overflow-y: auto; scroll-behavior: smooth; flex: 1;
          }
          .msg-in { align-self: flex-start; max-width: 75%; }
          .msg-out { align-self: flex-end; max-width: 75%; }
          .bubble-in {
              background: var(--bubble-in); border: 1.5px solid var(--chat-border);
              border-radius: 4px 12px 12px 12px; padding: 10px 14px;
              font-size: 13px; color: var(--c-text); line-height: 1.5;
          }
          .bubble-out {
              background: var(--bubble-out); border: 1.5px solid var(--chat-border);
              border-radius: 12px 4px 12px 12px; padding: 10px 14px;
              font-size: 13px; color: var(--bubble-out-text); line-height: 1.5;
          }
          .msg-time { font-size: 11px; color: var(--c-text-muted); margin-top: 3px; }
          .msg-time.right { text-align: right; }
          .intent-pill {
              display: inline-block; padding: 2px 10px; border-radius: 20px;
              font-size: 11px; font-weight: 700; margin-top: 4px; color: #fff;
          }
          .intent-Soporte     { background:#1e40af; }
          .intent-Facturacion { background:#d97706; }
          .intent-Ventas      { background:#047857; }
          .intent-Reclamos    { background:#b91c1c; }
          .copilot-inline {
              background: var(--chat-bg); border: 1.5px solid var(--chat-border); border-radius: 8px;
              padding: 10px 14px; margin-top: 6px; font-size: 12px; color: var(--c-text);
              line-height: 1.5; max-width: 80%; align-self: flex-start;
          }
          .copilot-inline-header {
              font-size: 10px; font-weight: 700; color: #1a56db;
              text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
          }
          .urgencia-inline {
              background: var(--urgency-bg); border: 1.5px solid var(--urgency-border); border-radius: 8px;
              padding: 8px 12px; font-size: 12px; color: var(--urgency-text); font-weight: 600;
              margin-top: 6px; align-self: flex-start;
          }
          .empty-state {
              text-align: center; padding: 60px 20px; color: var(--c-text-muted); font-size: 13px;
          }
          .empty-icon { font-size: 40px; margin-bottom: 12px; }
          ::-webkit-scrollbar { width: 5px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: var(--chat-border); border-radius: 4px; }
        </style>"""

        # Build chat body
        chat_body = ""
        if not st.session_state.chat_mensajes:
            chat_body = """
            <div class="empty-state">
              <div class="empty-icon">💬</div>
              <p>No hay mensajes aún.</p>
              <p style="margin-top:6px">Presione <strong>"Nuevo mensaje del cliente"</strong> para iniciar la conversación.</p>
            </div>"""
        else:
            for msg in st.session_state.chat_mensajes:
                ts = msg["timestamp"]
                if msg["tipo"] == "cliente":
                    intencion = msg.get("intencion", "")
                    urgencia  = msg.get("urgencia", False)
                    urg_html  = '<span class="intent-pill" style="background:#fee2e2;color:#991b1b;margin-left:4px">⚡ Urgente</span>' if urgencia else ''
                    icon      = INTERPRETACION[intencion]["icono"]

                    chat_body += f"""
                    <div class="msg-in">
                      <div class="bubble-in">{msg['texto']}</div>
                      <div style="margin-top:4px">
                        <span class="intent-pill intent-{intencion}">{icon} {intencion}</span>{urg_html}
                      </div>
                      <div class="msg-time">{ts}</div>
                    </div>"""

                else:
                    chat_body += f"""
                    <div class="msg-out">
                      <div class="bubble-out">{msg['texto']}</div>
                      <div class="msg-time right">{ts} ✓✓</div>
                    </div>"""

        # Calculate component height — bigger window
        n_msgs = len(st.session_state.chat_mensajes)
        component_height = max(400, min(750, 80 + n_msgs * 130))

        full_html = f"""{chat_css}
        <div class="chat-wrapper">
          <div class="chat-header-sim">
            <div class="chat-avatar">{st.session_state.chat_cliente[0]}</div>
            <div class="chat-user-info">
              <div class="chat-user-name">{st.session_state.chat_cliente}</div>
              <div class="chat-user-meta">Canal: {canal_emoji} {st.session_state.chat_canal} · ID: {client_id}</div>
            </div>
            <span class="chat-status-pill">● En línea</span>
          </div>
          <div class="messages-area" id="chat-area">{chat_body}
          </div>
        </div>
        <script>
          var c = document.getElementById('chat-area');
          if (c) c.scrollTop = c.scrollHeight;
        </script>"""

        import streamlit.components.v1 as components
        components.html(full_html, height=component_height, scrolling=False)

        # ── Barra de respuesta del agente (estilo composer Beex) ───────────────
        st.markdown(
            "<style>"
            "/* Custom styles to merge the Streamlit container with the chat iframe */"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) {"
            "    background-color: var(--c-bg) !important;"
            "    border: 1.5px solid var(--chat-border) !important;"
            "    border-top: 1px solid var(--chat-border) !important;"
            "    border-radius: 0 0 12px 12px !important;"
            "    padding: 16px 20px !important;"
            "    margin-top: -1.2rem !important;"
            "    box-shadow: 0 -1px 4px rgba(0,0,0,0.02) !important;"
            "    z-index: 10;"
            "    position: relative;"
            "}"
            "/* Force text colors for dark mode compatibility */"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) p,"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) span {"
            "    color: var(--c-text) !important;"
            "}"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) label {"
            "    color: var(--c-text-muted) !important;"
            "}"
            "/* Force input and button styles to look like sketch theme */"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) div[data-baseweb='input'],"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) div[data-baseweb='textarea'] {"
            "    background-color: var(--input-bg) !important;"
            "    border: 1.5px solid var(--chat-border) !important;"
            "}"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) input,"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) textarea {"
            "    color: var(--c-text) !important;"
            "}"
            "div[data-testid='stVerticalBlock']:has(> div.element-container .composer-marker) button[kind='secondary'] {"
            "    background-color: var(--btn-sec-bg) !important;"
            "    border: 1.5px solid var(--btn-sec-border) !important;"
            "    color: var(--c-text) !important;"
            "}"
            ".suggested-box {"
            "    background: var(--input-bg); border: 1.5px solid var(--chat-border); border-radius: 8px;"
            "    padding: 12px; font-size: 13px; color: var(--c-text); margin-bottom: 12px; line-height: 1.5;"
            "}"
            "div[data-testid='stTabs'] button { font-weight: 500; color: var(--c-text-muted); }"
            "div[data-testid='stTabs'] button[aria-selected='true'] { color: var(--beex-blue); }"
            "</style>", unsafe_allow_html=True
        )

        with st.container():
            st.markdown("<span class='composer-marker'></span>", unsafe_allow_html=True)
            tab_resp, tab_notas = st.tabs(["Respuesta sugerida", "Notas internas"])

        with tab_resp:
            # Buscar sugerencia
            resp_pre = ""
            if st.session_state.chat_mensajes and st.session_state.chat_mensajes[-1]["tipo"] == "cliente":
                last_intent = st.session_state.chat_mensajes[-1]["intencion"]
                faqs_resp   = buscar_faq(st.session_state.chat_mensajes[-1]["texto"], last_intent)
                resp_pre    = faqs_resp[0]["respuesta"] if faqs_resp else ""

            if resp_pre:
                st.markdown(f"<div class='suggested-box'>{resp_pre}</div>", unsafe_allow_html=True)
                
                col_send, col_edit, col_bad = st.columns([3, 2, 2])
                with col_send:
                    enviar_sug = st.button("Enviar respuesta", key="btn_env_sug", type="primary", use_container_width=True)
                with col_edit:
                    editar_sug = st.button("Editar", key="btn_edit_sug", use_container_width=True)
                with col_bad:
                    st.button("No es útil", key="btn_bad_sug", use_container_width=True)

                if enviar_sug:
                    st.session_state.chat_mensajes.append({
                        "tipo": "agente",
                        "texto": resp_pre,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    st.rerun()

                if editar_sug:
                    st.session_state.editar_texto = resp_pre
            else:
                st.info("No hay respuesta sugerida para este mensaje.")

            # Input manual
            texto_a_enviar = st.session_state.get("editar_texto", "")
            resp_col, send_col = st.columns([6, 1])
            with resp_col:
                resp_agente = st.text_input(
                    "Escribe un mensaje...",
                    value=texto_a_enviar,
                    key="resp_agente_txt",
                    label_visibility="collapsed",
                    placeholder="Escribe un mensaje..."
                )
            with send_col:
                enviar_resp = st.button("📤", key="btn_enviar_resp", type="primary", use_container_width=True)
                
            if enviar_resp and resp_agente.strip():
                if "editar_texto" in st.session_state:
                    del st.session_state.editar_texto
                st.session_state.chat_mensajes.append({
                    "tipo": "agente",
                    "texto": resp_agente.strip(),
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()

        with tab_notas:
            st.text_area("Añadir nota interna al cliente...", height=100, label_visibility="collapsed", placeholder="Escribe una nota interna...")
            st.button("Guardar nota", key="btn_guardar_nota")

        st.markdown("---")

        # ── Entrada del cliente ──────────────────────────────────────────────
        st.markdown("##### Mensaje del cliente")
        modo_col1, modo_col2 = st.columns(2)
        with modo_col1:
            modo_cliente = st.radio(
                "Modo de entrada:",
                ["Escribir mensaje manualmente", "Mensaje aleatorio"],
                horizontal=True,
                key="modo_entrada_chat",
                label_visibility="collapsed"
            )

        if modo_cliente == "Escribir mensaje manualmente":
            txt_col, send_txt_col = st.columns([5, 1])
            with txt_col:
                texto_manual = st.text_input(
                    "Mensaje del cliente:",
                    key="txt_cliente_manual",
                    placeholder="Escriba aquí como si fuera el cliente...",
                    label_visibility="collapsed"
                )
            with send_txt_col:
                enviar_manual = st.button("📨", key="btn_enviar_manual",
                                          type="primary", use_container_width=True,
                                          help="Enviar mensaje del cliente")

            if enviar_manual and texto_manual.strip():
                intencion_real = predecir_intencion(texto_manual)
                urgencia_real  = detectar_urgencia(texto_manual)
                ts_now = datetime.now().strftime("%H:%M")

                st.session_state.chat_mensajes.append({
                    "tipo": "cliente", "texto": texto_manual.strip(),
                    "intencion": intencion_real, "urgencia": urgencia_real,
                    "timestamp": ts_now
                })
                st.session_state.historial.append({
                    "timestamp": ts_now,
                    "canal": st.session_state.chat_canal,
                    "agente": "Chat Sim",
                    "mensaje": texto_manual[:100],
                    "intencion": intencion_real,
                    "prioridad": INTERPRETACION[intencion_real]["prioridad"],
                    "urgencia": urgencia_real,
                    "sla": INTERPRETACION[intencion_real]["sla_sugerido"]
                })
                st.session_state.total_sesion += 1
                st.session_state.ultima_clasificacion = {
                    "intencion": intencion_real,
                    "info": INTERPRETACION[intencion_real],
                    "urgencia": urgencia_real,
                    "texto": texto_manual.strip(),
                    "canal": st.session_state.chat_canal
                }
                st.rerun()

        else:  # Modo aleatorio
            nuevo_msg = st.button(
                "Generar mensaje aleatorio",
                type="primary",
                use_container_width=True,
                key="btn_nuevo_msg"
            )

            if nuevo_msg:
                cat = random.choice(list(MENSAJES_SIMULADOS.keys()))
                texto_cliente = random.choice(MENSAJES_SIMULADOS[cat])

                intencion_real = predecir_intencion(texto_cliente)
                urgencia_real  = detectar_urgencia(texto_cliente)
                ts_now = datetime.now().strftime("%H:%M")

                st.session_state.chat_mensajes.append({
                    "tipo": "cliente", "texto": texto_cliente,
                    "intencion": intencion_real, "urgencia": urgencia_real,
                    "timestamp": ts_now
                })
                st.session_state.historial.append({
                    "timestamp": ts_now,
                    "canal": st.session_state.chat_canal,
                    "agente": "Chat Sim",
                    "mensaje": texto_cliente[:100],
                    "intencion": intencion_real,
                    "prioridad": INTERPRETACION[intencion_real]["prioridad"],
                    "urgencia": urgencia_real,
                    "sla": INTERPRETACION[intencion_real]["sla_sugerido"]
                })
                st.session_state.total_sesion += 1
                st.session_state.ultima_clasificacion = {
                    "intencion": intencion_real,
                    "info": INTERPRETACION[intencion_real],
                    "urgencia": urgencia_real,
                    "texto": texto_cliente,
                    "canal": st.session_state.chat_canal
                }
                st.rerun()

    # ── Panel Copilot lateral del chat ───────────────────────────────────────
    with panel_col:
        st.markdown("### Copilot del Agente")

        if not st.session_state.chat_mensajes:
            st.info("El Copilot aparecerá aquí cuando el cliente envíe su primer mensaje.")
        else:
            ultimo_cliente = next(
                (m for m in reversed(st.session_state.chat_mensajes) if m["tipo"] == "cliente"),
                None
            )
            if ultimo_cliente:
                intent_actual = ultimo_cliente["intencion"]
                info_actual   = INTERPRETACION[intent_actual]
                script        = COPILOT_SCRIPTS[intent_actual]

                # Badge de intención
                st.markdown(
                    f"<span class='intent-pill intent-{intent_actual}' style='font-size:14px;padding:5px 14px'>"
                    f"{info_actual['icono']} {intent_actual} · {info_actual['prioridad']}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**SLA:** {info_actual['sla_sugerido']}")
                st.markdown("---")

                # Script de apertura
                st.markdown("**📢 Apertura recomendada:**")
                st.markdown(
                    f"<div class='copilot-box' style='font-size:13px'>{script['apertura']}</div>",
                    unsafe_allow_html=True
                )

                # Checklist
                st.markdown("**✅ Checklist:**")
                checklist_html = ""
                for tipo, item in script["checklist"]:
                    icon_cls = "checklist-icon-ok" if tipo == "ok" else "checklist-icon-warn"
                    icon_char = "✓" if tipo == "ok" else "⚑"
                    checklist_html += (
                        f"<div class='checklist-item'>"
                        f"<span class='{icon_cls}'>{icon_char}</span>"
                        f"<span>{item}</span></div>"
                    )
                st.markdown(
                    f"<div class='beex-card' style='padding:12px 16px'>{checklist_html}</div>",
                    unsafe_allow_html=True
                )

                # Respuesta sugerida (RAG) — estilo copilot-inline
                faqs_panel = buscar_faq(ultimo_cliente["texto"], intent_actual)
                if faqs_panel:
                    st.markdown(
                        f"<div style='background:#fafaf7;border:1.5px solid #e8a849;border-left:4px solid #e8a849;"
                        f"border-radius:8px;padding:12px 16px;margin:8px 0;font-size:13px;color:#555;line-height:1.6'>"
                        f"<div style='font-size:11px;font-weight:700;color:#b45309;text-transform:uppercase;"
                        f"letter-spacing:0.5px;margin-bottom:6px'>Copilot — Respuesta sugerida para {intent_actual}</div>"
                        f"{faqs_panel[0]['respuesta']}</div>",
                        unsafe_allow_html=True
                    )
                    # Botón para enviar la respuesta sugerida directamente al chat
                    if st.button("📤 Enviar respuesta sugerida al chat", key="btn_enviar_sugerida",
                                 use_container_width=True, type="primary"):
                        st.session_state.chat_mensajes.append({
                            "tipo": "agente",
                            "texto": faqs_panel[0]["respuesta"],
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        st.rerun()

                # Alerta urgencia
                if ultimo_cliente.get("urgencia"):
                    st.markdown(
                        "<div class='urgencia-banner' style='font-size:13px'>⚡ URGENCIA — "
                        "Escalar a supervisor · Fast-Track activo</div>",
                        unsafe_allow_html=True
                    )

                # Cierre recomendado
                st.markdown("**🔚 Cierre sugerido:**")
                st.markdown(
                    f"<div class='copilot-box' style='font-size:13px'>{script['cierre']}</div>",
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HISTORIAL (HU-01)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Historial unificado de la sesión")
    st.caption("Vista centralizada de todas las interacciones clasificadas (HU-01)")

    if not st.session_state.historial:
        st.info("Aún no hay clasificaciones en esta sesión.")
    else:
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        total = len(st.session_state.historial)
        urgentes = sum(1 for h in st.session_state.historial if h["urgencia"])
        canales_unicos = len(set(h["canal"] for h in st.session_state.historial))
        intencion_frecuente = pd.Series(
            [h["intencion"] for h in st.session_state.historial]
        ).value_counts().index[0]

        with h_col1: st.metric("Total interacciones", total)
        with h_col2: st.metric("⚡ Con urgencia", urgentes,
                               delta=f"{(urgentes/total*100):.0f}% del total" if total else None)
        with h_col3: st.metric("📡 Canales usados", canales_unicos)
        with h_col4: st.metric("📊 Categoría dominante", intencion_frecuente)

        st.markdown("---")
        df_hist = pd.DataFrame(st.session_state.historial)
        df_hist["urgencia"] = df_hist["urgencia"].map({True: "⚡ SÍ", False: "—"})
        df_hist.columns = ["Hora", "Canal", "Agente", "Mensaje (preview)", "Intención", "Prioridad", "Urgencia", "SLA"]

        st.dataframe(df_hist, use_container_width=True,
                     height=min(400, 60 + len(df_hist) * 35),
                     column_config={
                         "Hora": st.column_config.TextColumn("🕐 Hora", width="small"),
                         "Canal": st.column_config.TextColumn("📡 Canal", width="small"),
                         "Agente": st.column_config.TextColumn("👤 Agente", width="small"),
                         "Mensaje (preview)": st.column_config.TextColumn("💬 Mensaje", width="large"),
                         "Intención": st.column_config.TextColumn("🎯 Intención", width="small"),
                         "Prioridad": st.column_config.TextColumn("🚦 Prioridad", width="small"),
                         "Urgencia": st.column_config.TextColumn("⚡ Urgencia", width="small"),
                         "SLA": st.column_config.TextColumn("⏱️ SLA", width="small"),
                     })

        csv_data = df_hist.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("Descargar historial CSV", data=csv_data,
                           file_name=f"beex_historial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DASHBOARD (HU-02)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Dashboard de métricas en tiempo real")
    st.caption("Indicadores de rendimiento operativo de la sesión actual (HU-02)")

    if not st.session_state.historial:
        st.info("El dashboard se activa después de clasificar al menos 1 interacción.")
    else:
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            df_dash = pd.DataFrame(st.session_state.historial)
            total_d    = len(df_dash)
            urgentes_d = df_dash["urgencia"].sum()
            soporte_d  = (df_dash["intencion"] == "Soporte").sum()
            reclamos_d = (df_dash["intencion"] == "Reclamos").sum()
            ventas_d   = (df_dash["intencion"] == "Ventas").sum()
            factura_d  = (df_dash["intencion"] == "Facturacion").sum()

            d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)
            with d_col1: st.metric("📨 Total", total_d)
            with d_col2: st.metric("🔧 Soporte",     soporte_d,  delta=f"{(soporte_d/total_d*100):.0f}%")
            with d_col3: st.metric("⚠️ Reclamos",    reclamos_d, delta=f"{(reclamos_d/total_d*100):.0f}%", delta_color="inverse")
            with d_col4: st.metric("🛒 Ventas",       ventas_d,   delta=f"{(ventas_d/total_d*100):.0f}%")
            with d_col5: st.metric("⚡ Urgencias",    int(urgentes_d), delta=f"{(urgentes_d/total_d*100):.0f}%", delta_color="inverse")

            st.markdown("---")
            g_col1, g_col2 = st.columns(2)
            colores_pastel = {"Soporte":"#3b82f6","Facturacion":"#f59e0b","Ventas":"#10b981","Reclamos":"#ef4444"}

            with g_col1:
                conteo = df_dash["intencion"].value_counts().reset_index()
                conteo.columns = ["Intención", "Cantidad"]
                fig_donut = px.pie(conteo, names="Intención", values="Cantidad", hole=0.5,
                                   title="🎯 Distribución de intenciones",
                                   color="Intención", color_discrete_map=colores_pastel)
                fig_donut.update_layout(margin=dict(t=50,b=0,l=0,r=0),
                                        legend=dict(orientation="h",yanchor="bottom",y=-0.2))
                st.plotly_chart(fig_donut, use_container_width=True)

            with g_col2:
                conteo_canal = df_dash["canal"].value_counts().reset_index()
                conteo_canal.columns = ["Canal", "Cantidad"]
                fig_canal = px.bar(conteo_canal, x="Canal", y="Cantidad",
                                   title="📡 Interacciones por canal (HU-04)",
                                   color="Cantidad", color_continuous_scale="Blues", text_auto=True)
                fig_canal.update_layout(showlegend=False, margin=dict(t=50,b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_canal, use_container_width=True)

            g_col3, g_col4 = st.columns(2)
            with g_col3:
                df_tl = df_dash.copy()
                df_tl["#"] = range(1, len(df_tl)+1)
                fig_line = px.scatter(df_tl, x="#", y="intencion", color="intencion",
                                      title="🕐 Línea de tiempo",
                                      color_discrete_map=colores_pastel,
                                      labels={"#":"Interacción #","intencion":"Categoría"})
                fig_line.update_traces(marker_size=14)
                fig_line.update_layout(showlegend=False, margin=dict(t=50,b=0))
                st.plotly_chart(fig_line, use_container_width=True)

            with g_col4:
                st.markdown("#### ⏱️ SLA por categoría")
                sla_table = pd.DataFrame([
                    {"Categoría":"🔧 Soporte",     "SLA":"≤ 2 min",  "Cantidad":soporte_d,  "% Total":f"{(soporte_d/total_d*100):.0f}%"},
                    {"Categoría":"💰 Facturación", "SLA":"≤ 3 min",  "Cantidad":factura_d,  "% Total":f"{(factura_d/total_d*100):.0f}%"},
                    {"Categoría":"🛒 Ventas",      "SLA":"≤ 1 min",  "Cantidad":ventas_d,   "% Total":f"{(ventas_d/total_d*100):.0f}%"},
                    {"Categoría":"⚠️ Reclamos",   "SLA":"≤ 30 seg", "Cantidad":reclamos_d, "% Total":f"{(reclamos_d/total_d*100):.0f}%"},
                ])
                st.dataframe(sla_table, use_container_width=True, hide_index=True)

                no_urg = total_d - int(urgentes_d)
                fig_urg = go.Figure(go.Bar(
                    x=["Sin urgencia", "Con urgencia"], y=[no_urg, int(urgentes_d)],
                    marker_color=["#10b981","#ef4444"],
                    text=[no_urg, int(urgentes_d)], textposition="outside"
                ))
                fig_urg.update_layout(margin=dict(t=10,b=0), height=200, showlegend=False)
                st.plotly_chart(fig_urg, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🤖 Métricas del modelo en producción (APF3)")
            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            with m_col1: st.metric("Accuracy",  "94.56%")
            with m_col2: st.metric("Precision", "94.77%")
            with m_col3: st.metric("Recall",    "94.56%")
            with m_col4: st.metric("F1-Score",  "94.60%")
            with m_col5: st.metric("Train time","6.50 seg")
            with m_col6: st.metric("Inferencia","0.029 seg")

        except ImportError:
            st.error("⚠️ Plotly no está instalado. Ejecute: `pip install plotly`")
            df_fb = pd.DataFrame(st.session_state.historial)
            st.dataframe(df_fb["intencion"].value_counts().reset_index(), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — COPILOT & FAQ (HU-06, HU-07) — colores corregidos
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Panel Copilot & Respuestas Automáticas")
    st.caption("Asistencia en tiempo real: scripts guiados (HU-06) y base de conocimiento FAQ con RAG simplificado (HU-07).")

    if st.session_state.ultima_clasificacion is None:
        st.info("Clasifique primero una interacción (Clasificador o Chat Simulado) para activar el Copilot.")
    else:
        uc = st.session_state.ultima_clasificacion
        intencion_actual = uc["intencion"]
        info_actual      = uc["info"]
        texto_actual     = uc["texto"]
        canal_actual     = uc["canal"]
        script           = COPILOT_SCRIPTS[intencion_actual]

        # Banner de contexto
        st.markdown(
            f"<div class='beex-card'>"
            f"<p style='margin:0;font-size:13px;color:var(--c-text-muted)'>Última interacción clasificada · Canal: {canal_actual}</p>"
            f"<p style='font-size:20px;font-weight:700;color:{info_actual['color']}'>"
            f"{info_actual['icono']} Intención detectada: {intencion_actual}</p>"
            f"<p style='color:var(--c-text);font-style:italic'>\"{texto_actual[:120]}{'...' if len(texto_actual)>120 else ''}\"</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        cop_col1, cop_col2 = st.columns(2)

        # ── Panel Copilot (HU-06) ────────────────────────────────────────────
        with cop_col1:
            st.markdown("### Panel Copilot para el Agente")

            st.markdown("**📢 Script de apertura recomendado:**")
            st.markdown(
                f"<div class='copilot-box'>{script['apertura']}</div>",
                unsafe_allow_html=True
            )

            st.markdown("**Checklist de atención:**")
            checklist_html = ""
            for tipo, item in script["checklist"]:
                icon_cls  = "checklist-icon-ok" if tipo == "ok" else "checklist-icon-warn"
                icon_char = "✓" if tipo == "ok" else "⚑"
                checklist_html += (
                    f"<div class='checklist-item'>"
                    f"<span class='{icon_cls}'>{icon_char}</span>"
                    f"<span>{item}</span></div>"
                )
            st.markdown(
                f"<div class='beex-card' style='padding:12px 16px'>{checklist_html}</div>",
                unsafe_allow_html=True
            )

            st.markdown("**🔚 Frase de cierre recomendada:**")
            st.markdown(
                f"<div class='copilot-box'>{script['cierre']}</div>",
                unsafe_allow_html=True
            )

            if uc["urgencia"]:
                st.markdown(
                    "<div class='urgencia-banner'>⚡ URGENCIA DETECTADA — "
                    "Aplicar protocolo Fast-Track. Notificar supervisor y escalar "
                    "con código de prioridad ROJO.</div>",
                    unsafe_allow_html=True
                )

        # ── Panel FAQ / RAG (HU-07) ──────────────────────────────────────────
        with cop_col2:
            st.markdown("### Respuestas Automáticas Sugeridas (RAG)")
            st.caption(f"Base de conocimiento Beex · Categoría: {intencion_actual}")

            faqs = buscar_faq(texto_actual, intencion_actual)
            if faqs:
                for i, faq in enumerate(faqs, 1):
                    with st.expander(f"📄 {faq['pregunta']}", expanded=(i == 1)):
                        st.markdown(
                            f"<div class='faq-box'>{faq['respuesta']}</div>",
                            unsafe_allow_html=True
                        )
                        st.button("Usar esta respuesta", key=f"usar_faq_{i}",
                                  help="Copie esta respuesta para enviarla al cliente.")
            else:
                st.info("No se encontraron respuestas automáticas. El agente debe responder manualmente.")

            st.markdown("---")
            st.markdown("#### Buscar en base de conocimiento")
            busqueda_manual = st.text_input("Buscar por categoría:",
                                            placeholder="Ej: 'Reclamos' o 'Ventas'", key="busqueda_faq")
            if busqueda_manual:
                cat_busq = next((c for c in FAQ_BEEX if c.lower() in busqueda_manual.lower()), None)
                if cat_busq:
                    st.markdown(f"**Resultados para: {cat_busq}**")
                    for item in FAQ_BEEX[cat_busq]:
                        with st.expander(f"📄 {item['pregunta']}"):
                            st.markdown(
                                f"<div class='faq-box'>{item['respuesta']}</div>",
                                unsafe_allow_html=True
                            )
                else:
                    st.warning("Intente con: Soporte, Facturacion, Ventas o Reclamos.")

    # Explorador completo
    if st.session_state.ultima_clasificacion is not None:
        st.markdown("---")
        st.markdown("### Explorador completo de base de conocimiento")
        cat_sel = st.selectbox(
            "Seleccione una categoría:",
            options=list(FAQ_BEEX.keys()),
            format_func=lambda x: {"Soporte":"🔧 Soporte","Facturacion":"💰 Facturación",
                                    "Ventas":"🛒 Ventas","Reclamos":"⚠️ Reclamos"}.get(x, x),
            key="cat_explorador"
        )
        for item in FAQ_BEEX[cat_sel]:
            with st.expander(f"📄 {item['pregunta']}"):
                st.markdown(
                    f"<div class='faq-box'>{item['respuesta']}</div>",
                    unsafe_allow_html=True
                )

# ==============================================================================
# PIE DE PÁGINA
# ==============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:12px'>"
    "BEEX Suite — Proyecto Final · Innovación y Transformación Digital · UTP · 2026 | "
    "Modelo SVM · TF-IDF · Dataset sintético 1,500 interacciones · Ley N.° 29733"
    "</p>",
    unsafe_allow_html=True
)

# ==============================================================================
# PERSISTENCIA LOCAL AUTOMÁTICA
# ==============================================================================
try:
    with open("historial_persistente.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.historial, f, ensure_ascii=False, indent=2)
    with open("chat_persistente.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_mensajes, f, ensure_ascii=False, indent=2)
except Exception:
    pass
