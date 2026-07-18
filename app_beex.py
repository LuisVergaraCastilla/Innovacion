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
import random
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="BEEX - Clasificador de Intenciones",
    page_icon="beex",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS PERSONALIZADO — SISTEMA DE DISEÑO BEEX
# ==============================================================================
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════
       1. VARIABLES CSS — PALETA CORPORATIVA Y TEMAS (TAILWIND BEEX)
       ═══════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@100..900&family=Inter:wght@100..900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    :root {
        /* Paleta Primaria (Basada en Tailwind config) */
        --beex-blue:      #004ac6; /* primary */
        --beex-blue-dark: #003ea8; /* on-primary-fixed-variant */
        --beex-blue-light:#2563eb; /* primary-container */
        --beex-green:     #10b981; 
        --beex-green-dark:#047857;
        --beex-green-light:#34d399;
        --beex-orange:    #bc4800; /* tertiary-container */
        --beex-orange-light:#ffb596; /* tertiary-fixed-dim */
        --beex-red:       #ba1a1a; /* error */
        --beex-red-dark:  #93000a; /* on-error-container */
        --beex-red-light: #ffdad6; /* error-container */
        --beex-gray:      #5c647a; /* on-secondary-container */
        --beex-gray-dark: #434655; /* on-surface-variant */
        --beex-gray-light:#c3c6d7; /* outline-variant */

        /* Tema Claro */
        --c-bg:           #f8f9ff; /* background */
        --c-bg-secondary: #eff4ff; /* surface-container-low */
        --c-border:       #c3c6d7; /* outline-variant */
        --c-border-light: #e5eeff; /* surface-container */
        --c-text:         #0b1c30; /* on-background */
        --c-text-secondary:#434655; /* on-surface-variant */
        --c-text-muted:   #737686; /* outline */
        --c-shadow:       rgba(0, 0, 0, 0.05);
        --c-shadow-md:    rgba(0, 0, 0, 0.08);

        /* Chat */
        --chat-bg:        #f8f9ff; /* surface */
        --chat-border:    #d3e4fe; /* surface-variant */
        --bubble-in:      #ffffff; /* surface-container-lowest */
        --bubble-in-border:#c3c6d7;
        --bubble-out:     #004ac6; /* primary */
        --bubble-out-text:#ffffff; /* on-primary */
        --input-bg:       #ffffff;

        /* Componentes */
        --copilot-bg:     rgba(37, 99, 235, 0.06);
        --copilot-border: #2563eb;
        --faq-bg:         rgba(0, 74, 198, 0.06);
        --faq-border:     #004ac6;
        --urgency-bg:     linear-gradient(135deg, #ffdad6 0%, #ffb596 100%);
        --urgency-border: #ba1a1a;
        --urgency-text:   #93000a;

        /* Spacing */
        --space-xs: 4px;
        --space-sm: 8px;
        --space-md: 16px;
        --space-lg: 24px;
        --space-xl: 32px;

        /* Border Radius */
        --radius-sm: 0.25rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-full: 9999px;

        /* Tipografía */
        --font-sans: 'Inter', 'Geist', 'Segoe UI', sans-serif;
        --font-mono: 'Geist', monospace;
        --text-xs: 12px;
        --text-sm: 14px;
        --text-base: 16px;
        --text-lg: 18px;
        --text-xl: 24px;
        --text-2xl: 32px;
    }

    /* Dark Mode (adaptado) */
    @media (prefers-color-scheme: dark) {
        :root {
            --c-bg:           #0b1c30;
            --c-bg-secondary:#131b2e;
            --c-border:       #434655;
            --c-border-light:#213145;
            --c-text:         #ffffff;
            --c-text-secondary:#c3c6d7;
            --c-text-muted:   #737686;
            --c-shadow:       rgba(0, 0, 0, 0.2);
            --c-shadow-md:    rgba(0, 0, 0, 0.3);

            --chat-bg:        #0b1c30;
            --chat-border:    #434655;
            --bubble-in:      #131b2e;
            --bubble-in-border:#434655;
            --bubble-out:     #004ac6;
            --bubble-out-text:#ffffff;
            --input-bg:       #131b2e;

            --copilot-bg:     rgba(37, 99, 235, 0.1);
            --copilot-border: #2563eb;
            --faq-bg:         rgba(0, 74, 198, 0.1);
            --faq-border:     #004ac6;
            --urgency-bg:     linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
            --urgency-border: #ef4444;
            --urgency-text:   #fca5a5;
        }
    }

    /* ═══════════════════════════════════════════════════════════════
       2. TIPOGRAFÍA
       ═══════════════════════════════════════════════════════════════ */
    * { box-sizing: border-box; }
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-sans);
        font-weight: 700;
        color: var(--c-text);
        line-height: 1.2;
    }
    p, span, div, label { font-family: var(--font-sans); }

    /* ═══════════════════════════════════════════════════════════════
       3. BOTONES
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--beex-blue) !important;
        border: none !important;
        color: var(--bubble-out-text) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-md) !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
        padding: 12px 24px !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: var(--beex-blue-dark) !important;
        transform: scale(0.98) !important;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background: var(--c-bg) !important;
        border: 1px solid var(--c-border) !important;
        color: var(--c-text) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-md) !important;
        transition: all 0.2s ease !important;
        padding: 12px 24px !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: var(--beex-blue) !important;
        color: var(--beex-blue) !important;
        background: var(--c-bg-secondary) !important;
        transform: scale(0.98) !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       4. TABS
       ═══════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid var(--c-border);
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        padding: 0 16px;
        font-weight: 600;
        font-size: var(--text-sm);
        color: var(--c-text-muted);
        border-bottom: 4px solid transparent;
        border-radius: var(--radius-md) var(--radius-md) 0 0;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--beex-blue);
        background: var(--c-bg-secondary);
    }
    .stTabs [aria-selected="true"] {
        color: var(--beex-blue) !important;
        border-bottom-color: var(--beex-blue) !important;
        background: var(--c-bg-secondary) !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       5. CARDS
       ═══════════════════════════════════════════════════════════════ */
    .beex-card {
        background: var(--c-bg);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        border: 1px solid var(--c-border);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: var(--space-md);
        transition: all 0.2s ease;
    }
    .beex-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .beex-card-compact {
        background: var(--c-bg);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        border: 1px solid var(--c-border);
        margin-bottom: var(--space-sm);
    }

    /* ═══════════════════════════════════════════════════════════════
       6. BADGES Y PILLS
       ═══════════════════════════════════════════════════════════════ */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: var(--radius-full);
        font-weight: 600;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        letter-spacing: 0.05em;
    }
    .badge-soporte     { background:#dbe1ff; color:#00174b; } /* primary-fixed */
    .badge-facturacion { background:#ffede6; color:#7d2d00; } /* tertiary-container-ish */
    .badge-ventas      { background:#d3e4fe; color:#003ea8; } /* surface-container-highest */
    .badge-reclamos    { background:#ffdad6; color:#93000a; } /* error-container */

    /* ═══════════════════════════════════════════════════════════════
       7. BANNERS Y ALERTAS
       ═══════════════════════════════════════════════════════════════ */
    .urgencia-banner {
        background: var(--urgency-bg);
        border-left: 4px solid var(--urgency-border);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin-top: var(--space-md);
        font-weight: 600;
        color: var(--urgency-text);
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        animation: pulse-urgency 2s infinite;
    }
    @keyframes pulse-urgency {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }

    /* ═══════════════════════════════════════════════════════════════
       8. COPILOT Y FAQ BOXES
       ═══════════════════════════════════════════════════════════════ */
    .copilot-box {
        background: var(--copilot-bg);
        border-left: 4px solid var(--copilot-border);
        border-radius: var(--radius-md);
        padding: var(--space-md) var(--space-lg);
        margin: var(--space-sm) 0;
        font-size: var(--text-sm);
        color: var(--c-text);
        line-height: 1.7;
    }
    .faq-box {
        background: var(--faq-bg);
        border-left: 4px solid var(--faq-border);
        border-radius: var(--radius-md);
        padding: var(--space-md) var(--space-lg);
        margin: var(--space-sm) 0;
        font-size: var(--text-sm);
        color: var(--c-text);
        line-height: 1.7;
    }

    /* ═══════════════════════════════════════════════════════════════
       9. CHECKLIST
       ═══════════════════════════════════════════════════════════════ */
    .checklist-item {
        display: flex;
        align-items: flex-start;
        gap: var(--space-sm);
        padding: var(--space-sm) 0;
        font-size: var(--text-sm);
        color: var(--c-text-secondary);
        border-bottom: 1px solid var(--c-border-light);
        transition: background 0.15s ease;
    }
    .checklist-item:hover {
        background: var(--c-bg-secondary);
        margin: 0 calc(-1 * var(--space-sm));
        padding-left: var(--space-sm);
        padding-right: var(--space-sm);
        border-radius: var(--radius-sm);
    }
    .checklist-item:last-child { border-bottom: none; }
    .checklist-icon-ok   { color: var(--beex-green); font-weight: 700; flex-shrink: 0; }
    .checklist-icon-warn { color: var(--beex-red); font-weight: 700; flex-shrink: 0; }

    /* ═══════════════════════════════════════════════════════════════
       10. CHAT SIMULADO
       ═══════════════════════════════════════════════════════════════ */
    .chat-wrapper {
        background: var(--chat-bg);
        border: 1px solid var(--chat-border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        box-shadow: 0 2px 8px var(--c-shadow);
    }
    .chat-header-sim {
        background: var(--c-bg);
        border-bottom: 1px solid var(--chat-border);
        padding: var(--space-md);
        display: flex;
        align-items: center;
        gap: var(--space-md);
    }
    .chat-avatar {
        width: 42px; height: 42px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--beex-blue-light) 0%, var(--beex-blue) 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }
    .chat-user-info { flex: 1; }
    .chat-user-name { font-weight: 600; font-size: var(--text-base); color: var(--c-text); }
    .chat-user-meta { font-size: var(--text-xs); color: var(--c-text-muted); margin-top: 2px; }
    .chat-status-pill {
        background: rgba(5, 150, 105, 0.1);
        color: var(--beex-green);
        border: 1px solid var(--beex-green-light);
        border-radius: var(--radius-full);
        padding: 4px 12px;
        font-size: var(--text-xs);
        font-weight: 600;
    }
    .messages-area-sim {
        padding: var(--space-md);
        min-height: 200px;
        max-height: 520px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: var(--space-md);
        scroll-behavior: smooth;
    }
    .msg-in { align-self: flex-start; max-width: 75%; }
    .msg-out { align-self: flex-end; max-width: 75%; }
    .bubble-in {
        background: var(--bubble-in);
        border: 1px solid var(--bubble-in-border);
        border-radius: var(--radius-sm) var(--radius-lg) var(--radius-lg) var(--radius-lg);
        padding: var(--space-sm) var(--space-md);
        font-size: var(--text-sm);
        color: var(--c-text);
        line-height: 1.6;
    }
    .bubble-out {
        background: var(--bubble-out);
        border: 1px solid var(--beex-blue-dark);
        border-radius: var(--radius-lg) var(--radius-sm) var(--radius-lg) var(--radius-lg);
        padding: var(--space-sm) var(--space-md);
        font-size: var(--text-sm);
        color: var(--bubble-out-text);
        line-height: 1.6;
    }
    .msg-time-sim {
        font-size: var(--text-xs);
        color: var(--c-text-muted);
        margin-top: var(--space-xs);
    }
    .msg-time-sim.right { text-align: right; }

    /* ═══════════════════════════════════════════════════════════════
       11. INTENT PILLS
       ═══════════════════════════════════════════════════════════════ */
    .intent-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: var(--radius-full);
        font-size: var(--text-xs);
        font-weight: 600;
        margin-top: var(--space-xs);
    }
    .intent-Soporte     { background:#dbeafe; color:#1e40af; }
    .intent-Facturacion { background:#fef3c7; color:#92400e; }
    .intent-Ventas      { background:#d1fae5; color:#065f46; }
    .intent-Reclamos    { background:#fee2e2; color:#991b1b; }

    /* ═══════════════════════════════════════════════════════════════
       12. COPILOT INLINE
       ═══════════════════════════════════════════════════════════════ */
    .copilot-inline {
        background: var(--c-bg);
        border: 1px solid var(--chat-border);
        border-radius: var(--radius-md);
        padding: var(--space-sm) var(--space-md);
        margin-top: var(--space-sm);
        font-size: var(--text-xs);
        color: var(--c-text-secondary);
        line-height: 1.6;
        max-width: 85%;
    }
    .copilot-inline-header {
        font-size: 10px;
        font-weight: 700;
        color: var(--beex-blue);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: var(--space-xs);
    }
    .copilot-urgencia-inline {
        background: var(--urgency-bg);
        border: 1px solid var(--urgency-border);
        border-radius: var(--radius-md);
        padding: var(--space-sm) var(--space-md);
        font-size: var(--text-xs);
        color: var(--urgency-text);
        font-weight: 600;
        margin-top: var(--space-sm);
    }

    /* ═══════════════════════════════════════════════════════════════
       13. SIDEBAR
       ═══════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: var(--c-bg-secondary);
    }

    /* ═══════════════════════════════════════════════════════════════
       14. MÉTRICAS
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: var(--c-bg);
        border: 1px solid var(--c-border);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 2px 8px var(--c-shadow-md);
    }

    /* ═══════════════════════════════════════════════════════════════
       15. TABLA HISTORIAL
       ═══════════════════════════════════════════════════════════════ */
    .dataframe tbody tr:nth-child(even) { background-color: var(--c-bg-secondary); }
    .dataframe tbody tr:hover { background-color: rgba(37, 99, 235, 0.04); }

    /* ═══════════════════════════════════════════════════════════════
       16. UTILIDADES
       ═══════════════════════════════════════════════════════════════ */
    .beex-mt-xs { margin-top: var(--space-xs) !important; }
    .beex-mt-sm { margin-top: var(--space-sm) !important; }
    .beex-mt-md { margin-top: var(--space-md) !important; }
    .beex-mt-lg { margin-top: var(--space-lg) !important; }
    .beex-mb-xs { margin-bottom: var(--space-xs) !important; }
    .beex-mb-sm { margin-bottom: var(--space-sm) !important; }
    .beex-mb-md { margin-bottom: var(--space-md) !important; }
    .beex-mb-lg { margin-bottom: var(--space-lg) !important; }
    .beex-text-center { text-align: center !important; }
    .beex-text-muted { color: var(--c-text-muted) !important; }
    .beex-font-bold { font-weight: 700 !important; }
    .beex-divider {
        border: none;
        border-top: 1px solid var(--c-border);
        margin: var(--space-lg) 0;
    }

    /* ═══════════════════════════════════════════════════════════════
       17. ANIMACIONES
       ═══════════════════════════════════════════════════════════════ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .beex-fade-in { animation: fadeIn 0.3s ease-out; }

    /* Tabla historial zebra */
    .dataframe tbody tr:nth-child(even) { background-color: var(--c-bg-secondary); }

    /* ═══════════════════════════════════════════════════════════════
       18. SIDEBAR MENU (RADIO COMO LINKS)
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 8px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 12px 16px;
        background-color: transparent;
        border-radius: var(--radius-md);
        margin: 0;
        cursor: pointer;
        transition: all 0.2s ease;
        border-left: 4px solid transparent !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: var(--c-bg-secondary) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover div[data-testid="stMarkdownContainer"] p {
        color: var(--beex-blue) !important;
    }
    
    /* Estado Activo / Seleccionado */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: var(--c-bg-secondary) !important;
        border-left: 4px solid var(--beex-blue) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: var(--beex-blue) !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] div[data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p::before {
        font-variation-settings: "FILL" 1 !important;
        color: var(--beex-blue) !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-size: var(--text-base) !important;
        font-weight: 600;
        color: var(--c-text-secondary);
        margin: 0;
        display: flex !important;
        align-items: center;
        gap: 12px;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p::before {
        font-family: 'Material Symbols Outlined';
        font-size: 22px;
        font-weight: normal;
        font-style: normal;
        line-height: 1;
        display: inline-block;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: normal;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
        transition: all 0.2s ease;
        font-variation-settings: "FILL" 0;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1) div[data-testid="stMarkdownContainer"] p::before {
        content: 'dashboard';
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2) div[data-testid="stMarkdownContainer"] p::before {
        content: 'history';
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3) div[data-testid="stMarkdownContainer"] p::before {
        content: 'smart_toy';
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4) div[data-testid="stMarkdownContainer"] p::before {
        content: 'category';
    }
    
    /* Ocultar el círculo nativo de Streamlit de forma segura sin romper el click */
    [data-testid="stSidebar"] div[role="radiogroup"] label div:has(> [data-testid="stMarkdownContainer"]) > *:first-child {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        position: absolute !important;
        pointer-events: none !important;
    }
    
    /* Ocultar label del widget si existe */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }
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

def generar_respuesta_llm_local(texto_cliente: str, intencion_clasificada: str, historial_chat: list) -> tuple:
    import requests
    
    # 1. Recuperamos la respuesta local de la base de conocimiento como base / fallback
    contexto_faq = buscar_faq(texto_cliente, intencion_clasificada)
    contexto_str = "\n".join([f"Pregunta: {f['pregunta']}\nRespuesta: {f['respuesta']}" for f in contexto_faq[:2]])
    resp_fallback = contexto_faq[0]["respuesta"] if contexto_faq else "No hay respuestas recomendadas disponibles."
    
    script_guia = COPILOT_SCRIPTS.get(intencion_clasificada, {"apertura": "", "cierre": ""})
    
    system_prompt = f"""
    Eres un agente de atención al cliente de la empresa Beex. 
    Tu objetivo es responder de manera amable, directa y profesional.
    La intención detectada del cliente es: {intencion_clasificada}.
    
    Usa la siguiente información autorizada de la base de conocimiento para responder:
    {contexto_str}
    
    Lineamientos obligatorios:
    - Debes saludar si es el inicio de la conversación.
    - Sé muy claro y conciso.
    - Utiliza un tono corporativo formal.
    """
    
    # Formateamos el prompt en formato simple de diálogo
    prompt_completo = f"System: {system_prompt}\n"
    for msg in historial_chat[-4:]:
        role = "User" if msg["tipo"] == "cliente" else "Assistant"
        prompt_completo += f"{role}: {msg['texto']}\n"
    prompt_completo += f"User: {texto_cliente}\nAssistant: "
    
    try:
        # Hacemos la llamada al API local de Ollama (Recomendamos el modelo 'llama3' (8B))
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt_completo,
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=30 # Incrementamos el timeout a 30s para que la CPU de Ollama no muera en el intento
        )
        if response.status_code == 200:
            res_txt = response.json().get("response", "").strip()
            if res_txt:
                return res_txt, "Ollama (Llama 3)"
    except Exception:
        # Si Ollama no está activo o no tiene el modelo, retorna la respuesta estática RAG
        pass
        
    return resp_fallback, "Base de Conocimiento (Fallback)"

def renderizar_copilot_y_faq(intencion_actual, info_actual, texto_actual, canal_actual, uc, suffix="", compact=False):
    script = COPILOT_SCRIPTS[intencion_actual]

    if compact:
        st.markdown(
            """
            <div style='display:flex;align-items:center;gap:8px;padding-bottom:8px;border-bottom:1px solid var(--c-border);margin-bottom:12px;margin-top:24px'>
                <span class='material-symbols-outlined' style='color:var(--beex-blue);font-size:20px;font-variation-settings:"FILL" 1'>smart_toy</span>
                <h4 style='margin:0;font-size:14px;font-weight:700;color:var(--beex-blue)'>Agent Copilot (Asistente Rápido)</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("<span style='font-size:12px;font-weight:700;color:var(--c-text-secondary)'>📢 Script de Apertura:</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='copilot-box' style='padding:8px 12px;font-size:12px;margin:4px 0'>{script['apertura']}</div>", unsafe_allow_html=True)
        with col_s2:
            st.markdown("<span style='font-size:12px;font-weight:700;color:var(--c-text-secondary)'>🔚 Frase de Cierre:</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='copilot-box' style='padding:8px 12px;font-size:12px;margin:4px 0'>{script['cierre']}</div>", unsafe_allow_html=True)
            
        st.markdown("<span style='font-size:12px;font-weight:700;color:var(--c-text-secondary)'>📋 Checklist de atención:</span>", unsafe_allow_html=True)
        checklist_html = ""
        for tipo, item in script["checklist"]:
            icon_char = "✓" if tipo == "ok" else "⚑"
            icon_cls = "checklist-icon-ok" if tipo == "ok" else "checklist-icon-warn"
            checklist_html += f"<div style='font-size:12px;display:flex;gap:8px;align-items:center;margin:2px 0'><span class='{icon_cls}'>{icon_char}</span><span>{item}</span></div>"
        st.markdown(f"<div class='beex-card' style='padding:8px 12px;margin:4px 0 12px 0'>{checklist_html}</div>", unsafe_allow_html=True)
        
        faqs = buscar_faq(texto_actual, intencion_actual)
        if faqs:
            st.markdown("<span style='font-size:12px;font-weight:700;color:var(--c-text-secondary)'>📄 Respuesta sugerida más relevante:</span>", unsafe_allow_html=True)
            faq = faqs[0]
            st.markdown(f"<div class='faq-box' style='padding:8px 12px;font-size:12px;margin:4px 0 8px 0'>{faq['respuesta']}</div>", unsafe_allow_html=True)
            
        if uc["urgencia"]:
            st.markdown(
                "<div class='urgencia-banner' style='padding:8px 12px;font-size:12px;margin-top:8px'>⚡ URGENCIA DETECTADA — Aplicar protocolo Fast-Track.</div>",
                unsafe_allow_html=True
            )
    else:
        # ── Panel Copilot (HU-06) ────────────────────────────────────────────
        with st.container():
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
        with st.container():
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
                        st.button("Usar esta respuesta", key=f"usar_faq_{i}_{suffix}",
                                  help="Copie esta respuesta para enviarla al cliente.")
            else:
                st.info("No se encontraron respuestas automáticas. El agente debe responder manualmente.")

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
    if "ultima_clasificacion_manual" not in st.session_state:
        st.session_state.ultima_clasificacion_manual = None
    if "ultima_clasificacion_chat" not in st.session_state:
        st.session_state.ultima_clasificacion_chat = None
    if "sugerencia_actual" not in st.session_state:
        st.session_state.sugerencia_actual = ""
    if "composer_key" not in st.session_state:
        st.session_state.composer_key = "resp_agente_txt_0"
    if "sugerencia_proveedor" not in st.session_state:
        st.session_state.sugerencia_proveedor = ""
    if "total_sesion" not in st.session_state:
        st.session_state.total_sesion = len(st.session_state.historial)
    if "chat_cliente" not in st.session_state:
        st.session_state.chat_cliente = random.choice(NOMBRES_CLIENTES)
    if "chat_canal" not in st.session_state:
        st.session_state.chat_canal = random.choice(CANALES_CHAT)

inicializar_session_state()

# ==============================================================================
# DIÁLOGO DE CONFIRMACIÓN PARA NUEVA SESIÓN
# ==============================================================================
@st.dialog("Confirmar Nueva Sesión")
def dialog_nueva_sesion():
    st.write("Los chats y clasificaciones de la sesión actual que no hayan sido guardados se eliminarán permanentemente.")
    
    # Preparar datos para descargar
    historial = st.session_state.get("historial", [])
    chat_mensajes = st.session_state.get("chat_mensajes", [])
    
    # Ofrecer opciones de exportación
    st.markdown("#### Exportar datos actuales:")
    col1, col2 = st.columns(2)
    with col1:
        if historial:
            df = pd.DataFrame(historial)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Historial (CSV)",
                data=csv,
                file_name=f"historial_beex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button(
                label="Descargar Historial (CSV)",
                disabled=True,
                use_container_width=True,
                key="btn_descargar_historial_disabled"
            )
            
    with col2:
        if chat_mensajes:
            chat_json = json.dumps(chat_mensajes, indent=2, ensure_ascii=False)
            st.download_button(
                label="Descargar Chat (JSON)",
                data=chat_json,
                file_name=f"chat_beex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.button(
                label="Descargar Chat (JSON)",
                disabled=True,
                use_container_width=True,
                key="btn_descargar_chat_disabled"
            )
            
    st.markdown("---")
    
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("Cancelar", use_container_width=True):
            st.rerun()
            
    with col_confirm:
        if st.button("Sí, borrar todo", type="primary", use_container_width=True):
            # Crear backups antes de borrar
            import shutil
            if os.path.exists("historial_persistente.json"):
                try:
                    shutil.copy("historial_persistente.json", "historial_persistente_backup.json")
                except Exception:
                    pass
                os.remove("historial_persistente.json")
            if os.path.exists("chat_persistente.json"):
                try:
                    shutil.copy("chat_persistente.json", "chat_persistente_backup.json")
                except Exception:
                    pass
                os.remove("chat_persistente.json")
                
            for k in ["historial", "ultima_clasificacion", "total_sesion",
                      "chat_mensajes", "chat_cliente", "chat_canal"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:32px'>
            <div style='width:40px;height:40px;border-radius:8px;background:var(--beex-blue-light);color:white;display:flex;align-items:center;justify-content:center'>
                <span class='material-symbols-outlined' style='font-size:24px;font-variation-settings:"FILL" 1'>hive</span>
            </div>
            <div>
                <h1 style='margin:0;font-size:24px;line-height:28px;font-weight:700;color:var(--beex-blue);letter-spacing:-0.02em'>BEEX AI</h1>
                <p style='margin:4px 0 0 0;font-size:12px;font-weight:600;color:var(--beex-blue);display:flex;align-items:center;gap:4px;letter-spacing:0.05em;text-transform:uppercase'>
                    <span style='width:8px;height:8px;border-radius:50%;background:#10b981;box-shadow:0 0 8px rgba(16,185,129,0.5)'></span>
                    Model Active
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    
    # Navegación Principal
    menu = st.radio(
        "Navegación",
        ["Dashboard", "Historial", "Copilot", "Clasificador"],
        label_visibility="collapsed"
    )

    # Espaciador para enviar el resto al fondo
    st.markdown("<div style='margin-top: 150px'></div>", unsafe_allow_html=True)

    # Botón de Nueva Sesión (Primary)
    if st.button("+ New Session", type="primary", use_container_width=True, key="btn_limpiar_sesion"):
        dialog_nueva_sesion()

    # Botón para restaurar última sesión si hay backups
    if os.path.exists("historial_persistente_backup.json") or os.path.exists("chat_persistente_backup.json"):
        if st.button("🔄 Restore Last Session", type="secondary", use_container_width=True, key="btn_restaurar_backup"):
            import shutil
            if os.path.exists("historial_persistente_backup.json"):
                try:
                    shutil.copy("historial_persistente_backup.json", "historial_persistente.json")
                    os.remove("historial_persistente_backup.json")
                except Exception:
                    pass
            if os.path.exists("chat_persistente_backup.json"):
                try:
                    shutil.copy("chat_persistente_backup.json", "chat_persistente.json")
                    os.remove("chat_persistente_backup.json")
                except Exception:
                    pass
            
            # Forzar recarga de session state
            for k in ["historial", "ultima_clasificacion", "total_sesion", "chat_mensajes"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.toast("✅ Sesión restaurada con éxito", icon="🔄")
            st.rerun()

    # Perfil del Agente
    st.markdown(
        """
        <div style='display:flex;align-items:center;gap:12px;margin-top:16px;padding-top:16px;border-top:1px solid var(--c-border)'>
            <div style='width:36px;height:36px;border-radius:50%;background:var(--c-bg-secondary);border:1px solid var(--c-border);display:flex;align-items:center;justify-content:center;overflow:hidden'>
                <img src='https://lh3.googleusercontent.com/aida-public/AB6AXuD9AxP-2KR_3RpRcCNlk-N5zl6CILYdBTZhbDvn9NpE_tkPnRYL8MFwwQNLiph96f7NTzw1lcY2fWO18KfkYqmi84Ss5AUlEydy-bw53_L3n-oa8ccKuWhxOoqGCFGL_1Wzet1rqwrQ-GyiiPGHYQ8nBVfoXDIN-tJXcBDKdCoZ6anez0LsY4WLH9MVPwOU27OsREHQSiNyPnFrTtqYhpYrTjNs4evxW9JnfF1iUc_mi0xZgu_2ldvnRQ' style='width:100%;height:100%;object-fit:cover'>
            </div>
            <div style='display:flex;flex-direction:column'>
                <span style='font-size:14px;font-weight:600;color:var(--c-text);line-height:1'>Agent Admin</span>
                <span style='font-size:12px;font-weight:600;color:var(--c-text-muted);letter-spacing:0.05em;margin-top:4px'>ID: BX-994</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# RUTEO DE PÁGINAS PRINCIPALES
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — CLASIFICADOR
# ─────────────────────────────────────────────────────────────────────────────
if menu == "Clasificador":
    st.markdown(
        "<div style='margin-bottom:16px'>"
        "<h2 style='margin:0 0 4px 0;font-size:1.35rem'>Ingrese la interacción del cliente</h2>"
        "<p style='margin:0;color:var(--c-text-muted);font-size:0.9rem'>"
        "Escriba o pegue el mensaje del cliente (WhatsApp, transcripción, webchat, etc.)</p>"
        "</div>",
        unsafe_allow_html=True
    )

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
            st.session_state.ultima_clasificacion_manual = {
                "intencion": intencion, "info": info,
                "urgencia": urgencia, "texto": texto_usuario, "canal": canal_limpio
            }
            st.session_state.total_sesion += 1
        else:
            st.warning("⚠️ Por favor, ingrese un mensaje del cliente para clasificar.")

    # ── Resultado de clasificación y Copilot asistido (Persistente en sesión) ──
    if st.session_state.ultima_clasificacion_manual is not None:
        uc = st.session_state.ultima_clasificacion_manual
        intencion = uc["intencion"]
        info = uc["info"]
        urgencia = uc["urgencia"]
        texto_actual = uc["texto"]
        canal_actual = uc["canal"]

        st.markdown(
            f"""
            <div class='beex-card' style='background:var(--c-bg);border:1px solid var(--c-border);border-radius:var(--radius-lg);padding:24px;margin-top:16px;margin-bottom:16px'>
                <h3 style='margin:0 0 16px 0;font-size:1.15rem'>📋 Resultado de la Clasificación</h3>
                <div style='display:flex;align-items:center;gap:12px;margin-bottom:12px'>
                    <span style='font-size:28px'>{info['icono']}</span>
                    <span style='font-size:20px;font-weight:700;color:var(--c-text)'>{intencion}</span>
                </div>
                <p style='margin:0 0 16px 0;font-size:14px;color:var(--c-text)'><strong>Descripción:</strong> {info['descripcion']}</p>
                <div style='margin-top:16px;padding-top:16px;border-top:1px solid var(--c-border)'>
                    <h4 style='margin:0 0 8px 0;font-size:0.95rem;font-weight:700'>🎯 Acción de Enrutamiento Recomendada</h4>
                    <div style='background:var(--c-bg-secondary);padding:12px 16px;border-radius:8px;border:1px solid var(--c-border);font-size:13px;color:var(--c-text)'>
                        {info['accion_recomendada']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("⏱️ SLA Sugerido", info["sla_sugerido"])
        with col_b:
            st.metric("🚦 Prioridad", info["prioridad"])
        with col_c:
            st.metric("🚨 Bandera de Urgencia", "ACTIVADA" if urgencia else "Normal")

        if urgencia:
            st.markdown(
                "<div class='urgencia-banner'>"
                "⚡ <strong>ALERTA DE URGENCIA:</strong> Se detectaron términos de alta fricción. "
                "Se recomienda escalamiento inmediato o activación de protocolo Fast-Track."
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(
            "<div style='background:var(--c-bg-secondary);border:1px solid var(--c-border);"
            "border-radius:var(--radius-lg);padding:20px;margin-top:16px'>"
            "<h4 style='margin:0 0 12px 0;font-size:1rem'>📈 Métricas del Modelo (APF3)</h4>"
            "</div>",
            unsafe_allow_html=True
        )
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1: st.metric("Accuracy",    "94.56%")
        with col_m2: st.metric("Precision (W)", "94.77%")
        with col_m3: st.metric("Recall (W)",    "94.56%")
        with col_m4: st.metric("F1-Score (W)",  "94.60%")

        # Panel Copilot y RAG embebido directamente (Diseño Compacto)
        renderizar_copilot_y_faq(intencion, info, texto_actual, canal_actual, uc, suffix="clasificador", compact=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CHAT SIMULADO
# ─────────────────────────────────────────────────────────────────────────────
if menu == "Copilot":
    col_chat, col_copilot = st.columns([6.5, 3.5], gap="large")
    with col_chat:
        # ── Controles del chat (Ocultos visualmente, integrados en la cabecera) ──
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if st.button("🔄 Cambiar cliente", use_container_width=True):
                st.session_state.chat_cliente = random.choice(NOMBRES_CLIENTES)
                st.session_state.chat_canal   = random.choice(CANALES_CHAT)
                st.session_state.chat_mensajes = []
                st.session_state.ultima_clasificacion_chat = None
                if os.path.exists("chat_persistente.json"):
                    os.remove("chat_persistente.json")
                st.rerun()
        with ctrl_col2:
            if st.button("🗑️ Limpiar chat", use_container_width=True):
                st.session_state.chat_mensajes = []
                st.session_state.ultima_clasificacion_chat = None
                if os.path.exists("chat_persistente.json"):
                    os.remove("chat_persistente.json")
                st.rerun()

        # Cabecera de contacto estilo mockup
        iniciales = "".join([n[0] for n in st.session_state.chat_cliente.split()[:2]])
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;justify-content:space-between;padding:12px 24px;background:var(--c-bg);border:1px solid var(--c-border);border-radius:12px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05)'>
                <div style='display:flex;align-items:center;gap:12px'>
                    <div style='width:40px;height:40px;border-radius:50%;background:var(--c-bg-secondary);border:1px solid var(--c-border);display:flex;align-items:center;justify-content:center;font-weight:700;color:var(--c-text-muted)'>{iniciales}</div>
                    <div>
                        <h3 style='margin:0;font-size:16px;font-weight:700;color:var(--c-text)'>{st.session_state.chat_cliente}</h3>
                        <p style='margin:2px 0 0 0;font-size:12px;color:var(--c-text-muted);display:flex;align-items:center;gap:4px'>
                            <span class='material-symbols-outlined' style='font-size:14px'>chat</span> {st.session_state.chat_canal} · ID: W-9982
                        </p>
                    </div>
                </div>
                <div style='display:flex;align-items:center;gap:16px'>
                    <span style='background:#ffdad6;color:#93000a;padding:4px 12px;border-radius:9999px;font-size:12px;font-weight:600;display:flex;align-items:center;gap:4px'>
                        <span class='material-symbols-outlined' style='font-size:14px'>warning</span> Reclamo - Urgente
                    </span>
                    <span style='font-size:12px;font-weight:600;color:var(--c-text-muted)'>SLA 0:18 / 0:30</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ── Layout: chat usa todo el espacio izquierdo ─────────────
        chat_col = st.container()

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
                # Recuperar sugerencia inteligente usando LLM Local (Ollama) desde caché
                resp_pre = st.session_state.get("sugerencia_actual", "")
                if not resp_pre and st.session_state.chat_mensajes and st.session_state.chat_mensajes[-1]["tipo"] == "cliente":
                    last_intent  = st.session_state.chat_mensajes[-1]["intencion"]
                    texto_ultimo = st.session_state.chat_mensajes[-1]["texto"]
                    resp_pre, prov = generar_respuesta_llm_local(texto_ultimo, last_intent, st.session_state.chat_mensajes)
                    st.session_state.sugerencia_actual = resp_pre
                    st.session_state.sugerencia_proveedor = prov
 
                if resp_pre:
                    proveedor = st.session_state.get("sugerencia_proveedor", "Base de Conocimiento (Fallback)")
                    badge_bg = "rgba(59, 130, 246, 0.15)" if "Ollama" in proveedor else "rgba(107, 114, 128, 0.15)"
                    badge_color = "#3b82f6" if "Ollama" in proveedor else "#6b7280"
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
                        f"<span style='font-size:12px;font-weight:700;color:var(--c-text-muted)'>Respuesta Sugerida:</span>"
                        f"<span style='background:{badge_bg};color:{badge_color};padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>🤖 {proveedor}</span>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                    st.markdown(f"<div class='suggested-box'>{resp_pre}</div>", unsafe_allow_html=True)
                
                    col_send, col_edit, col_bad = st.columns([3, 2, 2])
                    with col_send:
                        enviar_sug = st.button("Enviar respuesta", key="btn_env_sug", type="primary", use_container_width=True)
                    with col_edit:
                        editar_sug = st.button("Editar", key="btn_edit_sug", use_container_width=True)
                    with col_bad:
                        bad_sug = st.button("No es útil", key="btn_bad_sug", use_container_width=True)

                    if enviar_sug:
                        st.session_state.chat_mensajes.append({
                            "tipo": "agente",
                            "texto": resp_pre,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        if "editar_texto" in st.session_state:
                            del st.session_state.editar_texto
                        st.session_state.sugerencia_actual = ""
                        st.session_state.composer_counter = st.session_state.get("composer_counter", 0) + 1
                        st.session_state.composer_key = f"resp_agente_txt_{st.session_state.composer_counter}"
                        st.rerun()

                    if editar_sug:
                        st.session_state.editar_texto = resp_pre
                        st.session_state.composer_counter = st.session_state.get("composer_counter", 0) + 1
                        st.session_state.composer_key = f"resp_agente_txt_{st.session_state.composer_counter}"
                        st.rerun()

                    if bad_sug:
                        st.toast("💡 Feedback registrado. Buscando mejores respuestas para entrenar el modelo.", icon="🧠")
                else:
                    st.info("No hay respuesta sugerida para este mensaje.")

                # Input manual (se usa composer_key dinámico para poder reiniciarlo/limpiarlo legalmente)
                texto_a_enviar = st.session_state.get("editar_texto", "")
                resp_col, send_col = st.columns([6, 1])
                with resp_col:
                    resp_agente = st.text_input(
                        "Escribe un mensaje...",
                        value=texto_a_enviar,
                        key=st.session_state.composer_key,
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
                    st.session_state.sugerencia_actual = ""
                    st.session_state.composer_counter = st.session_state.get("composer_counter", 0) + 1
                    st.session_state.composer_key = f"resp_agente_txt_{st.session_state.composer_counter}"
                    st.rerun()

            with tab_notas:
                nota = st.text_area("Añadir nota interna al cliente...", height=100, label_visibility="collapsed", placeholder="Escribe una nota interna...", key="nota_interna")
                guardar_nota = st.button("Guardar nota", key="btn_guardar_nota")
                if guardar_nota and nota.strip():
                    st.toast("📝 Nota interna guardada exitosamente en la ficha del cliente.", icon="✅")
                    st.session_state.nota_interna = ""
                    st.rerun()

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
                    st.session_state.ultima_clasificacion_chat = {
                        "intencion": intencion_real,
                        "info": INTERPRETACION[intencion_real],
                        "urgencia": urgencia_real,
                        "texto": texto_manual.strip(),
                        "canal": st.session_state.chat_canal
                    }
                    if "editar_texto" in st.session_state:
                        del st.session_state.editar_texto
                    # Pre-calcula la sugerencia local del LLM (Ollama)
                    with st.spinner("🤖 Consultando a Llama 3 local en Ollama..."):
                        sug, prov = generar_respuesta_llm_local(texto_manual.strip(), intencion_real, st.session_state.chat_mensajes)
                        st.session_state.sugerencia_actual = sug
                        st.session_state.sugerencia_proveedor = prov
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
                    st.session_state.ultima_clasificacion_chat = {
                        "intencion": intencion_real,
                        "info": INTERPRETACION[intencion_real],
                        "urgencia": urgencia_real,
                        "texto": texto_cliente,
                        "canal": st.session_state.chat_canal
                    }
                    if "editar_texto" in st.session_state:
                        del st.session_state.editar_texto
                    # Pre-calcula la sugerencia local del LLM (Ollama)
                    with st.spinner("🤖 Generando sugerencia con Llama 3 local en Ollama..."):
                        sug, prov = generar_respuesta_llm_local(texto_cliente, intencion_real, st.session_state.chat_mensajes)
                        st.session_state.sugerencia_actual = sug
                        st.session_state.sugerencia_proveedor = prov
                    st.rerun()



    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 3 — HISTORIAL (HU-01)
    # ─────────────────────────────────────────────────────────────────────────────
if menu == "Historial":
    st.markdown(
        """
        <div style='display:flex;align-items:center;justify-content:space-between;padding:16px 24px;background:var(--c-bg);border-bottom:1px solid var(--c-border);position:sticky;top:0;z-index:10;margin:-16px -16px 24px -16px'>
            <div style='display:flex;align-items:center;gap:24px'>
                <h2 style='margin:0;font-size:20px;font-weight:700;color:var(--beex-blue);letter-spacing:-0.01em'>BEEX AI Historial</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.historial:
        st.info("Aún no hay clasificaciones en esta sesión.")
    else:
        st.markdown("<span style='font-size:16px;font-weight:700;color:var(--c-text);margin-bottom:8px;display:block'>Historial de Interacciones</span>", unsafe_allow_html=True)
        filtro_intencion = st.pills(
            "Filtro",
            ["Todos", "Soporte", "Facturacion", "Ventas", "Reclamos"],
            default="Todos",
            label_visibility="collapsed"
        )

        with st.container():
            df_hist = pd.DataFrame(st.session_state.historial)
            
            # Aplicar filtro interactivo
            if filtro_intencion != "Todos":
                df_hist = df_hist[df_hist["intencion"] == filtro_intencion]
                
            if not df_hist.empty:
                df_hist["urgencia"] = df_hist["urgencia"].map({True: "⚡ SÍ", False: "—"})
                df_hist.columns = ["Hora", "Canal", "Agente", "Mensaje", "Intención", "Prioridad", "Urgencia", "SLA"]
                
                st.dataframe(df_hist[["Hora", "Canal", "Agente", "Intención", "Urgencia"]], use_container_width=True,
                             height=min(400, 60 + len(df_hist) * 35),
                             hide_index=True)

                csv_data = df_hist.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("Descargar historial CSV", data=csv_data,
                                   file_name=f"beex_historial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv", use_container_width=True)
            else:
                st.info("No hay interacciones para esta categoría.")



# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DASHBOARD (HU-02)
# ─────────────────────────────────────────────────────────────────────────────
if menu == "Dashboard":
    st.markdown(
        """
        <div style='display:flex;align-items:center;justify-content:space-between;padding:16px 24px;background:var(--c-bg);border-bottom:1px solid var(--c-border);position:sticky;top:0;z-index:10;margin:-16px -16px 24px -16px'>
            <div>
                <h2 style='margin:0;font-size:24px;font-weight:600;color:var(--c-text);letter-spacing:-0.01em'>Dashboard Operativo</h2>
                <p style='margin:4px 0 0 0;font-size:14px;color:var(--c-text-muted)'>Monitoreo en tiempo real de interacciones y SLAs.</p>
            </div>
            <div style='display:flex;gap:12px;align-items:center;background:var(--c-bg-secondary);border:1px solid var(--c-border);padding:4px 16px;border-radius:9999px'>
                <span style='font-size:12px;font-weight:600;color:var(--beex-green);display:flex;align-items:center;gap:4px'>● Modelo Activo</span>
                <span style='color:var(--c-border)'>|</span>
                <span style='font-size:12px;font-weight:600;color:var(--beex-blue)'>📊 94.5% Accuracy</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
            
            # Cálculo de intención dominante
            intencion_counts = df_dash["intencion"].value_counts()
            intencion_dominante = intencion_counts.idxmax() if not intencion_counts.empty else "N/A"
            pct_dominante = (intencion_counts.max() / total_d * 100) if total_d > 0 else 0

            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            with d_col1:
                st.markdown(f"<div class='beex-card'><p class='beex-text-muted' style='font-weight:600;font-size:12px;margin:0'>TOTAL INTERACCIONES</p><h3 style='margin:8px 0;font-size:32px;color:var(--c-text)'>{total_d}</h3></div>", unsafe_allow_html=True)
            with d_col2:
                st.markdown(f"<div class='beex-card'><p class='beex-text-muted' style='font-weight:600;font-size:12px;margin:0'>SLA CUMPLIDO</p><h3 style='margin:8px 0;font-size:32px;color:var(--c-text)'>92.8%</h3></div>", unsafe_allow_html=True)
            with d_col3:
                st.markdown(f"<div class='beex-card' style='background:#ffdad6;border-color:#ba1a1a'><p style='color:#93000a;font-weight:600;font-size:12px;margin:0'>ALERTAS DE URGENCIA</p><h3 style='margin:8px 0;font-size:32px;color:#93000a'>{urgentes_d}</h3></div>", unsafe_allow_html=True)
            with d_col4:
                st.markdown(f"<div class='beex-card'><p class='beex-text-muted' style='font-weight:600;font-size:12px;margin:0'>INTENCIÓN DOMINANTE</p><h3 style='margin:8px 0;font-size:24px;color:var(--c-text)'>{intencion_dominante}</h3><p style='margin:0;color:var(--c-text-muted);font-size:12px;font-weight:600'>{pct_dominante:.0f}% del volumen total</p></div>", unsafe_allow_html=True)

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
if menu == "Copilot":
    with col_copilot:
        st.markdown(
            """
            <div style='display:flex;align-items:center;justify-content:space-between;padding-bottom:16px;border-bottom:1px solid var(--c-border);margin-bottom:16px'>
                <div style='display:flex;align-items:center;gap:8px'>
                    <span class='material-symbols-outlined' style='color:var(--beex-blue);font-variation-settings:"FILL" 1'>smart_toy</span>
                    <h3 style='margin:0;font-size:16px;font-weight:700;color:var(--beex-blue)'>Agent Copilot</h3>
                </div>
                <span style='background:rgba(16, 185, 129, 0.1);color:var(--beex-green);padding:4px 12px;border-radius:9999px;font-size:10px;font-weight:700;border:1px solid rgba(16, 185, 129, 0.2);display:flex;align-items:center;gap:4px'>
                    ● ACTIVO
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.ultima_clasificacion_chat is None:
            st.info("Clasifique primero una interacción (Clasificador o Chat Simulado) para activar el Copilot.")
        else:
            uc = st.session_state.ultima_clasificacion_chat
            intencion_actual = uc["intencion"]
            info_actual      = uc["info"]
            texto_actual     = uc["texto"]
            canal_actual     = uc["canal"]
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

            # Usar la función helper para renderizar Copilot y FAQ
            renderizar_copilot_y_faq(intencion_actual, info_actual, texto_actual, canal_actual, uc, suffix="copilot")

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
        if st.session_state.ultima_clasificacion_chat is not None:
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
st.markdown(
    "<div style='background:var(--c-bg-secondary);border:1px solid var(--c-border);"
    "border-radius:var(--radius-lg);padding:20px;margin-top:32px;text-align:center'>"
    "<p style='margin:0 0 4px 0;color:var(--c-text-muted);font-size:0.85rem'>"
    "BEEX Suite — Proyecto Final · Innovación y Transformación Digital · UTP · 2026</p>"
    "<p style='margin:0;color:var(--c-text-muted);font-size:0.75rem'>"
    "Modelo SVM · TF-IDF · Dataset sintético 1,500 interacciones · Ley N.° 29733</p>"
    "</div>",
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
