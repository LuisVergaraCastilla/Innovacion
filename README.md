# BEEX — Suite de Inteligencia Conversacional

Este es el prototipo de la **Suite de Inteligencia Conversacional BEEX** desarrollado para el proyecto final del curso de Innovación y Transformación Digital de la UTP. 

Esta aplicación basada en Streamlit incluye un simulador de contact center con un motor de inteligencia artificial (modelo SVM) para la clasificación de intenciones, un Copilot en tiempo real para agentes, y un dashboard de métricas operativas.

## Requisitos Previos

Asegúrate de tener instalado [Python 3.9 o superior](https://www.python.org/downloads/) en tu sistema.

## Instalación y Ejecución

Sigue estos pasos para desplegar el proyecto localmente:

### 1. Clonar el repositorio
Si aún no tienes el código en tu máquina, clona el repositorio e ingresa a la carpeta del proyecto:
```bash
git clone <URL_DEL_REPOSITORIO>
cd Innovacion
```

### 2. Crear un Entorno Virtual (Opcional pero recomendado)
Es una buena práctica crear un entorno virtual para no tener conflictos con otras librerías de tu sistema:
```bash
python -m venv venv

# Activar el entorno virtual en Windows:
venv\Scripts\activate

# Activar el entorno virtual en macOS/Linux:
source venv/bin/activate
```

### 3. Instalar las dependencias
Instala todas las librerías necesarias ejecutando el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
Levanta el servidor local de Streamlit con el siguiente comando:
```bash
streamlit run app_beex.py
```

Automáticamente se abrirá una pestaña en tu navegador web por defecto (generalmente en `http://localhost:8501`) con la aplicación BEEX lista para usar.

## Estructura Principal
- `app_beex.py`: Código principal de la interfaz web y la lógica en Streamlit.
- `modelo_intenciones.pkl`: Modelo SVM pre-entrenado (scikit-learn) para la detección de intenciones y fricciones.
- `requirements.txt`: Lista de dependencias del proyecto.
- `historial_persistente.json` y `chat_persistente.json`: Archivos generados automáticamente que guardan el estado de la sesión actual de manera persistente.
