import streamlit as st
import requests
import pandas as pd
from utils import show_detailed_results, list_tables
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Songs Recommender", layout="wide", page_icon="🎧")

st.sidebar.write("Hola!👋 Te ayudaré a encontrar canciones similares a las que tienes en mente.")
page = st.sidebar.radio("Selecciona una opción:", [
    "🔍 Consulta SQL",
    "🎧 Buscar por Audio",
    "📁 Gestión de Tablas"
])

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stRadio > div { flex-direction: row; }
        .stDataFrame { max-height: 400px; overflow-y: auto; }
    </style>
""", unsafe_allow_html=True)

st.title("🎧 Songs Intelligent Recommender System")

if page == "🔍 Consulta SQL":

# --------------------------------- CONSULTA SQL -------------------------------
    available_tables = list_tables("data")
    selected_table = st.selectbox("📂 Elige la tabla:", available_tables, index=available_tables.index("Audio") if "Audio" in available_tables else 0)

    metadata_path = Path("data") / selected_table / "metadata.csv"
    try:
        metadata_df = pd.read_csv(metadata_path, nrows=1)  # Solo primera fila para mostrar columnas
        columns = metadata_df.columns.tolist()
        st.info(f"🧩 Features disponibles en **{selected_table}**:\n\n`{', '.join(columns)}`")

        default_query = f"SELECT track_name, track_artist FROM {selected_table} WHERE lyrics LIKE 'love' LIMIT 5;"
        user_query = st.text_area("Run your query:", default_query, height=100)

        if st.button("Execute"):
            with st.spinner("Procesando..."):
                print("Query:", user_query)
                res = requests.post(f"{API_URL}/search_sql", json={"query": user_query})
                if res.status_code == 200:
                    data = res.json()
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df)
                    else:
                        st.warning("No se encontraron resultados.")
                else:
                    st.error("Error al interpretar la consulta.")
    except Exception as e:
        st.error(f"Error al leer metadatos: {e}")

# --------------------------------- BUSQUEDA POR AUDIO -------------------------------

elif page == "🎧 Buscar por Audio":

    uploaded_file = st.file_uploader("🔊 Carga un archivo (.mp3 o .wav)", type=["mp3", "wav"])
    top_k = st.sidebar.slider("¿Top-K resultados?", 1, 15, 5)

    if st.button("Buscar canciones similares por letra"):
        if uploaded_file:
            with st.spinner("Procesando audio..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                res = requests.post(f"{API_URL}/search_audio", files=files, params={"k": top_k})
                if res.status_code == 200:
                    results = res.json()
                    if results:
                        df = pd.DataFrame(results)
                        st.success("Resultados:")
                        st.dataframe(df)
                    else:
                        st.warning("No se encontraron canciones similares.")
                else:
                    st.error("Error al procesar el audio.")
        else:
            st.warning("Primero sube un archivo de audio.")

# --------------------------------- GESTION DE TABLAS -------------------------------

elif page == "📁 Gestión de Tablas":
    table_name = st.text_input("Nombre de la nueva tabla (ej: Videos):")

    if st.button("Crear tabla"):
        res = requests.post(f"{API_URL}/create_table", params={"name": table_name})
        st.success(f"Tabla creada: {res.json()['path']}")

    st.markdown("---")
    st.subheader("📤 Insertar un CSV")
    csv_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
    if st.button("Insertar CSV"):
        if table_name and csv_file:
            files = {"file": (csv_file.name, csv_file, "text/csv")}
            res = requests.post(f"{API_URL}/insert_csv", files=files, params={"table": table_name})
            if res.status_code == 200:
                st.success(f"Se indexaron {res.json()['n_docs']} documentos.")
            else:
                st.error("Error al indexar el CSV.")
