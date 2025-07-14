import os
from pathlib import Path
import pandas as pd
import streamlit as st

def list_tables(base_path="data"):
    tables = []
    for path in Path(base_path).iterdir():
        if path.is_dir() and (path / "metadata.csv").exists():
            tables.append(path.name)
    return sorted(tables)

def show_sql_query_results(results, table_name: str, selected_fields: list) -> pd.DataFrame:
    try:
        metadata_path = Path("data") / table_name / "metadata.csv"
        metadata_df = pd.read_csv(metadata_path)
        metadata_df.set_index("track_id", inplace=True)
    except Exception as e:
        st.error(f"⚠️ Error al cargar metadata de la tabla '{table_name}': {e}")
        return pd.DataFrame()

    rows = []
    for item in results:
        track_id = item["doc_id"]
        score = round(item["score"], 4)

        if track_id in metadata_df.index:
            row = metadata_df.loc[track_id]

            # Solo incluir las columnas solicitadas
            filtered = {field: row.get(field, "") for field in selected_fields}
            filtered["score"] = score  # siempre puedes incluir el score
            rows.append(filtered)

    if not rows:
        st.warning("No se encontraron resultados coincidentes.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def show_detailed_results(results, table_name: str):
    try:
        metadata_path = Path("data") / table_name / "metadata.csv"
        metadata_df = pd.read_csv(metadata_path)
        metadata_df.set_index("track_id", inplace=True)
    except Exception as e:
        st.error(f"⚠️ Error al cargar metadata de la tabla {table_name}: {e}")
        return

    enriched = []
    for item in results:
        track_id = item["doc_id"]
        score = round(item["score"], 4)

        if track_id in metadata_df.index:
            row = metadata_df.loc[track_id]
            enriched.append({
                "🎵 Título": row.get("track_name", ""),
                "👤 Artista": row.get("track_artist", ""),
                "🎧 Audio": f"audio/{track_id}.mp3",
                "📈 Score": score,
                "🎤 Letra": row.get("lyrics", "")[:150] + "..." if isinstance(row.get("lyrics", ""), str) else ""
            })

    if enriched:
        for item in enriched:
            st.markdown(f"### {item['🎵 Título']} — {item['👤 Artista']}")
            st.write(f"**Score de similitud:** {item['📈 Score']}")
            st.write(f"**Letra (fragmento):** _{item['🎤 Letra']}_")

            audio_path = f"static/audio/{Path(item['🎧 Audio']).name}"
            if os.path.exists(audio_path):
                st.audio(audio_path)
            else:
                st.warning("🎧 Archivo de audio no encontrado.")
            st.markdown("---")
    else:
        st.warning("No se encontraron resultados para mostrar.")
