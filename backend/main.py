from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.indexing.search import SPIMISearcher
from backend.indexing.spimi import SPIMIIndexer
from backend.audio_processing import transcribe_audio
from backend.ai_query_parser import parse_sql_query
from backend.indexing.preprocessor import preprocess
from backend.models import SearchResponse, SQLQuery
from typing import List
import pandas as pd
import csv
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search", response_model=List[SearchResponse])
def search(q: str, table: str = Query(...), k: int = 5):
    index_path = f"data/{table}/index.json"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"Índice para tabla '{table}' no encontrado.")
    
    searcher = SPIMISearcher(index_path=index_path)
    results = searcher.search(q, top_k=k)
    return [{"doc_id": doc_id, "score": score} for doc_id, score in results]

@app.post("/search_audio", response_model=List[SearchResponse])
async def search_from_audio(file: UploadFile = File(...), table: str = Query(...), k: int = 5):
    index_path = f"data/{table}/index.json"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"Índice para tabla '{table}' no encontrado.")
    
    text = transcribe_audio(await file.read())
    searcher = SPIMISearcher(index_path=index_path)
    results = searcher.search(text, top_k=k)
    return [{"doc_id": doc_id, "score": score} for doc_id, score in results]

# @app.post("/search_sql", response_model=List[SearchResponse])
# def search_from_sql(payload: SQLQuery):
#     table_name, query_text, k, selected_fields = parse_sql_query(payload.query)
#     index_path = f"data/{table_name}/index.json"

#     if not query_text or not os.path.exists(index_path):
#         raise HTTPException(status_code=400, detail="Consulta inválida o índice no encontrado.")

#     searcher = SPIMISearcher(index_path=index_path)
#     results = searcher.search(query_text, top_k=k)

#     # ⬇️ Incluye también los campos seleccionados como parte de la respuesta (opcional)
#     return [{"doc_id": doc_id, "score": score} for doc_id, score in results]


@app.post("/search_sql")
def search_from_sql(payload: SQLQuery):
    table_name, query_text, k, selected_fields = parse_sql_query(payload.query)
    index_path = f"data/{table_name}/index.json"
    metadata_path = f"data/{table_name}/metadata.csv"

    if not query_text or not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise HTTPException(status_code=400, detail="Consulta inválida o recursos no encontrados.")

    # Buscar
    searcher = SPIMISearcher(index_path=index_path)
    results = searcher.search(query_text, top_k=k)

    # Cargar metadata
    df = pd.read_csv(metadata_path)
    df.set_index("track_id", inplace=True)

    output = []
    for doc_id, score in results:
        if doc_id in df.index:
            row = df.loc[doc_id]
            enriched = {f: row.get(f, "") for f in selected_fields}
            enriched["score"] = score
            output.append(enriched)

    return output


@app.post("/create_table")
def create_table(name: str):
    path = f"data/{name}"
    os.makedirs(path, exist_ok=True)
    return {"status": "created", "path": path}

@app.post("/insert_csv")
def insert_csv(table: str, file: UploadFile = File(...)):
    path = f"data/{table}"
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "metadata.csv")

    with open(csv_path, "wb") as f:
        f.write(file.file.read())

    docs = {}
    with open(csv_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            doc_id = row.get("track_id") or row.get("id") or str(hash(str(row)))
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
            docs[doc_id] = text

    indexer = SPIMIIndexer(output_path=os.path.join(path, "index.json"))
    indexer.index_documents(docs)
    
    return {"status": "indexed", "n_docs": len(docs)}
