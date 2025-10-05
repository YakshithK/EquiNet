from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

app = FastAPI(title="EquiNet API", description="Inclusive Knowledge Retrieval System")

# Load model and data at startup
print("[INFO] Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("[INFO] Loading FAISS index...")
index_path = os.path.join("faiss_index", "equinet_faiss.index")
index = faiss.read_index(index_path)

print("[INFO] Loading embedded dataset...")
with open(os.path.join("faiss_index", "embedded_dataset.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

print("[INFO] Backend ready!")


class Result(BaseModel):
    text: str
    source: Optional[str]
    metadata: Optional[dict]


@app.get("/")
async def root():
    return {"message": "EquiNet API is running. Use /query endpoint."}


@app.get("/query", response_model=List[Result])
async def query_equiNet(
    q: str = Query(..., description="Search query for marginalized knowledge"),
    k: int = Query(5, description="Number of results to return")
):
    query_vec = model.encode([q])
    D, I = index.search(np.array(query_vec).astype("float32"), k)

    results = []
    for idx in I[0]:
        results.append({
            "text": data[idx]["text"][:500] + "...",
            "source": data[idx].get("source"),
            "metadata": data[idx].get("metadata", {})
        })
    return results
