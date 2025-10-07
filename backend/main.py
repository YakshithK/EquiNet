from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq  # Changed to Groq

# -------------------------
# CONFIG
# -------------------------
FAISS_INDEX_FILE = "faiss_index/equinet_faiss.index"
METADATA_FILE = "faiss_index/output.json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

# Initialize Groq client
client = Groq(api_key="")

# -------------------------
# LOAD INDEX + METADATA
# -------------------------
print("[INFO] Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"[INFO] Loaded {len(metadata)} metadata entries.")

def get_snippet_from_db(snippet_id: str):
    """
    Retrieve a snippet by its ID from the loaded data.
    Returns None if not found.
    """
    for snippet in metadata:
        if snippet.get('id') == snippet_id:
            return snippet
    return None

# -------------------------
# EMBEDDING MODEL
# -------------------------
model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------
# FASTAPI APP
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_equinet(req: QueryRequest):
    query_embedding = model.encode([req.query], show_progress_bar=False)
    query_vector = np.array(query_embedding, dtype="float32")
    
    # Retrieve top-K
    D, I = index.search(query_vector, TOP_K)
    results = []
    for i, score in zip(I[0], D[0]):
        entry = metadata[i]
        fairness_score = entry.get("fairness_score", 1.0)
        print(entry)
        results.append({
            "id": entry["id"],
            "text": entry["text"],
            "source": entry.get("source"),
            "domain": entry.get("domain"),
            "language": entry.get("language"),
            "cluster": entry.get("cluster"),
            "fairness_score": fairness_score,
            "similarity": float(score)
        })
    print(results)
    
    # Pass retrieved context to LLM
    context = "\n\n".join([metadata[i]["text"] + ": " + metadata[i]["source"] for i in I[0]])
    prompt = f"Based on the following context:\n{context}\n\nAnswer: {req.query}"
    
    # Groq API call
    llm_response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",  # or "mixtral-8x7b-32768", "llama-3.3-70b-versatile"
        messages=[
            {"role": "system", "content": "You are a fairness-aware knowledge assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    answer = llm_response.choices[0].message.content
    
    return {
        "query": req.query,
        "results": results,
        "answer": answer
    }

@app.get("/get-all")
async def sphere_data():
    with open("faiss_index/output.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/get-info")
async def get_info(id: str):
    snippet = get_snippet_from_db(id)
    
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    
    return {
        "id": snippet["id"],
        "source": snippet["source"],
        "domain": snippet["domain"],
        "language": snippet["language"],
        "text": snippet["text"],
        "fairness_score": snippet["fairness_score"],
        "cluster": snippet["cluster_id"],
        "metadata": {
            "author": snippet["author"],
            "date": snippet["publication_date"],
            "url": snippet["url"],
        }
    }