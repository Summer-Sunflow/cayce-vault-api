from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from meilisearch import Client
import openai
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
app = FastAPI(title="Cayce Vault API")

# CORS: Allow your Vercel frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cayce-vault-frontend.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Meilisearch setup
MEILISEARCH_URL = os.getenv("MEILISEARCH_URL")
MEILISEARCH_KEY = os.getenv("MEILISEARCH_MASTER_KEY")
meili = Client(MEILISEARCH_URL, MEILISEARCH_KEY)

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Index names
PRECISION_INDEX = "cayce-vault"
INSIGHT_INDEX = "cayce_chunks"

# ----------------------------
# DATA MODELS
# ----------------------------
class SearchRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    id: str
    reading_id: str
    text: str
    date: str = ""
    category: str = ""

class InsightResponse(BaseModel):
    answer: str
    sources: list[str]

# ----------------------------
# ENDPOINTS
# ----------------------------

@app.post("/search/precision", response_model=list[SearchResult])
async def precision_search(request: SearchRequest):
    try:
        index = meili.index(PRECISION_INDEX)
        results = index.search(request.query, {
            "limit": 10,
            "attributesToRetrieve": ["reading_id", "reading_text", "date", "category"]
        })
        formatted = []
        for hit in results["hits"]:
            formatted.append(SearchResult(
                id=hit.get("id", hit.get("reading_id", "")),
                reading_id=hit.get("reading_id", ""),
                text=hit.get("reading_text", ""),
                date=hit.get("date", ""),
                category=hit.get("category", "")
            ))
        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meilisearch error: {str(e)}")

@app.post("/search/insight", response_model=InsightResponse)
async def insight_search(request: SearchRequest):
    try:
        index = meili.index(INSIGHT_INDEX)
        results = index.search(request.query, {
            "limit": 5,
            "hybrid": {"embedder": "openai-embedder"},
            "attributesToRetrieve": ["reading_id", "text"]
        })
        
        sources = []
        context = ""
        for hit in results["hits"]:
            rid = hit.get("reading_id", "Unknown")
            text = hit.get("text", "")
            sources.append(rid)
            context += f"[{rid}] {text}\n\n"
        
        if not context.strip():
            return InsightResponse(answer="No relevant readings found.", sources=[])

        prompt = (
            "You are Edgar Cayce's wisdom assistant. Based ONLY on the provided Cayce readings below, "
            "answer the user's question with compassion, clarity, and spiritual insight. "
            "Cite reading numbers like [294-12] when possible.\n\n"
            f"Readings:\n{context}\n"
            f"User question: {request.query}\n\n"
            "Answer:"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        answer = response.choices[0].message["content"].strip()
        return InsightResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight error: {str(e)}")

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "meilisearch": bool(meili.health()),
        "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    }

