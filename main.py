from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from meilisearch import Client
from openai import OpenAI
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
app = FastAPI(title="Cayce Vault API")

# CORS: Allow your Vercel frontend
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
MEILISEARCH_MASTER_KEY = os.getenv("MEILISEARCH_MASTER_KEY")
meili = Client(MEILISEARCH_URL, MEILISEARCH_MASTER_KEY)

# Index names — must match Meilisearch exactly
PRECISION_INDEX = "cayce_vault"
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
        print(f"Precision search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Meilisearch error: {str(e)}")

@app.post("/search/insight", response_model=InsightResponse)
async def insight_search(request: SearchRequest):
    try:
        # Validate env vars at runtime
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY in environment")
        if not os.getenv("MEILISEARCH_MASTER_KEY"):
            raise HTTPException(status_code=500, detail="Missing MEILISEARCH_MASTER_KEY in environment")

        index = meili.index(INSIGHT_INDEX)
        results = index.search(request.query, {
            "limit": 8,
            "hybrid": {"embedder": "OpenAI_Embedder"},
            "attributesToRetrieve": ["reading_id", "text"]
        })
        
        sources = []
        context = ""
        for hit in results["hits"]:
            rid = hit.get("reading_id", "Unknown")
            text = hit.get("text", "")
            if rid not in sources:
                sources.append(rid)
                context += f"[{rid}] {text}\n\n"
        
        if not context.strip():
            return InsightResponse(answer="No relevant readings found for this query.", sources=[])

        prompt = (
            "You are a compassionate spiritual guide channeling Edgar Cayce’s wisdom. "
            "Based SOLELY on the Cayce readings provided below, offer a thoughtful, "
            "nurturing, and deeply insightful response to the user’s question.\n\n"
            
            "Guidelines:\n"
            "- Begin with a gentle acknowledgment of the seeker’s intent\n"
            "- Weave together key themes from multiple readings (cite as [294-12])\n"
            "- Include practical suggestions or meditative practices when relevant\n"
            "- Close with an uplifting, soul-centered reflection\n"
            "- Write in warm, flowing prose (not bullet points)\n"
            "- Be thorough — aim for a meaningful paragraph or two\n\n"
            
            f"Relevant Cayce Readings:\n{context}\n"
            f"User Question: \"{request.query}\"\n\n"
            "Your Response:"
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1400,
            temperature=0.75
        )

        answer = response.choices[0].message.content.strip()
        return InsightResponse(answer=answer, sources=sources)

    except Exception as e:
        print(f"Insight error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insight error: {str(e)}")

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
async def health_check():
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "ok",
        "meilisearch": bool(meili.health()),
        "openai": "configured" if openai_configured else "missing"
    }
