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
        "https://cayce-vault-frontend.vercel.app",  # ← FIXED: no extra spaces
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
            return InsightResponse(answer="No relevant Readings found for this query.", sources=[])

        # ✅ FULLY COMPLIANT PROMPT — aligned with ECF guidelines
        prompt = (
            "You are an AI research assistant exploring Edgar Cayce’s archival Readings (held by the Edgar Cayce Foundation). "
            "Based SOLELY on the provided Readings below, offer a clear, respectful, and insightful synthesis that honors the spiritual depth of the material.\n\n"
            
            "Guidelines:\n"
            "- Always capitalize 'Reading' and 'Readings' when referring to Edgar Cayce's channeled material\n"
            "- Maintain a tone of deep respect toward the wisdom in the Readings\n"
            "- Do NOT address the user personally (avoid 'you', 'we', 'beloved', 'Seeker', etc.)\n"
            "- Do NOT combine health recommendations unless they appear together in a single Reading\n"
            "- Do NOT claim Cayce 'favored,' 'often said,' or 'loved to point out' — only report what is present\n"
            "- If a concept appears in multiple retrieved Readings, you may note it as 'frequent,' 'recurring,' or 'referenced across many Readings' — but only cite the specific Reading IDs that were provided\n"
            "- When describing content, use direct quotes when possible, or closely paraphrase using the Reading’s own terminology — do not substitute modern interpretations (e.g., say 'reflect on impressions' not 'journal')\n"
            "- Do NOT invent prayers, rituals, or practices not explicitly in the source\n"
            "- Cite Reading numbers like [294-12]\n"
            "- Write in warm, flowing prose (not bullet points)\n"
            "- Close with one open-ended, research-oriented question that naturally follows from the themes (e.g., 'How might the concept of... be explored further in the Readings?')\n\n"
            
            f"Relevant Cayce Readings:\n{context}\n"
            f"User Question: \"{request.query}\"\n\n"
            "Response:"
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.65
        )

        answer = response.choices[0].message.content.strip()

        # ✅ REQUIRED DISCLAIMERS (per ECF) — with spacing and visual separation
        disclaimer = (
            "\n\n\n"  # Three line breaks for generous spacing
            "<small>"
            "---\n"
            "© Edgar Cayce Foundation. All Readings are copyrighted and used for research purposes only.\n"
            "Health-related information reflects historical practices and may be outdated or unsafe without medical supervision. "
            "Consult a licensed physician before applying any health advice. This tool does not replace professional mental health or medical care.\n"
            "Responses are AI-generated and may not accurately reflect the original source material or the views of the Edgar Cayce organizations."
            "</small>"
        )

        full_answer = answer + disclaimer
        return InsightResponse(answer=full_answer, sources=sources)

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
