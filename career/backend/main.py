from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from schemas import (
    CareerRecommendRequest,
    CareerRecommendResponse,
    RightsQueryRequest,
    RightsQueryResponse,
)
from services import CareerRecommenderService, WorkplaceRightsService

services_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load MiniLM once and share across Feature A and Feature B
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    services_store["career"] = CareerRecommenderService(embed_model)
    services_store["rights"] = WorkplaceRightsService(embed_model)
    yield
    services_store.clear()


app = FastAPI(
    title="HER360 Career Module API",
    description="Feature A (Explainable Career Re-entry Recommender) & Feature B (Source-Grounded Workplace Rights Assistant - India)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React / Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "occupations_loaded": len(services_store["career"].occ_codes_ordered),
        "legal_provisions_loaded": len(services_store["rights"].corpus_df),
    }


@app.get("/api/career/skills")
def get_available_skills():
    """Returns the 35 valid O*NET core skills for frontend dropdowns/chips."""
    return {"core_skills": services_store["career"].skill_columns}


@app.post("/api/career/recommend", response_model=CareerRecommendResponse)
def recommend_careers(payload: CareerRecommendRequest):
    try:
        return services_store["career"].recommend(
            summary_text=payload.summary_text,
            core_skills=payload.core_skills,
            software_tools=payload.software_tools,
            preferred_major_groups=payload.preferred_major_groups,
            work_mode=payload.work_mode,
            weekly_hours=payload.weekly_hours,
            top_k=payload.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/career/rights-assistant", response_model=RightsQueryResponse)
def query_workplace_rights(payload: RightsQueryRequest):
    try:
        return services_store["rights"].ask(query=payload.query, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))