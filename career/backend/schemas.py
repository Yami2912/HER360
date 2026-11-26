from typing import List, Optional
from pydantic import BaseModel, Field


# --- Feature A: Career Re-entry & Skill-Transfer ---
class CareerRecommendRequest(BaseModel):
    summary_text: str = Field(
        ...,
        min_length=10,
        description="Brief summary of prior work experience, education, or career interests."
    )
    core_skills: List[str] = Field(
        default_factory=list,
        description="List of O*NET 35 core skills the user possesses."
    )
    software_tools: List[str] = Field(
        default_factory=list,
        description="Software tools or technologies the user knows (e.g., Python, Excel, SQL)."
    )
    preferred_major_groups: List[str] = Field(
        default_factory=list,
        description="Optional 2-digit O*NET major group codes (e.g., '15' for Tech, '21' for Counseling)."
    )
    work_mode: str = Field(default="Flexible / Remote")
    weekly_hours: int = Field(default=8, ge=1, le=40)
    top_k: int = Field(default=3, ge=1, le=10)


class RoadmapPhase(BaseModel):
    Phase: str
    Actions: List[str]


class RestartRoadmap(BaseModel):
    Target_Occupation: str
    Match_Score: float
    Stated_Preferences: dict
    Why_Recommended: str
    Phases: List[RoadmapPhase]


class CareerRecommendationItem(BaseModel):
    onet_soc_code: str
    title: str
    hybrid_score: float
    matched_core_skills: List[str]
    priority_skill_gaps: List[str]
    matched_tools: List[str]
    top_hot_tech_to_learn: List[str]
    related_transitions: List[str]
    restart_roadmap: RestartRoadmap


class CareerRecommendResponse(BaseModel):
    model_version: str
    weights: dict
    recommendations: List[CareerRecommendationItem]


# --- Feature B: Workplace Rights Assistant (India) ---
class RightsQueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Workplace rights question in plain language.")
    top_k: int = Field(default=2, ge=1, le=5)


class LegalCitation(BaseModel):
    doc_id: str
    act_name: str
    section: str
    official_channel: str
    source_url: str
    last_verified_date: str
    similarity_score: float


class RightsQueryResponse(BaseModel):
    query: str
    status: str  # "ANSWERED" or "ABSTAINED"
    confidence_score: float
    answer: str
    citations: List[LegalCitation]
    disclaimer: str