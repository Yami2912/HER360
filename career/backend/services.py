import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data"

LOCKED_WEIGHTS = (0.5, 0.2, 0.2, 0.1)  # (semantic, skill, tfidf, preference)
ABSTENTION_THRESHOLD = 0.44
LEGAL_DISCLAIMER = (
    "Disclaimer: This response is for general workplace-rights awareness in India and does not constitute "
    "legal advice. HER360 is not an official complaint portal. For formal action or case-specific guidance, "
    "consult a qualified legal professional or use official Government of India channels."
)


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if (s_max - s_min) < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - s_min) / (s_max - s_min)


class CareerRecommenderService:
    """Feature A: Locked-in Preference-Aware Hybrid Recommender + Skill-Gap & Roadmap Generator."""

    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model

        # Load finalized O*NET datasets
        self.occ_final = pd.read_csv(DATA_DIR / "HER360_Occupations_Final_910.csv")
        self.career_skills_35 = pd.read_csv(DATA_DIR / "HER360_Career_Skills35_Processed.csv")
        self.sw_910 = pd.read_csv(DATA_DIR / "HER360_Software_Skills_910.csv")
        self.rel_final = pd.read_csv(DATA_DIR / "HER360_Related_Occupations_910.csv")

        self.occ_final["Occupation_Text"] = self.occ_final["Occupation_Text"].fillna("")
        self.occ_codes_ordered = self.occ_final["O*NET-SOC Code"].values
        self.occ_titles_map = dict(zip(self.occ_final["O*NET-SOC Code"], self.occ_final["Title"]))
        self.occ_major_groups = np.array([code[:2] for code in self.occ_codes_ordered])

        # 1. TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.occ_tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.occ_final["Occupation_Text"])

        # 2. Dense semantic embeddings (MiniLM)
        self.occ_embeddings = self.embed_model.encode(
            self.occ_final["Occupation_Text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 3. Weighted 35-skill matrix
        skills_df = self.career_skills_35.copy()
        skills_df["Skill_Weight"] = np.where(
            skills_df["Not_Relevant"] == "Y",
            0.0,
            (skills_df["Importance"] / 5.0) * (skills_df["Level"] / 7.0),
        )
        occ_skill_pivot = (
            skills_df.pivot(index="O*NET-SOC Code", columns="Element Name", values="Skill_Weight")
            .reindex(self.occ_codes_ordered)
            .fillna(0.0)
        )
        self.skill_columns = sorted(occ_skill_pivot.columns.tolist())
        occ_skill_pivot = occ_skill_pivot[self.skill_columns]
        self.skill_to_idx = {name: i for i, name in enumerate(self.skill_columns)}
        self.occ_skill_matrix = occ_skill_pivot.values

        # 4. Software tools & Hot Tech lookup
        self.occ_tools_map: Dict[str, List[tuple]] = {code: [] for code in self.occ_codes_ordered}
        for row in self.sw_910[["O*NET-SOC Code", "Workplace Example", "Hot Technology"]].itertuples(index=False):
            tool_name = str(row[1]).strip().lower()
            if tool_name and tool_name != "nan":
                self.occ_tools_map[row[0]].append((tool_name, row[2] == "Y"))

        # 5. Required skills per occupation (Importance >= 3.0)
        req_skills_df = self.career_skills_35[
            (self.career_skills_35["Not_Relevant"] != "Y") & (self.career_skills_35["Importance"] >= 3.0)
        ].copy()
        req_skills_df["Priority"] = (req_skills_df["Importance"] / 5.0) * (req_skills_df["Level"] / 7.0)
        req_skills_df = req_skills_df.sort_values(["O*NET-SOC Code", "Priority"], ascending=[True, False])
        self.occ_req_skills_map = req_skills_df.groupby("O*NET-SOC Code")["Element Name"].apply(list).to_dict()

        # 6. Related career transitions lookup
        tier_order = {"Primary-Short": 0, "Primary-Long": 1, "Supplemental": 2}
        rel_sorted = self.rel_final.copy()
        rel_sorted["Tier_Rank"] = rel_sorted["Relatedness Tier"].map(tier_order).fillna(3)
        rel_sorted = rel_sorted.sort_values(["O*NET-SOC Code", "Tier_Rank"])
        self.occ_related_map = (
            rel_sorted.groupby("O*NET-SOC Code")["Related O*NET-SOC Code"]
            .apply(lambda s: [self.occ_titles_map[c] for c in s[:3] if c in self.occ_titles_map])
            .to_dict()
        )

    def _profile_to_text(self, summary_text: str, core_skills: List[str], software_tools: List[str]) -> str:
        return f"{summary_text} Skills: {', '.join(core_skills)}. Tools: {', '.join(software_tools)}."

    def _build_roadmap(self, rec: dict, work_mode: str, weekly_hours: int) -> dict:
        title = rec["title"]
        code = rec["onet_soc_code"]
        matched_skills = rec["matched_core_skills"]
        skill_gaps = rec["priority_skill_gaps"][:3]
        matched_tools = rec["matched_tools"]
        hot_tech_gaps = rec["top_hot_tech_to_learn"][:3]
        related = rec["related_transitions"]

        pace_label = "4–6 weeks" if weekly_hours >= 10 else "6–10 weeks"

        return {
            "Target_Occupation": f"{title} ({code})",
            "Match_Score": rec["hybrid_score"],
            "Stated_Preferences": {
                "Work_Mode": work_mode,
                "Study_Pace": f"{weekly_hours} hrs/week (Est. {pace_label} per upskilling phase)",
            },
            "Why_Recommended": (
                f"Strong alignment with your existing skills ({', '.join(matched_skills[:3]) or 'transferable profile'}) "
                f"and tools ({', '.join(matched_tools[:3]) or 'general workplace tools'})."
            ),
            "Phases": [
                {
                    "Phase": "Phase 1: Refresh & Position Existing Strengths (Weeks 1–2)",
                    "Actions": [
                        f"Highlight your matched core skills on your resume: {', '.join(matched_skills) or 'Core transferable strengths'}.",
                        f"Refresh hands-on fluency in tools you already know: {', '.join(matched_tools) or 'Relevant workplace software'}.",
                        f"Frame past experience around outcomes relevant to {title} roles.",
                    ],
                },
                {
                    "Phase": f"Phase 2: Close Priority Skill & Tool Gaps ({pace_label})",
                    "Actions": [
                        f"Strengthen top missing O*NET core skills: {', '.join(skill_gaps) or 'No major core gaps detected'}.",
                        f"Learn high-demand (Hot Technology) tools for this role: {', '.join(hot_tech_gaps) or 'No major tool gaps detected'}.",
                    ],
                },
                {
                    "Phase": "Phase 3: Proof of Work & Re-entry Applications (Final 2–4 Weeks)",
                    "Actions": [
                        f"Build 1 small portfolio artifact or case study combining {matched_tools[0] if matched_tools else 'your core domain'} with {hot_tech_gaps[0] if hot_tech_gaps else 'a target workflow'}.",
                        f"Target {work_mode} openings and structured returnship programs for {title}.",
                        f"Keep adjacent backup paths open: {', '.join(related) if related else 'Explore related O*NET roles'}.",
                    ],
                },
            ],
        }

    def recommend(
        self,
        summary_text: str,
        core_skills: List[str],
        software_tools: List[str],
        preferred_major_groups: List[str],
        work_mode: str = "Flexible / Remote",
        weekly_hours: int = 8,
        top_k: int = 3,
    ) -> dict:
        profile_text = self._profile_to_text(summary_text, core_skills, software_tools)

        # Component 1: Semantic
        q_emb = self.embed_model.encode([profile_text], normalize_embeddings=True)[0]
        sem_scores = _minmax_norm(np.array(np.dot(self.occ_embeddings, q_emb), dtype=float))

        # Component 2: Weighted Skills + Software Tools
        user_vec = np.zeros((1, len(self.skill_columns)), dtype=float)
        for s in core_skills:
            if s in self.skill_to_idx:
                user_vec[0, self.skill_to_idx[s]] = 1.0
        core_scores = (
            cosine_similarity(user_vec, self.occ_skill_matrix).flatten()
            if user_vec.sum() > 0
            else np.zeros(len(self.occ_codes_ordered), dtype=float)
        )

        clean_tools = [t.strip().lower() for t in software_tools if t.strip()]
        tool_patterns = [re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)") for t in clean_tools]
        sw_scores = np.zeros(len(self.occ_codes_ordered), dtype=float)
        if clean_tools:
            max_possible = len(clean_tools) * 1.2
            for idx, code in enumerate(self.occ_codes_ordered):
                matched_weight = 0.0
                for pat in tool_patterns:
                    best = 0.0
                    for tool_text, is_hot in self.occ_tools_map[code]:
                        if pat.search(tool_text):
                            best = max(best, 1.2 if is_hot else 1.0)
                    matched_weight += best
                sw_scores[idx] = matched_weight / max_possible
        skill_scores = _minmax_norm(0.5 * core_scores + 0.5 * sw_scores)

        # Component 3: TF-IDF
        kw_scores = _minmax_norm(
            cosine_similarity(self.tfidf_vectorizer.transform([profile_text]), self.occ_tfidf_matrix).flatten()
        )

        # Component 4: Explicit Preference
        prefs = set(preferred_major_groups)
        pref_scores = (
            np.array([1.0 if g in prefs else 0.0 for g in self.occ_major_groups], dtype=float)
            if prefs
            else np.zeros(len(self.occ_codes_ordered), dtype=float)
        )

        a, b, g, d = LOCKED_WEIGHTS
        final_scores = a * sem_scores + b * skill_scores + g * kw_scores + d * pref_scores
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        user_skills_set = set(core_skills)
        recommendations = []

        for idx in top_indices:
            code = self.occ_codes_ordered[idx]
            title = self.occ_titles_map[code]

            req_skills = self.occ_req_skills_map.get(code, [])[:8]
            matched_skills = [s for s in req_skills if s in user_skills_set]
            missing_skills = [s for s in req_skills if s not in user_skills_set][:5]

            matched_tools, missing_hot_tools = [], []
            for tool_text, is_hot in self.occ_tools_map.get(code, []):
                if any(pat.search(tool_text) for pat in tool_patterns):
                    if tool_text not in matched_tools:
                        matched_tools.append(tool_text)
                elif is_hot and tool_text not in missing_hot_tools:
                    missing_hot_tools.append(tool_text)

            item = {
                "onet_soc_code": code,
                "title": title,
                "hybrid_score": round(float(final_scores[idx]), 4),
                "matched_core_skills": matched_skills,
                "priority_skill_gaps": missing_skills,
                "matched_tools": matched_tools[:5],
                "top_hot_tech_to_learn": missing_hot_tools[:5],
                "related_transitions": self.occ_related_map.get(code, []),
            }
            item["restart_roadmap"] = self._build_roadmap(item, work_mode, weekly_hours)
            recommendations.append(item)

        return {
            "model_version": "HER360-FeatureA-Hybrid-v1",
            "weights": {"semantic": a, "skills": b, "tfidf": g, "preference": d},
            "recommendations": recommendations,
        }


class WorkplaceRightsService:
    """Feature B: Source-Grounded Workplace Rights Assistant (India) with Safe Abstention."""

    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model

        with open(DATA_DIR / "HER360_Workplace_Rights_Corpus_v1.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)

        self.corpus_df = pd.DataFrame(corpus)
        self.corpus_df["search_text"] = (
            self.corpus_df["act_name"] + " | " +
            self.corpus_df["section"] + " | " +
            self.corpus_df["topic"] + ": " +
            self.corpus_df["text"]
        )
        self.corpus_embeddings = self.embed_model.encode(
            self.corpus_df["search_text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def ask(self, query: str, top_k: int = 2, threshold: float = ABSTENTION_THRESHOLD) -> dict:
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)[0]
        scores = np.array(np.dot(self.corpus_embeddings, q_emb), dtype=float)
        top_idx = np.argsort(scores)[::-1][:top_k]
        top_score = float(scores[top_idx[0]])

        if top_score < threshold:
            return {
                "query": query,
                "status": "ABSTAINED",
                "confidence_score": round(top_score, 4),
                "answer": (
                    "I cannot provide a reliable, source-grounded answer to this question from my verified "
                    "Indian workplace-rights corpus. Please consult a qualified legal professional or check "
                    "official Government of India portals (India Code: https://www.indiacode.nic.in/, "
                    "Ministry of Labour & Employment: https://labour.gov.in/)."
                ),
                "citations": [],
                "disclaimer": LEGAL_DISCLAIMER,
            }

        citations = []
        answer_parts = []
        for idx in top_idx:
            if float(scores[idx]) >= threshold - 0.05:
                row = self.corpus_df.iloc[idx]
                answer_parts.append(f"[{row['section']} - {row['topic']}]: {row['text']}")
                citations.append({
                    "doc_id": row["doc_id"],
                    "act_name": row["act_name"],
                    "section": row["section"],
                    "official_channel": row["official_channel"],
                    "source_url": row["source_url"],
                    "last_verified_date": row["last_verified_date"],
                    "similarity_score": round(float(scores[idx]), 4),
                })

        return {
            "query": query,
            "status": "ANSWERED",
            "confidence_score": round(top_score, 4),
            "answer": " ".join(answer_parts),
            "citations": citations,
            "disclaimer": LEGAL_DISCLAIMER,
        }