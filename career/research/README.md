# HER360 — Career Module Research (`feature/career`)

This directory contains the research notebooks, structured datasets, provenance metadata, and sample outputs for the **HER360 Career Module**. Following our project scoping and uniqueness analysis, the module is strictly focused on two research-backed features:

1. **Feature A:** AI-Powered, Explainable Career Re-entry and Skill-Transfer Recommendation System for Women
2. **Feature B:** Source-Grounded AI Assistant for Women's Workplace Rights and Legal Awareness in India

---

## Directory Structure

| File | Description |
| :--- | :--- |
| `HER360_Career_Reentry_Research.ipynb` | End-to-end Feature A notebook: O*NET preprocessing, 6-model benchmark, error analysis, skill-gap analyzer, and 3-phase restart roadmap generator. |
| `dataset_provenance.json` | Schema, row counts, and scope limitations for the processed O*NET datasets used in Feature A. |
| `HER360_FeatureA_Sample_Output.json` | Sample end-to-end JSON output (recommended role, matched skills, priority gaps, hot-tech gaps, and 3-phase restart roadmap). |
| `HER360_Workplace_Rights_Research_pynb.ipynb` | End-to-end Feature B notebook: verified Indian legal corpus creation, retrieval comparison, abstention calibration, and grounded Q&A evaluation. |
| `HER360_Workplace_Rights_Corpus_v1.json` | 14 structured Indian workplace-rights provisions with Act, Section, Official Channel, Source URL, and `last_verified_date` metadata. |
| `HER360_FeatureB_Sample_Output.json` | Sample grounded assistant outputs demonstrating SHe-Box citation (in-scope) and safe abstention (out-of-scope). |

---

## Feature A: AI-Powered, Explainable Career Re-entry & Skill-Transfer Recommender

### 1. Dataset & Benchmark Design
* **Source Taxonomy:** O*NET Database processed into 4 relational tables covering **910 occupations** (`occ_final`), **35 core transferable skills** with Importance/Level ratings (`career_skills_35`), **software tools & Hot Technologies** (`sw_910`), and **related career transitions** (`rel_final`).
* **Evaluation Split (No Data Leakage):** 16 structured scenario profiles split cleanly into **8 Validation profiles** (used strictly for hyperparameter/weight tuning) and **8 Held-Out Test profiles** (locked until final evaluation).

### 2. Hybrid Scoring Formulation
The winning **Preference-Aware Hybrid** model combines four min-max normalized signals:

$$S_{\text{hybrid}} = 0.50 \cdot S_{\text{semantic}} + 0.20 \cdot S_{\text{skill}} + 0.20 \cdot S_{\text{tfidf}} + 0.10 \cdot S_{\text{preference}}$$

* $S_{\text{semantic}}$: Cosine similarity using Sentence Transformers (`all-MiniLM-L6-v2`)
* $S_{\text{skill}}$: Weighted 35-skill vector similarity $\left(\frac{\text{Importance}}{5} \times \frac{\text{Level}}{7}\right)$ + word-boundary-safe software/Hot-Tech matching
* $S_{\text{tfidf}}$: Unigram + bigram TF-IDF cosine similarity
* $S_{\text{preference}}$: Binary match against the user's explicitly stated O*NET Major Group preferences (defaults to `0.0` if omitted)

### 3. Held-Out Test Set Results ($K = 5$)

| Model | Precision@5 | Recall@5 | nDCG@5 | MAP@5 | Unique Jobs@5 | Coverage@5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. TF-IDF + Cosine | 0.300 | 0.5000 | 0.4472 | 0.3059 | 38 | 4.18% |
| 2. BM25 | 0.175 | 0.2917 | 0.2772 | 0.1962 | 38 | 4.18% |
| 3. Sentence Transformer (MiniLM) | 0.400 | 0.6562 | 0.6551 | 0.5017 | 36 | 3.96% |
| 4. Weighted Skill Matching | 0.175 | 0.3125 | 0.2529 | 0.1608 | 37 | 4.07% |
| 5a. Hybrid (No Pref) `(0.7, 0.3, 0.0, 0.0)` | 0.400 | 0.6562 | 0.6735 | 0.5337 | 35 | 3.85% |
| **5b. Hybrid (Pref-Aware) `(0.5, 0.2, 0.2, 0.1)`** | **0.400** | **0.6562** | **0.7131** | **0.5806** | **37** | **4.07%** |

### 4. Key Findings & Explainable Outputs
* **Ranking Improvement:** While `MiniLM`, `Hybrid 5a`, and `Hybrid 5b` achieved the same Recall@5 (`0.6562`), `Hybrid 5b` pushed true relevant jobs significantly higher in the ranking (`nDCG@5`: `0.6551` $\rightarrow$ `0.7131`; `MAP@5`: `0.5017` $\rightarrow$ `0.5806`).
* **Rank-#1 Accuracy & Stability:** Achieved **8/8 Rank-#1 accuracy** on the unseen test set with **zero regressions** against standalone `MiniLM` (improved on 5/8 profiles, tied on 3/8).
* **Explainability & Roadmap:** Each recommendation outputs **Matched Core Skills**, **Priority Skill Gaps** ($\text{Importance} \ge 3.0$), **Matched Tools**, **Top Hot Technologies to Learn**, **Related Career Transitions**, and a **3-Phase Personalized Career-Restart Roadmap** tailored to user-stated work mode and weekly study hours.

### 5. Feature A Limitations & Ethical Guardrails
* **US Taxonomy Localization:** O*NET is a US occupational taxonomy; job titles, demand signals, and tool distributions require validation and localization for the Indian job market.
* **Benchmark Size:** The v1 held-out test set contains 8 curated profiles; results represent strong preliminary evidence rather than statistically significant proof of real-world generalization.
* **Fairness Guardrails:** Career-break duration is never penalized, and sensitive personal attributes (marital status, motherhood, caregiving) are never inferred or used as negative ranking signals.

---

## Feature B: Source-Grounded Workplace Rights Assistant (India)

### 1. Verified Legal Corpus & Benchmark
* **Corpus (`v1`):** 14 structured provisions verified on `2026-11-25` across:
  * **POSH Act, 2013** (Sections 2(n), 4, 6, 9, 11, 12, 16)
  * **Government of India SHe-Box Portal** (`https://shebox.wcd.gov.in/`)
  * **Maternity Benefit Act, 1961 (2017 Amendment)** (Sections 5, 5(5), 11A, 12)
  * **Code on Social Security, 2020** (Chapter VI, Sections 59–72)
  * **Equal Remuneration Act, 1976 / Code on Wages, 2019** (Section 4 / Section 3)
* **Evaluation Benchmark:** 16 labeled queries consisting of **12 In-Scope** everyday employee questions and **4 Out-of-Scope** abstention tests (Indian Income Tax 80C, mutual consent divorce/alimony, US FMLA leave, and startup ESOP taxation).

### 2. Retrieval Comparison (12 In-Scope Queries, $K = 3$)

| Retriever | Recall@3 | Precision@3 | MRR |
| :--- | :---: | :---: | :---: |
| 1. TF-IDF + Cosine | 0.9583 | 0.3889 | 0.8889 |
| 2. BM25 | 0.7917 | 0.3333 | 0.7917 |
| **3. Dense (`all-MiniLM-L6-v2`)** | **0.9167** | **0.3611** | **1.0000** |

### 3. Safe Abstention & Grounding Evaluation
Using `all-MiniLM-L6-v2` with a calibrated cosine similarity threshold ($\tau = 0.44$):
* **Out-of-Scope Abstention Rate:** **100.00%** (4/4 safely abstained, including the borderline US FMLA query at `0.4350`).
* **Top-1 Citation Accuracy (on Answered Queries):** **100.00%** (10/10 cited the exact expected statutory provision).
* **In-Scope Answer Rate:** **83.33%** (10/12; two lower-similarity phrasing queries—`Q03` at `0.4162` and `Q07` at `0.3925`—abstained conservatively, reflecting a deliberate safety-over-coverage tradeoff).

### 4. Feature B Safeguards & Limitations
* **Mandatory Citations:** Every answered query returns the exact **Act, Section, Official Channel, Source URL, and Last-Verified Date**.
* **Non-Replacement & Disclaimer:** Every response includes an explicit disclaimer stating that HER360 provides general awareness, does not constitute legal advice, and does not replace official complaint portals such as **SHe-Box** (`https://shebox.wcd.gov.in/`).
* **Next Expansion Step:** Expand the corpus with state-specific Shops & Establishments provisions and validate statutory summaries with a qualified Indian legal reviewer.