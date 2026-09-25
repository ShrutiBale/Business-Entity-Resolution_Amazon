# EntityResolve — Business Entity Resolution Demo
### Amazon ML Challenge 2026 | Judge-Facing Explainability Layer

> **This is a visualization/explainability layer over the ML pipeline — not the competition pipeline itself.**
> The production pipeline runs separately: `normalization → candidate retrieval/blocking → features → LightGBM → threshold → matches`

---

## Overview

A polished judge-facing demo that makes the business entity resolution pipeline's decisions **transparent and explainable**. Given business records from 3 independent sources (S1, S2, S3) with noisy, inconsistent fields, the system determines which S2/S3 records refer to the same real-world business as each S1 entity.

**Key design principles:**
- S1 may match 0, 1, or many S2/S3 records — multi-match is first-class
- Country is open-set — France (unseen test domain) is handled without France-specific training
- Precision-heavy F₀.₅ metric — false merges cost more than missed links
- No external lookup, geocoding, or external data augmentation

---

## Architecture

```
business-entity-resolution-demo/
├── backend/           # FastAPI (Python 3.11+)
│   ├── main.py        # FastAPI app + all endpoints
│   ├── models.py      # Pydantic schemas
│   ├── data_loader.py # Loads JSON files once at startup
│   ├── services/
│   │   └── explanation_service.py  # Deterministic feature-based explanations
│   ├── data/          # JSON data files (swap for real pipeline exports)
│   │   ├── s1_entities.json
│   │   ├── candidates.json
│   │   ├── decisions.json
│   │   ├── metrics.json
│   │   └── experiment_runs.json
│   └── requirements.txt
├── frontend/          # React + Vite
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── pages/
│           ├── Home.jsx            # Pipeline diagram + showcase shortcuts
│           ├── EntityBrowser.jsx   # Search + filter entity table
│           ├── EntityDetail.jsx    # Per-entity trace, evidence, explanation
│           ├── MetricsDashboard.jsx# F0.5, charts, error taxonomy
│           ├── Experiments.jsx     # Experiment run comparison
│           └── Methodology.jsx     # Full methodology write-up
└── scripts/
    └── generate_mock_data.py       # Regenerates all data files
```

---

## Setup & Run

### 1. Backend

```bash
cd backend
pip install fastapi uvicorn pydantic
uvicorn main:app --reload --port 8000
```

Backend runs at **http://localhost:8000**

### 2. Frontend

```bash
cd frontend
npm install
npm run dev -- --port 3000
```

Frontend runs at **http://localhost:3000**

### 3. Regenerate mock data (optional)

```bash
python3 scripts/generate_mock_data.py
```

---

## API Summary

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check + loaded counts |
| `GET /api/s1?search=&case_tag=&country=&match_state=` | List/filter S1 entities |
| `GET /api/s1/{entity_id}` | Entity detail with candidates, decision, explanation |
| `GET /api/metrics` | Precomputed metrics (F₀.₅, blocking recall, etc.) |
| `GET /api/experiments` | Experiment run log |
| `GET /api/showcase` | Entity IDs for each required showcase case |

---

## Data Schemas

### `s1_entities.json`
```json
[{
  "entity_id": "S1-00001",
  "business_name": "...",
  "business_address": "...",
  "country": "US|India|France",
  "case_tag": "singleton|single_match|multi_match|chain_collision|transliteration|address_variation|france",
  "case_description": "..."
}]
```

### `candidates.json`
```json
{
  "S1-00001": [{
    "entity_id": "S2-...",
    "source": "S2|S3",
    "business_name": "...",
    "business_address": "...",
    "country": "...",
    "final_score": 0.94,
    "selected": true,
    "name_similarity": 0.99,
    "address_similarity": 0.92,
    "composite_similarity": 0.96,
    "country_relation": "same|different|missing",
    "pin_postal_match": true,
    "house_number_match": true,
    "tfidf_name_rank": 1,
    "tfidf_address_rank": 2,
    "blocking_passes": ["exact_key", "tfidf_name"],
    "blocking_pass_count": 2,
    "name_frequency": 0.03,
    "numeric_evidence": 0.20,
    "cross_source_support": true,
    "notes": "..."
  }]
}
```

### `metrics.json`
```json
{
  "is_mock_data": true,
  "macro_f05": 0.847,
  "singleton_f05": 0.913,
  "non_singleton_f05": 0.798,
  "blocking_recall": 0.963,
  ...
}
```

---

## Replacing Mock Data with Real Pipeline Exports

**No code changes required.** Replace the 5 JSON files in `backend/data/` with real exports from your pipeline:

1. `s1_entities.json` — export your S1 test entities (or a curated demo sample)
2. `candidates.json` — export your `candidate_pairs` with per-pair feature values attached
3. `decisions.json` — export your threshold decisions per S1
4. `metrics.json` — export from your validation run log; set `"is_mock_data": false`
5. `experiment_runs.json` — export from your experiment tracking (Section 29 of the approach)

Set `"is_mock_data": false` in `metrics.json` to remove the **DEMO DATA** banner.

---

## Showcase Cases Covered

| Case | Entity ID | Description |
|---|---|---|
| Singleton | S1-00001 | Candidates retrieved but all correctly rejected |
| Clean Match | S1-00002 | One high-confidence exact match |
| Noisy Match | S1-00003 | Abbreviations + address variations matched |
| Multi-Match | S1-00004 | 3 S2/S3 records legitimately match one S1 |
| Chain Collision | S1-00005 | Same name, different branches — only 1 accepted |
| Address Variation | S1-00006 | W 34th St ↔ West 34th Street |
| Transliteration | S1-00007 | Sharma ↔ Sarma phonetic blocking |
| France (unseen) | S1-00008 | French entity, open-set model |
| France multi-match | S1-00009 | French chain with two matching records |
| Generic Name | S1-00010 | "Star Hotel" — common name, all rejected |

---

## 3–5 Minute Judge Demo Flow

1. **Home** (`/`) — Show pipeline diagram animating through stages. Explain: 3 sources, 0/1/many matches, F₀.₅, open-set country.

2. **Chain Collision** (`/entities/S1-00005`) — Show "Cafe Coffee Day" with 4 candidates. Name similarity = 1.0 for 3 of them, but only MG Road branch (PIN 560001) is accepted. Explain how address/PIN evidence overrides perfect name match.

3. **Multi-Match** (`/entities/S1-00004`) — Show TechZone Electronics with 3 accepted matches across S2+S3. Emphasize that multi-match is first-class — not forced to one.

4. **France** (`/entities/S1-00008`) — Show the UNSEEN DOMAIN badge. Pipeline trained on US/India only. Boulangerie Dupont matched correctly via open-set normalization.

5. **Metrics** (`/metrics`) — Show Macro F₀.₅ = 0.847. Point to the hero metric, explain it's entity-level (not pair-level), includes singletons. Show error taxonomy.

6. **Experiments** (`/experiments`) — Show RUN-001 (F₀.₅ 0.52) → RUN-004 (F₀.₅ 0.847). Each step was measured — no arbitrary complexity added.

---

## Important Notes

- **DEMO DATA — NOT REAL VALIDATION** is displayed throughout when `is_mock_data=true`
- The demo is read-only — no live retraining or live full-scale blocking
- Explanations are deterministic and feature-based — no LLM used
- No external APIs, geocoding, or business registries are accessed
