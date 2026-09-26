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
  "country": "US | India | France",
  "case_tag": "singleton | single_match | multi_match | chain_collision | transliteration | address_variation | france",
  "case_description": "...",
  "ground_truth_source": "train_holdout | test_unverified"
}]
```

**`ground_truth_source` provenance:** `"train_holdout"` means the entity was drawn from the training set and its ground-truth matches were verified during S1-grouped CV + untouched holdout evaluation. `"test_unverified"` means the entity is from the test set — test ground truth is never available to any participant (not withheld, genuinely nonexistent), so everything shown for these examples is a model **prediction**, not a verified answer. Every `france`-tagged entity is `"test_unverified"`.

---

### `candidates.json`

```json
{
  "S1-00001": [{
    "entity_id": "S2-...",
    "source": "S2 | S3",
    "business_name": "...",
    "business_address": "...",
    "country": "...",
    "final_score": 0.94,
    "selected": true,
    "name_similarity": 0.99,
    "address_similarity": 0.92,
    "composite_similarity": 0.96,
    "country_relation": "same | different | missing",
    "pin_postal_match": true,
    "house_number_match": true,
    "street_name_similarity": 0.97,
    "unit_suite_match": null,
    "tfidf_name_rank": 1,
    "tfidf_address_rank": 2,
    "tfidf_composite_rank": 1,
    "blocking_passes": ["exact_key", "tfidf_name"],
    "blocking_pass_count": 2,
    "name_frequency_s1": 0.04,
    "name_frequency_s2": 0.04,
    "name_frequency_s3": null,
    "numeric_evidence": 0.20,
    "cross_source_support": {
      "s2_s3_name_similarity": 0.91,
      "s2_s3_address_similarity": 0.88,
      "numeric_agreement": true
    },
    "cluster_id": null,
    "cluster_size": 1,
    "notes": "..."
  }]
}
```

**Schema notes:**
- **`name_frequency_s1/s2/s3`** (approch.md §11.5): split by source because `name_frequency_s1` must be computed fold-locally during CV (S1 partitions define the folds), while `name_frequency_s2/s3` are computed source-wide. A collapsed scalar would silently re-introduce fold leakage. `null` when a source has no candidate for this pair.
- **`street_name_similarity`** (approch.md §4.2, §11.2): component-level street name similarity (float). Required alongside whole-string `address_similarity`.
- **`unit_suite_match`** (approch.md §4.2): boolean when a suite/unit number is present in both records; `null` when not applicable to this address pair (not missing — explicitly not applicable).
- **`tfidf_composite_rank`** (approch.md §11.4): rank in the composite TF-IDF retrieval channel (third channel alongside name and address). `null` when the candidate was not retrieved via TF-IDF composite.
- **`cross_source_support`** (approch.md §13): structured triangulation object — populated only when both an S2 and an S3 candidate exist for the same S1. `null` otherwise. Never used as a decision mechanism — supporting evidence context only. A boolean would collapse graduated evidence into something closer to a verdict.
- **`cluster_id` / `cluster_size`** (approch.md §3.4): near-duplicate cluster tagging within the candidate's source. A cluster identifies near-identical records within S2 or within S3 (never across sources). Tagged as **evidence/context only** — never auto-merged, never treated as proof of same real-world entity.

---

### `decisions.json`

Keyed by `entity_id`. Computed server-side from the `candidates.json` accepted-set — **not stored redundantly in `s1_entities.json`**. The `match_state` filter on `GET /api/s1` is computed dynamically from this file at query time (see `main.py: _state(eid)`), keeping a single source of truth.

```json
{
  "S1-00001": {
    "entity_id": "S1-00001",
    "has_any_match_probability": 0.35,
    "global_threshold": 0.60,
    "top1_score": 0.42,
    "top2_score": 0.38,
    "top1_top2_margin": 0.04,
    "n_candidates": 2,
    "n_above_threshold": 0,
    "predicted_matches": [],
    "decision_notes": "No match predicted. Top candidate score 0.42 is below threshold 0.60."
  }
}
```

---

### `experiment_runs.json`

Array of experiment runs, newest best last. Used by the Experiments page.

```json
[{
  "run_id": "RUN-001",
  "description": "Baseline: exact name match only",
  "blocking_config": "Exact normalized name key",
  "tfidf_k": null,
  "blocking_recall": 0.51,
  "mean_candidates": 1.4,
  "p95_candidates": 6,
  "max_candidates": 14,
  "threshold": 0.50,
  "macro_f05": 0.52,
  "precision": 0.89,
  "recall": 0.41,
  "singleton_f05": 0.91,
  "non_singleton_f05": 0.38,
  "notes": "Low recall. Blocking is the bottleneck. Must add TF-IDF retrieval."
}]
```

---

### `metrics.json`

```json
{
  "is_mock_data": true,
  "macro_f05": 0.847,
  "singleton_f05": 0.913,
  "non_singleton_f05": 0.798,
  "blocking_recall": 0.963,
  "pair_precision": 0.871,
  "pair_recall": 0.812,
  "mean_candidates_per_s1": 4.2,
  "p95_candidates_per_s1": 18.0,
  "max_candidates": 47,
  "prediction_distribution": {"zero_matches": 0.31, "one_match": 0.48, "two_plus_matches": 0.21},
  "error_taxonomy": { "BLOCKING_MISS": 4, ... },
  "validation_note": "DEMO DATA — NOT REAL VALIDATION."
}
```

Set `"is_mock_data": false` to remove the **DEMO DATA** banner from all pages.

---

## Replacing Mock Data with Real Pipeline Exports

**No code changes required.** Replace the 5 JSON files in `backend/data/` with real exports from your pipeline:

1. `s1_entities.json` — export your S1 test entities (or a curated demo sample)
2. `candidates.json` — export your `candidate_pairs` with per-pair feature values attached
3. `decisions.json` — export your threshold decisions per S1
4. `metrics.json` — export from your validation run log; set `"is_mock_data": false`
5. `experiment_runs.json` — export from your experiment tracking

---

## Showcase Cases Covered

| Case | Entity ID | Ground Truth | Description |
|---|---|---|---|
| Singleton | S1-00001 | ✓ train_holdout | Candidates retrieved but all correctly rejected |
| Clean Match | S1-00002 | ✓ train_holdout | One high-confidence exact match |
| Noisy Match | S1-00003 | ✓ train_holdout | Abbreviations + address variations matched |
| Multi-Match | S1-00004 | ✓ train_holdout | 3 S2/S3 records legitimately match one S1 |
| Chain Collision | S1-00005 | ✓ train_holdout | Same name, different branches — only 1 accepted |
| Address Variation | S1-00006 | ✓ train_holdout | street_name_similarity high, unit_suite_match=False reveals discrepancy |
| Transliteration | S1-00007 | ✓ train_holdout | Sharma ↔ Sarma phonetic blocking |
| France (unseen) | S1-00008 | ⚠ test_unverified | Prediction only — no ground truth exists for any participant |
| France multi-match | S1-00009 | ⚠ test_unverified | Prediction only — unseen-domain generalisation |
| Generic Name | S1-00010 | ✓ train_holdout | "Star Hotel" — common name, all rejected |

---

## 3–5 Minute Judge Demo Flow

1. **Home** (`/`) — Show pipeline diagram animating through stages. Explain: 3 sources, 0/1/many matches, F₀.₅, open-set country. Point out the **Training vs. Test Provenance** note at the bottom — this is important context.

2. **Chain Collision** (`/entities/S1-00005`) — Show "Cafe Coffee Day" with 4 candidates. Name similarity = 1.0 for 3 of them, but only MG Road branch (PIN 560001) is accepted. The feature table shows `name_frequency_s1` is very low — distinctive name yet multiple branches. Address/PIN evidence overrides perfect name match.

3. **Address Variation** (`/entities/S1-00006`) — Show "Liberty Pharmaceuticals". Point to `street_name_similarity=0.95` (high) but `unit_suite_match=False` (Suite 4B vs 4C). Whole-address similarity (0.74) is diluted — component-level parsing reveals the exact source of the discrepancy.

4. **Multi-Match** (`/entities/S1-00004`) — Show TechZone Electronics with 3 accepted matches across S2+S3. Note the near-duplicate cluster badge on S3-40031/S3-40089 — these are near-identical within S3, but correctly kept as separate candidates (never auto-merged). Cross-source triangulation shows S2↔S3 name/addr similarity.

5. **France** (`/entities/S1-00008`) — Show the **"⚠ TEST PREDICTION — not verifiable"** badge. The match IDs are shown as orange ⚠ predictions, not green ✓ verified. Explain: France is genuinely test data — no ground truth exists for any participant. The pipeline generalises via open-set normalization design.

6. **Metrics** (`/metrics`) — Show Macro F₀.₅ = 0.847. Entity-level, includes singletons. Show error taxonomy.

7. **Experiments** (`/experiments`) — RUN-001 (F₀.₅ 0.52) → RUN-004 (F₀.₅ 0.847). Each step was measured — no arbitrary complexity added.

---

## Important Notes

- **DEMO DATA — NOT REAL VALIDATION** is displayed throughout when `is_mock_data=true`
- The demo is read-only — no live retraining or live full-scale blocking
- Explanations are deterministic and feature-based — no LLM used
- No external APIs, geocoding, or business registries are accessed
- France showcase entities show **predictions only** — test ground truth is inherently unavailable to all participants
