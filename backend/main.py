"""FastAPI backend for the Business Entity Resolution Demo."""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data_loader import DataStore, get_store
from services.explanation_service import generate_explanation


# ─── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all data once at startup
    store = get_store()
    print(f"✅ Loaded {store.entity_count} S1 entities, {store.candidate_count} candidates")
    yield


app = FastAPI(
    title="Business Entity Resolution Demo",
    description="Amazon ML Challenge 2026 — Judge-facing explainability layer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    store = get_store()
    return {
        "status": "ok",
        "entities_loaded": store.entity_count,
        "candidates_loaded": store.candidate_count,
    }


# ─── S1 Entity Listing ───────────────────────────────────────────────────────

@app.get("/api/s1")
def list_s1(
    search: Optional[str] = Query(None, description="Filter by name or ID (case-insensitive)"),
    case_tag: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    match_state: Optional[str] = Query(None, description="singleton | matched | multi_match | no_match"),
):
    store = get_store()
    results = store.entities_list

    if search:
        q = search.lower()
        results = [
            e for e in results
            if q in e["entity_id"].lower() or q in e["business_name"].lower()
        ]

    if case_tag:
        results = [e for e in results if e["case_tag"] == case_tag]

    if country:
        results = [e for e in results if e["country"].lower() == country.lower()]

    if match_state:
        def _state(eid):
            dec = store.decisions.get(eid, {})
            n = len(dec.get("predicted_matches", []))
            if n == 0:
                return "no_match"
            if n == 1:
                return "matched"
            return "multi_match"

        results = [e for e in results if _state(e["entity_id"]) == match_state]

    return {"total": len(results), "entities": results}


# ─── S1 Entity Detail ────────────────────────────────────────────────────────

@app.get("/api/s1/{entity_id}")
def get_entity_detail(entity_id: str):
    store = get_store()

    entity = store.entities.get(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    candidates = store.candidates.get(entity_id, [])
    decision = store.decisions.get(entity_id, {
        "entity_id": entity_id,
        "has_any_match_probability": 0.0,
        "global_threshold": 0.60,
        "top1_score": 0.0,
        "top2_score": 0.0,
        "top1_top2_margin": 0.0,
        "n_candidates": 0,
        "n_above_threshold": 0,
        "predicted_matches": [],
        "decision_notes": "No candidates retrieved.",
    })

    explanation = generate_explanation(entity, candidates, decision)

    return {
        "entity": entity,
        "candidates": candidates,
        "decision": decision,
        "explanation": explanation,
    }


# ─── Metrics ─────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics():
    store = get_store()
    return store.metrics


# ─── Experiments ─────────────────────────────────────────────────────────────

@app.get("/api/experiments")
def get_experiments():
    store = get_store()
    return {"runs": store.experiments}


# ─── Showcase shortcuts ───────────────────────────────────────────────────────

@app.get("/api/showcase")
def get_showcase():
    """Return the entity_id for each required showcase case."""
    store = get_store()
    showcase = {}
    tag_priority = {
        "singleton": None,
        "single_match": None,
        "multi_match": None,
        "chain_collision": None,
        "transliteration": None,
        "address_variation": None,
        "france": None,
    }
    for e in store.entities_list:
        tag = e["case_tag"]
        if tag in tag_priority and tag_priority[tag] is None:
            tag_priority[tag] = e["entity_id"]

    return {
        "singleton": tag_priority.get("singleton"),
        "single_match": tag_priority.get("single_match"),
        "multi_match": tag_priority.get("multi_match"),
        "chain_collision": tag_priority.get("chain_collision"),
        "noisy_match": tag_priority.get("transliteration"),
        "france": tag_priority.get("france"),
        "address_variation": tag_priority.get("address_variation"),
    }
