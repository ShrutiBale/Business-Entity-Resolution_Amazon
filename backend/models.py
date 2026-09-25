"""Pydantic models for the Business Entity Resolution Demo API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class S1Entity(BaseModel):
    entity_id: str
    business_name: str
    business_address: str
    country: str
    case_tag: str
    case_description: str


class Candidate(BaseModel):
    entity_id: str
    source: str
    business_name: str
    business_address: str
    country: str
    final_score: float
    selected: bool
    name_similarity: float
    address_similarity: float
    composite_similarity: float
    country_relation: str
    pin_postal_match: bool
    house_number_match: bool
    tfidf_name_rank: Optional[int] = None
    tfidf_address_rank: Optional[int] = None
    blocking_passes: List[str] = []
    blocking_pass_count: int = 0
    name_frequency: float
    numeric_evidence: float
    cross_source_support: bool
    notes: str = ""


class Decision(BaseModel):
    entity_id: str
    has_any_match_probability: float
    global_threshold: float
    top1_score: float
    top2_score: float
    top1_top2_margin: float
    n_candidates: int
    n_above_threshold: int
    predicted_matches: List[str]
    decision_notes: str


class EntityDetail(BaseModel):
    entity: S1Entity
    candidates: List[Candidate]
    decision: Decision
    explanation: str


class PredictionDistribution(BaseModel):
    zero_matches: float
    one_match: float
    two_plus_matches: float


class Metrics(BaseModel):
    is_mock_data: bool
    macro_f05: float
    singleton_f05: float
    non_singleton_f05: float
    blocking_recall: float
    pair_precision: float
    pair_recall: float
    mean_candidates_per_s1: float
    p95_candidates_per_s1: float
    max_candidates: int
    prediction_distribution: PredictionDistribution
    error_taxonomy: Dict[str, int]
    validation_note: str


class ExperimentRun(BaseModel):
    run_id: str
    description: str
    blocking_config: str
    tfidf_k: Optional[int]
    blocking_recall: float
    mean_candidates: float
    p95_candidates: int
    max_candidates: int
    threshold: float
    macro_f05: float
    precision: float
    recall: float
    singleton_f05: float
    non_singleton_f05: float
    notes: str


class EntityListResponse(BaseModel):
    total: int
    entities: List[S1Entity]


class HealthResponse(BaseModel):
    status: str
    entities_loaded: int
    candidates_loaded: int
