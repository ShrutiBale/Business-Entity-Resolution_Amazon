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
    # Fix 8: distinguish verified training examples from unverifiable test examples
    ground_truth_source: str  # "train_holdout" | "test_unverified"


class CrossSourceSupport(BaseModel):
    """Fix 2: structured triangulation evidence (approch.md §13).
    Only populated when both an S2 and S3 candidate exist for the same S1.
    Never a decision mechanism — supporting evidence only.
    """
    s2_s3_name_similarity: float
    s2_s3_address_similarity: float
    numeric_agreement: bool


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
    # Address component-level features (Fix 4 — approch.md §4.2, §11.2)
    pin_postal_match: bool
    house_number_match: bool
    street_name_similarity: float          # new: component-level street name
    unit_suite_match: Optional[bool] = None  # new: null when not applicable
    # TF-IDF channel ranks (Fix 3 — approch.md §11.4)
    tfidf_name_rank: Optional[int] = None
    tfidf_address_rank: Optional[int] = None
    tfidf_composite_rank: Optional[int] = None  # new: third channel
    blocking_passes: List[str] = []
    blocking_pass_count: int = 0
    # Fix 1: split name_frequency by source (approch.md §11.5)
    name_frequency_s1: float   # fold-local during CV
    name_frequency_s2: float   # source-wide (S2 never partitioned by fold)
    name_frequency_s3: float   # source-wide (S3 never partitioned by fold)
    numeric_evidence: float
    # Fix 2: structured cross-source triangulation (null when only one source)
    cross_source_support: Optional[CrossSourceSupport] = None
    # Fix 5: near-duplicate cluster tagging (approch.md §3.4) — context only
    cluster_id: Optional[str] = None   # within-source cluster ID
    cluster_size: int = 1              # 1 = no near-duplicates found
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
