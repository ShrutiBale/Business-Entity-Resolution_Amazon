"""Data loader — reads JSON files once at startup."""

import json
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).parent / "data"


def load_json(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DataStore:
    def __init__(self):
        raw_entities: List[dict] = load_json("s1_entities.json")
        self.entities: Dict[str, dict] = {e["entity_id"]: e for e in raw_entities}
        self.entities_list: List[dict] = raw_entities

        raw_candidates: dict = load_json("candidates.json")
        self.candidates: Dict[str, List[dict]] = raw_candidates

        raw_decisions: dict = load_json("decisions.json")
        self.decisions: Dict[str, dict] = raw_decisions

        self.metrics: dict = load_json("metrics.json")

        raw_experiments: List[dict] = load_json("experiment_runs.json")
        self.experiments: List[dict] = raw_experiments

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def candidate_count(self) -> int:
        return sum(len(v) for v in self.candidates.values())


_store: DataStore | None = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
    return _store
