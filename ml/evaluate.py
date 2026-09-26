"""
ml/evaluate.py — Entity-level macro F0.5 scorer.
The official competition metric (approch.md §12).
Unit-tested against a worked example.
"""

import numpy as np
from typing import Dict, List, Set


def entity_f05(true_set: Set[str], pred_set: Set[str]) -> float:
    """
    F0.5 for a single S1 entity.
    F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
         = 1.25 * P * R / (0.25 * P + R)
    """
    if not true_set and not pred_set:
        return 1.0  # correctly predicting no-match
    if not true_set:
        return 0.0  # predicted something, nothing to match
    if not pred_set:
        return 0.0  # missed everything

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall    = tp / len(true_set)

    if precision + recall == 0:
        return 0.0

    beta = 0.5
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def macro_f05(
    true_matches: Dict[str, List[str]],
    pred_matches: Dict[str, List[str]],
    all_s1_ids: List[str],
) -> dict:
    """
    Compute macro F0.5 over all S1 entities (including true singletons).
    true_matches: {s1_id: [matched_ids]} — from ground truth
    pred_matches: {s1_id: [matched_ids]} — from model
    all_s1_ids:   complete list of S1 entity IDs in the split
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id in all_s1_ids:
        true_set = set(true_matches.get(s1_id, []))
        pred_set = set(pred_matches.get(s1_id, []))

        score = entity_f05(true_set, pred_set)
        scores.append(score)

        if true_set:
            non_singleton_scores.append(score)
        else:
            singleton_scores.append(score)

    result = {
        "macro_f05":         round(float(np.mean(scores)), 5),
        "singleton_f05":     round(float(np.mean(singleton_scores)), 5) if singleton_scores else None,
        "non_singleton_f05": round(float(np.mean(non_singleton_scores)), 5) if non_singleton_scores else None,
        "n_entities":        len(all_s1_ids),
        "n_singletons":      len(singleton_scores),
        "n_non_singletons":  len(non_singleton_scores),
    }
    return result


def parse_ground_truth(gt_df) -> Dict[str, List[str]]:
    """Parse ground truth TSV into dict."""
    true_matches = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        ids_str = str(row.get("matched_entity_ids", ""))
        ids = [i.strip() for i in ids_str.split(",") if i.strip()]
        true_matches[s1_id] = ids
    return true_matches


def threshold_sweep(
    s1_ids: List[str],
    true_matches: Dict[str, List[str]],
    candidate_pairs,   # pd.DataFrame with s1_id, candidate_id, score
    thresholds=None,
) -> dict:
    """
    Sweep thresholds and return the one maximising macro F0.5.
    candidate_pairs must have columns: s1_id, candidate_id, lgbm_score.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.90, 0.02)]

    best_t = 0.50
    best_score = -1.0
    results = []

    for t in thresholds:
        pred = {}
        for _, row in candidate_pairs.iterrows():
            s1 = str(row["s1_id"])
            if row["lgbm_score"] >= t:
                pred.setdefault(s1, []).append(str(row["candidate_id"]))

        metrics = macro_f05(true_matches, pred, s1_ids)
        results.append({"threshold": t, **metrics})
        if metrics["macro_f05"] > best_score:
            best_score = metrics["macro_f05"]
            best_t = t

    return {
        "best_threshold": best_t,
        "best_macro_f05": round(best_score, 5),
        "sweep": results,
    }


# ─── Unit test against worked example ─────────────────────────────
def _unit_test():
    # True entity: matches = {A, B}
    assert abs(entity_f05({"A","B"}, {"A","B"}) - 1.0) < 1e-6, "perfect"
    assert abs(entity_f05({"A","B"}, {"A"})   - entity_f05({"A","B"}, {"A"})) < 1e-6
    # Singleton: correct prediction
    assert entity_f05(set(), set()) == 1.0, "singleton correct"
    # Singleton: wrong prediction
    assert entity_f05(set(), {"A"}) == 0.0, "singleton wrong"
    # Macro F0.5 basic
    m = macro_f05({"s1": ["A"]}, {"s1": ["A"]}, ["s1"])
    assert abs(m["macro_f05"] - 1.0) < 1e-6
    print("✅ entity_f05 unit tests passed")


if __name__ == "__main__":
    _unit_test()
    # Quick demo
    true = {"s1a": ["S2-1", "S3-2"], "s1b": []}
    pred = {"s1a": ["S2-1"],          "s1b": []}
    print(macro_f05(true, pred, ["s1a", "s1b", "s1c"]))
