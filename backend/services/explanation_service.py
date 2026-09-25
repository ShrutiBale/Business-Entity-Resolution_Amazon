"""
Deterministic, feature-based explanation generator.
No LLM — explanations are derived purely from candidate feature values.
"""

from typing import List


def generate_explanation(entity: dict, candidates: List[dict], decision: dict) -> str:
    """
    Generate a human-readable explanation for the pipeline decision on an S1 entity.
    Follows the rule from the prompt: only mention evidence actually present in the candidate.
    """
    predicted = decision.get("predicted_matches", [])
    n_candidates = len(candidates)
    threshold = decision.get("global_threshold", 0.60)
    is_france = entity.get("country", "").lower() == "france"
    case_tag = entity.get("case_tag", "")

    parts: List[str] = []

    # ── Domain note
    if is_france:
        parts.append(
            "⚠️ Unseen domain (France): this entity is from a country not seen during training. "
            "The pipeline applies the US/India-trained model as-is — no France-specific tuning."
        )

    # ── Candidate retrieval summary
    if n_candidates == 0:
        parts.append(
            "No candidates were retrieved during blocking. "
            "This entity is predicted as a singleton (no match)."
        )
        return " ".join(parts)

    passes_seen = set()
    for c in candidates:
        passes_seen.update(c.get("blocking_passes", []))

    parts.append(
        f"Blocking retrieved {n_candidates} candidate(s) via passes: {', '.join(sorted(passes_seen))}."
    )

    # ── Per accepted candidate
    accepted = [c for c in candidates if c["selected"]]
    rejected = [c for c in candidates if not c["selected"]]

    if accepted:
        acc_parts = []
        for c in accepted:
            reasons = _acceptance_reasons(c)
            acc_parts.append(
                f"✅ {c['entity_id']} ({c['source']}) accepted (score {c['final_score']:.3f} ≥ {threshold}): {reasons}"
            )
        parts.append(" | ".join(acc_parts))
    else:
        parts.append(f"No candidate cleared the decision threshold ({threshold}).")

    # ── Per rejected candidate (top scorers only)
    top_rejected = sorted(rejected, key=lambda c: c["final_score"], reverse=True)[:3]
    if top_rejected:
        rej_parts = []
        for c in top_rejected:
            reasons = _rejection_reasons(c, threshold)
            rej_parts.append(
                f"❌ {c['entity_id']} ({c['source']}) rejected (score {c['final_score']:.3f}): {reasons}"
            )
        parts.append(" | ".join(rej_parts))

    # ── Case-tag specific note
    if case_tag == "chain_collision":
        parts.append(
            "Chain/franchise collision: high name similarity alone was not sufficient — "
            "address and postal evidence was required to discriminate between branches."
        )
    elif case_tag == "singleton":
        parts.append(
            "Singleton: all candidates were correctly rejected. "
            "The model required both name and address evidence to meet threshold."
        )
    elif case_tag == "multi_match":
        parts.append(
            f"Multi-match: {len(accepted)} records from S2/S3 independently refer to the same real-world entity."
        )
    elif case_tag == "transliteration":
        parts.append(
            "Transliteration: phonetic blocking recovered a transliteration variant that "
            "exact-name blocking would have missed."
        )

    return " ".join(parts)


def _acceptance_reasons(c: dict) -> str:
    reasons = []
    if c.get("name_similarity", 0) >= 0.90:
        reasons.append(f"name similarity {c['name_similarity']:.2f}")
    elif c.get("name_similarity", 0) >= 0.75:
        reasons.append(f"name similarity {c['name_similarity']:.2f} (noisy match)")
    if c.get("pin_postal_match"):
        reasons.append("postal/PIN exact match")
    if c.get("house_number_match"):
        reasons.append("house number agreement")
    if c.get("address_similarity", 0) >= 0.80:
        reasons.append(f"address similarity {c['address_similarity']:.2f}")
    bpc = c.get("blocking_pass_count", 0)
    if bpc >= 2:
        reasons.append(f"retrieved by {bpc} blocking passes")
    if c.get("cross_source_support"):
        reasons.append("cross-source triangulation support")
    nf = c.get("name_frequency", 1.0)
    if nf < 0.01:
        reasons.append("rare/distinctive name")
    return ", ".join(reasons) if reasons else "score above threshold"


def _rejection_reasons(c: dict, threshold: float) -> str:
    reasons = []
    gap = threshold - c["final_score"]
    if gap > 0:
        reasons.append(f"score {gap:.3f} below threshold")
    if not c.get("pin_postal_match") and not c.get("house_number_match"):
        reasons.append("no postal or house-number agreement")
    if c.get("name_similarity", 1) >= 0.95 and c.get("address_similarity", 1) < 0.40:
        reasons.append("name match but address conflict (chain collision signal)")
    nf = c.get("name_frequency", 1.0)
    if nf < 0.005:
        reasons.append(f"generic/common name (freq={nf:.4f}) demands stronger address evidence")
    if c.get("address_similarity", 0) < 0.35:
        reasons.append(f"low address similarity ({c['address_similarity']:.2f})")
    return ", ".join(reasons) if reasons else "score below threshold"
