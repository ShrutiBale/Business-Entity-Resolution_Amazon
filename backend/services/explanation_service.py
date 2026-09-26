"""
Deterministic, feature-based explanation generator.
No LLM — explanations are derived purely from candidate feature values.
Aligned with approch.md Fixes 1-5 and Fix 8 from modified_prompt.md.
"""

from typing import List, Optional


def generate_explanation(entity: dict, candidates: List[dict], decision: dict) -> str:
    predicted = decision.get("predicted_matches", [])
    n_candidates = len(candidates)
    threshold = decision.get("global_threshold", 0.60)
    is_france = entity.get("country", "").lower() == "france"
    case_tag = entity.get("case_tag", "")
    gt_source = entity.get("ground_truth_source", "train_holdout")

    parts: List[str] = []

    # ── Fix 8: domain provenance note ───────────────────────────
    if gt_source == "test_unverified":
        parts.append(
            "⚠️ TEST-DERIVED EXAMPLE: This entity is from the test set. "
            "No ground truth exists for any test entity — not withheld from us, genuinely nonexistent for all participants. "
            "Everything shown here is what the frozen US/India-trained pipeline *predicts* on genuinely unseen data. "
            "France-specific training was never performed; the model generalises via open-set design."
        )
    elif is_france:
        parts.append(
            "⚠️ Unseen domain (France): this entity is from a country not seen during training. "
            "The pipeline applies the US/India-trained model as-is — no France-specific tuning."
        )

    # ── Retrieval summary ────────────────────────────────────────
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

    accepted = [c for c in candidates if c["selected"]]
    rejected = [c for c in candidates if not c["selected"]]

    # ── Per accepted candidate ───────────────────────────────────
    if accepted:
        acc_parts = []
        for c in accepted:
            reasons = _acceptance_reasons(c, threshold, gt_source)
            verb = "predicted match" if gt_source == "test_unverified" else "accepted"
            acc_parts.append(
                f"✅ {c['entity_id']} ({c['source']}) {verb} "
                f"(score {c['final_score']:.3f} ≥ {threshold}): {reasons}"
            )
        parts.append(" | ".join(acc_parts))
    else:
        parts.append(f"No candidate cleared the decision threshold ({threshold}).")

    # ── Top rejected candidates ──────────────────────────────────
    top_rejected = sorted(rejected, key=lambda c: c["final_score"], reverse=True)[:3]
    if top_rejected:
        rej_parts = []
        for c in top_rejected:
            reasons = _rejection_reasons(c, threshold)
            rej_parts.append(
                f"❌ {c['entity_id']} ({c['source']}) rejected "
                f"(score {c['final_score']:.3f}): {reasons}"
            )
        parts.append(" | ".join(rej_parts))

    # ── Case-tag specific note ───────────────────────────────────
    if case_tag == "chain_collision":
        parts.append(
            "Chain/franchise collision: name_frequency_s1/s2/s3 are all very low (highly distinctive), "
            "yet the name appears multiple times across branches. "
            "Address/postal evidence was required to discriminate — name similarity alone was insufficient."
        )
    elif case_tag == "singleton":
        parts.append(
            "Singleton: all candidates correctly rejected. "
            "The model required both name and address/numeric evidence to meet threshold."
        )
    elif case_tag == "multi_match":
        parts.append(
            f"Multi-match: {len(accepted)} records from S2/S3 independently refer to the same real-world entity. "
            "One-to-one matching is never assumed."
        )
    elif case_tag == "transliteration":
        parts.append(
            "Transliteration: phonetic blocking recovered a variant that "
            "exact-name blocking would have missed. "
            "address_similarity and PIN/postal agreement provided the confirming evidence."
        )
    elif case_tag == "address_variation":
        parts.append(
            "Address variation: component-level features (street_name_similarity, unit_suite_match) "
            "reveal why whole-string address similarity alone can be misleading."
        )

    return " ".join(parts)


def _acceptance_reasons(c: dict, threshold: float, gt_source: str) -> str:
    reasons = []
    if c.get("name_similarity", 0) >= 0.90:
        reasons.append(f"name sim {c['name_similarity']:.2f}")
    elif c.get("name_similarity", 0) >= 0.75:
        reasons.append(f"name sim {c['name_similarity']:.2f} (noisy match)")
    if c.get("pin_postal_match"):
        reasons.append("postal/PIN exact match")
    if c.get("house_number_match"):
        reasons.append("house number agreement")
    if (c.get("street_name_similarity") or 0) >= 0.80:
        reasons.append(f"street name sim {c['street_name_similarity']:.2f}")
    if c.get("address_similarity", 0) >= 0.80:
        reasons.append(f"address sim {c['address_similarity']:.2f}")
    bpc = c.get("blocking_pass_count", 0)
    if bpc >= 2:
        reasons.append(f"retrieved by {bpc} blocking passes")
    # Fix 2: use structured cross_source_support
    css = c.get("cross_source_support")
    if css and isinstance(css, dict):
        if css.get("numeric_agreement"):
            reasons.append(f"cross-src triangulation (name {css['s2_s3_name_similarity']:.2f}, addr {css['s2_s3_address_similarity']:.2f}, numeric ✓)")
    # Fix 1: use source-split name frequencies
    nf_s1 = c.get("name_frequency_s1", 1.0)
    if nf_s1 < 0.01:
        reasons.append(f"rare/distinctive name (S1 freq {nf_s1:.4f})")
    if c.get("cluster_size", 1) > 1:
        reasons.append(f"near-duplicate cluster ({c['cluster_size']} records — context only, not auto-merged)")
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
    # Fix 1: prefer s1 frequency for rejection reasoning (fold-local)
    nf_s1 = c.get("name_frequency_s1", 1.0)
    if nf_s1 < 0.005:
        reasons.append(
            f"generic/very-common name (S1 freq={nf_s1:.4f}) demands stronger address evidence"
        )
    if c.get("address_similarity", 0) < 0.35:
        reasons.append(f"low address sim ({c['address_similarity']:.2f})")
    # Fix 4: unit/suite mismatch
    if c.get("unit_suite_match") is False:
        reasons.append("unit/suite mismatch")
    return ", ".join(reasons) if reasons else "score below threshold"
