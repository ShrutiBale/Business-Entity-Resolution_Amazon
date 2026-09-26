"""
ml/features.py — Real pair features for every candidate pair.
Implements all features from approch.md §11.
"""

from typing import Optional
import numpy as np

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False

try:
    from rapidfuzz import fuzz as rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from normalization import country_relation


# ─── String similarity helpers ──────────────────────────────────────
def _levenshtein_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.ratio(a, b) / 100.0
    # fallback: basic edit distance
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return 1.0 - prev[lb] / max(la, lb)


def _jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if HAS_JELLYFISH:
        try:
            return jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            return 0.0
    return 0.0


def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _token_overlap(a: str, b: str) -> int:
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    return len(ta & tb)


def _containment(a: str, b: str) -> float:
    """Is a contained in b or vice versa?"""
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / min(len(ta), len(tb))


def _phonetic_match(a: str, b: str) -> int:
    if not a or not b or not HAS_JELLYFISH:
        return 0
    try:
        return int(jellyfish.metaphone(a[:80]) == jellyfish.metaphone(b[:80]))
    except Exception:
        return 0


# ─── Pair feature extraction ────────────────────────────────────────
def compute_pair_features(s1_row: dict, cand_row: dict,
                           tfidf_name_score: float = 0.0,
                           tfidf_addr_score: float = 0.0,
                           tfidf_comp_score: float = 0.0,
                           blocking_passes: str = "",
                           blocking_pass_count: int = 0,
                           tfidf_name_rank: Optional[int] = None,
                           tfidf_addr_rank: Optional[int] = None,
                           tfidf_comp_rank: Optional[int] = None,
                           name_freq_s1: float = 1.0,
                           name_freq_s2: float = 1.0,
                           name_freq_s3: float = 1.0) -> dict:
    """
    Returns a flat dict of all pair features.
    s1_row and cand_row should be pre-normalized (normalize_row output).
    """
    feats = {}

    # ── Name features (§11.1) ────────────────────────────────────
    n1 = s1_row.get("name_norm", "") or ""
    n2 = cand_row.get("name_norm", "") or ""
    n1_raw = s1_row.get("name_raw", "") or ""
    n2_raw = cand_row.get("name_raw", "") or ""
    n1_legal = s1_row.get("name_legal", "") or ""
    n2_legal = cand_row.get("name_legal", "") or ""
    n1_st  = s1_row.get("name_sorted_tokens", "") or ""
    n2_st  = cand_row.get("name_sorted_tokens", "") or ""

    feats["name_raw_exact"]    = int(n1_raw.lower().strip() == n2_raw.lower().strip())
    feats["name_norm_exact"]   = int(n1 == n2)
    feats["name_legal_exact"]  = int(n1_legal == n2_legal)
    feats["name_levenshtein"]  = _levenshtein_sim(n1, n2)
    feats["name_jaro_winkler"] = _jaro_winkler(n1, n2)
    feats["name_token_jaccard"] = _token_jaccard(n1, n2)
    feats["name_token_overlap"] = _token_overlap(n1, n2)
    feats["name_containment"]   = _containment(n1, n2)
    feats["name_tfidf_cosine"]  = float(tfidf_name_score)
    feats["name_phonetic_match"] = _phonetic_match(n1, n2)
    feats["name_sorted_token_match"] = int(n1_st == n2_st and bool(n1_st))
    feats["name_len_diff"]      = abs(len(n1) - len(n2))
    # name frequency protection (§11.5)
    feats["name_freq_s1"]  = float(name_freq_s1)
    feats["name_freq_s2"]  = float(name_freq_s2)
    feats["name_freq_s3"]  = float(name_freq_s3)
    feats["name_missing_s1"] = int(s1_row.get("name_missing", False))
    feats["name_missing_cand"] = int(cand_row.get("name_missing", False))

    # ── Address features (§11.2) ─────────────────────────────────
    a1 = s1_row.get("addr_norm", "") or ""
    a2 = cand_row.get("addr_norm", "") or ""
    feats["addr_full_lev"]       = _levenshtein_sim(a1, a2)
    feats["addr_token_jaccard"]  = _token_jaccard(a1, a2)
    feats["addr_tfidf_cosine"]   = float(tfidf_addr_score)
    feats["comp_tfidf_cosine"]   = float(tfidf_comp_score)

    # Postal/PIN
    p1 = s1_row.get("postal_code", "") or ""
    p2 = cand_row.get("postal_code", "") or ""
    feats["postal_exact"]    = int(bool(p1) and p1 == p2)
    feats["postal_partial"]  = int(bool(p1) and bool(p2) and p1[:3] == p2[:3])
    feats["postal_missing"]  = int(not p1 or not p2)

    # House number
    h1 = s1_row.get("house_number", "") or ""
    h2 = cand_row.get("house_number", "") or ""
    feats["house_num_exact"]   = int(bool(h1) and h1 == h2)
    feats["house_num_missing"] = int(not h1 or not h2)

    # Street name
    st1 = s1_row.get("street_name", "") or ""
    st2 = cand_row.get("street_name", "") or ""
    feats["street_name_sim"] = _levenshtein_sim(st1, st2)

    # Unit/suite
    u1 = s1_row.get("unit_suite", "") or ""
    u2 = cand_row.get("unit_suite", "") or ""
    feats["unit_suite_exact"]   = int(bool(u1) and bool(u2) and u1 == u2)
    feats["unit_suite_conflict"] = int(bool(u1) and bool(u2) and u1 != u2)
    feats["unit_suite_missing"] = int(not u1 or not u2)

    # Numeric overlap
    nums1 = set(s1_row.get("numeric_tokens", []) or [])
    nums2 = set(cand_row.get("numeric_tokens", []) or [])
    feats["numeric_overlap"]      = len(nums1 & nums2)
    feats["numeric_contradiction"] = int(bool(nums1) and bool(nums2) and not (nums1 & nums2))

    feats["addr_missing_s1"]   = int(s1_row.get("addr_missing", False))
    feats["addr_missing_cand"] = int(cand_row.get("addr_missing", False))

    # ── Country (§11 / §4.3) ─────────────────────────────────────
    cr = country_relation(s1_row.get("country", ""), cand_row.get("country", ""))
    feats["same_country"]      = int(cr == "same_country")
    feats["different_country"] = int(cr == "different_country")
    feats["country_missing"]   = int(cr == "country_missing")

    # ── Cross-field interactions (§11.3) ─────────────────────────
    feats["name_and_addr_both_strong"] = int(feats["name_levenshtein"] >= 0.85 and feats["addr_full_lev"] >= 0.80)
    feats["name_strong_addr_weak"]     = int(feats["name_levenshtein"] >= 0.85 and feats["addr_full_lev"] < 0.40)
    feats["common_name_postal_agree"]  = int(feats["name_freq_s1"] > 0.01 and feats["postal_exact"])
    feats["common_name_house_agree"]   = int(feats["name_freq_s1"] > 0.01 and feats["house_num_exact"])

    # ── Retrieval evidence (§11.4) ────────────────────────────────
    feats["blocking_pass_count"]   = int(blocking_pass_count)
    feats["tfidf_name_rank"]       = int(tfidf_name_rank) if tfidf_name_rank is not None else -1
    feats["tfidf_addr_rank"]       = int(tfidf_addr_rank) if tfidf_addr_rank is not None else -1
    feats["tfidf_comp_rank"]       = int(tfidf_comp_rank) if tfidf_comp_rank is not None else -1
    feats["pass_A"]  = int("A" in blocking_passes)
    feats["pass_B"]  = int("B" in blocking_passes)
    feats["pass_C"]  = int("C" in blocking_passes)
    feats["pass_D"]  = int("D" in blocking_passes)
    feats["pass_E"]  = int("E" in blocking_passes)
    feats["tfidf_name_retrieved"] = int("TFIDF_name" in blocking_passes)
    feats["tfidf_addr_retrieved"] = int("TFIDF_addr" in blocking_passes)

    return feats


def get_feature_names() -> list:
    """Return ordered list of feature names (stable ordering for LightGBM)."""
    dummy_s1 = {k: "" for k in ["name_raw","name_norm","name_legal","name_sorted_tokens","name_phonetic","house_number","street_name","unit_suite","postal_code","country","addr_norm"]}
    dummy_s1.update({"name_missing": False, "addr_missing": False, "numeric_tokens": []})
    return list(compute_pair_features(dummy_s1, dummy_s1).keys())


if __name__ == "__main__":
    print("Feature names:", get_feature_names())
    print("Feature count:", len(get_feature_names()))
