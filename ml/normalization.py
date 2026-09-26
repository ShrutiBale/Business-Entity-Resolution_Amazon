"""
ml/normalization.py — Safe, non-destructive vectorized normalization.
Creates multiple views; never replaces the original.
Aligned with approch.md §4.

Key design: preprocess_df_fast() works on full DataFrames with pandas
string ops — 10-50x faster than per-row apply for large datasets.
"""

import re
import unicodedata
from typing import Optional

try:
    from unidecode import unidecode as _unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False

import pandas as pd
import numpy as np

# ─── Legal suffix normalization vocabulary ─────────────────────────
LEGAL_PAIRS = [
    (r"\bcorp\b", "corporation"),
    (r"\bpvt\b", "private"),
    (r"\bltd\b", "limited"),
    (r"\bllc\b", "llc"),
    (r"\binc\b", "incorporated"),
    (r"\b&\b", "and"),
    # French legal forms (kept as-is — normalised only)
]

# ─── Address abbreviation vocabulary ──────────────────────────────
ADDR_PAIRS = [
    (r"\bst\b", "street"),
    (r"\brd\b", "road"),
    (r"\bblvd\b", "boulevard"),
    (r"\bav\b", "avenue"),
    (r"\bave\b", "avenue"),
    (r"\bdr\b", "drive"),
    (r"\bln\b", "lane"),
    (r"\bct\b", "court"),
    (r"\bpl\b", "place"),
    (r"\bpkwy\b", "parkway"),
    (r"\bhwy\b", "highway"),
    (r"\bn\b", "north"),
    (r"\bs\b", "south"),
    (r"\be\b", "east"),
    (r"\bw\b", "west"),
    (r"\bbd\b", "boulevard"),
]

# Pre-compiled regex patterns
_PUNCT_RE   = re.compile(r"[^\w\s-]")
_SPACE_RE   = re.compile(r"\s+")
_ADDR_PUNCT = re.compile(r"[^\w\s,.-]")
US_ZIP_RE    = re.compile(r'\b\d{5}(?:-\d{4})?\b')
INDIA_PIN_RE = re.compile(r'\b[1-9]\d{5}\b')
GENERIC_NUM  = re.compile(r'\b\d{4,6}\b')
HOUSE_NUM_RE = re.compile(r'^\s*(\d+[a-zA-Z]?)\b')
UNIT_RE      = re.compile(r'\b(?:suite|ste|unit|apt|#)\s*([a-z0-9]+)')


def _accent_fold(s: str) -> str:
    if not s:
        return ""
    if HAS_UNIDECODE:
        return _unidecode(s)
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()


# ─── Vectorized DataFrame normalization (fast path for large DFs) ──
def preprocess_df_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized normalization for full DataFrames.
    10-50x faster than per-row apply on large datasets.
    Returns original columns PLUS all normalized view columns.
    """
    out = df.copy()
    name_col = "business_name" if "business_name" in df.columns else None
    addr_col = "business_address" if "business_address" in df.columns else None
    ctry_col = "country" if "country" in df.columns else None

    # ── Name normalization ──────────────────────────────────────────
    if name_col:
        raw = out[name_col].fillna("").astype(str)
        out["name_raw"]     = raw
        out["name_missing"] = raw.str.strip() == ""

        lower = raw.str.lower().str.strip()
        if HAS_UNIDECODE:
            folded = lower.apply(_accent_fold)
        else:
            folded = lower.apply(lambda s: unicodedata.normalize("NFD", s).encode("ascii","ignore").decode())

        clean = folded.str.replace(_PUNCT_RE, " ", regex=True).str.replace(_SPACE_RE, " ", regex=True).str.strip()
        out["name_lower"] = lower
        out["name_norm"]  = clean

        legal = clean.copy()
        for pat, repl in LEGAL_PAIRS:
            legal = legal.str.replace(pat, repl, regex=True)
        legal = legal.str.replace(_SPACE_RE, " ", regex=True).str.strip()
        out["name_legal"] = legal

        tokens = clean.str.split()
        out["name_tokens"]        = tokens
        out["name_sorted_tokens"] = tokens.apply(lambda t: " ".join(sorted(t)) if isinstance(t, list) else "")

        # Phonetic — must be per-row but only on non-empty
        if HAS_JELLYFISH:
            out["name_phonetic"] = clean.apply(
                lambda s: jellyfish.metaphone(s[:80]) if s else ""
            )
        else:
            out["name_phonetic"] = ""
    else:
        for col in ["name_raw","name_lower","name_norm","name_legal","name_tokens","name_sorted_tokens","name_phonetic"]:
            out[col] = ""
        out["name_missing"] = True

    # ── Address normalization ───────────────────────────────────────
    if addr_col:
        araw = out[addr_col].fillna("").astype(str)
        out["addr_raw"]     = araw
        out["addr_missing"] = araw.str.strip() == ""

        ctry = out[ctry_col].fillna("").str.lower() if ctry_col else pd.Series([""] * len(out))

        alower = araw.str.lower().str.strip()
        if HAS_UNIDECODE:
            afolded = alower.apply(_accent_fold)
        else:
            afolded = alower.apply(lambda s: unicodedata.normalize("NFD", s).encode("ascii","ignore").decode())

        aclean = afolded.str.replace(_ADDR_PUNCT, " ", regex=True).str.replace(_SPACE_RE, " ", regex=True).str.strip()
        aexpanded = aclean.copy()
        for pat, repl in ADDR_PAIRS:
            aexpanded = aexpanded.str.replace(pat, repl, regex=True)
        aexpanded = aexpanded.str.replace(_SPACE_RE, " ", regex=True).str.strip()
        out["addr_norm"]   = aexpanded
        out["addr_tokens"] = aexpanded.str.split()

        # Postal code — vectorized per country group
        postal = pd.Series([""] * len(out), index=out.index)
        us_mask    = ctry.str.contains("us|united states|usa", regex=True, na=False)
        india_mask = ctry.str.contains("india", na=False)
        postal[us_mask]    = araw[us_mask].apply(lambda s: (m.group() if (m := US_ZIP_RE.search(s)) else ""))
        postal[india_mask] = araw[india_mask].apply(lambda s: (m.group() if (m := INDIA_PIN_RE.search(s)) else ""))
        rest_mask = ~us_mask & ~india_mask
        postal[rest_mask]  = araw[rest_mask].apply(lambda s: (m.group() if (m := GENERIC_NUM.search(s)) else ""))
        # Fallback generic for missed
        fallback = postal == ""
        postal[fallback] = araw[fallback].apply(lambda s: (m.group() if (m := GENERIC_NUM.search(s)) else ""))
        out["postal_code"] = postal

        # House number
        out["house_number"] = araw.str.strip().apply(
            lambda s: (m.group(1) if (m := HOUSE_NUM_RE.match(s)) else "")
        )

        # Unit/suite
        out["unit_suite"] = alower.apply(
            lambda s: (m.group(1) if (m := UNIT_RE.search(s)) else "")
        )

        # Street name heuristic
        def _street(row):
            hn = row["house_number"]
            an = row["addr_norm"]
            if hn and len(an.split()) > 2:
                remaining = an.replace(hn, "", 1).strip()
                parts = remaining.split(",")
                return parts[0].strip() if parts else ""
            return ""
        out["street_name"] = out[["house_number","addr_norm"]].apply(_street, axis=1)

        # Numeric tokens
        out["numeric_tokens"] = araw.apply(lambda s: re.findall(r'\b\d+\b', s))
    else:
        for col in ["addr_raw","addr_norm","addr_tokens","house_number","street_name","unit_suite","postal_code","numeric_tokens"]:
            out[col] = ""
        out["addr_missing"] = True
        out["numeric_tokens"] = [[] for _ in range(len(out))]

    # ── Country ────────────────────────────────────────────────────
    if ctry_col:
        out["country_norm"] = out[ctry_col].fillna("").apply(normalize_country)
    else:
        out["country_norm"] = ""

    return out


# ─── Per-row normalize (still used for single entity lookups) ──────
def normalize_name(name: str) -> dict:
    raw = name.strip()
    if not raw:
        return {
            "name_raw": "", "name_lower": "", "name_norm": "",
            "name_legal": "", "name_phonetic": "", "name_sorted_tokens": "",
            "name_tokens": [], "name_missing": True,
        }
    lower   = raw.lower()
    folded  = _accent_fold(lower)
    clean   = _PUNCT_RE.sub(" ", folded)
    clean   = _SPACE_RE.sub(" ", clean).strip()
    legal   = clean
    for pat, repl in LEGAL_PAIRS:
        legal = re.sub(pat, repl, legal)
    legal = _SPACE_RE.sub(" ", legal).strip()
    tokens = clean.split()
    phonetic = ""
    if HAS_JELLYFISH and clean:
        try:
            phonetic = jellyfish.metaphone(clean[:80])
        except Exception:
            pass
    return {
        "name_raw": raw, "name_lower": lower, "name_norm": clean,
        "name_legal": legal, "name_phonetic": phonetic,
        "name_sorted_tokens": " ".join(sorted(tokens)),
        "name_tokens": tokens, "name_missing": False,
    }


def normalize_address(addr: str, country: str = "") -> dict:
    raw = addr.strip()
    if not raw:
        return {
            "addr_raw": "", "addr_norm": "", "addr_tokens": [],
            "house_number": "", "street_name": "", "unit_suite": "",
            "postal_code": "", "numeric_tokens": [], "addr_missing": True,
        }
    lower   = raw.lower()
    folded  = _accent_fold(lower)
    clean   = _ADDR_PUNCT.sub(" ", folded)
    clean   = _SPACE_RE.sub(" ", clean).strip()
    expanded = clean
    for pat, repl in ADDR_PAIRS:
        expanded = re.sub(pat, repl, expanded)
    expanded = _SPACE_RE.sub(" ", expanded).strip()
    tokens   = expanded.split()

    postal = ""
    c = (country or "").strip().lower()
    if c in ("us", "united states", "usa"):
        m = US_ZIP_RE.search(raw)
        if m: postal = m.group()
    elif c == "india":
        m = INDIA_PIN_RE.search(raw)
        if m: postal = m.group()
    if not postal:
        m = GENERIC_NUM.search(raw)
        if m: postal = m.group()

    house_number = ""
    hm = HOUSE_NUM_RE.match(raw.strip())
    if hm: house_number = hm.group(1)

    unit_suite = ""
    um = UNIT_RE.search(lower)
    if um: unit_suite = um.group(1)

    street_name = ""
    if house_number and len(tokens) > 2:
        remaining = expanded.replace(house_number, "", 1).strip()
        parts = remaining.split(",")
        if parts: street_name = parts[0].strip()

    return {
        "addr_raw": raw, "addr_norm": expanded, "addr_tokens": tokens,
        "house_number": house_number, "street_name": street_name,
        "unit_suite": unit_suite, "postal_code": postal,
        "numeric_tokens": re.findall(r'\b\d+\b', raw),
        "addr_missing": False,
    }


def normalize_country(country: str) -> str:
    aliases = {
        "united states": "US", "usa": "US", "u.s.a.": "US", "u.s.": "US",
        "india": "India", "ind": "India",
        "france": "France", "fr": "France",
    }
    return aliases.get(country.strip().lower(), country.strip())


def country_relation(c1: str, c2: str) -> str:
    n1 = normalize_country(c1)
    n2 = normalize_country(c2)
    if not n1 or not n2:
        return "country_missing"
    return "same_country" if n1 == n2 else "different_country"


def normalize_row(row: dict) -> dict:
    """Normalize a single entity dict (used for individual lookups)."""
    name_views = normalize_name(row.get("business_name", ""))
    addr_views = normalize_address(row.get("business_address", ""), row.get("country", ""))
    country_norm = normalize_country(row.get("country", ""))
    return {**row, **name_views, **addr_views, "country_norm": country_norm}


if __name__ == "__main__":
    import time
    import pandas as pd
    # Benchmark: compare per-row vs vectorized on 10K rows
    sample = pd.DataFrame([{
        "entity_id": f"S1-{i}", "business_name": f"Test Corp {i} Ltd",
        "business_address": f"{i} Main St, City, 10001", "country": "US"
    } for i in range(10_000)])

    t0 = time.time()
    slow = sample.apply(lambda r: normalize_row(r.to_dict()), axis=1)
    print(f"Per-row apply (10K): {time.time()-t0:.2f}s")

    t0 = time.time()
    fast = preprocess_df_fast(sample)
    print(f"Vectorized fast   (10K): {time.time()-t0:.2f}s")
    print(f"Fast cols: {[c for c in fast.columns if c not in sample.columns]}")
