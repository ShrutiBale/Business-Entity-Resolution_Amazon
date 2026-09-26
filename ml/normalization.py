"""
ml/normalization.py — Safe, non-destructive normalization.
Creates multiple views; never replaces the original.
Aligned with approch.md §4.
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

# ─── Legal suffix normalization vocabulary ─────────────────────────
LEGAL_MAP = {
    r"\bcorp\b": "corporation",
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bllc\b": "llc",
    r"\binc\b": "incorporated",
    r"\b&\b": "and",
    # French legal forms
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\beurl\b": "eurl",
    r"\bsa\b": "sa",
    r"\bsci\b": "sci",
}

# ─── Address abbreviation vocabulary ──────────────────────────────
ADDR_MAP = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bblvd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    # French
    r"\bbd\b": "boulevard",
    r"\brue\b": "rue",
}

# ─── Postal patterns ───────────────────────────────────────────────
US_ZIP_RE    = re.compile(r'\b\d{5}(?:-\d{4})?\b')
INDIA_PIN_RE = re.compile(r'\b[1-9]\d{5}\b')
GENERIC_NUM  = re.compile(r'\b\d{4,6}\b')
HOUSE_NUM_RE = re.compile(r'^\s*(\d+[a-zA-Z]?)\b')


def _unicode_norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _accent_fold(s: str) -> str:
    if HAS_UNIDECODE:
        return _unidecode(s)
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()


def normalize_name(name: str) -> dict:
    """Return multiple non-destructive normalized views of a business name."""
    raw = name.strip()
    if not raw:
        return {
            "name_raw": "",
            "name_lower": "",
            "name_norm": "",
            "name_legal": "",
            "name_phonetic": "",
            "name_sorted_tokens": "",
            "name_tokens": [],
            "name_missing": True,
        }

    lower = raw.lower()
    unidecoded = _accent_fold(lower)
    # remove punctuation except hyphens and spaces
    clean = re.sub(r"[^\w\s-]", " ", unidecoded)
    clean = re.sub(r"\s+", " ", clean).strip()

    # legal suffix normalization
    legal = clean
    for pat, repl in LEGAL_MAP.items():
        legal = re.sub(pat, repl, legal)
    legal = re.sub(r"\s+", " ", legal).strip()

    tokens = clean.split()
    sorted_tokens = " ".join(sorted(tokens))

    phonetic = ""
    if HAS_JELLYFISH and clean:
        try:
            phonetic = jellyfish.metaphone(clean[:80])
        except Exception:
            phonetic = ""

    return {
        "name_raw": raw,
        "name_lower": lower,
        "name_norm": clean,
        "name_legal": legal,
        "name_phonetic": phonetic,
        "name_sorted_tokens": sorted_tokens,
        "name_tokens": tokens,
        "name_missing": False,
    }


def normalize_address(addr: str, country: str = "") -> dict:
    """Return parsed and normalized address components."""
    raw = addr.strip()
    if not raw:
        return {
            "addr_raw": "",
            "addr_norm": "",
            "addr_tokens": [],
            "house_number": "",
            "street_name": "",
            "unit_suite": "",
            "postal_code": "",
            "numeric_tokens": [],
            "addr_missing": True,
        }

    lower = raw.lower()
    unidecoded = _accent_fold(lower)
    clean = re.sub(r"[^\w\s,.-]", " ", unidecoded)
    clean = re.sub(r"\s+", " ", clean).strip()

    # abbreviation expansion
    expanded = clean
    for pat, repl in ADDR_MAP.items():
        expanded = re.sub(pat, repl, expanded)
    expanded = re.sub(r"\s+", " ", expanded).strip()

    tokens = expanded.split()

    # postal extraction
    postal = ""
    c = (country or "").strip().lower()
    if c in ("us", "united states"):
        m = US_ZIP_RE.search(raw)
        if m:
            postal = m.group()
    elif c == "india":
        m = INDIA_PIN_RE.search(raw)
        if m:
            postal = m.group()
    else:
        m = GENERIC_NUM.search(raw)
        if m:
            postal = m.group()
    if not postal:
        m = GENERIC_NUM.search(raw)
        if m:
            postal = m.group()

    # house number (leading digits)
    house_number = ""
    hm = HOUSE_NUM_RE.match(raw.strip())
    if hm:
        house_number = hm.group(1)

    # unit/suite
    unit_suite = ""
    um = re.search(r'\b(?:suite|ste|unit|apt|#)\s*([a-z0-9]+)', lower)
    if um:
        unit_suite = um.group(1)

    # street name (heuristic: tokens between house number and city-like token)
    street_name = ""
    if house_number and len(tokens) > 2:
        remaining = expanded.replace(house_number, "", 1).strip()
        parts = remaining.split(",")
        if parts:
            street_name = parts[0].strip()

    numeric_tokens = re.findall(r'\b\d+\b', raw)

    return {
        "addr_raw": raw,
        "addr_norm": expanded,
        "addr_tokens": tokens,
        "house_number": house_number,
        "street_name": street_name,
        "unit_suite": unit_suite,
        "postal_code": postal,
        "numeric_tokens": numeric_tokens,
        "addr_missing": False,
    }


def normalize_country(country: str) -> str:
    """Return a normalized country label (open-set — never hard-filter on this)."""
    aliases = {
        "united states": "US", "usa": "US", "u.s.a.": "US", "u.s.": "US",
        "india": "India", "ind": "India",
        "france": "France", "fr": "France",
    }
    c = country.strip().lower()
    return aliases.get(c, country.strip())


def country_relation(c1: str, c2: str) -> str:
    """Relational country evidence — never used as a hard filter."""
    n1 = normalize_country(c1)
    n2 = normalize_country(c2)
    if not n1 or not n2:
        return "country_missing"
    if n1 == n2:
        return "same_country"
    return "different_country"


def normalize_row(row: dict) -> dict:
    """Normalize a single entity dict. Returns original + all views."""
    name_views = normalize_name(row.get("business_name", ""))
    addr_views = normalize_address(row.get("business_address", ""), row.get("country", ""))
    country_norm = normalize_country(row.get("country", ""))
    return {**row, **name_views, **addr_views, "country_norm": country_norm}


if __name__ == "__main__":
    sample = {"entity_id": "S1-TEST", "business_name": "Cafe Coffee Day Pvt Ltd",
               "business_address": "12 MG Rd, Bangalore, 560001", "country": "India"}
    result = normalize_row(sample)
    import json
    print(json.dumps({k: v for k, v in result.items() if k not in ("name_tokens","addr_tokens","numeric_tokens")}, indent=2, ensure_ascii=False))
