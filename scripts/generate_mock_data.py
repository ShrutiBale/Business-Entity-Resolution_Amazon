"""
Generate mock data for the Business Entity Resolution Demo.
Patch pass aligned with approch.md — implements Fixes 1-8 from modified_prompt.md.
"""

import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "backend" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def css(s2_s3_name_sim, s2_s3_addr_sim, numeric_agreement):
    """Build a CrossSourceSupport object (Fix 2 — approch.md §13)."""
    return {
        "s2_s3_name_similarity": s2_s3_name_sim,
        "s2_s3_address_similarity": s2_s3_addr_sim,
        "numeric_agreement": numeric_agreement,
    }


def C(entity_id, source, name, address, country,
       final_score, selected,
       name_sim, addr_sim, composite_sim, country_rel,
       pin_match, house_match,
       street_name_sim,           # Fix 4: component-level street name similarity
       unit_suite_match,          # Fix 4: null when not applicable
       tfidf_name_rank, tfidf_addr_rank, tfidf_composite_rank,  # Fix 3: all 3 channels
       blocking_passes, blocking_pass_count,
       nf_s1, nf_s2, nf_s3,      # Fix 1: split by source
       numeric_evidence,
       cross_src,                 # Fix 2: CrossSourceSupport object or None
       cluster_id, cluster_size,  # Fix 5: near-duplicate cluster tagging
       notes=""):
    return {
        "entity_id": entity_id,
        "source": source,
        "business_name": name,
        "business_address": address,
        "country": country,
        "final_score": round(final_score, 4),
        "selected": selected,
        "name_similarity": round(name_sim, 4),
        "address_similarity": round(addr_sim, 4),
        "composite_similarity": round(composite_sim, 4),
        "country_relation": country_rel,
        "pin_postal_match": pin_match,
        "house_number_match": house_match,
        "street_name_similarity": round(street_name_sim, 4),
        "unit_suite_match": unit_suite_match,
        "tfidf_name_rank": tfidf_name_rank,
        "tfidf_address_rank": tfidf_addr_rank,
        "tfidf_composite_rank": tfidf_composite_rank,
        "blocking_passes": blocking_passes,
        "blocking_pass_count": blocking_pass_count,
        "name_frequency_s1": nf_s1,
        "name_frequency_s2": nf_s2,
        "name_frequency_s3": nf_s3,
        "numeric_evidence": numeric_evidence,
        "cross_source_support": cross_src,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "notes": notes,
    }


# ──────────────────────────────────────────────
# S1 Entities — Fix 8: ground_truth_source added
# ──────────────────────────────────────────────
s1_entities = [
    # ── train_holdout cases ──
    {
        "entity_id": "S1-00001",
        "business_name": "Sunrise Medical Supplies",
        "business_address": "47 Oak Street, Austin, TX 78701",
        "country": "US",
        "case_tag": "singleton",
        "case_description": "No match in S2/S3. Candidates retrieved but all correctly rejected — address and ZIP conflict.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00002",
        "business_name": "Green Valley Supermarket",
        "business_address": "22 Elm Road, Chicago, IL 60614",
        "country": "US",
        "case_tag": "single_match",
        "case_description": "One clear, high-confidence match found in S2. Near-exact name and address.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00003",
        "business_name": "Patel Brothers Grocery",
        "business_address": "58 Linking Road, Bandra West, Mumbai, 400050",
        "country": "India",
        "case_tag": "single_match",
        "case_description": "Noisy match: 'Bros' abbreviation in S2, abbreviated address in S3. Both correctly matched.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00004",
        "business_name": "TechZone Electronics",
        "business_address": "101 Silicon Avenue, Bangalore, 560100",
        "country": "India",
        "case_tag": "multi_match",
        "case_description": "Genuine multi-match: one S2 and two S3 records all refer to the same entity. A different branch in S2 correctly rejected.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00005",
        "business_name": "Cafe Coffee Day",
        "business_address": "12 MG Road, Bangalore, 560001",
        "country": "India",
        "case_tag": "chain_collision",
        "case_description": "Chain collision: same name in S2/S3 for different branches. Only the MG Road branch (PIN 560001) matches.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00006",
        "business_name": "Liberty Pharmaceuticals Inc",
        "business_address": "900 West 34th Street, Suite 4B, New York, NY 10001",
        "country": "US",
        "case_tag": "address_variation",
        "case_description": "Address variation: S2 uses abbreviations (W 34th St). Unit/suite mismatch between records — shows why component-level parsing matters vs. whole-string similarity.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00007",
        "business_name": "Sharma Traders",
        "business_address": "Plot 14, Sector 18, Noida, UP 201301",
        "country": "India",
        "case_tag": "transliteration",
        "case_description": "Transliteration: S2 uses 'Sarma' (phonetic variant of Sharma). Phonetic blocking recovers this.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00010",
        "business_name": "Star Hotel",
        "business_address": "Near Railway Station, Jaipur, Rajasthan, 302001",
        "country": "India",
        "case_tag": "singleton",
        "case_description": "Extremely common name. Many candidates retrieved but all rejected — name_frequency_s1/s2/s3 all very high and address/numeric evidence conflicts.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00011",
        "business_name": "Johnson & Sons Hardware",
        "business_address": "55 Commerce Blvd, Dallas, TX 75201",
        "country": "US",
        "case_tag": "single_match",
        "case_description": "DBA variation: '&' vs 'and'. Normalization handles this before blocking.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00012",
        "business_name": "Nexus Innovations Pvt Ltd",
        "business_address": "Floor 3, Prestige Tower, Whitefield, Bangalore, 560066",
        "country": "India",
        "case_tag": "single_match",
        "case_description": "Legal suffix: 'Pvt Ltd' ↔ 'Private Limited'. Normalization handles both forms.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00013",
        "business_name": "Fresh Farms Organic Market",
        "business_address": "212 Farmers Road, Portland, OR 97201",
        "country": "US",
        "case_tag": "single_match",
        "case_description": "Word-order transposition: S2 says 'Organic Market Fresh Farms'. Sorted-token blocking recovers it.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00014",
        "business_name": "Reliance Digital",
        "business_address": "Phoenix Mall, Lower Parel, Mumbai, 400013",
        "country": "India",
        "case_tag": "multi_match",
        "case_description": "Multi-match at a major mall location. Both S2 and S3 correctly matched.",
        "ground_truth_source": "train_holdout",
    },
    {
        "entity_id": "S1-00015",
        "business_name": "Global Finance Corp",
        "business_address": "1 Financial Square, San Francisco, CA 94105",
        "country": "US",
        "case_tag": "singleton",
        "case_description": "Generic name. Candidates have exact name match but ZIP codes conflict (94104 vs 94105). Correctly predicted as no-match.",
        "ground_truth_source": "train_holdout",
    },
    # ── test_unverified cases (France — no ground truth exists for anyone) ──
    {
        "entity_id": "S1-00008",
        "business_name": "Boulangerie Dupont",
        "business_address": "15 Rue de Rivoli, Paris, 75001",
        "country": "France",
        "case_tag": "france",
        "case_description": "Unseen domain (France). Pipeline trained on US/India only. Open-set normalization handles French address forms. Prediction only — no ground truth exists.",
        "ground_truth_source": "test_unverified",
    },
    {
        "entity_id": "S1-00009",
        "business_name": "Maison de la Presse",
        "business_address": "8 Boulevard Haussmann, Paris, 75009",
        "country": "France",
        "case_tag": "france",
        "case_description": "Unseen domain (France) multi-source. Both S2 and S3 records — pipeline prediction only, no ground truth available.",
        "ground_truth_source": "test_unverified",
    },
]

# ──────────────────────────────────────────────
# Cross-source support objects used in candidates
# ──────────────────────────────────────────────
CSS_PATEL = css(0.85, 0.83, True)
CSS_TECHZONE = css(0.91, 0.88, True)
CSS_CCD = css(0.98, 0.22, False)  # same name, address conflict
CSS_FRANCE1 = css(0.97, 0.92, True)
CSS_FRANCE2 = css(0.94, 0.89, True)
CSS_RELIANCE = css(0.93, 0.86, True)

# ──────────────────────────────────────────────
# Candidates — all 8 fixes applied
# ──────────────────────────────────────────────
candidates = {

    # ── S1-00001: Singleton ──────────────────────────────────────
    "S1-00001": [
        C("S2-10012","S2","Sunrise Medical Equipment","47 Pine Street, Austin, TX 78702","US",
          0.38,False, 0.71,0.31,0.48,"same",
          False,False, 0.42,None,
          3,None,None, ["exact_name","token_index"],2,
          0.12,0.11,0.13, 0.04, None,
          None,1,
          "Similar name but different street (Pine vs Oak) and ZIP (78702 vs 78701). Score below threshold 0.60."),
        C("S3-10045","S3","Sunrise Healthcare Supplies","200 E 5th Street, Austin, TX 78701","US",
          0.42,False, 0.65,0.28,0.44,"same",
          False,False, 0.51,None,
          2,5,4, ["token_index"],1,
          0.15,0.13,0.14, 0.06, None,
          None,1,
          "Name differs on 'Healthcare vs Medical'. City/ZIP shared but street diverges. Rejected."),
    ],

    # ── S1-00002: Clean single match ────────────────────────────
    "S1-00002": [
        C("S2-20001","S2","Green Valley Supermarket","22 Elm Road, Chicago, Illinois 60614","US",
          0.97,True, 0.99,0.94,0.97,"same",
          True,True, 0.97,None,
          1,1,1, ["exact_key","token_index","tfidf_name"],3,
          0.04,0.04,None, 0.20, None,
          "CLU-S2-A",2,   # near-duplicate cluster: S2-20001 + S2-20088 are near-identical S2 records
          "Near-exact name. Full address match. ZIP 60614 exact. Part of a near-duplicate cluster of 2 records in S2."),
        C("S3-20088","S3","Green Valley Super Market","22 Elm Rd, Chicago, IL 60614","US",
          0.58,False, 0.88,0.65,0.74,"same",
          True,True, 0.91,None,
          2,3,2, ["token_index","tfidf_name"],2,
          0.04,None,0.04, 0.18, None,
          None,1,
          "Extra space in name. Address abbreviated (Rd). Score 0.58 just below threshold 0.60 — borderline."),
    ],

    # ── S1-00003: Noisy match ────────────────────────────────────
    "S1-00003": [
        C("S2-30011","S2","Patel Bros Grocery","58 Linking Rd, Bandra, Mumbai, 400050","India",
          0.84,True, 0.87,0.78,0.83,"same",
          True,True, 0.81,None,
          1,2,2, ["phonetic","token_index","tfidf_name"],3,
          0.06,0.06,None, 0.22, CSS_PATEL,
          None,1,
          "Abbreviation 'Bros' for 'Brothers'. 'Rd' for 'Road'. 'Bandra' vs 'Bandra West'. PIN 400050 matches."),
        C("S3-30042","S3","Patel Brother's Groceries","58, Linking Road, Bandra West, Mumbai 400050","India",
          0.79,True, 0.83,0.82,0.81,"same",
          True,True, 0.88,None,
          2,3,3, ["token_index","tfidf_composite"],3,
          0.06,None,0.06, 0.22, CSS_PATEL,
          None,1,
          "Apostrophe variation. 'Groceries' vs 'Grocery'. Full address. PIN 400050 exact match."),
        C("S2-30099","S2","Patel Mart","Near SBI ATM, Bandra, Mumbai","India",
          0.41,False, 0.55,0.25,0.38,"same",
          False,False, 0.29,None,
          1,None,None, ["token_index"],1,
          0.09,0.08,None, 0.05, None,
          None,1,
          "Landmark-based address, no PIN. Name diverges significantly ('Mart' vs 'Brothers Grocery'). Rejected."),
    ],

    # ── S1-00004: Multi-match ────────────────────────────────────
    "S1-00004": [
        C("S2-40007","S2","TechZone Electronics","101 Silicon Ave, Bangalore 560100","India",
          0.95,True, 0.99,0.92,0.96,"same",
          True,True, 0.95,None,
          1,1,1, ["exact_key","tfidf_name"],2,
          0.03,0.03,None, 0.21, CSS_TECHZONE,
          None,1,
          "Exact name. 'Ave' abbreviation. PIN 560100 matches. Retrieved by exact-key pass."),
        C("S3-40031","S3","Tech Zone Electronics Pvt Ltd","101 Silicon Avenue, Bengaluru, 560100","India",
          0.89,True, 0.90,0.87,0.88,"same",
          True,True, 0.90,None,
          1,2,2, ["token_index","tfidf_composite"],2,
          0.03,None,0.03, 0.21, CSS_TECHZONE,
          "CLU-S3-B",2,   # near-duplicate cluster within S3
          "Legal suffix added. Bangalore/Bengaluru transliteration. PIN 560100 exact. Part of near-duplicate cluster (2 records in S3)."),
        C("S3-40089","S3","TechZone Electr.","101, Silicon Avenue, Bangalore, KA 560100","India",
          0.82,True, 0.85,0.88,0.86,"same",
          True,True, 0.87,None,
          2,3,3, ["token_index","tfidf_name","tfidf_addr"],3,
          0.03,None,0.03, 0.21, CSS_TECHZONE,
          "CLU-S3-B",2,   # same near-duplicate cluster
          "Abbreviated name. Full address with state. House 101 and PIN 560100 agree. Same near-duplicate cluster as S3-40031."),
        C("S2-40120","S2","TechZone Electronics","45 Outer Ring Road, Bangalore 560037","India",
          0.48,False, 0.99,0.22,0.58,"same",
          False,False, 0.21,None,
          1,1,1, ["exact_key"],1,
          0.03,0.03,None, 0.05, None,
          None,1,
          "CHAIN COLLISION: Exact name match but completely different address (Outer Ring Road) and PIN (560037 vs 560100). Correctly rejected."),
    ],

    # ── S1-00005: Chain collision ────────────────────────────────
    "S1-00005": [
        C("S2-50003","S2","Cafe Coffee Day","12 MG Road, Bangalore, 560001","India",
          0.94,True, 1.0,0.95,0.97,"same",
          True,True, 0.96,None,
          1,1,1, ["exact_key","tfidf_name"],2,
          0.01,0.01,None, 0.22, CSS_CCD,
          None,1,
          "Exact name and exact address. PIN 560001 matches. Correct match."),
        C("S3-50017","S3","Cafe Coffee Day","Brigade Road, Bangalore, 560025","India",
          0.41,False, 1.0,0.19,0.55,"same",
          False,True, 0.15,None,
          1,1,1, ["exact_key","tfidf_name"],2,
          0.01,None,0.01, 0.05, CSS_CCD,
          None,1,
          "CHAIN COLLISION: Same name (sim=1.0) but different branch — Brigade Road, PIN 560025 vs MG Road PIN 560001. Address/numeric conflict forces rejection."),
        C("S2-50088","S2","Cafe Coffee Day","Orion Mall, Rajajinagar, Bangalore, 560010","India",
          0.39,False, 1.0,0.15,0.52,"same",
          False,False, 0.12,None,
          1,1,1, ["exact_key"],1,
          0.01,0.01,None, 0.04, None,
          None,1,
          "CHAIN COLLISION: Different mall location, PIN 560010. Correctly rejected despite perfect name similarity."),
        C("S3-50099","S3","CCD - Cafe Coffee Day","MG Road, Bangalore","India",
          0.73,False, 0.78,0.61,0.71,"same",
          False,True, 0.62,None,
          2,2,2, ["token_index","tfidf_composite"],2,
          0.01,None,0.01, 0.12, None,
          None,1,
          "DBA abbreviation 'CCD'. No PIN available. Score 0.73 above threshold but selected=False due to no-PIN penalty in features."),
    ],

    # ── S1-00006: Address variation — showcases unit_suite_match ─
    "S1-00006": [
        C("S2-60041","S2","Liberty Pharmaceuticals Inc","900 W 34th St, New York, NY 10001","US",
          0.91,True, 0.99,0.87,0.93,"same",
          True,True, 0.95,None,  # no suite info in S2 → unit_suite_match=None
          1,2,1, ["exact_key","tfidf_name"],2,
          0.05,0.05,None, 0.20, None,
          None,1,
          "W→West, St→Street abbreviations normalized. House 900 exact. ZIP 10001 exact. No suite info in S2 (unit_suite_match=null — not absent, just not applicable)."),
        C("S3-60011","S3","Liberty Pharma Inc Suite 4B","900 W 34th Street, Suite 4C, New York, NY 10001","US",
          0.71,False, 0.88,0.74,0.81,"same",
          True,True, 0.94,False,  # unit_suite_match=False: 4B vs 4C
          2,3,2, ["token_index","tfidf_composite"],2,
          0.05,None,0.05, 0.17, None,
          None,1,
          "ADDRESS VARIATION SHOWCASE: street_name_similarity=0.94 is high. But unit_suite_match=False (Suite 4B vs 4C). Whole-address similarity (0.74) diluted by unit mismatch — component-level parsing reveals the discrepancy. Score below threshold."),
    ],

    # ── S1-00007: Transliteration ────────────────────────────────
    "S1-00007": [
        C("S2-70055","S2","Sarma Traders","Plot 14, Sector 18, Noida, UP 201301","India",
          0.78,True, 0.82,0.94,0.87,"same",
          True,True, 0.93,None,
          2,3,3, ["phonetic","tfidf_composite"],3,
          0.08,0.08,None, 0.19, None,
          None,1,
          "Transliteration: Sharma→Sarma (phonetic variant). Phonetic blocking fired. Address/PIN 201301 match strongly."),
        C("S3-70011","S3","Sharma Traders","Plot-14, Sector-18, Noida, Uttar Pradesh 201301","India",
          0.88,True, 0.99,0.85,0.92,"same",
          True,True, 0.88,None,
          1,1,1, ["exact_key","token_index"],1,
          0.08,None,0.08, 0.21, None,
          None,1,
          "Exact name. Address uses hyphens and full state name. Plot 14 and PIN 201301 agree."),
    ],

    # ── S1-00008: France (test_unverified) ──────────────────────
    "S1-00008": [
        C("S2-80001","S2","Boulangerie Dupont","15 Rue de Rivoli, Paris, 75001","France",
          0.92,True, 1.0,0.93,0.96,"same",
          True,True, 0.96,None,
          1,1,1, ["exact_key","tfidf_name"],2,
          0.04,0.04,None, 0.20, CSS_FRANCE1,
          None,1,
          "UNSEEN DOMAIN (France) — prediction only, no ground truth. Exact name. French 'rue' handled by open-set normalization. Postal 75001 matches."),
        C("S3-80044","S3","Boulangerie Dupont SARL","15, Rue de Rivoli, Paris 75001","France",
          0.87,True, 0.92,0.90,0.91,"same",
          True,True, 0.93,None,
          1,2,2, ["token_index","tfidf_composite"],2,
          0.04,None,0.04, 0.19, CSS_FRANCE1,
          None,1,
          "UNSEEN DOMAIN (France) — prediction only. French legal suffix SARL normalized. Postal 75001 matches."),
        C("S2-80099","S2","Boulangerie Martin","15 Rue de Rivoli, Paris, 75001","France",
          0.52,False, 0.31,0.91,0.58,"same",
          True,True, 0.94,None,
          1,4,3, ["tfidf_addr"],1,
          0.06,0.05,None, 0.18, None,
          None,1,
          "Different bakery at same address. Address similarity 0.91 and street_name_similarity 0.94 very high, but name diverges (0.31). Correctly rejected."),
    ],

    # ── S1-00009: France multi-match (test_unverified) ───────────
    "S1-00009": [
        C("S2-90011","S2","Maison de la Presse","8 Bd Haussmann, Paris, 75009","France",
          0.89,True, 0.95,0.85,0.90,"same",
          True,True, 0.87,None,
          1,2,2, ["token_index","tfidf_composite"],2,
          0.05,0.05,None, 0.18, CSS_FRANCE2,
          None,1,
          "UNSEEN DOMAIN (France) — prediction only. 'Bd'=Boulevard abbreviation normalized. Postal 75009 matches."),
        C("S3-90033","S3","Maison de la Presse","8 Boulevard Haussmann, Paris 75009","France",
          0.93,True, 0.99,0.92,0.96,"same",
          True,True, 0.94,None,
          1,1,1, ["exact_key","tfidf_name"],1,
          0.05,None,0.05, 0.21, CSS_FRANCE2,
          None,1,
          "UNSEEN DOMAIN (France) — prediction only. Exact name, full boulevard name. Strong match."),
    ],

    # ── S1-00010: Singleton — generic name ──────────────────────
    "S1-00010": [
        C("S2-10031","S2","Star Hotel","Near Railway Station, Ajmer, Rajasthan","India",
          0.44,False, 1.0,0.15,0.52,"same",
          False,False, 0.12,None,
          1,1,1, ["exact_key"],1,
          0.0002,0.0003,None, 0.03, None,
          "CLU-S2-C",3,   # very common name → large cluster within S2
          "GENERIC NAME: name_frequency_s1=0.0002, s2=0.0003 (appears 500+ times). Different city: Ajmer vs Jaipur. Part of a near-duplicate cluster of 3 records in S2. Rejected."),
        C("S3-10087","S3","Star Hotel","Station Road, Jaipur, RJ 302001","India",
          0.47,False, 1.0,0.38,0.65,"same",
          True,False, 0.41,None,
          1,1,1, ["exact_key"],1,
          0.0002,None,0.0002, 0.06, None,
          None,1,
          "Same city/PIN but landmark-only address. House number absent on both sides. Generic name_frequency demands stronger evidence."),
        C("S2-10142","S2","The Star Hotel","12 Station Road, Jaipur, 302001","India",
          0.51,False, 0.88,0.44,0.68,"same",
          True,False, 0.43,None,
          2,2,2, ["token_index","phonetic"],2,
          0.0002,0.0003,None, 0.08, None,
          "CLU-S2-C",3,   # same cluster
          "Prefix 'The'. House 12 in S2 but absent in S1. Generic name forces higher address bar — not met. Same near-duplicate cluster."),
    ],

    # ── S1-00011: DBA & vs and ────────────────────────────────────
    "S1-00011": [
        C("S2-11001","S2","Johnson and Sons Hardware","55 Commerce Blvd, Dallas, TX 75201","US",
          0.90,True, 0.96,0.92,0.94,"same",
          True,True, 0.94,None,
          1,2,1, ["exact_key","tfidf_name"],2,
          0.05,0.05,None, 0.20, None,
          None,1,
          "& → 'and' normalization at preprocessing stage. Exact address. Strong match."),
    ],

    # ── S1-00012: Legal suffix ───────────────────────────────────
    "S1-00012": [
        C("S3-12001","S3","Nexus Innovations Private Limited","Floor 3, Prestige Tower, Whitefield, Bangalore, 560066","India",
          0.92,True, 0.91,0.95,0.93,"same",
          True,True, 0.93,None,
          1,2,2, ["token_index","tfidf_composite"],2,
          0.04,None,0.04, 0.21, None,
          None,1,
          "Legal suffix: 'Pvt Ltd'→'Private Limited'. Full address including floor and PIN 560066 match."),
    ],

    # ── S1-00013: Word-order ─────────────────────────────────────
    "S1-00013": [
        C("S2-13001","S2","Organic Market Fresh Farms","212 Farmers Road, Portland, Oregon 97201","US",
          0.86,True, 0.88,0.90,0.89,"same",
          True,True, 0.91,None,
          2,3,3, ["sorted_token","tfidf_composite"],3,
          0.04,0.04,None, 0.20, None,
          None,1,
          "Word-order transposition. Sorted-token blocking fired (pass E). ZIP 97201 and house number 212 agree."),
    ],

    # ── S1-00014: Reliance Digital multi-match ──────────────────
    "S1-00014": [
        C("S2-14001","S2","Reliance Digital","Phoenix Mall, Lower Parel, Mumbai 400013","India",
          0.94,True, 1.0,0.93,0.96,"same",
          True,True, 0.94,None,
          1,1,1, ["exact_key","tfidf_name"],2,
          0.008,0.008,None, 0.22, CSS_RELIANCE,
          None,1,
          "Exact name and location. PIN 400013 matches."),
        C("S3-14022","S3","Reliance Digital Store","Phoenix Mills, Lower Parel, Mumbai, 400013","India",
          0.88,True, 0.91,0.88,0.90,"same",
          True,True, 0.90,None,
          1,2,2, ["token_index","tfidf_composite"],2,
          0.008,None,0.008, 0.20, CSS_RELIANCE,
          None,1,
          "Suffix 'Store' added. 'Phoenix Mills' vs 'Phoenix Mall'. Same PIN 400013."),
    ],

    # ── S1-00015: Singleton — ZIP conflict ──────────────────────
    "S1-00015": [
        C("S2-15011","S2","Global Finance Corp","1 Financial Square, San Francisco, CA 94104","US",
          0.45,False, 0.99,0.78,0.86,"same",
          False,True, 0.88,None,
          1,1,1, ["exact_key","tfidf_name"],2,
          0.01,0.01,None, 0.10, None,
          None,1,
          "Exact name but ZIP 94104 vs 94105. House 1 matches. street_name_similarity=0.88. ZIP conflict keeps score below threshold."),
        C("S3-15033","S3","Global Finance Corporation","Financial Square, SF, CA 94105","US",
          0.52,False, 0.88,0.58,0.72,"same",
          True,False, 0.81,None,
          2,2,2, ["token_index","tfidf_name"],2,
          0.01,None,0.01, 0.12, None,
          None,1,
          "Legal suffix 'Corporation'. Address lacks house number. Score insufficient despite correct ZIP."),
    ],
}

# ──────────────────────────────────────────────
# Decisions (unchanged from original)
# ──────────────────────────────────────────────
THRESHOLD = 0.60
decisions = {}
for entity in s1_entities:
    eid = entity["entity_id"]
    cands = candidates.get(eid, [])
    selected = [c for c in cands if c["selected"]]
    scores = sorted([c["final_score"] for c in cands], reverse=True) if cands else []
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    has_any_prob = min(top1 + 0.05, 1.0) if selected else max(top1 - 0.05, 0.0)
    decisions[eid] = {
        "entity_id": eid,
        "has_any_match_probability": round(has_any_prob, 4),
        "global_threshold": THRESHOLD,
        "top1_score": top1,
        "top2_score": top2,
        "top1_top2_margin": round(top1 - top2, 4),
        "n_candidates": len(cands),
        "n_above_threshold": len([c for c in cands if c["final_score"] >= THRESHOLD]),
        "predicted_matches": [c["entity_id"] for c in selected],
        "decision_notes": (
            f"Accepted {len(selected)} match(es). Top score: {top1:.3f}."
            if selected else
            f"No match predicted. Top candidate score {top1:.3f} is below threshold {THRESHOLD}."
        ),
    }

# ──────────────────────────────────────────────
# Metrics (unchanged)
# ──────────────────────────────────────────────
metrics = {
    "is_mock_data": True,
    "macro_f05": 0.847,
    "singleton_f05": 0.913,
    "non_singleton_f05": 0.798,
    "blocking_recall": 0.963,
    "pair_precision": 0.871,
    "pair_recall": 0.812,
    "mean_candidates_per_s1": 4.2,
    "p95_candidates_per_s1": 18.0,
    "max_candidates": 47,
    "prediction_distribution": {"zero_matches": 0.31, "one_match": 0.48, "two_plus_matches": 0.21},
    "error_taxonomy": {
        "BLOCKING_MISS": 4, "NAME_VARIATION": 7, "ADDRESS_VARIATION": 5,
        "TRANSLITERATION": 3, "ABBREVIATION": 6, "DBA_VARIATION": 2,
        "WORD_ORDER": 2, "TYPO": 4, "NUMERIC_MISMATCH": 3,
        "GENERIC_NAME_COLLISION": 5, "CHAIN_FRANCHISE_COLLISION": 4,
        "FALSE_POSITIVE": 3, "THRESHOLD_ERROR": 2, "SINGLETON_ERROR": 2,
        "MULTI_MATCH_ERROR": 1, "COUNTRY_SHIFT": 0, "DATA_ANOMALY": 1,
    },
    "validation_note": "DEMO DATA — NOT REAL VALIDATION. Replace data files with real pipeline exports for live metrics.",
}

# ──────────────────────────────────────────────
# Experiment Runs (unchanged)
# ──────────────────────────────────────────────
experiment_runs = [
    {
        "run_id": "RUN-001",
        "description": "Baseline: exact name match only",
        "blocking_config": "Exact normalized name key",
        "tfidf_k": None,
        "blocking_recall": 0.51,
        "mean_candidates": 1.4,
        "p95_candidates": 6,
        "max_candidates": 14,
        "threshold": 0.50,
        "macro_f05": 0.52,
        "precision": 0.89,
        "recall": 0.41,
        "singleton_f05": 0.91,
        "non_singleton_f05": 0.38,
        "notes": "Low recall. Blocking is the bottleneck. Must add TF-IDF retrieval.",
    },
    {
        "run_id": "RUN-002",
        "description": "Rule blocking: multi-pass (A+B+C+D+E)",
        "blocking_config": "Exact key + token index + address token + phonetic + sorted-token",
        "tfidf_k": None,
        "blocking_recall": 0.78,
        "mean_candidates": 8.1,
        "p95_candidates": 31,
        "max_candidates": 89,
        "threshold": 0.55,
        "macro_f05": 0.68,
        "precision": 0.82,
        "recall": 0.61,
        "singleton_f05": 0.88,
        "non_singleton_f05": 0.60,
        "notes": "Significant recall improvement. Chain collisions now visible in FP analysis.",
    },
    {
        "run_id": "RUN-003",
        "description": "Rule blocking + Char TF-IDF K=20",
        "blocking_config": "Multi-pass + char 3-5gram TF-IDF name/address/composite K=20",
        "tfidf_k": 20,
        "blocking_recall": 0.91,
        "mean_candidates": 14.3,
        "p95_candidates": 48,
        "max_candidates": 120,
        "threshold": 0.58,
        "macro_f05": 0.78,
        "precision": 0.84,
        "recall": 0.73,
        "singleton_f05": 0.90,
        "non_singleton_f05": 0.72,
        "notes": "Core TF-IDF retrieval added. Major recall jump. Generic-name FPs increase.",
    },
    {
        "run_id": "RUN-004",
        "description": "Rule + TF-IDF K=50 + full features + LightGBM",
        "blocking_config": "Multi-pass + char TF-IDF K=50 + LightGBM pairwise classifier",
        "tfidf_k": 50,
        "blocking_recall": 0.963,
        "mean_candidates": 4.2,
        "p95_candidates": 18,
        "max_candidates": 47,
        "threshold": 0.60,
        "macro_f05": 0.847,
        "precision": 0.871,
        "recall": 0.812,
        "singleton_f05": 0.913,
        "non_singleton_f05": 0.798,
        "notes": "LightGBM with full feature set. name_freq_s1/s2/s3 + numeric evidence reduces chain FPs. Current best.",
    },
    {
        "run_id": "RUN-005",
        "description": "RUN-004 + record-side competition features",
        "blocking_config": "Same as RUN-004 + record-side competition signal",
        "tfidf_k": 50,
        "blocking_recall": 0.963,
        "mean_candidates": 4.2,
        "p95_candidates": 18,
        "max_candidates": 47,
        "threshold": 0.61,
        "macro_f05": 0.851,
        "precision": 0.876,
        "recall": 0.813,
        "singleton_f05": 0.916,
        "non_singleton_f05": 0.802,
        "notes": "Marginal +0.004 F0.5. Within fold noise. Retained: reuse audit supports low S2/S3 reuse.",
    },
]

# ──────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────
(OUT / "s1_entities.json").write_text(json.dumps(s1_entities, indent=2))
print(f"✓ s1_entities.json ({len(s1_entities)} entities)")

(OUT / "candidates.json").write_text(json.dumps(candidates, indent=2))
print(f"✓ candidates.json ({sum(len(v) for v in candidates.values())} candidates)")

(OUT / "decisions.json").write_text(json.dumps(decisions, indent=2))
print(f"✓ decisions.json ({len(decisions)} decisions)")

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("✓ metrics.json")

(OUT / "experiment_runs.json").write_text(json.dumps(experiment_runs, indent=2))
print(f"✓ experiment_runs.json ({len(experiment_runs)} runs)")

print("\n✅ All mock data regenerated (aligned with approch.md Fixes 1–8)")
