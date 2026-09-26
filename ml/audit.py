"""
ml/audit.py — Phase 1 data audit.
Produces all 5 required audit outputs (approch.md §3).
Run BEFORE building the full matcher.
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd

from data_loader import load_all_train, load_all_test

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _null_rates(df: pd.DataFrame) -> dict:
    return {c: int((df[c] == "").sum()) for c in df.columns}


def _country_dist(df: pd.DataFrame) -> dict:
    if "country" not in df.columns:
        return {}
    return df["country"].value_counts().to_dict()


def _name_len_stats(df: pd.DataFrame) -> dict:
    if "business_name" not in df.columns:
        return {}
    lens = df["business_name"].str.len()
    return {"mean": round(float(lens.mean()), 1), "max": int(lens.max()), "p95": int(lens.quantile(0.95))}


def audit_structural(s1, s2, s3, gt):
    print("\n=== STRUCTURAL AUDIT ===")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"\n--- {name} ---")
        print(f"  rows: {len(df):,}")
        print(f"  columns: {list(df.columns)}")
        nulls = _null_rates(df)
        print(f"  missing (empty str): {nulls}")
        print(f"  country dist: {_country_dist(df)}")
        print(f"  name length: {_name_len_stats(df)}")
        dupe_ids = df["entity_id"].duplicated().sum()
        print(f"  duplicate entity_ids: {dupe_ids}")
        exact_dupes = df.duplicated().sum()
        print(f"  exact duplicate rows: {exact_dupes}")
    print(f"\n--- Ground Truth ---")
    print(f"  rows: {len(gt):,}")
    print(f"  columns: {list(gt.columns)}")


def audit_match_cardinality(s1, gt):
    print("\n=== MATCH CARDINALITY ===")
    # Parse matched_entity_ids into sets
    gt = gt.copy()
    gt["match_list"] = gt["matched_entity_ids"].apply(
        lambda x: [i.strip() for i in x.split(",") if i.strip()] if x else []
    )
    gt["match_count"] = gt["match_list"].apply(len)

    matched_s1_ids = set(gt["source1_entity_id"])
    singleton_count = len(s1) - len(matched_s1_ids)

    dist = Counter(gt["match_count"].tolist())
    dist[0] = singleton_count

    print(f"  0 matches (singleton): {dist[0]:,}")
    print(f"  1 match:  {dist.get(1, 0):,}")
    print(f"  2+ matches: {sum(v for k,v in dist.items() if k >= 2):,}")

    # By country — merge S1 country
    merged = gt.merge(s1[["entity_id","country"]].rename(columns={"entity_id":"source1_entity_id"}), on="source1_entity_id", how="left")
    by_country = merged.groupby("country")["match_count"].value_counts().unstack(fill_value=0)
    print(f"\n  Match count by country:\n{by_country}")

    return gt


def audit_record_reuse(gt):
    print("\n=== S2/S3 RECORD REUSE ===")
    all_ids = []
    for ids_str in gt["matched_entity_ids"]:
        for eid in ids_str.split(","):
            eid = eid.strip()
            if eid:
                all_ids.append(eid)
    id_counts = Counter(all_ids)
    reused = {k: v for k, v in id_counts.items() if v > 1}
    print(f"  Total unique S2/S3 IDs in GT: {len(id_counts):,}")
    print(f"  IDs appearing in >1 S1's GT: {len(reused):,}")
    if reused:
        print(f"  Top reused: {sorted(reused.items(), key=lambda x: -x[1])[:5]}")
    else:
        print("  → Mostly unique. Record-side competition signal may be applicable.")
    return reused


def audit_near_duplicates(df: pd.DataFrame, source_name: str, sample=5000) -> pd.Series:
    """Light near-duplicate detection using normalized name on a sample."""
    print(f"\n=== NEAR-DUPLICATE AUDIT ({source_name}, sample {sample}) ===")
    sample_df = df.sample(min(sample, len(df)), random_state=42).copy()
    sample_df["norm_name"] = sample_df["business_name"].str.lower().str.strip()
    dupes = sample_df[sample_df.duplicated(subset=["norm_name","business_address"], keep=False)]
    print(f"  Near-identical (name+addr exact, sample): {len(dupes):,}")
    return dupes


def audit_compute_estimate(s1, s2, s3):
    print("\n=== COMPUTE ESTIMATE ===")
    n_s1 = len(s1)
    n_s2 = len(s2)
    n_s3 = len(s3)
    k = 50  # typical TF-IDF K
    est_candidates = n_s1 * k * 2  # two sources
    est_pairs = est_candidates
    print(f"  S1={n_s1:,}, S2={n_s2:,}, S3={n_s3:,}")
    print(f"  With K={k}: ~{est_candidates:,} raw candidates before dedup")
    print(f"  Estimated pair-feature volume: ~{est_pairs:,} rows")
    print(f"  TF-IDF matrix: ~{(n_s2+n_s3):,} docs × vocab")
    # Memory rough estimate
    print(f"  Approx pair-feature RAM: ~{est_pairs * 60 / 1e9:.1f} GB at 60 bytes/row")
    if est_pairs > 50_000_000:
        print("  ⚠ Candidate volume may be high — consider smaller K or chunk processing")


def run_full_audit():
    print("Loading data …")
    s1, s2, s3, gt = load_all_train()
    ts1, ts2, ts3 = load_all_test()

    audit_structural(s1, s2, s3, gt)
    gt_enriched = audit_match_cardinality(s1, gt)
    reused = audit_record_reuse(gt)
    audit_near_duplicates(s2, "S2")
    audit_near_duplicates(s3, "S3")
    audit_compute_estimate(s1, s2, s3)

    print("\n=== TEST DATA ===")
    print(f"  Test S1: {len(ts1):,} | S2: {len(ts2):,} | S3: {len(ts3):,}")
    if "country" in ts1.columns:
        print(f"  Test S1 countries: {ts1['country'].value_counts().to_dict()}")

    summary = {
        "train_s1_rows": len(s1),
        "train_s2_rows": len(s2),
        "train_s3_rows": len(s3),
        "train_gt_rows": len(gt),
        "test_s1_rows": len(ts1),
        "test_s2_rows": len(ts2),
        "test_s3_rows": len(ts3),
        "s2_s3_record_reuse_count": len(reused),
    }
    out = ARTIFACTS / "audit_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n✅ Audit complete. Summary saved to {out}")
    return summary


if __name__ == "__main__":
    run_full_audit()
