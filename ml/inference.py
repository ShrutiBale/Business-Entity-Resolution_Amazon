"""
ml/inference.py — Frozen model test inference.
Runs the frozen pipeline on real test TSVs and produces:
  output/candidate_pairs.tsv   — exact final candidate set
  output/matching_results.tsv  — final predictions

OFFLINE ONLY — never called from FastAPI.
"""

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_loader import load_all_test
from normalization import normalize_row
from blocking import run_blocking, preprocess_df, TFIDFRetriever
from features import compute_pair_features, get_feature_names
from train import build_feature_matrix, build_name_frequency

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
OUTPUT    = Path(__file__).parent.parent / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)


def run_inference():
    print("=== LOADING ARTIFACTS ===")
    config_path = ARTIFACTS / "config.json"
    model_path  = ARTIFACTS / "model.txt"

    if not model_path.exists():
        print(f"❌ No trained model found at {model_path}. Run ml/train.py first.")
        sys.exit(1)

    config = json.loads(config_path.read_text())
    threshold = config["global_threshold"]
    print(f"  Global threshold: {threshold}")

    booster = lgb.Booster(model_file=str(model_path))

    # ── Load pre-saved name frequencies (from training artifact) ───────────
    freq_path = ARTIFACTS / "name_frequencies.pkl"
    if freq_path.exists():
        print("  Loading name frequencies from saved artifact …")
        with open(freq_path, "rb") as f:
            freq_data = pickle.load(f)
        freq_s1 = freq_data["freq_s1"]
        freq_s2 = freq_data["freq_s2"]
        freq_s3 = freq_data["freq_s3"]
        print(f"  Loaded: {len(freq_s1):,} S1 names, {len(freq_s2):,} S2 names, {len(freq_s3):,} S3 names")
    else:
        print("  ⚠ name_frequencies.pkl not found — re-running train.py will save it.")
        print("  Falling back to loading train data (slow) …")
        from data_loader import load_all_train
        s1_train_raw, s2_train_raw, s3_train_raw, _ = load_all_train()
        s1_tr = preprocess_df(s1_train_raw)
        s2_tr = preprocess_df(s2_train_raw)
        s3_tr = preprocess_df(s3_train_raw)
        freq_s1, freq_s2, freq_s3 = build_name_frequency(s1_tr, s2_tr, s3_tr)

    # ── Load test data ────────────────────────────────────────────
    print("\n=== LOADING TEST DATA ===")
    ts1_raw, ts2_raw, ts3_raw = load_all_test()
    print(f"  Test S1: {len(ts1_raw):,} | S2: {len(ts2_raw):,} | S3: {len(ts3_raw):,}")

    ts1 = preprocess_df(ts1_raw)
    ts2 = preprocess_df(ts2_raw)
    ts3 = preprocess_df(ts3_raw)

    s23_map = {str(r["entity_id"]): r for r in pd.concat([ts2, ts3]).to_dict("records")}
    s1_map  = {str(r["entity_id"]): r for r in ts1.to_dict("records")}

    # ── Run blocking on test ──────────────────────────────────────
    print("\n=== TEST BLOCKING ===")
    # Load existing retriever fitted on TRAIN; transform test
    retriever_path = ARTIFACTS / "tfidf_retriever.pkl"
    if retriever_path.exists():
        print("  Using saved TF-IDF retriever (train-only fitted, never refitted on test)")
        retriever = TFIDFRetriever.load(retriever_path)
    else:
        print("  ⚠ No retriever found — refitting on test S2+S3 (less ideal)")
        retriever = None

    candidates = run_blocking(ts1, ts2, ts3, K=50, save_retriever=False)
    print(f"  Test candidates: {len(candidates):,}")

    # ── Build feature matrix ──────────────────────────────────────
    print("\n=== BUILDING TEST FEATURES ===")
    feat_names = get_feature_names()
    rows = []
    pair_meta = []  # (s1_id, cand_id)

    for _, row in candidates.iterrows():
        s1_id   = str(row["s1_id"])
        cand_id = str(row["candidate_id"])
        s1_row  = s1_map.get(s1_id)
        c_row   = s23_map.get(cand_id)
        if s1_row is None or c_row is None:
            continue
        n_leg = s1_row.get("name_legal", "")
        feats = compute_pair_features(
            s1_row, c_row,
            blocking_passes=str(row.get("blocking_passes", "")),
            blocking_pass_count=int(row.get("blocking_pass_count", 0)),
            name_freq_s1=freq_s1.get(n_leg, 1e-6),
            name_freq_s2=freq_s2.get(c_row.get("name_legal", ""), 1e-6),
            name_freq_s3=freq_s3.get(c_row.get("name_legal", ""), 1e-6),
        )
        rows.append([feats[f] for f in feat_names])
        pair_meta.append((s1_id, cand_id))

    X_test = np.array(rows, dtype=np.float32)
    print(f"  Feature matrix: {X_test.shape}")

    # ── LightGBM predictions ──────────────────────────────────────
    print("\n=== SCORING WITH FROZEN MODEL ===")
    scores = booster.predict(X_test)

    # ── Build candidate_pairs.tsv ─────────────────────────────────
    # Exactly one row per test S1; comma-separated candidate IDs
    print("\n=== BUILDING candidate_pairs.tsv ===")
    all_test_s1_ids = ts1["entity_id"].tolist()
    cand_by_s1: dict = {s: [] for s in all_test_s1_ids}

    for (s1_id, cand_id), score in zip(pair_meta, scores):
        cand_by_s1.setdefault(s1_id, []).append((cand_id, float(score)))

    cand_rows = []
    match_rows = []

    for s1_id in all_test_s1_ids:
        all_cands = cand_by_s1.get(s1_id, [])
        # deduplicate
        seen_cids = {}
        for cid, score in all_cands:
            if cid not in seen_cids or score > seen_cids[cid]:
                seen_cids[cid] = score

        cand_ids_str = ",".join(seen_cids.keys())
        cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_ids_str})

        # predictions above threshold
        matched = [cid for cid, sc in seen_cids.items() if sc >= threshold]
        match_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matched)})

    cand_df  = pd.DataFrame(cand_rows)
    match_df = pd.DataFrame(match_rows)

    cand_path  = OUTPUT / "candidate_pairs.tsv"
    match_path = OUTPUT / "matching_results.tsv"

    cand_df.to_csv(cand_path,  sep="\t", index=False)
    match_df.to_csv(match_path, sep="\t", index=False)

    n_with_match = (match_df["matched_entity_ids"] != "").sum()
    print(f"  candidate_pairs.tsv: {len(cand_df):,} rows → {cand_path}")
    print(f"  matching_results.tsv: {len(match_df):,} rows ({n_with_match:,} with at least one match) → {match_path}")
    print("\n✅ Inference complete.")

    return {"candidate_pairs_path": str(cand_path), "matching_results_path": str(match_path),
            "threshold": threshold, "n_test_s1": len(all_test_s1_ids),
            "n_with_match": int(n_with_match)}


if __name__ == "__main__":
    run_inference()


