"""
ml/train.py — Train LightGBM on real candidate pairs.
Uses S1-grouped K-fold cross-validation (approch.md §11, §12, §13).
Saves frozen model artifact.

OFFLINE ONLY — never called from FastAPI.
"""

import json
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from data_loader import load_all_train
from normalization import normalize_row
from blocking import run_blocking, preprocess_df, measure_blocking_recall
from features import compute_pair_features, get_feature_names
from evaluate import parse_ground_truth, macro_f05, threshold_sweep

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5
LGBM_PARAMS = {
    "objective":       "binary",
    "metric":          "auc",
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "lambda_l1":       0.1,
    "lambda_l2":       0.1,
    "random_state":    RANDOM_SEED,
    "n_jobs":          -1,
    "verbose":         -1,
}


def build_name_frequency(s1_df, s2_df, s3_df):
    """Compute name frequency per source (approch.md §11.5)."""
    def freq(df, col="name_legal"):
        counts = df[col].value_counts()
        total = len(df)
        return (counts / total).to_dict()

    freq_s1 = freq(s1_df)
    freq_s2 = freq(s2_df)
    freq_s3 = freq(s3_df)
    return freq_s1, freq_s2, freq_s3


def build_feature_matrix(
    candidate_pairs: pd.DataFrame,
    s1_map: dict, cand_map: dict,
    freq_s1: dict, freq_s2: dict, freq_s3: dict,
    gt_pairs: set = None,
    chunk_size: int = 50_000,
) -> tuple:
    """
    Build feature matrix + labels from candidate pairs.
    Processes in chunks to avoid memory blowups on large candidate sets.
    """
    feat_names = get_feature_names()
    all_rows   = []
    all_labels = []
    all_groups = []

    n = len(candidate_pairs)
    for start in range(0, n, chunk_size):
        chunk = candidate_pairs.iloc[start:start + chunk_size]
        if start % 200_000 == 0 and start > 0:
            print(f"    features: {start:,}/{n:,} …", flush=True)
        for _, row in chunk.iterrows():
            s1_id   = str(row["s1_id"])
            cand_id = str(row["candidate_id"])
            s1_row  = s1_map.get(s1_id)
            c_row   = cand_map.get(cand_id)
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
            all_rows.append([feats[f] for f in feat_names])
            if gt_pairs is not None:
                all_labels.append(1 if (s1_id, cand_id) in gt_pairs else 0)
            all_groups.append(s1_id)

    X = np.array(all_rows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int8) if gt_pairs is not None else None
    return X, y, all_groups


def run_training(max_train_rows: int = None):
    """
    Full training pipeline:
    1. Load real data
    2. Preprocess + normalize
    3. Run blocking
    4. Build training pairs
    5. Generate features
    6. S1-grouped K-fold validation
    7. Tune global F0.5 threshold
    8. Retrain on full data
    9. Save artifacts
    """
    t0 = time.time()
    print("=== LOADING DATA ===")
    s1_raw, s2_raw, s3_raw, gt = load_all_train()

    if max_train_rows:
        print(f"  ⚠ DEV MODE: using only first {max_train_rows:,} S1 rows")
        sample_ids = set(s1_raw["entity_id"].tolist()[:max_train_rows])
        s1_raw = s1_raw[s1_raw["entity_id"].isin(sample_ids)].reset_index(drop=True)
        gt     = gt[gt["source1_entity_id"].isin(sample_ids)].reset_index(drop=True)
        # Also sample S2/S3 proportionally (10x ratio) so preprocessing is fast
        s2_n   = min(max_train_rows * 10, len(s2_raw))
        s3_n   = min(max_train_rows * 10, len(s3_raw))
        s2_raw = s2_raw.sample(s2_n, random_state=42).reset_index(drop=True)
        s3_raw = s3_raw.sample(s3_n, random_state=42).reset_index(drop=True)
        print(f"  DEV S2 sample: {s2_n:,} | S3 sample: {s3_n:,}")

    print("=== PREPROCESSING ===")
    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)

    s1_map   = {str(r["entity_id"]): r for r in s1.to_dict("records")}
    s2_map   = {str(r["entity_id"]): r for r in s2.to_dict("records")}
    s3_map   = {str(r["entity_id"]): r for r in s3.to_dict("records")}
    cand_map = {**s2_map, **s3_map}

    print("=== BLOCKING ===")
    # In dev mode use smaller K to keep candidate volume tractable
    blocking_K = 20 if max_train_rows else 50
    candidates = run_blocking(s1, s2, s3, K=blocking_K, save_retriever=True)
    blocking_stats = measure_blocking_recall(candidates, gt)
    print(f"  Blocking recall: {blocking_stats['blocking_recall']:.4f} (dev mode: low because S2/S3 sampled)")
    print(f"  Candidates: {blocking_stats['total_candidate_pairs']:,}")

    print("=== PARSING GROUND TRUTH ===")
    true_matches = parse_ground_truth(gt)
    gt_pairs = set()
    for s1_id, mids in true_matches.items():
        for mid in mids:
            gt_pairs.add((str(s1_id), str(mid)))

    # Cap candidate pairs in dev mode to avoid multi-hour feature computation
    if max_train_rows and len(candidates) > 200_000:
        print(f"  DEV CAP: sampling 200K from {len(candidates):,} candidate pairs")
        # Vectorized: build key column and check against gt_pairs set
        candidates["_key"] = candidates["s1_id"].astype(str) + "|" + candidates["candidate_id"].astype(str)
        gt_keys  = {f"{s}|{c}" for s, c in gt_pairs}
        pos_mask = candidates["_key"].isin(gt_keys)
        pos_cands = candidates[pos_mask].drop(columns=["_key"])
        neg_pool  = candidates[~pos_mask].drop(columns=["_key"])
        n_neg_keep = min(200_000 - len(pos_cands), len(neg_pool))
        neg_cands  = neg_pool.sample(n_neg_keep, random_state=42)
        candidates = pd.concat([pos_cands, neg_cands], ignore_index=True)
        print(f"  After DEV CAP: {len(candidates):,} pairs ({len(pos_cands)} pos + {len(neg_cands)} neg)")

    candidates.to_csv(ARTIFACTS / "train_candidates.tsv", sep="\t", index=False)

    # Force-add any missed positives with NaN retrieval features
    missed = gt_pairs - set(zip(candidates["s1_id"].astype(str), candidates["candidate_id"].astype(str)))
    if missed:
        print(f"  Force-injecting {len(missed)} missed positive pairs")
        extra = pd.DataFrame([
            {"s1_id": s, "candidate_id": c, "blocking_passes": "", "blocking_pass_count": 0}
            for s, c in missed
        ])
        candidates = pd.concat([candidates, extra], ignore_index=True)

    print("=== COMPUTING NAME FREQUENCIES ===")
    freq_s1, freq_s2, freq_s3 = build_name_frequency(s1, s2, s3)
    # Save frequencies so inference never needs to reload all train data
    freq_path = ARTIFACTS / "name_frequencies.pkl"
    with open(freq_path, "wb") as f:
        pickle.dump({"freq_s1": freq_s1, "freq_s2": freq_s2, "freq_s3": freq_s3}, f)
    print(f"  Name frequencies saved → {freq_path}")

    print("=== BUILDING FEATURE MATRIX ===")
    X, y, groups = build_feature_matrix(candidates, s1_map, cand_map, freq_s1, freq_s2, freq_s3, gt_pairs)
    print(f"  Shape: {X.shape} | Positives: {y.sum():,} ({100*y.mean():.2f}%)")

    # ── S1-grouped K-fold CV ─────────────────────────────────────
    print("=== S1-GROUPED K-FOLD VALIDATION ===")
    group_arr = np.array(groups)
    unique_groups = list(dict.fromkeys(groups))  # preserve order
    group_idx = {g: i for i, g in enumerate(unique_groups)}
    group_int = np.array([group_idx[g] for g in groups])

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_s1_ids = list(s1_map.keys())
    fold_metrics = []
    oof_scores = np.zeros(len(X))

    # Scale positive weight to handle class imbalance
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    spw   = max(1, n_neg // max(n_pos, 1))
    print(f"  Class balance: {n_pos} pos / {n_neg} neg → scale_pos_weight={spw}")

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, group_int)):
        print(f"  Fold {fold_i+1}/{N_FOLDS} …", end=" ", flush=True)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        fold_params = {**LGBM_PARAMS, "scale_pos_weight": spw}
        model = lgb.LGBMClassifier(**fold_params)
        # Use new LightGBM API (eval_X/eval_y) to avoid deprecation warning
        try:
            model.fit(X_tr, y_tr,
                      eval_X=X_val, eval_y=y_val,
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        except TypeError:
            # Fallback for older LightGBM versions
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_scores[val_idx] = val_probs

        # Entity-level F0.5 for this fold
        val_groups = group_arr[val_idx]
        val_s1_ids = list(dict.fromkeys(val_groups))
        val_true = {s: true_matches.get(s, []) for s in val_s1_ids}
        val_cands = candidates.iloc[val_idx].copy()
        val_cands["lgbm_score"] = val_probs

        sweep = threshold_sweep(val_s1_ids, val_true, val_cands)
        fold_metrics.append(sweep)
        print(f"best F0.5={sweep['best_macro_f05']:.4f} @ t={sweep['best_threshold']}")

    best_thresholds = [m["best_threshold"] for m in fold_metrics]
    global_threshold = round(float(np.median(best_thresholds)), 2)
    mean_f05 = round(float(np.mean([m["best_macro_f05"] for m in fold_metrics])), 5)
    print(f"\n  CV macro F0.5 (mean): {mean_f05}")
    print(f"  Global threshold (median of folds): {global_threshold}")

    # ── Retrain on full data ─────────────────────────────────────
    print("\n=== TRAINING FINAL MODEL ON FULL DATA ===")
    final_model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "scale_pos_weight": spw})
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    model_path = ARTIFACTS / "model.txt"
    final_model.booster_.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    config = {
        "global_threshold": global_threshold,
        "cv_macro_f05":     mean_f05,
        "blocking_recall":  blocking_stats["blocking_recall"],
        "n_train_s1":       len(s1),
        "n_candidates":     len(candidates),
        "feature_names":    get_feature_names(),
        "lgbm_params":      LGBM_PARAMS,
        "elapsed_sec":      round(time.time() - t0, 1),
        "artifacts": {
            "model":             "artifacts/model.txt",
            "tfidf_retriever":   "artifacts/tfidf_retriever.pkl",
            "name_frequencies":  "artifacts/name_frequencies.pkl",
            "train_candidates":  "artifacts/train_candidates.tsv",
        }
    }
    (ARTIFACTS / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  Config saved → {ARTIFACTS / 'config.json'}")
    print(f"\n✅ Training complete in {config['elapsed_sec']}s")
    return config


if __name__ == "__main__":
    import sys
    # Quick dev run: python3 train.py --dev 5000
    dev_rows = None
    if "--dev" in sys.argv:
        idx = sys.argv.index("--dev")
        dev_rows = int(sys.argv[idx + 1])
    run_training(max_train_rows=dev_rows)
