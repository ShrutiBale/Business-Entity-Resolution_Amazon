"""
ml/blocking.py — High-recall multi-pass candidate generation.
Implements Passes A-G + character TF-IDF retrieval (approch.md §5, §6).
Runs OFFLINE — never called from FastAPI request handlers.
"""

import pickle
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from normalization import normalize_row

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# ─── Preprocess dataframe ──────────────────────────────────────────
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.apply(lambda row: normalize_row(row.to_dict()), axis=1)
    out  = pd.DataFrame(rows.tolist())
    return out


# ─── Pass A — exact structured keys ───────────────────────────────
def _build_key_index(df: pd.DataFrame) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        keys = []
        # name + postal
        if row.get("name_legal") and row.get("postal_code"):
            keys.append(f"NP:{row['name_legal'][:40]}|{row['postal_code']}")
        # name + house number
        if row.get("name_legal") and row.get("house_number"):
            keys.append(f"NH:{row['name_legal'][:40]}|{row['house_number']}")
        # sorted-token name + postal
        if row.get("name_sorted_tokens") and row.get("postal_code"):
            keys.append(f"SP:{row['name_sorted_tokens'][:40]}|{row['postal_code']}")
        for k in keys:
            idx[k].append(str(row["entity_id"]))
    return idx


def pass_A(s1_df: pd.DataFrame, s23_df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    s23_idx = _build_key_index(s23_df)
    pairs = []
    for _, row in s1_df.iterrows():
        eid = str(row["entity_id"])
        keys = []
        if row.get("name_legal") and row.get("postal_code"):
            keys.append(f"NP:{row['name_legal'][:40]}|{row['postal_code']}")
        if row.get("name_legal") and row.get("house_number"):
            keys.append(f"NH:{row['name_legal'][:40]}|{row['house_number']}")
        if row.get("name_sorted_tokens") and row.get("postal_code"):
            keys.append(f"SP:{row['name_sorted_tokens'][:40]}|{row['postal_code']}")
        seen = set()
        for k in keys:
            for cid in s23_idx.get(k, []):
                if cid not in seen:
                    pairs.append((eid, cid, "A"))
                    seen.add(cid)
    return pairs


# ─── Pass B/C — token index ─────────────────────────────────────────
def _build_token_index(df: pd.DataFrame, field: str, max_df_frac: float = 0.10) -> Dict[str, List[str]]:
    """Build inverted token index. Excludes tokens appearing in >max_df_frac of docs (common tokens)."""
    # Vectorized: explode tokens then group by token
    tokens_col = df[field].apply(lambda x: x if isinstance(x, list) else (x.split() if isinstance(x, str) else []))
    exploded = pd.DataFrame({"entity_id": df["entity_id"].repeat(tokens_col.str.len()), "token": [t for toks in tokens_col for t in toks]})
    if exploded.empty:
        return {}
    # Filter: min length 3, not too common
    exploded = exploded[exploded["token"].str.len() >= 3]
    doc_freq = exploded.drop_duplicates().groupby("token")["entity_id"].count()
    max_df   = int(len(df) * max_df_frac)
    valid_tokens = doc_freq[doc_freq <= max_df].index
    filtered = exploded[exploded["token"].isin(valid_tokens)]
    idx = filtered.groupby("token")["entity_id"].apply(list).to_dict()
    return {k: [str(v) for v in vs] for k, vs in idx.items()}


def _token_pass(s1_df: pd.DataFrame, s23_idx: Dict[str, List[str]],
                field: str, pass_label: str, max_per_entity: int = 100) -> List[Tuple]:
    pairs = []
    for _, row in s1_df.iterrows():
        eid = str(row["entity_id"])
        tokens = row.get(field, [])
        if isinstance(tokens, str):
            tokens = tokens.split()
        seen: Set[str] = set()
        counts: Dict[str, int] = defaultdict(int)
        for t in tokens:
            if len(t) < 3:
                continue
            for cid in s23_idx.get(t, []):
                counts[cid] += 1
        # take top-scored by overlap count
        top = sorted(counts.items(), key=lambda x: -x[1])[:max_per_entity]
        for cid, _ in top:
            if cid not in seen:
                pairs.append((eid, cid, pass_label))
                seen.add(cid)
    return pairs


# ─── Pass D — phonetic ─────────────────────────────────────────────
def _phonetic_pass(s1_df: pd.DataFrame, s23_df: pd.DataFrame) -> List[Tuple]:
    s23_idx: Dict[str, List[str]] = defaultdict(list)
    for _, row in s23_df.iterrows():
        ph = row.get("name_phonetic", "")
        if ph:
            s23_idx[ph].append(str(row["entity_id"]))

    pairs = []
    for _, row in s1_df.iterrows():
        eid = str(row["entity_id"])
        ph = row.get("name_phonetic", "")
        if ph:
            for cid in s23_idx.get(ph, []):
                pairs.append((eid, cid, "D"))
    return pairs


# ─── Pass E — sorted-token ─────────────────────────────────────────
def _sorted_token_pass(s1_df: pd.DataFrame, s23_df: pd.DataFrame) -> List[Tuple]:
    s23_idx: Dict[str, List[str]] = defaultdict(list)
    for _, row in s23_df.iterrows():
        st = row.get("name_sorted_tokens", "")
        if st:
            s23_idx[st].append(str(row["entity_id"]))

    pairs = []
    for _, row in s1_df.iterrows():
        eid = str(row["entity_id"])
        st = row.get("name_sorted_tokens", "")
        if st:
            for cid in s23_idx.get(st, []):
                pairs.append((eid, cid, "E"))
    return pairs


# ─── Character TF-IDF retrieval ────────────────────────────────────
class TFIDFRetriever:
    def __init__(self, ngram_range=(3, 5), max_features=200_000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.matrices = {}
        self.id_maps: Dict[str, List[str]] = {}

    def fit_transform(self, s23_df: pd.DataFrame, channels=("name_norm", "addr_norm")):
        for ch in channels:
            texts = s23_df[ch].fillna("").tolist()
            ids   = s23_df["entity_id"].tolist()
            vect = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                sublinear_tf=True,
            )
            mat = vect.fit_transform(texts)
            mat = normalize(mat, norm="l2", copy=False)
            self.vectorizers[ch] = vect
            self.matrices[ch] = mat
            self.id_maps[ch] = [str(i) for i in ids]
        return self

    def query(self, texts: List[str], channel: str, K: int = 50) -> List[List[Tuple[str, float]]]:
        vect = self.vectorizers[channel]
        mat  = self.matrices[channel]
        ids  = self.id_maps[channel]
        q_mat = normalize(vect.transform(texts), norm="l2")
        sims = (q_mat @ mat.T).toarray()
        results = []
        for row in sims:
            top_idx = row.argsort()[::-1][:K]
            results.append([(ids[i], float(row[i])) for i in top_idx if row[i] > 0])
        return results

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "TFIDFRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)


def tfidf_pass(s1_df: pd.DataFrame, retriever: TFIDFRetriever,
               K: int = 50) -> List[Tuple[str, str, str, float]]:
    """Query all TF-IDF channels and return (s1_id, s23_id, pass_label, score)."""
    pairs = []
    channels = list(retriever.vectorizers.keys())
    for ch in channels:
        texts = s1_df.get(ch, pd.Series([""] * len(s1_df))).fillna("").tolist()
        results = retriever.query(texts, ch, K=K)
        for i, (_, row) in enumerate(s1_df.iterrows()):
            s1_id = str(row["entity_id"])
            for cid, score in results[i]:
                label = f"TFIDF_{ch[:8]}"
                pairs.append((s1_id, cid, label, score))
    return pairs


# ─── Full blocking pipeline ────────────────────────────────────────
def run_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                 K: int = 50, save_retriever: bool = True,
                 retriever_path: Path = ARTIFACTS / "tfidf_retriever.pkl"
                 ) -> pd.DataFrame:
    """
    Run all passes on preprocessed DataFrames.
    Returns a DataFrame with columns: s1_id, candidate_id, blocking_passes, blocking_pass_count.
    """
    print("  Preprocessing S2 + S3 …")
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

    print("  Building token indexes …")
    s23_name_idx = _build_token_index(s23_df, "name_tokens")
    s23_addr_idx = _build_token_index(s23_df, "addr_tokens")

    print("  Running rule-based passes …")
    all_pairs: List[Tuple] = []
    all_pairs += pass_A(s1_df, s23_df)
    all_pairs += _token_pass(s1_df, s23_name_idx, "name_tokens", "B")
    all_pairs += _token_pass(s1_df, s23_addr_idx, "addr_tokens", "C")
    all_pairs += _phonetic_pass(s1_df, s23_df)
    all_pairs += _sorted_token_pass(s1_df, s23_df)
    print(f"  Rule passes: {len(all_pairs):,} (s1, cand) pairs before dedup")

    print("  Fitting TF-IDF retriever …")
    channels = ["name_norm", "addr_norm"]
    for ch in channels:
        if ch not in s23_df.columns:
            s23_df[ch] = ""

    retriever = TFIDFRetriever()
    retriever.fit_transform(s23_df, channels=channels)
    if save_retriever:
        retriever.save(retriever_path)
        print(f"  TF-IDF retriever saved → {retriever_path}")

    print("  Running TF-IDF retrieval …")
    for ch in channels:
        if ch not in s1_df.columns:
            s1_df[ch] = ""

    tfidf_pairs = tfidf_pass(s1_df, retriever, K=K)
    for s1_id, cid, label, score in tfidf_pairs:
        all_pairs.append((s1_id, cid, label))
    print(f"  After TF-IDF: {len(all_pairs):,} pairs before dedup")

    # ── Union + dedup ───────────────────────────────────────────────
    print("  Deduplicating …")
    pair_passes: Dict[Tuple[str,str], Set[str]] = defaultdict(set)
    for p in all_pairs:
        pair_passes[(p[0], p[1])].add(p[2])

    records = []
    for (s1_id, cid), passes in pair_passes.items():
        records.append({
            "s1_id": s1_id,
            "candidate_id": cid,
            "blocking_passes": ",".join(sorted(passes)),
            "blocking_pass_count": len(passes),
        })

    result = pd.DataFrame(records)
    print(f"  Final candidate pairs after dedup: {len(result):,}")
    return result


def measure_blocking_recall(candidate_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """Measure how many true positive pairs were retrieved."""
    # Build set of true pairs
    true_pairs: Set[Tuple[str, str]] = set()
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        for cid in str(row["matched_entity_ids"]).split(","):
            cid = cid.strip()
            if cid:
                true_pairs.add((s1_id, cid))

    retrieved_pairs = set(
        zip(candidate_df["s1_id"].astype(str), candidate_df["candidate_id"].astype(str))
    )

    retrieved_true = true_pairs & retrieved_pairs
    recall = len(retrieved_true) / len(true_pairs) if true_pairs else 0.0

    stats = {
        "blocking_recall": round(recall, 4),
        "true_pairs_total": len(true_pairs),
        "true_pairs_retrieved": len(retrieved_true),
        "true_pairs_missed": len(true_pairs) - len(retrieved_true),
        "total_candidate_pairs": len(candidate_df),
        "mean_candidates_per_s1": round(candidate_df.groupby("s1_id").size().mean(), 2),
        "p95_candidates_per_s1": int(candidate_df.groupby("s1_id").size().quantile(0.95)),
        "max_candidates_per_s1": int(candidate_df.groupby("s1_id").size().max()),
    }
    return stats


if __name__ == "__main__":
    from data_loader import load_all_train
    s1_raw, s2_raw, s3_raw, gt = load_all_train()
    print("Preprocessing …")
    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)
    print("Running blocking …")
    candidates = run_blocking(s1, s2, s3, K=50)
    stats = measure_blocking_recall(candidates, gt)
    print("\nBlocking recall stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    candidates.to_csv(Path(__file__).parent.parent / "artifacts" / "train_candidates.tsv", sep="\t", index=False)
    print("✅ Saved train candidates.")
