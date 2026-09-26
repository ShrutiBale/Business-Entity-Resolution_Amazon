"""
ml/data_loader.py — Load real TSV files from the dataset directory.
Always reads with sep='\t'. Preserves original fields.
"""

import pandas as pd
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "student_resource" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR  = DATASET_DIR / "test"


def load_train_source1() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str).fillna("")

def load_train_source2() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str).fillna("")

def load_train_source3() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str).fillna("")

def load_train_ground_truth() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

def load_test_source1() -> pd.DataFrame:
    return pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str).fillna("")

def load_test_source2() -> pd.DataFrame:
    return pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str).fillna("")

def load_test_source3() -> pd.DataFrame:
    return pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str).fillna("")


def load_all_train():
    return (
        load_train_source1(),
        load_train_source2(),
        load_train_source3(),
        load_train_ground_truth(),
    )


def load_all_test():
    return (
        load_test_source1(),
        load_test_source2(),
        load_test_source3(),
    )


if __name__ == "__main__":
    s1, s2, s3, gt = load_all_train()
    print(f"S1: {len(s1):,} rows | columns: {list(s1.columns)}")
    print(f"S2: {len(s2):,} rows | columns: {list(s2.columns)}")
    print(f"S3: {len(s3):,} rows | columns: {list(s3.columns)}")
    print(f"GT: {len(gt):,} rows | columns: {list(gt.columns)}")
    ts1, ts2, ts3 = load_all_test()
    print(f"Test S1: {len(ts1):,} | S2: {len(ts2):,} | S3: {len(ts3):,}")
