"""
ml/submission.py — Generate and validate the final submission files.
Runs the official validator if available.
"""

import subprocess
import sys
from pathlib import Path

OUTPUT    = Path(__file__).parent.parent / "output"
VALIDATOR = Path(__file__).parent.parent.parent / "student_resource" / "utils" / "validate_submission.py"
DATASET   = Path(__file__).parent.parent.parent / "student_resource" / "dataset"


def validate():
    """Run the official validator."""
    candidate_path = OUTPUT / "candidate_pairs.tsv"
    matching_path  = OUTPUT / "matching_results.tsv"
    test_dir       = DATASET / "test"

    if not candidate_path.exists():
        print(f"❌ {candidate_path} not found. Run inference.py first.")
        sys.exit(1)
    if not matching_path.exists():
        print(f"❌ {matching_path} not found. Run inference.py first.")
        sys.exit(1)
    if not VALIDATOR.exists():
        print(f"⚠ Validator not found at {VALIDATOR}. Skipping official validation.")
        return

    print(f"Running official validator …")
    cmd = [
        sys.executable, str(VALIDATOR),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(result.returncode)
    print("✅ Validator passed.")


def check_outputs():
    """Basic sanity checks on output files."""
    import pandas as pd

    for fname in ("candidate_pairs.tsv", "matching_results.tsv"):
        path = OUTPUT / fname
        if not path.exists():
            print(f"❌ {fname} missing")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        print(f"\n{fname}:")
        print(f"  rows: {len(df):,}")
        print(f"  columns: {list(df.columns)}")
        print(f"  sample:")
        print(df.head(3).to_string(index=False))

        # Check no S1 IDs inside candidate lists
        id_col = "candidate_entity_ids" if "candidate_entity_ids" in df.columns else "matched_entity_ids"
        s1_in_cands = df[id_col].str.contains("S1-", na=False).sum()
        if s1_in_cands:
            print(f"  ⚠ WARNING: {s1_in_cands} rows have S1 IDs inside {id_col}")
        else:
            print(f"  ✓ No S1 IDs inside {id_col}")


if __name__ == "__main__":
    check_outputs()
    validate()
