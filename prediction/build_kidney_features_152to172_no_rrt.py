"""Create the spatial feature table without recipient RRT modality."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/kidney_features_merged_152to172_canonical_four_groups.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/kidney_features_merged_152to172_canonical_no_rrt.csv"
REMOVED_COL = "recipient_rrt_modality"


def build_table(input_csv=DEFAULT_INPUT, output_csv=DEFAULT_OUTPUT):
    df = pd.read_csv(input_csv)
    if REMOVED_COL not in df:
        raise ValueError(f"{REMOVED_COL} is missing from {input_csv}")
    if df["case_num"].duplicated().any():
        raise ValueError("Duplicate case IDs in the canonical table")
    result = df.drop(columns=REMOVED_COL)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    table = build_table(args.input_csv, args.output_csv)
    print(f"Wrote {len(table)} cases and {len(table.columns)} columns to {args.output_csv}")


if __name__ == "__main__":
    main()

