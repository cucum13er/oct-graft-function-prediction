import argparse
import os
import re

import numpy as np
import pandas as pd


DEFAULT_DIR = "data/processed"
DEFAULT_CURVE_CSV = os.path.join(
    DEFAULT_DIR, "case152_case172_core_lumen_features_curve_depth.csv"
)
DEFAULT_LUMEN_CSV = os.path.join(DEFAULT_DIR, "georgetown_real_case_features_all.csv")
DEFAULT_CLINICAL_CSV = os.path.join(DEFAULT_DIR, "kidney_features_X_152to172.csv")
DEFAULT_OUT_CSV = os.path.join(DEFAULT_DIR, "kidney_features_merged_152to172.csv")

DROP_LUMEN_META_COLS = [
    "num_bscans_total",
    "num_bscans_used",
    "num_lumens_total",
    "num_nnd_total",
]


OLD_CLINICAL_ORDER = [
    "age",
    "init_creatinine",
    "term_creatinine",
    "kdpi",
    "cold_ischemia_min",
    "sex_male",
    "donor_dcd",
    "race_code",
    "num_glom",
    "num_glob_scl",
    "pct_glob_scl",
    "num_arteries",
    "ifta",
    "vascular_disease",
    "arteriolar_hyalinosis",
    "cortical_necrosis",
    "fibrin_thrombi",
    "IGF(0)/DGF(1)",
]


def normalize_case_id(value):
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        return ""
    return f"KTX{int(match.group(1))}"


def parse_binary_outcome(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    return 1 if "dgf" in text else 0


def read_lumen_features(path):
    df = pd.read_csv(path)
    if "case_num" not in df.columns:
        raise ValueError(f"{path} is missing required column: case_num")
    df = df.copy()
    df["case_num"] = df["case_num"].apply(normalize_case_id)
    df = df.drop(columns=[c for c in DROP_LUMEN_META_COLS if c in df.columns])
    return df


def read_curve_depth_features(path):
    df = pd.read_csv(path)
    if "case" not in df.columns:
        raise ValueError(f"{path} is missing required column: case")
    df = df.copy()
    df["case_num"] = df["case"].apply(normalize_case_id)
    drop_cols = ["case", "source_csv", "missing_columns"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def read_clinical_features(path):
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} is missing required column: sample_id")
    df = df.copy()
    df["case_num"] = df["sample_id"].apply(normalize_case_id)

    if "DGF or IGF" in df.columns:
        df["IGF(0)/DGF(1)"] = df["DGF or IGF"].apply(parse_binary_outcome)

    df = df.drop(columns=[c for c in ["sample_id", "DGF or IGF"] if c in df.columns])
    return df


def require_unique_cases(name, df):
    dup = df.loc[df["case_num"].duplicated(), "case_num"].tolist()
    if dup:
        raise ValueError(f"{name} has duplicate case_num values: {sorted(set(dup))}")


def ordered_columns(df):
    lumen_cols = [
        c
        for c in df.columns
        if c
        not in {
            "case_num",
            *OLD_CLINICAL_ORDER,
        }
    ]
    clinical_cols = [c for c in OLD_CLINICAL_ORDER if c in df.columns]
    remaining_cols = [
        c for c in df.columns if c not in {"case_num", *lumen_cols, *clinical_cols}
    ]
    return ["case_num"] + lumen_cols + clinical_cols + remaining_cols


def merge_features(args):
    lumen_df = read_lumen_features(args.lumen_csv)
    curve_df = read_curve_depth_features(args.curve_csv)
    clinical_df = read_clinical_features(args.clinical_csv)

    require_unique_cases("lumen_csv", lumen_df)
    require_unique_cases("curve_csv", curve_df)
    require_unique_cases("clinical_csv", clinical_df)

    merged = lumen_df.merge(curve_df, on="case_num", how=args.how)
    merged = merged.merge(clinical_df, on="case_num", how=args.how)
    merged = merged[ordered_columns(merged)].sort_values("case_num").reset_index(drop=True)

    return merged, lumen_df, curve_df, clinical_df


def print_case_summary(merged, lumen_df, curve_df, clinical_df):
    merged_cases = set(merged["case_num"])
    inputs = {
        "lumen": set(lumen_df["case_num"]),
        "curve_depth": set(curve_df["case_num"]),
        "clinical": set(clinical_df["case_num"]),
    }
    print("Merged cases:", sorted(merged_cases))
    for name, cases in inputs.items():
        missing = sorted(cases - merged_cases)
        if missing:
            print(f"{name} cases not in merged output:", missing)


def main():
    parser = argparse.ArgumentParser(
        description="Merge OCT lumen, curve-depth, and clinical features for KTX152-KTX172."
    )
    parser.add_argument("--curve-csv", default=DEFAULT_CURVE_CSV)
    parser.add_argument("--lumen-csv", default=DEFAULT_LUMEN_CSV)
    parser.add_argument("--clinical-csv", default=DEFAULT_CLINICAL_CSV)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--how",
        default="inner",
        choices=["inner", "left", "right", "outer"],
        help="Merge mode. Default inner keeps cases present in all three input CSVs.",
    )
    args = parser.parse_args()

    merged, lumen_df, curve_df, clinical_df = merge_features(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv, "shape:", merged.shape)
    print_case_summary(merged, lumen_df, curve_df, clinical_df)


if __name__ == "__main__":
    main()
