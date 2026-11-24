"""Add complete clinical fields to the existing 18-case OCT table."""

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_CSV = ROOT / "data/processed/kidney_features_merged_152to172.csv"
CLINICAL_CSV = ROOT / "data/processed/kidney_features_X_complete.csv"
OUTPUT_CSV = ROOT / "data/processed/kidney_features_merged_152to172_canonical_four_groups.csv"
LABEL = "IGF(0)/DGF(1)"
RECIPIENT_COLS = [
    "recipient_age",
    "recipient_bmi",
    "recipient_sex_male",
    "recipient_race_code",
    "recipient_previous_kidney_txp",
    "recipient_rrt_modality",
    "recipient_diabetes",
    "recipient_etiology_htn",
    "recipient_etiology_dm",
    "recipient_etiology_pckd",
    "recipient_etiology_gn",
    "recipient_etiology_other",
]
CANONICAL_DONOR_COLS = [
    "donor_age",
    "donor_init_creatinine",
    "donor_term_creatinine",
    "donor_cold_ischemia_min",
    "donor_kdpi",
    "donor_sex_male",
    "donor_dcd",
    "donor_cause_of_death_stroke",
    "donor_race_code",
    "donor_htn",
]
DONOR_EXTRA_COLS = ["donor_cause_of_death_stroke", "donor_htn"]
DONOR_SOURCE_MAP = {
    "age": "donor_age",
    "init_creatinine": "donor_init_creatinine",
    "term_creatinine": "donor_term_creatinine",
    "kdpi": "donor_kdpi",
    "cold_ischemia_min": "donor_cold_ischemia_min",
    "sex_male": "donor_sex_male",
    "donor_dcd": "donor_dcd",
    "race_code": "donor_race_code",
}


def normalize_case_id(value):
    match = re.fullmatch(r"\s*KTX\s*0*(\d+)\s*", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid case ID: {value!r}")
    return f"KTX{int(match.group(1))}"


def build_table(
    base_csv=BASE_CSV, clinical_csv=CLINICAL_CSV, output_csv=OUTPUT_CSV,
    use_complete_donor=False, canonical_clinical=True,
):
    added_donor = (CANONICAL_DONOR_COLS if canonical_clinical else
                   DONOR_EXTRA_COLS if use_complete_donor else [])
    base = pd.read_csv(base_csv)
    clinical = pd.read_csv(clinical_csv)
    if canonical_clinical:
        morphology = [col for col in base if col.startswith(("minor_", "ecc_", "sol_"))]
        spatial = [col for col in base if col.startswith(("anterior_", "posterior_"))]
        if not morphology or not spatial:
            raise ValueError("Both OCT morphology and spatial features are required")
        base = base[["case_num", *morphology, *spatial, LABEL]].copy()
    for name, frame, required in [
        ("OCT table", base, ["case_num", LABEL]),
        ("complete clinical table", clinical, ["sample_id", "DGF or IGF", *added_donor, *RECIPIENT_COLS,
                                             *(DONOR_SOURCE_MAP.values() if use_complete_donor else [])]),
    ]:
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    base = base.copy()
    clinical = clinical.copy()
    base["case_num"] = base["case_num"].map(normalize_case_id)
    clinical["case_num"] = clinical["sample_id"].map(normalize_case_id)
    for name, frame in [("OCT", base), ("clinical", clinical)]:
        duplicates = frame.loc[frame["case_num"].duplicated(), "case_num"].tolist()
        if duplicates:
            raise ValueError(f"Duplicate {name} case IDs: {duplicates}")

    existing = set(base.columns) & set([*added_donor, *RECIPIENT_COLS])
    if existing:
        raise ValueError(f"Clinical columns already present in OCT table: {sorted(existing)}")
    source = clinical[["case_num", "DGF or IGF", *added_donor, *RECIPIENT_COLS]]
    merged = base.merge(source, on="case_num", how="left", validate="one_to_one", indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", "case_num"].tolist()
    if unmatched:
        raise ValueError(f"Cases missing from complete clinical table: {unmatched}")
    merged = merged.drop(columns="_merge")

    clinical_label = pd.to_numeric(merged["DGF or IGF"], errors="coerce")
    base_label = pd.to_numeric(merged[LABEL], errors="coerce")
    mismatched = merged.loc[clinical_label.notna() & (clinical_label != base_label), "case_num"].tolist()
    if mismatched:
        raise ValueError(f"Outcome disagreement for cases: {mismatched}")
    merged = merged.drop(columns="DGF or IGF")
    if use_complete_donor and not canonical_clinical:
        complete = clinical.set_index("case_num").loc[merged["case_num"]]
        for target, source_col in DONOR_SOURCE_MAP.items():
            merged[target] = complete[source_col].to_numpy()
    columns = [col for col in base.columns if col != LABEL] + added_donor + RECIPIENT_COLS + [LABEL]
    merged = merged[columns]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, default=BASE_CSV)
    parser.add_argument("--clinical-csv", type=Path, default=CLINICAL_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--use-complete-donor", action="store_true")
    parser.add_argument("--legacy-clinical", action="store_true")
    args = parser.parse_args()
    table = build_table(args.base_csv, args.clinical_csv, args.output_csv,
                        use_complete_donor=args.use_complete_donor,
                        canonical_clinical=not args.legacy_clinical)
    print(f"Wrote {len(table)} cases and {len(table.columns)} columns to {args.output_csv}")


if __name__ == "__main__":
    main()
