"""
Build a single raw clinical CSV from the donor and recipient workbook sheets.

The input workbook is the clinical team's export.  The output keeps only
pre-transplant donor/recipient fields plus the DGF/IGF outcome label.
"""

from __future__ import annotations

import argparse
import math
import re

import pandas as pd


DEFAULT_INPUT_XLSX = "data/raw/clinical_data.xlsx"
DEFAULT_OUTPUT_CSV = "data/processed/all_raw_kidney_data_complete.csv"

DONOR_SHEET = "1) Donor Data 4.30.2026"
RECIPIENT_SHEET = "2) Recipient Data 4.30.2026"

DONOR_COLUMN_MAP = {
    "Unnamed: 0": "sample_id",
    "Age": "donor_Age",
    "Sex": "donor_Sex",
    "Race/Ethnicity": "donor_Race/Ethnicity",
    "Living/Cadaver Donor": "donor_Living/Cadaver Donor",
    "DCD or DBD": "donor_DCD or DBD",
    "Cause of Death ": "donor_Cause of Death",
    "History Of Hypertension": "donor_History Of Hypertension",
    "Intitial Creatinine": "donor_Intitial Creatinine",
    "Terminal Creatnine": "donor_Terminal Creatinine",
    "Time of Cold ischemia\n( In Minutes)": "donor_Time of Cold ischemia ( In Minutes)",
    "KDPI": "donor_KDPI",
    "Number of Glomeruli": "donor_Number of Glomeruli",
    "# of globally sclerosed glomeruli": "donor_# of globally sclerosed glomeruli",
    "% of globally scleorosed glomeruli": "donor_% of globally sclerosed glomeruli",
    "Interstitial fibrosis (and tubular atrophy) = IFTA": (
        "donor_Interstitial fibrosis (and tubular atrophy) = IFTA"
    ),
    "Vascular disease": "donor_Vascular disease",
    "Arterioral hylinosis": "donor_Arterioral hylinosis",
    "Cortical necrosis": "donor_Cortical necrosis",
    "Fibrin thrombi": "donor_Fibrin thrombi",
    "Number of arteries": "donor_Number of arteries",
}

RECIPIENT_COLUMN_MAP = {
    "Study ID": "sample_id",
    "Age": "recipient_Age",
    "Gender": "recipient_Sex",
    "BMI": "recipient_BMI",
    "Etiology of ESRD": "recipient_Etiology of ESRD",
    "Previous Transplant?": "recipient_Previous Transplant?",
    "Pre dialysis (CKD) or ESRD at time of txp": (
        "recipient_Pre dialysis (CKD) or ESRD at time of txp"
    ),
    "PD or HD if on RRTx": "recipient_PD or HD if on RRTx",
    "Diabetes?": "recipient_Diabetes?",
    "HLA Profile": "recipient_HLA Profile",
    "DGF or IGF": "DGF or IGF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge donor and recipient clinical workbook sheets into one raw CSV."
    )
    parser.add_argument("--input-xlsx", default=DEFAULT_INPUT_XLSX)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--donor-sheet", default=DONOR_SHEET)
    parser.add_argument("--recipient-sheet", default=RECIPIENT_SHEET)
    return parser.parse_args()


def normalize_case_id(value: object) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"KT(?:X)?\s*0*(\d+)", str(value), re.IGNORECASE)
    if match is None:
        return ""
    return f"KTX{int(match.group(1))}"


def normalize_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if not isinstance(value, str):
        return value
    text = value.replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"\b(White|Black or African American|African American|Other|Asian|Unknown|Not Reported)\s*/\s*",
        r"\1/",
        text,
    )
    text = re.sub(r"(?<=\d)\$(?=\s*$)", "%", text)
    text = re.sub(r"%{2,}", "%", text)
    text = re.sub(r"\?+", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("Affrican", "African").replace("Africam", "African")
    text = text.replace("Non Hispanic", "Non-Hispanic")
    text = text.replace("pre diialysis", "pre dialysis")
    text = text.replace("proteinurea", "proteinuria")
    text = text.replace("bengin", "benign")
    return text if text else pd.NA


def parse_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    text = re.sub(r"[%$]+$", "", text)
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def parse_fraction_or_percent(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    has_percent = "%" in text
    number = parse_number(text)
    if number is None:
        return None
    return number / 100 if has_percent else number


def format_number(value: float | int | None) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def normalize_glomeruli_columns(df: pd.DataFrame) -> None:
    total_column = "donor_Number of Glomeruli"
    sclerosed_column = "donor_# of globally sclerosed glomeruli"
    percent_column = "donor_% of globally sclerosed glomeruli"

    for index, row in df.iterrows():
        total = parse_number(row[total_column])
        sclerosed = parse_number(row[sclerosed_column])
        source_percent = parse_fraction_or_percent(row[percent_column])

        if total is None or sclerosed is None or total <= 0:
            continue

        # Some count cells are formatted as 0.01/0.04 while the source percent
        # makes clear they mean 1/4 globally sclerosed glomeruli.
        if 0 < sclerosed < 1 and source_percent is not None:
            inferred_count = round(source_percent * total)
            if inferred_count >= 1:
                inferred_percent = round(inferred_count / total, 2)
                if abs(inferred_percent - round(source_percent, 2)) <= 0.01:
                    sclerosed = inferred_count

        df.at[index, total_column] = format_number(total)
        df.at[index, sclerosed_column] = format_number(sclerosed)
        df.at[index, percent_column] = format_number(round(sclerosed / total, 2))


def fill_pathology_defaults(df: pd.DataFrame) -> None:
    arteriolar_hyalinosis_column = "donor_Arterioral hylinosis"
    df[arteriolar_hyalinosis_column] = df[arteriolar_hyalinosis_column].fillna("None")
    df[arteriolar_hyalinosis_column] = df[arteriolar_hyalinosis_column].replace(
        {
            "none": "None",
            "Mild to Moderate (1 arteriole)": "Mild to moderate (1 arteriole)",
            "mild to moderate (1 arteriole)": "Mild to moderate (1 arteriole)",
        }
    )


def require_columns(df: pd.DataFrame, columns: list[str], sheet_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{sheet_name} is missing required columns: {missing}")


def build_recipient_race_ethnicity(df: pd.DataFrame) -> pd.Series:
    race = df["Race"].map(normalize_text)
    ethnicity = df["Ethnicity"].map(normalize_text)

    def combine(values: tuple[object, object]) -> object:
        race_value, ethnicity_value = values
        parts = [
            str(part)
            for part in (race_value, ethnicity_value)
            if not pd.isna(part) and str(part).strip()
        ]
        return "/".join(parts) if parts else pd.NA

    return pd.Series(zip(race, ethnicity), index=df.index).map(combine)


def normalize_donor_cause_of_death(value: object) -> object:
    text = normalize_text(value)
    if pd.isna(text):
        return pd.NA
    if re.search(r"\b(stroke|cva)\b", str(text), re.IGNORECASE):
        return "Stroke"
    return "Other"


def main() -> None:
    args = parse_args()

    donor_raw = pd.read_excel(
        args.input_xlsx, sheet_name=args.donor_sheet, keep_default_na=False
    )
    recipient_raw = pd.read_excel(
        args.input_xlsx, sheet_name=args.recipient_sheet, keep_default_na=False
    )

    require_columns(donor_raw, list(DONOR_COLUMN_MAP), args.donor_sheet)
    require_columns(
        recipient_raw,
        list(RECIPIENT_COLUMN_MAP) + ["Race", "Ethnicity"],
        args.recipient_sheet,
    )

    donor = donor_raw[list(DONOR_COLUMN_MAP)].rename(columns=DONOR_COLUMN_MAP)
    donor["donor_Cause of Death"] = donor["donor_Cause of Death"].map(
        normalize_donor_cause_of_death
    )
    recipient = recipient_raw[list(RECIPIENT_COLUMN_MAP)].rename(
        columns=RECIPIENT_COLUMN_MAP
    )
    recipient.insert(
        recipient.columns.get_loc("recipient_BMI"),
        "recipient_Race/Ethnicity",
        build_recipient_race_ethnicity(recipient_raw),
    )

    donor["sample_id"] = donor["sample_id"].map(normalize_case_id)
    recipient["sample_id"] = recipient["sample_id"].map(normalize_case_id)

    donor = donor[donor["sample_id"] != ""].copy()
    recipient = recipient[recipient["sample_id"] != ""].copy()

    if donor["sample_id"].duplicated().any():
        duplicates = donor.loc[donor["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate donor sample_id values: {duplicates}")
    if recipient["sample_id"].duplicated().any():
        duplicates = recipient.loc[
            recipient["sample_id"].duplicated(), "sample_id"
        ].tolist()
        raise ValueError(f"Duplicate recipient sample_id values: {duplicates}")

    merged = donor.merge(
        recipient,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    for column in merged.columns:
        if merged[column].dtype == "object":
            merged[column] = merged[column].map(normalize_text)

    fill_pathology_defaults(merged)
    normalize_glomeruli_columns(merged)

    merged = merged.sort_values(
        by="sample_id", key=lambda series: series.str.extract(r"(\d+)").astype(int)[0]
    )
    merged.to_csv(args.output_csv, index=False)

    donor_only = sorted(set(donor["sample_id"]) - set(recipient["sample_id"]))
    recipient_only = sorted(set(recipient["sample_id"]) - set(donor["sample_id"]))
    print(f"Saved {args.output_csv} shape={merged.shape}")
    print(f"Donor rows: {len(donor)} Recipient rows: {len(recipient)}")
    print(f"Matched rows: {len(merged)}")
    print(f"Donor-only IDs: {donor_only}")
    print(f"Recipient-only IDs: {recipient_only}")


if __name__ == "__main__":
    main()
