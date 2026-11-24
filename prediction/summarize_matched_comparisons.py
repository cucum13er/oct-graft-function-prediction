"""Reproduce the matched ROC-AUC comparisons from Supplementary Section 2.2."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_FOLDERS = {
    "Logistic regression": "logistic_corr_auc_quota_grid",
    "Random forest": "random_forest_corr_auc_quota_grid",
    "Extra Trees": "extra_trees_corr_auc_quota_grid",
    "XGBoost": "xgboost_corr_auc_quota_grid",
    "CatBoost": "catboost_corr_auc_quota_grid",
    "TabPFN": "tabpfn_corr_auc_quota_grid_more",
}
COMPARISONS = {
    52: ("clinical_recipient", "clinical_recipient_oct", ("donor", "recipient"), "oct", ("2", "3", "4")),
    33: ("clinical_oct", "clinical_biopsy_oct", ("donor", "oct"), "biopsy", ("4", "all")),
}


def normalize_parameters(value):
    parameters = json.loads(value)
    if "l1_ratio" in parameters and "penalty" not in parameters:
        ratio = parameters.pop("l1_ratio")
        parameters["penalty"] = {0: "l2", 1: "l1"}[ratio]
    if "C" in parameters:
        parameters["C"] = float(parameters["C"])
    return json.dumps(parameters, sort_keys=True)


def paired_runs(summary_path, cohort):
    baseline, addition, shared, added_group, added_quotas = COMPARISONS[cohort]
    quota_column = f"{added_group}_k_grid"
    shared_columns = [f"{group}_k_grid" for group in shared]
    matching_columns = ["model_parameters", *shared_columns]
    dtype = {column: str for column in [quota_column, *shared_columns]}
    runs = pd.read_csv(summary_path, dtype=dtype)
    runs = runs.loc[runs["status"].eq("ok") & runs["n_cases"].eq(cohort)].copy()
    runs["model_parameters"] = runs["param_json_grid"].map(normalize_parameters)
    if "logistic" in summary_path.parent.name:
        allowed_c = {0.01, 0.1, 1, 10}
        runs = runs.loc[runs["model_parameters"].map(lambda value: json.loads(value)["C"] in allowed_c)]
    original = runs.loc[runs["feature_group"].eq(baseline), [*matching_columns, "auc_roc"]]
    extended = runs.loc[
        runs["feature_group"].eq(addition) & runs[quota_column].isin(added_quotas),
        [*matching_columns, quota_column, "auc_roc"],
    ]
    if original.empty or extended.empty or extended.duplicated([*matching_columns, quota_column]).any():
        raise ValueError(f"Missing or repeated comparison settings in {summary_path}")
    pairs = extended.merge(original, on=matching_columns, suffixes=("_with_group", "_baseline"),
                           how="left", validate="many_to_one")
    if pairs["auc_roc_baseline"].isna().any() or len(pairs) != len(original) * len(added_quotas):
        raise ValueError(f"Incomplete matched comparison grid in {summary_path}")
    pairs["delta_auc"] = pairs["auc_roc_with_group"] - pairs["auc_roc_baseline"]
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-52", type=Path, default=ROOT / "results/models/52")
    parser.add_argument("--results-33", type=Path, default=ROOT / "results/models/33")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/matched_comparisons")
    args = parser.parse_args()
    summaries = []
    all_pairs = []
    for cohort, result_dir in ((52, args.results_52), (33, args.results_33)):
        for name, folder in MODEL_FOLDERS.items():
            pairs = paired_runs(result_dir / folder / "quota_grid_summary.csv", cohort)
            pairs.insert(0, "classifier", name)
            pairs.insert(0, "kidneys", cohort)
            all_pairs.append(pairs)
            delta = pairs["delta_auc"]
            same = np.isclose(delta, 0, atol=1e-12, rtol=0)
            summaries.append({
                "kidneys": cohort, "classifier": name, "grid_size": pairs["model_parameters"].nunique(),
                "pairs": len(pairs), "higher": int(((delta > 0) & ~same).sum()),
                "same": int(same.sum()), "lower": int(((delta < 0) & ~same).sum()),
                "median_delta_auc": float(delta.median()),
            })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out_dir / "paired_auc_summary.csv", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(args.out_dir / "paired_auc_runs.csv", index=False)


if __name__ == "__main__":
    main()
