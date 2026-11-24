import argparse
import json
import os
import re
from pathlib import Path


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler


DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DEFAULT_OUT_DIR_BASE = Path(__file__).resolve().parents[1] / "results" / "corr_auc_dim_reduction_feature_selection_with_etiology"

LABEL_COL = "IGF(0)/DGF(1)"
ID_COL = "case_num"
GLOM_RATIO_COL = "donor_pct_glob_scl"
MIN_VALID_GLOM = 35

CLINICAL_BASE_COLS = [
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

BIOPSY_COLS = [
    GLOM_RATIO_COL,
    "donor_num_arteries",
    "donor_ifta_level",
    "donor_vascular_disease_level",
    "donor_arteriolar_hyalinosis_level",
    "donor_cortical_necrosis",
    "donor_fibrin_thrombi",
]

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
    # "recipient_cpra",
    # "recipient_hla_class_i",
    # "recipient_hla_class_ii",
]

# OCT_PREFIXES = ("minor_", "ecc_", "sol_", "nnd_")
OCT_PREFIXES = ("minor_", "ecc_", "sol_")

CONTINUOUS_COLS = {
    "donor_age",
    "donor_init_creatinine",
    "donor_term_creatinine",
    "donor_cold_ischemia_min",
    "donor_kdpi",
    GLOM_RATIO_COL,
    "donor_num_arteries",
    "recipient_age",
    "recipient_bmi",
    "recipient_cpra",
}

BINARY_COLS = {
    "donor_sex_male",
    "donor_dcd",
    "donor_cause_of_death_stroke",
    "donor_htn",
    "donor_cortical_necrosis",
    "donor_fibrin_thrombi",
    "recipient_sex_male",
    "recipient_previous_kidney_txp",
    "recipient_diabetes",
    "recipient_etiology_htn",
    "recipient_etiology_dm",
    "recipient_etiology_pckd",
    "recipient_etiology_gn",
    "recipient_etiology_other",
}

MANUAL_ONE_HOT_FAMILIES = {
    "recipient_etiology": {
        "recipient_etiology_htn",
        "recipient_etiology_dm",
        "recipient_etiology_pckd",
        "recipient_etiology_gn",
        "recipient_etiology_other",
    },
}

NOMINAL_CODE_COLS = {
    "donor_race_code",
    "recipient_race_code",
    "recipient_rrt_modality",
}

ORDINAL_COLS = {
    "donor_ifta_level",
    "donor_vascular_disease_level",
    "donor_arteriolar_hyalinosis_level",
    "recipient_hla_class_i",
    "recipient_hla_class_ii",
}


DATASETS = {
    "complete_all": DEFAULT_DIR / "kidney_features_complete_clinical_oct_all_available.csv",
    "complete_biopsy": DEFAULT_DIR / "kidney_features_complete_clinical_oct_biopsy.csv",
}

FEATURE_GROUPS = [
    "clinical",
    "biopsy",
    "recipient",
    "oct",
    "clinical_recipient",
    "clinical_biopsy",
    "clinical_oct",
    "biopsy_recipient",
    "biopsy_oct",
    "recipient_oct",
    "clinical_biopsy_recipient",
    "clinical_biopsy_oct",
    "clinical_recipient_oct",
    "biopsy_recipient_oct",
    "clinical_biopsy_recipient_oct",
]

NORMALIZATION_MODES = ["zscore", "minmax", "current"]
DIM_REDUCTION_METHODS = ["none", "pca"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run LOOCV logistic-regression experiments for clinical, biopsy, and "
            "OCT feature combinations."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. If omitted, uses "
            "results/corr_auc_dim_reduction_feature_selection_with_etiology_<threshold-mode>_<normalization>."
        ),
    )
    parser.add_argument(
        "--donor-k",
        default="all",
        help="Maximum donor clinical features selected after pruning/ranking. Use 'all' to keep all post-pruning.",
    )
    parser.add_argument(
        "--biopsy-k",
        default="all",
        help="Maximum biopsy features selected after pruning/ranking. Use 'all' to keep all post-pruning.",
    )
    parser.add_argument(
        "--recipient-k",
        default="all",
        help="Maximum recipient clinical features selected after pruning/ranking. Use 'all' to keep all post-pruning.",
    )
    parser.add_argument(
        "--oct-k",
        default="3",
        help="Maximum OCT features selected after pruning/ranking. Use 'all' to keep all post-pruning.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.80,
        help="Absolute Spearman correlation threshold for within-block pruning.",
    )
    parser.add_argument(
        "--one-hot",
        default="false",
        choices=["true", "false"],
        help=(
            "Set true to one-hot encode nominal and ordinal columns. Pruning "
            "skips pairs inside the same one-hot family."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        default="fixed",
        choices=["fixed", "youden"],
        help="Threshold for reported class predictions. AUC/AP use probabilities.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--model-params",
        default="",
        help=(
            "Optional JSON object overriding logistic-regression defaults, "
            "for example '{\"C\": 1, \"penalty\": \"l1\"}'."
        ),
    )
    parser.add_argument(
        "--normalization",
        default="zscore",
        choices=NORMALIZATION_MODES,
        help=(
            "Preprocessing mode. zscore standardizes all model features, minmax "
            "scales all model features to 0-1, and current uses the grouped "
            "clinical scaling rules in this script."
        ),
    )
    parser.add_argument(
        "--dim-reduction",
        default="pca",
        choices=DIM_REDUCTION_METHODS,
        help=(
            "Dimension reduction method applied after corr/AUC feature selection. "
            "Use none to reproduce the non-PCA path."
        ),
    )
    parser.add_argument(
        "--dim-reduction-min-groups",
        type=int,
        default=3,
        help="Minimum number of active feature blocks required before applying dimension reduction.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.90,
        help="Fraction of variance retained by PCA when --dim-reduction pca is active.",
    )
    parser.add_argument(
        "--cases",
        default="1-172",
        help=(
            "Case IDs to keep. Accepts comma lists and ranges such as "
            "'38,46,47-155'. Default keeps KTX1-KTX172."
        ),
    )
    parser.add_argument(
        "--cohort",
        default="biopsy",
        choices=["both", "all", "biopsy"],
        help=(
            "Which complete-data cohort to run: 'all' for all-available OCT "
            "experiments that do not use biopsy features, 'biopsy' for the full "
            "biopsy-compatible seven-experiment grid, or 'both'."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional single experiment run_name to execute after cohort filtering.",
    )
    parser.add_argument(
        "--flat-run-dir",
        default="false",
        choices=["true", "false"],
        help=(
            "Write a single --run-name directly under --out-dir instead of "
            "--out-dir/run_name. Intended for quota-grid wrappers."
        ),
    )
    parser.add_argument(
        "--save-model-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save model_input_full_data_preprocessed.csv inside each experiment "
            "run directory. Use --no-save-model-input to disable."
        ),
    )
    parser.add_argument(
        "--model-input-check-csv",
        default=None,
        help=(
            "Optional path for one standalone preprocessed model-input CSV for "
            "format checking. Relative paths are saved under --out-dir."
        ),
    )
    parser.add_argument(
        "--model-input-check-dataset",
        default="complete_biopsy",
        choices=sorted(DATASETS),
        help="Dataset used with --model-input-check-csv.",
    )
    parser.add_argument(
        "--model-input-check-feature-group",
        default="clinical_biopsy_oct",
        choices=FEATURE_GROUPS,
        help="Feature group used with --model-input-check-csv.",
    )
    parser.add_argument(
        "--model-input-check-only",
        default="false",
        choices=["true", "false"],
        help="Write --model-input-check-csv and exit before running experiments.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def parse_block_k(value, arg_name):
    if str(value).strip().lower() == "all":
        return None
    k = int(value)
    if k < 0:
        raise ValueError(f"{arg_name} must be a nonnegative integer or 'all'")
    return k


def normalize_case_id(value):
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        return ""
    return f"KTX{int(match.group(1))}"


def parse_case_selection(value):
    selected = set()
    if value is None or str(value).strip() == "":
        return selected

    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"\d+\s*-\s*\d+", token):
            start_s, end_s = token.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if start <= 0 or end <= 0:
                raise ValueError("Case numbers must be positive.")
            if end < start:
                raise ValueError(f"Invalid case range: {token}")
            selected.update(f"KTX{i}" for i in range(start, end + 1))
        else:
            case_id = normalize_case_id(token)
            if not case_id:
                raise ValueError(f"Could not parse case token: {token}")
            selected.add(case_id)
    return selected


def case_sort_value(case_id):
    match = re.search(r"(\d+)", str(case_id))
    return int(match.group(1)) if match else 10**9


def parse_model_params(value):
    if value is None or str(value).strip() == "":
        return {}
    try:
        params = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--model-params must be a valid JSON object") from exc
    if not isinstance(params, dict):
        raise ValueError("--model-params must decode to a JSON object")
    return params


def merged_lr_params(model_params=None, max_iter=5000, random_state=0):
    params = {"solver": "liblinear", "class_weight": "balanced"}
    params.update(model_params or {})
    params.setdefault("max_iter", max_iter)
    params.setdefault("random_state", random_state)
    return params


def make_lr(max_iter=5000, random_state=0, model_params=None):
    return LogisticRegression(**merged_lr_params(model_params, max_iter, random_state))


def fixed_scale_values(values, offset=0.0, scale=1.0):
    return (values - offset) / scale


def fixed_scale_pipeline(offset, scale, impute_strategy="median"):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            (
                "scaler",
                FunctionTransformer(
                    fixed_scale_values,
                    kw_args={"offset": offset, "scale": scale},
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocessing_groups(feature_names):
    feature_names = list(feature_names)
    continuous = [
        col
        for col in feature_names
        if col in CONTINUOUS_COLS or col.startswith(OCT_PREFIXES)
    ]
    race_code = [
        col
        for col in feature_names
        if col in NOMINAL_CODE_COLS and col.endswith("race_code")
    ]
    rrt_code = [
        col
        for col in feature_names
        if col in NOMINAL_CODE_COLS and col == "recipient_rrt_modality"
    ]
    hla_percent = [
        col
        for col in feature_names
        if col in ORDINAL_COLS and col in {"recipient_hla_class_i", "recipient_hla_class_ii"}
    ]
    severity_code = [
        col
        for col in feature_names
        if col in ORDINAL_COLS and col not in set(hla_percent)
    ]
    passthrough = [
        col
        for col in feature_names
        if col
        not in set(continuous)
        | set(race_code)
        | set(rrt_code)
        | set(severity_code)
        | set(hla_percent)
    ]
    return continuous, race_code, rrt_code, severity_code, hla_percent, passthrough


def one_hot_groups(feature_names):
    feature_names = list(feature_names)
    one_hot = [col for col in feature_names if col in NOMINAL_CODE_COLS or col in ORDINAL_COLS]
    numeric = [col for col in feature_names if col not in set(one_hot)]
    return numeric, one_hot


def numeric_pipeline(normalization):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if normalization == "zscore":
        steps.append(("scaler", StandardScaler()))
    elif normalization == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif normalization != "current":
        raise ValueError(f"Unknown normalization mode: {normalization}")
    return Pipeline(steps)


def make_preprocessor(feature_names, normalization="current", one_hot=False):
    feature_names = list(feature_names)
    if one_hot:
        numeric, one_hot_cols = one_hot_groups(feature_names)
        transformers = []
        if numeric:
            if normalization == "current":
                continuous, _, _, _, _, passthrough = preprocessing_groups(numeric)
                if continuous:
                    transformers.append(
                        (
                            "continuous",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            continuous,
                        )
                    )
                if passthrough:
                    transformers.append(("binary_ordinal", SimpleImputer(strategy="median"), passthrough))
            else:
                transformers.append(("numeric", numeric_pipeline(normalization), numeric))
        if one_hot_cols:
            transformers.append(
                (
                    "one_hot",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", make_one_hot_encoder()),
                        ]
                    ),
                    one_hot_cols,
                )
            )
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    if normalization == "zscore":
        return ColumnTransformer(
            transformers=[
                (
                    "zscore",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    feature_names,
                )
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    if normalization == "minmax":
        return ColumnTransformer(
            transformers=[
                (
                    "minmax",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", MinMaxScaler()),
                        ]
                    ),
                    feature_names,
                )
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    if normalization != "current":
        raise ValueError(f"Unknown normalization mode: {normalization}")

    continuous, race_code, rrt_code, severity_code, hla_percent, passthrough = (
        preprocessing_groups(feature_names)
    )
    transformers = []
    if continuous:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous,
            )
        )
    if race_code:
        transformers.append(
            (
                "race_code",
                fixed_scale_pipeline(offset=0.0, scale=5.0, impute_strategy="most_frequent"),
                race_code,
            )
        )
    if rrt_code:
        transformers.append(
            (
                "rrt_code",
                fixed_scale_pipeline(offset=0.0, scale=2.0, impute_strategy="most_frequent"),
                rrt_code,
            )
        )
    if severity_code:
        transformers.append(
            (
                "severity_code",
                fixed_scale_pipeline(offset=0.0, scale=3.0),
                severity_code,
            )
        )
    if hla_percent:
        transformers.append(
            (
                "hla_percent",
                fixed_scale_pipeline(offset=0.0, scale=100.0),
                hla_percent,
            )
        )
    if passthrough:
        transformers.append(
            (
                "binary_ordinal",
                SimpleImputer(strategy="median"),
                passthrough,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transformed_feature_names(preprocessor):
    return list(preprocessor.get_feature_names_out())


def fit_preprocessed_model(
    X_df,
    y,
    feature_names,
    max_iter,
    random_state,
    normalization,
    one_hot=False,
):
    preprocessor = make_preprocessor(feature_names, normalization, one_hot=one_hot)
    X_processed = preprocessor.fit_transform(X_df)
    model = make_lr(max_iter=max_iter, random_state=random_state)
    model.fit(X_processed, y)
    return preprocessor, model, X_processed, transformed_feature_names(preprocessor)


def save_preprocessed_model_input(run_dir, df, X_df, y, feature_names, normalization, one_hot=False):
    preprocessor = make_preprocessor(feature_names, normalization, one_hot=one_hot)
    X_processed = preprocessor.fit_transform(X_df)
    model_feature_names = transformed_feature_names(preprocessor)

    model_input = pd.DataFrame(X_processed, columns=model_feature_names)
    model_input.insert(0, LABEL_COL, y)
    model_input.insert(0, ID_COL, df[ID_COL].to_numpy())
    model_input.to_csv(run_dir / "model_input_full_data_preprocessed.csv", index=False)


def save_full_data_dim_reduction_outputs(
    run_dir,
    df,
    X_df,
    y,
    feature_names,
    feature_group,
    normalization,
    one_hot,
    block_k,
    corr_threshold,
    dim_reduction,
    dim_reduction_min_groups,
    pca_variance,
):
    active_blocks = group_blocks(feature_group)
    if not should_apply_dim_reduction(dim_reduction, active_blocks, dim_reduction_min_groups):
        return {
            "dim_reduction_applied_full_data": False,
            "n_features_before_dim_reduction_full_data": np.nan,
            "n_pca_components_full_data": np.nan,
        }

    preprocessor = make_preprocessor(feature_names, normalization, one_hot=one_hot)
    X_processed = preprocessor.fit_transform(X_df)
    model_feature_names = transformed_feature_names(preprocessor)
    selected_features, _, full_rank_df, _ = select_features_on_processed_data(
        X_processed,
        y,
        model_feature_names,
        raw_block_map=feature_to_block_map(df),
        raw_feature_names=feature_names,
        active_blocks=active_blocks,
        block_k=block_k,
        corr_threshold=corr_threshold,
        one_hot=one_hot,
    )
    pd.DataFrame({"feature": selected_features}).to_csv(
        run_dir / "selected_features_before_dim_reduction_full_data.csv",
        index=False,
    )
    if not full_rank_df.empty:
        full_rank_df.to_csv(run_dir / "feature_ranking_before_dim_reduction_full_data.csv", index=False)

    if len(selected_features) <= 1:
        return {
            "dim_reduction_applied_full_data": False,
            "n_features_before_dim_reduction_full_data": int(len(selected_features)),
            "n_pca_components_full_data": int(len(selected_features)),
        }

    selected_idx = [model_feature_names.index(feature) for feature in selected_features]
    X_selected = X_processed[:, selected_idx]
    X_reduced, _, transformer, pca_summary, component_names = fit_transform_dim_reduction(
        X_selected,
        X_selected[:1],
        method=dim_reduction,
        pca_variance=pca_variance,
    )
    if transformer is None:
        return {
            "dim_reduction_applied_full_data": False,
            "n_features_before_dim_reduction_full_data": int(len(selected_features)),
            "n_pca_components_full_data": int(len(selected_features)),
        }

    pca_summary.to_csv(run_dir / "pca_explained_variance_full_data.csv", index=False)
    loading_rows = []
    for component_idx, component_name in enumerate(component_names):
        for feature, loading in zip(selected_features, transformer.components_[component_idx]):
            loading_rows.append(
                {
                    "component": component_name,
                    "component_rank": component_idx + 1,
                    "feature": feature,
                    "loading": float(loading),
                    "abs_loading": float(abs(loading)),
                    "explained_variance_ratio": float(
                        transformer.explained_variance_ratio_[component_idx]
                    ),
                    "cumulative_explained_variance_ratio": float(
                        np.sum(transformer.explained_variance_ratio_[: component_idx + 1])
                    ),
                }
            )
    pd.DataFrame(loading_rows).sort_values(
        ["component_rank", "abs_loading", "feature"],
        ascending=[True, False, True],
    ).to_csv(run_dir / "pca_loadings_full_data.csv", index=False)
    model_input = pd.DataFrame(X_reduced, columns=component_names)
    model_input.insert(0, LABEL_COL, y)
    model_input.insert(0, ID_COL, df[ID_COL].to_numpy())
    model_input.to_csv(run_dir / "model_input_full_data_dim_reduction.csv", index=False)
    return {
        "dim_reduction_applied_full_data": True,
        "n_features_before_dim_reduction_full_data": int(len(selected_features)),
        "n_pca_components_full_data": int(len(component_names)),
    }


def preprocessed_model_input_frame(df, feature_group, normalization, one_hot=False):
    features = feature_columns(df, feature_group)
    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[LABEL_COL].to_numpy(dtype=int)
    preprocessor = make_preprocessor(features, normalization, one_hot=one_hot)
    X_processed = preprocessor.fit_transform(X_df)
    model_feature_names = transformed_feature_names(preprocessor)

    model_input = pd.DataFrame(X_processed, columns=model_feature_names)
    model_input.insert(0, LABEL_COL, y)
    model_input.insert(0, ID_COL, df[ID_COL].to_numpy())
    return model_input


def prepare_experiment_df(df, dataset_name):
    df = df.copy()
    n_excluded_low_glom = 0
    if dataset_name == "complete_biopsy":
        if "donor_num_glom" not in df.columns:
            raise ValueError("complete_biopsy dataset missing donor_num_glom for biopsy adequacy filtering")
        keep = pd.to_numeric(df["donor_num_glom"], errors="coerce") > MIN_VALID_GLOM
        n_excluded_low_glom = int((~keep).sum())
        df = df[keep].copy()
    return df, n_excluded_low_glom


def save_model_input_check_csv(datasets, args):
    if not args.model_input_check_csv:
        return

    df, _ = prepare_experiment_df(
        datasets[args.model_input_check_dataset],
        args.model_input_check_dataset,
    )
    model_input = preprocessed_model_input_frame(
        df,
        args.model_input_check_feature_group,
        args.normalization,
        one_hot=args.one_hot,
    )
    output_path = Path(args.model_input_check_csv)
    if not output_path.is_absolute():
        output_path = Path(args.out_dir) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_input.to_csv(output_path, index=False)
    print("Saved model input check:", output_path, "shape:", model_input.shape)


def read_dataset(path, selected_cases=None):
    df = pd.read_csv(path)
    required = {ID_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df[ID_COL] = df[ID_COL].apply(normalize_case_id)
    if selected_cases:
        df = df[df[ID_COL].isin(selected_cases)].copy()
    df = df[df[LABEL_COL].notna()].copy()
    if {"donor_num_glom", "donor_num_glob_scl"}.issubset(df.columns):
        num_glom = pd.to_numeric(df["donor_num_glom"], errors="coerce")
        num_glob_scl = pd.to_numeric(df["donor_num_glob_scl"], errors="coerce")
        df[GLOM_RATIO_COL] = np.divide(
            num_glob_scl,
            num_glom,
            out=np.full(len(df), np.nan, dtype=float),
            where=num_glom > 0,
        )
    df = df.sort_values(ID_COL, key=lambda s: s.map(case_sort_value)).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def oct_cols(df):
    return [col for col in df.columns if col.startswith(OCT_PREFIXES)]


def existing_cols(df, cols):
    return [col for col in cols if col in df.columns]


def feature_blocks(df):
    return {
        "donor_clinical": existing_cols(df, CLINICAL_BASE_COLS),
        "biopsy": existing_cols(df, BIOPSY_COLS),
        "recipient_clinical": existing_cols(df, RECIPIENT_COLS),
        "oct": oct_cols(df),
    }


def group_blocks(group):
    if group == "clinical":
        return ["donor_clinical"]
    if group == "biopsy":
        return ["biopsy"]
    if group == "recipient":
        return ["recipient_clinical"]
    if group == "oct":
        return ["oct"]
    if group == "clinical_recipient":
        return ["donor_clinical", "recipient_clinical"]
    if group == "clinical_biopsy":
        return ["donor_clinical", "biopsy"]
    if group == "clinical_oct":
        return ["donor_clinical", "oct"]
    if group == "biopsy_recipient":
        return ["biopsy", "recipient_clinical"]
    if group == "biopsy_oct":
        return ["biopsy", "oct"]
    if group == "recipient_oct":
        return ["recipient_clinical", "oct"]
    if group == "clinical_biopsy_recipient":
        return ["donor_clinical", "biopsy", "recipient_clinical"]
    if group == "clinical_biopsy_oct":
        return ["donor_clinical", "biopsy", "oct"]
    if group == "clinical_recipient_oct":
        return ["donor_clinical", "recipient_clinical", "oct"]
    if group == "biopsy_recipient_oct":
        return ["biopsy", "recipient_clinical", "oct"]
    if group == "clinical_biopsy_recipient_oct":
        return ["donor_clinical", "biopsy", "recipient_clinical", "oct"]
    raise ValueError(f"Unknown feature group: {group}")


def feature_columns(df, group):
    blocks = feature_blocks(df)
    cols = []
    for block in group_blocks(group):
        cols.extend(blocks[block])

    missing_all = [col for col in cols if col not in df.columns]
    if missing_all:
        raise ValueError(f"Feature group {group} missing columns: {missing_all}")
    return cols


def feature_to_block_map(df):
    out = {}
    for block, cols in feature_blocks(df).items():
        for col in cols:
            out[col] = block
    return out


def transformed_feature_block(feature, raw_block_map):
    if feature in raw_block_map:
        return raw_block_map[feature]
    matches = [
        (raw_feature, block)
        for raw_feature, block in raw_block_map.items()
        if feature.startswith(f"{raw_feature}_")
    ]
    if not matches:
        return "unknown"
    return max(matches, key=lambda item: len(item[0]))[1]


def transformed_feature_family(feature, raw_features, one_hot):
    for family, members in MANUAL_ONE_HOT_FAMILIES.items():
        if feature in members:
            return family
    if one_hot:
        one_hot_raw = [col for col in raw_features if col in NOMINAL_CODE_COLS or col in ORDINAL_COLS]
        matches = [col for col in one_hot_raw if feature == col or feature.startswith(f"{col}_")]
        if matches:
            return max(matches, key=len)
    return feature


def direction_free_auc(feature_values, y):
    values = np.asarray(feature_values, dtype=float)
    ok = np.isfinite(values) & np.isfinite(y)
    if ok.sum() < 3 or len(np.unique(y[ok])) < 2 or len(np.unique(values[ok])) < 2:
        return 0.5, 0.5
    auc = float(roc_auc_score(y[ok], values[ok]))
    return auc, max(auc, 1.0 - auc)


def univariate_auc_table(X, y, feature_names, block_map):
    rows = []
    for idx, feature in enumerate(feature_names):
        auc, auc_direction_free = direction_free_auc(X[:, idx], y)
        rows.append(
            {
                "block": block_map.get(feature, "unknown"),
                "feature": feature,
                "auc": auc,
                "auc_direction_free": auc_direction_free,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["block", "auc_direction_free", "feature"],
        ascending=[True, False, True],
    )


def corr_prune_auc_select(X, y, features, block, k, corr_threshold, feature_family_map):
    features = list(features)
    if not features or k == 0:
        return [], [], pd.DataFrame(), pd.DataFrame()

    auc_table = univariate_auc_table(
        X,
        y,
        features,
        {feature: block for feature in features},
    )
    score = auc_table.set_index("feature")["auc_direction_free"].to_dict()
    auc_raw = auc_table.set_index("feature")["auc"].to_dict()
    keep = set(features)
    decisions = []
    corr = pd.DataFrame(X, columns=features).corr(method="spearman").abs()

    for i, left in enumerate(features):
        if left not in keep:
            continue
        for right in features[i + 1 :]:
            if right not in keep:
                continue
            rho = corr.loc[left, right]
            if not np.isfinite(rho) or rho < corr_threshold:
                continue
            same_family = (
                feature_family_map.get(left) == feature_family_map.get(right)
                and feature_family_map.get(left) != left
            )
            if same_family:
                decisions.append(
                    {
                        "block": block,
                        "feature_a": left,
                        "feature_b": right,
                        "feature_a_family": feature_family_map.get(left),
                        "feature_b_family": feature_family_map.get(right),
                        "abs_spearman": float(rho),
                        "retained_feature": "",
                        "removed_feature": "",
                        "feature_a_auc_direction_free": score.get(left, np.nan),
                        "feature_b_auc_direction_free": score.get(right, np.nan),
                        "reason": "skip_same_one_hot_family",
                    }
                )
                continue
            left_key = (score.get(left, 0.5), left)
            right_key = (score.get(right, 0.5), right)
            if left_key >= right_key:
                retained, removed = left, right
            else:
                retained, removed = right, left
            keep.remove(removed)
            decisions.append(
                {
                    "block": block,
                    "feature_a": left,
                    "feature_b": right,
                    "feature_a_family": feature_family_map.get(left, left),
                    "feature_b_family": feature_family_map.get(right, right),
                    "abs_spearman": float(rho),
                    "retained_feature": retained,
                    "removed_feature": removed,
                    "feature_a_auc_direction_free": score.get(left, np.nan),
                    "feature_b_auc_direction_free": score.get(right, np.nan),
                    "reason": "corr_prune_lower_auc",
                }
            )
            if removed == left:
                break

    ranked = [feature for feature in auc_table["feature"].tolist() if feature in keep]
    if k is not None:
        ranked = ranked[: min(k, len(ranked))]

    rank_table = auc_table[auc_table["feature"].isin(keep)].copy()
    rank_table["rank_after_pruning"] = rank_table.groupby("block")["auc_direction_free"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    rank_table["selected"] = rank_table["feature"].isin(ranked)
    rank_table["auc"] = rank_table["feature"].map(auc_raw)

    return ranked, sorted(keep), pd.DataFrame(decisions), rank_table


def should_apply_dim_reduction(dim_reduction, active_blocks, min_groups):
    return dim_reduction != "none" and len(active_blocks) >= min_groups


def make_pca_feature_names(n_components):
    return [f"pca_pc{i}" for i in range(1, n_components + 1)]


def fit_transform_dim_reduction(X_train, X_test, method, pca_variance):
    if method == "none":
        return X_train, X_test, None, None, []
    if method != "pca":
        raise ValueError(f"Unsupported dimension reduction method: {method}")
    if X_train.shape[1] <= 1:
        return X_train, X_test, None, None, []

    transformer = PCA(n_components=pca_variance, svd_solver="full")
    X_train_reduced = transformer.fit_transform(X_train)
    X_test_reduced = transformer.transform(X_test)
    component_names = make_pca_feature_names(X_train_reduced.shape[1])
    rows = []
    for i, name in enumerate(component_names):
        rows.append(
            {
                "component": name,
                "explained_variance_ratio": float(transformer.explained_variance_ratio_[i]),
                "cumulative_explained_variance_ratio": float(
                    np.sum(transformer.explained_variance_ratio_[: i + 1])
                ),
            }
        )
    return X_train_reduced, X_test_reduced, transformer, pd.DataFrame(rows), component_names


def select_features_on_processed_data(
    X_processed,
    y,
    model_feature_names,
    raw_block_map,
    raw_feature_names,
    active_blocks,
    block_k,
    corr_threshold,
    one_hot,
):
    processed_block_map = {
        feature: transformed_feature_block(feature, raw_block_map)
        for feature in model_feature_names
    }
    feature_family_map = {
        feature: transformed_feature_family(feature, raw_feature_names, one_hot)
        for feature in model_feature_names
    }
    selected_features = []
    prune_rows = []
    rank_rows = []

    for block in active_blocks:
        block_features = [
            feature for feature in model_feature_names if processed_block_map.get(feature) == block
        ]
        block_idx = [model_feature_names.index(feature) for feature in block_features]
        selected, _, prune_df, rank_df = corr_prune_auc_select(
            X_processed[:, block_idx],
            y,
            block_features,
            block=block,
            k=block_k[block],
            corr_threshold=corr_threshold,
            feature_family_map=feature_family_map,
        )
        selected_features.extend(selected)
        if not prune_df.empty:
            prune_rows.extend(prune_df.to_dict("records"))
        if not rank_df.empty:
            rank_rows.extend(rank_df.to_dict("records"))

    return selected_features, pd.DataFrame(prune_rows), pd.DataFrame(rank_rows), processed_block_map


def loocv_corr_auc_predict(
    df,
    X_df,
    y,
    feature_names,
    feature_group,
    max_iter,
    random_state,
    normalization,
    one_hot,
    block_k,
    corr_threshold,
    dim_reduction,
    dim_reduction_min_groups,
    pca_variance,
    model_params=None,
):
    loo = LeaveOneOut()
    p1 = np.zeros(len(y), dtype=float)
    selected_counts = pd.Series(dtype=int)
    fold_rows = []
    prune_rows = []
    rank_rows = []
    coef_rows = []
    dim_reduction_rows = []
    raw_block_map = feature_to_block_map(df)
    active_blocks = group_blocks(feature_group)
    apply_dim_reduction = should_apply_dim_reduction(
        dim_reduction,
        active_blocks,
        dim_reduction_min_groups,
    )

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_df), start=1):
        X_train = X_df.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X_df.iloc[test_idx]
        test_case = df.iloc[test_idx[0]][ID_COL]

        preprocessor = make_preprocessor(feature_names, normalization, one_hot=one_hot)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        model_feature_names = transformed_feature_names(preprocessor)
        processed_block_map = {
            feature: transformed_feature_block(feature, raw_block_map)
            for feature in model_feature_names
        }
        feature_family_map = {
            feature: transformed_feature_family(feature, feature_names, one_hot)
            for feature in model_feature_names
        }

        selected_features = []
        for block in active_blocks:
            block_features = [
                feature for feature in model_feature_names if processed_block_map.get(feature) == block
            ]
            block_idx = [model_feature_names.index(feature) for feature in block_features]
            selected, retained_after_pruning, prune_df, rank_df = corr_prune_auc_select(
                X_train_processed[:, block_idx],
                y_train,
                block_features,
                block=block,
                k=block_k[block],
                corr_threshold=corr_threshold,
                feature_family_map=feature_family_map,
            )
            selected_features.extend(selected)

            for rank_df_local in [rank_df]:
                if not rank_df_local.empty:
                    rank_df_local = rank_df_local.copy()
                    rank_df_local.insert(0, "fold", fold)
                    rank_df_local.insert(1, "heldout_case", test_case)
                    rank_rows.extend(rank_df_local.to_dict("records"))
            if not prune_df.empty:
                prune_df = prune_df.copy()
                prune_df.insert(0, "fold", fold)
                prune_df.insert(1, "heldout_case", test_case)
                prune_rows.extend(prune_df.to_dict("records"))

            fold_rows.append(
                {
                    "fold": fold,
                    "heldout_case": test_case,
                    "block": block,
                    "n_input_features": len(block_features),
                    "n_retained_after_pruning": len(retained_after_pruning),
                    "k": "all" if block_k[block] is None else block_k[block],
                    "n_selected": len(selected),
                    "selected_features": ";".join(selected),
                }
            )
            for feature in selected:
                selected_counts.loc[feature] = int(selected_counts.get(feature, 0)) + 1

        if not selected_features:
            raise ValueError(f"{feature_group} fold {fold}: no features selected")
        selected_idx = np.array([model_feature_names.index(feature) for feature in selected_features])
        X_train_model = X_train_processed[:, selected_idx]
        X_test_model = X_test_processed[:, selected_idx]
        model_input_features = list(selected_features)
        fold_dim_reduction_applied = False

        if apply_dim_reduction:
            (
                X_train_model,
                X_test_model,
                dim_transformer,
                dim_summary,
                model_input_features,
            ) = fit_transform_dim_reduction(
                X_train_model,
                X_test_model,
                method=dim_reduction,
                pca_variance=pca_variance,
            )
            fold_dim_reduction_applied = dim_transformer is not None
            if dim_summary is not None and not dim_summary.empty:
                dim_summary = dim_summary.copy()
                dim_summary.insert(0, "fold", fold)
                dim_summary.insert(1, "heldout_case", test_case)
                dim_summary.insert(2, "method", dim_reduction)
                dim_summary.insert(3, "n_features_before_dim_reduction", len(selected_features))
                dim_reduction_rows.extend(dim_summary.to_dict("records"))

        model = make_lr(
            max_iter=max_iter,
            random_state=random_state,
            model_params=model_params,
        )
        model.fit(X_train_model, y_train)
        p1[test_idx[0]] = model.predict_proba(X_test_model)[0, 1]

        for feature, coef in zip(model_input_features, model.coef_[0]):
            coef_rows.append(
                {
                    "fold": fold,
                    "heldout_case": test_case,
                    "block": "dim_reduction" if fold_dim_reduction_applied else processed_block_map.get(feature, "unknown"),
                    "feature": feature,
                    "coef": float(coef),
                }
            )

    return (
        p1,
        selected_counts,
        pd.DataFrame(fold_rows),
        pd.DataFrame(prune_rows),
        pd.DataFrame(rank_rows),
        pd.DataFrame(coef_rows),
        pd.DataFrame(dim_reduction_rows),
    )


def feature_stability(rank_df, coef_df, n_folds):
    if rank_df.empty:
        return pd.DataFrame()
    selected_rank = rank_df[rank_df["selected"]].copy()
    rows = []
    for (block, feature), group in rank_df.groupby(["block", "feature"], sort=True):
        selected_group = selected_rank[
            (selected_rank["block"] == block) & (selected_rank["feature"] == feature)
        ]
        coef_group = coef_df[
            (coef_df["block"] == block) & (coef_df["feature"] == feature)
        ] if not coef_df.empty else pd.DataFrame()
        nonzero = coef_group["coef"].abs() > 0 if not coef_group.empty else pd.Series(dtype=bool)
        pos = (coef_group["coef"] > 0).sum() if not coef_group.empty else 0
        neg = (coef_group["coef"] < 0).sum() if not coef_group.empty else 0
        rows.append(
            {
                "block": block,
                "feature": feature,
                "selection_count": int(len(selected_group)),
                "selection_frequency": float(len(selected_group) / n_folds),
                "median_rank_after_pruning": float(group["rank_after_pruning"].median()),
                "rank_iqr": float(group["rank_after_pruning"].quantile(0.75) - group["rank_after_pruning"].quantile(0.25)),
                "median_auc_direction_free": float(group["auc_direction_free"].median()),
                "median_auc": float(group["auc"].median()),
                "nonzero_coef_count": int(nonzero.sum()) if len(nonzero) else 0,
                "coef_positive_count": int(pos),
                "coef_negative_count": int(neg),
                "coef_sign_consistency": float(max(pos, neg) / len(coef_group)) if len(coef_group) else np.nan,
                "median_coef": float(coef_group["coef"].median()) if len(coef_group) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_count", "median_auc_direction_free", "block", "feature"],
        ascending=[False, False, True, True],
    )


def threshold_predictions(y, p1, threshold_mode, fixed_threshold):
    fpr, tpr, roc_thr = roc_curve(y, p1)
    youden_j = tpr - fpr
    youden_threshold = float(roc_thr[np.argmax(youden_j)])

    if threshold_mode == "youden":
        threshold = youden_threshold
    else:
        threshold = fixed_threshold

    yhat = (p1 >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "threshold_mode": threshold_mode,
        "threshold_used": float(threshold),
        "youden_threshold": youden_threshold,
        "accuracy": float(accuracy_score(y, yhat)),
        "sensitivity_label1": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity_label0": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return yhat, metrics, fpr, tpr, roc_thr, youden_j


def pr_outputs(y, p1):
    precision, recall, pr_thr = precision_recall_curve(y, p1)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_idx = int(np.argmax(f1)) if len(f1) else 0
    return {
        "precision": precision,
        "recall": recall,
        "threshold": pr_thr,
        "f1": f1,
        "max_f1": float(f1[best_idx]) if len(f1) else np.nan,
        "max_f1_threshold": float(pr_thr[best_idx]) if len(pr_thr) else np.nan,
        "average_precision": float(average_precision_score(y, p1)),
        "positive_rate": float(np.mean(y)),
    }


def save_plots(run_dir, run_name, y, p1, fpr, tpr, pr, skip_plots):
    if skip_plots:
        return

    fig = plt.figure()
    auc = roc_auc_score(y, p1)
    plt.plot(fpr, tpr, label=f"LOOCV ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(run_name)
    plt.legend()
    plt.grid(True)
    fig.savefig(run_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(run_dir / "roc_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(pr["recall"], pr["precision"], label=f"LOOCV PR (AP={pr['average_precision']:.3f})")
    plt.axhline(pr["positive_rate"], linestyle="--", label=f"No skill ({pr['positive_rate']:.3f})")
    plt.xlabel("Recall for DGF")
    plt.ylabel("Precision for DGF")
    plt.title(run_name)
    plt.legend()
    plt.grid(True)
    fig.savefig(run_dir / "pr_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(run_dir / "pr_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def run_experiment(exp, datasets, args, block_k, model_params):
    df, n_excluded_low_glom = prepare_experiment_df(
        datasets[exp["dataset"]],
        exp["dataset"],
    )

    features = feature_columns(df, exp["feature_group"])
    run_name = exp["name"]
    run_dir = Path(args.out_dir) if args.flat_run_dir == "true" else Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if len(df) < 3 or df[LABEL_COL].nunique() < 2:
        metrics = {
            "run_name": run_name,
            "dataset": exp["dataset"],
            "input_csv": str(DATASETS[exp["dataset"]]),
            "feature_group": exp["feature_group"],
            "normalization": args.normalization,
            "one_hot": args.one_hot,
            "selector": "corr_auc",
            "ml_method": "logistic",
            "model_params": json.dumps(
                merged_lr_params(model_params, args.max_iter, args.random_state),
                sort_keys=True,
            ),
            "corr_threshold": args.corr_threshold,
            "dim_reduction": args.dim_reduction,
            "dim_reduction_min_groups": args.dim_reduction_min_groups,
            "pca_variance": args.pca_variance if args.dim_reduction == "pca" else None,
            "dim_reduction_applied": False,
            "donor_k": "all" if block_k["donor_clinical"] is None else block_k["donor_clinical"],
            "biopsy_k": "all" if block_k["biopsy"] is None else block_k["biopsy"],
            "recipient_k": "all" if block_k["recipient_clinical"] is None else block_k["recipient_clinical"],
            "oct_k": "all" if block_k["oct"] is None else block_k["oct"],
            "case_filter": args.cases,
            "min_valid_glom": MIN_VALID_GLOM if exp["dataset"] == "complete_biopsy" else None,
            "n_excluded_low_glom": n_excluded_low_glom,
            "status": "skipped",
            "skip_reason": "filtered dataset has fewer than 3 cases or only one label class",
            "n_cases": int(len(df)),
            "n_features_input": int(len(features)),
            "n_features_model": 0,
            "n_positive_label1": int(df[LABEL_COL].sum()) if len(df) else 0,
            "n_negative_label0": int((df[LABEL_COL] == 0).sum()) if len(df) else 0,
        }
        pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
        with open(run_dir / "metrics_and_config.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics

    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[LABEL_COL].to_numpy(dtype=int)

    if len(np.unique(y)) != 2:
        raise ValueError(f"{run_name}: need both classes for ROC/AUC")

    if args.save_model_input:
        save_preprocessed_model_input(
            run_dir,
            df,
            X_df,
            y,
            features,
            args.normalization,
            one_hot=args.one_hot,
        )

    (
        p1,
        selected_counts,
        fold_selections,
        pruning_decisions,
        fold_auc_ranks,
        coefficients,
        fold_dim_reduction,
    ) = loocv_corr_auc_predict(
        df,
        X_df,
        y,
        features,
        exp["feature_group"],
        max_iter=args.max_iter,
        random_state=args.random_state,
        normalization=args.normalization,
        one_hot=args.one_hot,
        block_k=block_k,
        corr_threshold=args.corr_threshold,
        dim_reduction=args.dim_reduction,
        dim_reduction_min_groups=args.dim_reduction_min_groups,
        pca_variance=args.pca_variance,
        model_params=model_params,
    )
    yhat, threshold_metrics, fpr, tpr, roc_thr, youden_j = threshold_predictions(
        y,
        p1,
        args.threshold_mode,
        args.fixed_threshold,
    )
    pr = pr_outputs(y, p1)
    auc = float(roc_auc_score(y, p1))

    predictions = df[[ID_COL, LABEL_COL]].copy()
    predictions["p_label1_loocv"] = p1
    predictions["pred_loocv"] = yhat
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thr,
            "youden_j": youden_j,
        }
    ).to_csv(run_dir / "roc_curve_points.csv", index=False)

    pd.DataFrame(
        {
            "recall": pr["recall"],
            "precision": pr["precision"],
            "threshold": np.r_[pr["threshold"], np.nan],
            "f1": np.r_[pr["f1"], np.nan],
        }
    ).to_csv(run_dir / "pr_curve_points.csv", index=False)

    preprocessor = make_preprocessor(features, args.normalization, one_hot=args.one_hot)
    X_processed = preprocessor.fit_transform(X_df)
    model_feature_names = transformed_feature_names(preprocessor)
    raw_block_map = feature_to_block_map(df)
    full_auc = univariate_auc_table(
        X_processed,
        y,
        model_feature_names,
        {
            feature: transformed_feature_block(feature, raw_block_map)
            for feature in model_feature_names
        },
    )
    full_auc["loocv_selected_count"] = full_auc["feature"].map(selected_counts).fillna(0).astype(int)
    full_auc.to_csv(run_dir / "univariate_auc_by_group.csv", index=False)
    full_auc.to_csv(run_dir / "feature_ranking_full_data.csv", index=False)

    fold_selections.to_csv(run_dir / "fold_selections.csv", index=False)
    pruning_decisions.to_csv(run_dir / "correlation_pruning_decisions.csv", index=False)
    fold_auc_ranks.to_csv(run_dir / "fold_auc_ranks.csv", index=False)
    coefficients.to_csv(run_dir / "fold_coefficients.csv", index=False)
    fold_dim_reduction.to_csv(run_dir / "fold_dim_reduction.csv", index=False)
    stability = feature_stability(fold_auc_ranks, coefficients, len(y))
    stability.to_csv(run_dir / "feature_stability.csv", index=False)
    selected_raw_features = stability[stability["selection_count"] > 0]["feature"].tolist()
    pd.DataFrame({"feature": selected_raw_features}).to_csv(
        run_dir / "selected_features_before_dim_reduction.csv",
        index=False,
    )

    dim_reduction_applied = not fold_dim_reduction.empty
    if dim_reduction_applied:
        selected_features = sorted(fold_dim_reduction["component"].unique().tolist())
        pd.DataFrame({"feature": selected_features}).to_csv(run_dir / "selected_features.csv", index=False)
    else:
        selected_features = selected_raw_features
        pd.DataFrame({"feature": selected_features}).to_csv(run_dir / "selected_features.csv", index=False)

    full_data_dim_metrics = save_full_data_dim_reduction_outputs(
        run_dir,
        df,
        X_df,
        y,
        features,
        exp["feature_group"],
        args.normalization,
        args.one_hot,
        block_k,
        args.corr_threshold,
        args.dim_reduction,
        args.dim_reduction_min_groups,
        args.pca_variance,
    )

    save_plots(run_dir, run_name, y, p1, fpr, tpr, pr, args.skip_plots)

    metrics = {
        "run_name": run_name,
        "dataset": exp["dataset"],
        "input_csv": str(DATASETS[exp["dataset"]]),
        "feature_group": exp["feature_group"],
        "normalization": args.normalization,
        "one_hot": args.one_hot,
        "selector": "corr_auc",
        "ml_method": "logistic",
        "model_params": json.dumps(
            merged_lr_params(model_params, args.max_iter, args.random_state),
            sort_keys=True,
        ),
        "corr_threshold": args.corr_threshold,
        "dim_reduction": args.dim_reduction,
        "dim_reduction_min_groups": args.dim_reduction_min_groups,
        "pca_variance": args.pca_variance if args.dim_reduction == "pca" else None,
        "dim_reduction_applied": bool(dim_reduction_applied),
        "donor_k": "all" if block_k["donor_clinical"] is None else block_k["donor_clinical"],
        "biopsy_k": "all" if block_k["biopsy"] is None else block_k["biopsy"],
        "recipient_k": "all" if block_k["recipient_clinical"] is None else block_k["recipient_clinical"],
        "oct_k": "all" if block_k["oct"] is None else block_k["oct"],
        "case_filter": args.cases,
        "min_valid_glom": MIN_VALID_GLOM if exp["dataset"] == "complete_biopsy" else None,
        "n_excluded_low_glom": n_excluded_low_glom,
        "status": "ok",
        "n_cases": int(len(y)),
        "n_features_input": int(len(features)),
        "n_features_before_dim_reduction": float(fold_selections.groupby("fold")["n_selected"].sum().mean()),
        "n_features_model": float(
            fold_dim_reduction.groupby("fold")["component"].nunique().mean()
            if dim_reduction_applied
            else fold_selections.groupby("fold")["n_selected"].sum().mean()
        ),
        "n_pca_components_mean_loocv": float(
            fold_dim_reduction.groupby("fold")["component"].nunique().mean()
            if dim_reduction_applied
            else np.nan
        ),
        "n_unique_features_selected_before_dim_reduction": int(len(selected_raw_features)),
        "n_unique_features_selected": int(len(selected_features)),
        "n_positive_label1": int(y.sum()),
        "n_negative_label0": int((y == 0).sum()),
        "auc_roc": auc,
        "average_precision": pr["average_precision"],
        "no_skill_positive_rate": pr["positive_rate"],
        "max_f1": pr["max_f1"],
        "max_f1_threshold": pr["max_f1_threshold"],
        "raw_features": features,
        "features": selected_features,
        "features_before_dim_reduction": selected_raw_features,
        **full_data_dim_metrics,
        **threshold_metrics,
    }
    with open(run_dir / "metrics_and_config.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "features"}]).to_csv(
        run_dir / "metrics.csv",
        index=False,
    )
    return metrics


def experiment_specs():
    specs = [
        {
            "name": "data_all_donor",
            "dataset": "complete_all",
            "feature_group": "clinical",
            "cohort": "all",
        },
        {
            "name": "data_all_recipient",
            "dataset": "complete_all",
            "feature_group": "recipient",
            "cohort": "all",
        },
        {
            "name": "data_all_oct",
            "dataset": "complete_all",
            "feature_group": "oct",
            "cohort": "all",
        },
        {
            "name": "data_all_donor_recipient",
            "dataset": "complete_all",
            "feature_group": "clinical_recipient",
            "cohort": "all",
        },
        {
            "name": "data_all_donor_oct",
            "dataset": "complete_all",
            "feature_group": "clinical_oct",
            "cohort": "all",
        },
        {
            "name": "data_all_recipient_oct",
            "dataset": "complete_all",
            "feature_group": "recipient_oct",
            "cohort": "all",
        },
        {
            "name": "data_all_donor_recipient_oct",
            "dataset": "complete_all",
            "feature_group": "clinical_recipient_oct",
            "cohort": "all",
        },
        {
            "name": "data_withBiopsy_donor",
            "dataset": "complete_biopsy",
            "feature_group": "clinical",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_biopsy",
            "dataset": "complete_biopsy",
            "feature_group": "biopsy",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_recipient",
            "dataset": "complete_biopsy",
            "feature_group": "recipient",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_oct",
            "dataset": "complete_biopsy",
            "feature_group": "oct",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_donor_recipient",
            "dataset": "complete_biopsy",
            "feature_group": "clinical_recipient",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_donor_biopsy",
            "dataset": "complete_biopsy",
            "feature_group": "clinical_biopsy",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_donor_oct",
            "dataset": "complete_biopsy",
            "feature_group": "clinical_oct",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_biopsy_recipient",
            "dataset": "complete_biopsy",
            "feature_group": "biopsy_recipient",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_biopsy_oct",
            "dataset": "complete_biopsy",
            "feature_group": "biopsy_oct",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_recipient_oct",
            "dataset": "complete_biopsy",
            "feature_group": "recipient_oct",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_donor_biopsy_recipient",
            "dataset": "complete_biopsy",
            "feature_group": "clinical_biopsy_recipient",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_donor_biopsy_oct",
            "dataset": "complete_biopsy",
            "feature_group": "clinical_biopsy_oct",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_donor_recipient_oct",
            "dataset": "complete_biopsy",
            "feature_group": "clinical_recipient_oct",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_biopsy_recipient_oct",
            "dataset": "complete_biopsy",
            "feature_group": "biopsy_recipient_oct",
            "cohort": "biopsy",
        },
        {
            "name": "data_withBiopsy_donor_biopsy_recipient_oct",
            "dataset": "complete_biopsy",
            "feature_group": "clinical_biopsy_recipient_oct",
            "cohort": "biopsy",
        },
    ]
    return specs


def filter_experiment_specs(specs, cohort):
    if cohort == "both":
        return specs
    return [spec for spec in specs if spec["cohort"] == cohort]


def filter_run_name(specs, run_name):
    if not run_name:
        return specs
    filtered = [spec for spec in specs if spec["name"] == run_name]
    if not filtered:
        raise ValueError(f"Unknown run_name after cohort/dim-reduction filtering: {run_name}")
    return filtered


def filter_dim_reduction_specs(specs, dim_reduction, min_groups):
    if dim_reduction == "none":
        return specs
    return [
        spec
        for spec in specs
        if should_apply_dim_reduction(
            dim_reduction,
            group_blocks(spec["feature_group"]),
            min_groups,
        )
    ]


def update_summary(out_dir, summary):
    path = Path(out_dir) / "summary_metrics.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if "run_name" in existing.columns:
            existing = existing[~existing["run_name"].isin(summary["run_name"])].copy()
            combined = pd.concat([existing, summary], ignore_index=True, sort=False)
        else:
            combined = summary.copy()
    else:
        combined = summary.copy()

    sort_cols = [col for col in ["n_cases", "run_name"] if col in combined.columns]
    if sort_cols:
        ascending = [False if col == "n_cases" else True for col in sort_cols]
        combined = combined.sort_values(sort_cols, ascending=ascending)

    combined.to_csv(path, index=False)
    return path


def main():
    args = parse_args()
    args.one_hot = args.one_hot == "true"
    model_params = parse_model_params(args.model_params)
    if args.dim_reduction_min_groups < 1:
        raise ValueError("--dim-reduction-min-groups must be positive")
    if args.dim_reduction == "pca" and not (0.0 < args.pca_variance < 1.0):
        raise ValueError("--pca-variance must be in (0, 1) for PCA")
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR_BASE.with_name(
            f"{DEFAULT_OUT_DIR_BASE.name}_{args.threshold_mode}_{args.normalization}"
        )
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.model_input_check_only == "true" and not args.model_input_check_csv:
        raise ValueError("--model-input-check-only requires --model-input-check-csv")
    block_k = {
        "donor_clinical": parse_block_k(args.donor_k, "--donor-k"),
        "biopsy": parse_block_k(args.biopsy_k, "--biopsy-k"),
        "recipient_clinical": parse_block_k(args.recipient_k, "--recipient-k"),
        "oct": parse_block_k(args.oct_k, "--oct-k"),
    }
    selected_cases = parse_case_selection(args.cases)

    specs = filter_experiment_specs(experiment_specs(), args.cohort)
    specs = filter_dim_reduction_specs(specs, args.dim_reduction, args.dim_reduction_min_groups)
    specs = filter_run_name(specs, args.run_name)
    required = {spec["dataset"] for spec in specs}
    if args.model_input_check_csv:
        required.add(args.model_input_check_dataset)
    datasets = {
        name: read_dataset(path, selected_cases=selected_cases)
        for name, path in DATASETS.items()
        if name in required
    }
    print(f"Case filter: {args.cases}")
    print(f"Normalization: {args.normalization}")
    print(
        "Selector: corr_auc; "
        f"corr_threshold={args.corr_threshold}; block_k={block_k}; "
        f"dim_reduction={args.dim_reduction}; "
        f"dim_reduction_min_groups={args.dim_reduction_min_groups}; "
        f"pca_variance={args.pca_variance}"
    )
    print(f"ML method: logistic; params={merged_lr_params(model_params, args.max_iter, args.random_state)}")
    for name, df in datasets.items():
        print(
            f"{name}: {len(df)} cases, "
            f"label counts={df[LABEL_COL].value_counts().sort_index().to_dict()}"
        )
    save_model_input_check_csv(datasets, args)
    if args.model_input_check_only == "true":
        return

    rows = []
    if args.flat_run_dir == "true" and len(specs) != 1:
        raise ValueError("--flat-run-dir true requires --run-name to select exactly one experiment")
    print(
        f"Cohort filter: {args.cohort}; "
        f"dim-reduction eligible specs: {len(specs)}; running {len(specs)} experiment specs"
    )

    for exp in specs:
        rows.append(run_experiment(exp, datasets, args, block_k, model_params))

    summary = pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in rows])
    summary = summary.sort_values(["n_cases", "run_name"], ascending=[False, True])
    summary_path = update_summary(args.out_dir, summary)
    print("Updated summary:", summary_path)
    display_cols = [
        "run_name",
        "status",
        "n_cases",
        "n_features_model",
        "auc_roc",
        "average_precision",
        "accuracy",
    ]
    display_cols = [col for col in display_cols if col in summary.columns]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
