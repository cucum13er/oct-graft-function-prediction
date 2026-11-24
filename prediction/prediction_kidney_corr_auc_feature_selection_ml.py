import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut

import prediction_kidney_corr_auc_feature_selection as base
from ml_model_factory import (
    SUPPORTED_ML_METHODS,
    create_model,
    merged_model_params,
    parse_model_params,
    predict_positive_probability,
)


DEFAULT_OUT_DIR_BASE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "corr_auc_feature_selection_ml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run corr_auc feature selection with selectable ML classifiers."
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ml-method", default="logistic", choices=SUPPORTED_ML_METHODS)
    parser.add_argument(
        "--model-params",
        default="",
        help="Optional JSON object overriding model defaults, e.g. '{\"max_depth\": 2}'.",
    )
    parser.add_argument("--donor-k", default="all")
    parser.add_argument("--biopsy-k", default="all")
    parser.add_argument("--recipient-k", default="all")
    parser.add_argument("--oct-k", default="3")
    parser.add_argument("--corr-threshold", type=float, default=0.80)
    parser.add_argument("--one-hot", default="false", choices=["true", "false"])
    parser.add_argument("--threshold-mode", default="fixed", choices=["fixed", "youden"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--normalization", default="zscore", choices=base.NORMALIZATION_MODES)
    parser.add_argument("--cases", default="1-172")
    parser.add_argument("--cohort", default="biopsy", choices=["both", "all", "biopsy"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--flat-run-dir", default="false", choices=["true", "false"])
    parser.add_argument("--save-model-input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def loocv_corr_auc_ml_predict(
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
    ml_method,
    model_params,
):
    loo = LeaveOneOut()
    p1 = np.zeros(len(y), dtype=float)
    selected_counts = pd.Series(dtype=int)
    fold_rows = []
    prune_rows = []
    rank_rows = []
    coef_rows = []
    raw_block_map = base.feature_to_block_map(df)
    active_blocks = base.group_blocks(feature_group)

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_df), start=1):
        X_train = X_df.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X_df.iloc[test_idx]
        test_case = df.iloc[test_idx[0]][base.ID_COL]

        preprocessor = base.make_preprocessor(feature_names, normalization, one_hot=one_hot)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        model_feature_names = base.transformed_feature_names(preprocessor)
        processed_block_map = {
            feature: base.transformed_feature_block(feature, raw_block_map)
            for feature in model_feature_names
        }
        feature_family_map = {
            feature: base.transformed_feature_family(feature, feature_names, one_hot)
            for feature in model_feature_names
        }

        selected_features = []
        for block in active_blocks:
            block_features = [
                feature for feature in model_feature_names if processed_block_map.get(feature) == block
            ]
            block_idx = [model_feature_names.index(feature) for feature in block_features]
            selected, retained_after_pruning, prune_df, rank_df = base.corr_prune_auc_select(
                X_train_processed[:, block_idx],
                y_train,
                block_features,
                block=block,
                k=block_k[block],
                corr_threshold=corr_threshold,
                feature_family_map=feature_family_map,
            )
            selected_features.extend(selected)

            if not rank_df.empty:
                rank_df = rank_df.copy()
                rank_df.insert(0, "fold", fold)
                rank_df.insert(1, "heldout_case", test_case)
                rank_rows.extend(rank_df.to_dict("records"))
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
        model = create_model(
            ml_method,
            random_state=random_state,
            max_iter=max_iter,
            y_train=y_train,
            override_params=model_params,
        )
        model.fit(X_train_processed[:, selected_idx], y_train)
        p1[test_idx[0]] = predict_positive_probability(model, X_test_processed[:, selected_idx])[0]

        if ml_method == "logistic" and hasattr(model, "coef_"):
            for feature, coef in zip(selected_features, model.coef_[0]):
                coef_rows.append(
                    {
                        "fold": fold,
                        "heldout_case": test_case,
                        "block": processed_block_map.get(feature, "unknown"),
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
    )


def run_experiment(exp, datasets, args, block_k, model_params):
    df, n_excluded_low_glom = base.prepare_experiment_df(
        datasets[exp["dataset"]],
        exp["dataset"],
    )
    features = base.feature_columns(df, exp["feature_group"])
    run_name = exp["name"]
    run_dir = Path(args.out_dir) if args.flat_run_dir == "true" else Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_common = {
        "run_name": run_name,
        "dataset": exp["dataset"],
        "input_csv": str(base.DATASETS[exp["dataset"]]),
        "feature_group": exp["feature_group"],
        "normalization": args.normalization,
        "one_hot": args.one_hot,
        "selector": "corr_auc",
        "ml_method": args.ml_method,
        "model_params": json.dumps(merged_model_params(args.ml_method, model_params), sort_keys=True),
        "corr_threshold": args.corr_threshold,
        "donor_k": "all" if block_k["donor_clinical"] is None else block_k["donor_clinical"],
        "biopsy_k": "all" if block_k["biopsy"] is None else block_k["biopsy"],
        "recipient_k": "all" if block_k["recipient_clinical"] is None else block_k["recipient_clinical"],
        "oct_k": "all" if block_k["oct"] is None else block_k["oct"],
        "case_filter": args.cases,
        "min_valid_glom": base.MIN_VALID_GLOM if exp["dataset"] == "complete_biopsy" else None,
        "n_excluded_low_glom": n_excluded_low_glom,
    }

    if len(df) < 3 or df[base.LABEL_COL].nunique() < 2:
        metrics = {
            **metrics_common,
            "status": "skipped",
            "skip_reason": "filtered dataset has fewer than 3 cases or only one label class",
            "n_cases": int(len(df)),
            "n_features_input": int(len(features)),
            "n_features_model": 0,
            "n_positive_label1": int(df[base.LABEL_COL].sum()) if len(df) else 0,
            "n_negative_label0": int((df[base.LABEL_COL] == 0).sum()) if len(df) else 0,
        }
        pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
        with open(run_dir / "metrics_and_config.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics

    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[base.LABEL_COL].to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{run_name}: need both classes for ROC/AUC")

    if args.save_model_input:
        base.save_preprocessed_model_input(
            run_dir,
            df,
            X_df,
            y,
            features,
            args.normalization,
            one_hot=args.one_hot,
        )

    p1, selected_counts, fold_selections, pruning_decisions, fold_auc_ranks, coefficients = (
        loocv_corr_auc_ml_predict(
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
            ml_method=args.ml_method,
            model_params=model_params,
        )
    )
    yhat, threshold_metrics, fpr, tpr, roc_thr, youden_j = base.threshold_predictions(
        y,
        p1,
        args.threshold_mode,
        args.fixed_threshold,
    )
    pr = base.pr_outputs(y, p1)
    auc = float(roc_auc_score(y, p1))

    predictions = df[[base.ID_COL, base.LABEL_COL]].copy()
    predictions["p_label1_loocv"] = p1
    predictions["pred_loocv"] = yhat
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr, "youden_j": youden_j}).to_csv(
        run_dir / "roc_curve_points.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "recall": pr["recall"],
            "precision": pr["precision"],
            "threshold": np.r_[pr["threshold"], np.nan],
            "f1": np.r_[pr["f1"], np.nan],
        }
    ).to_csv(run_dir / "pr_curve_points.csv", index=False)

    preprocessor = base.make_preprocessor(features, args.normalization, one_hot=args.one_hot)
    X_processed = preprocessor.fit_transform(X_df)
    model_feature_names = base.transformed_feature_names(preprocessor)
    raw_block_map = base.feature_to_block_map(df)
    full_auc = base.univariate_auc_table(
        X_processed,
        y,
        model_feature_names,
        {
            feature: base.transformed_feature_block(feature, raw_block_map)
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
    stability = base.feature_stability(fold_auc_ranks, coefficients, len(y))
    stability.to_csv(run_dir / "feature_stability.csv", index=False)
    selected_features = stability[stability["selection_count"] > 0]["feature"].tolist()
    pd.DataFrame({"feature": selected_features}).to_csv(run_dir / "selected_features.csv", index=False)

    base.save_plots(run_dir, run_name, y, p1, fpr, tpr, pr, args.skip_plots)

    metrics = {
        **metrics_common,
        "status": "ok",
        "n_cases": int(len(y)),
        "n_features_input": int(len(features)),
        "n_features_model": float(fold_selections.groupby("fold")["n_selected"].sum().mean()),
        "n_unique_features_selected": int(len(selected_features)),
        "n_positive_label1": int(y.sum()),
        "n_negative_label0": int((y == 0).sum()),
        "auc_roc": auc,
        "average_precision": float(average_precision_score(y, p1)),
        "no_skill_positive_rate": pr["positive_rate"],
        "max_f1": pr["max_f1"],
        "max_f1_threshold": pr["max_f1_threshold"],
        "raw_features": features,
        "features": selected_features,
        **threshold_metrics,
    }
    with open(run_dir / "metrics_and_config.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "features"}]).to_csv(
        run_dir / "metrics.csv",
        index=False,
    )
    return metrics


def main():
    args = parse_args()
    args.one_hot = args.one_hot == "true"
    model_params = parse_model_params(args.model_params)
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR_BASE.with_name(
            f"{DEFAULT_OUT_DIR_BASE.name}_{args.ml_method}_{args.threshold_mode}_{args.normalization}"
        )
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    block_k = {
        "donor_clinical": base.parse_block_k(args.donor_k, "--donor-k"),
        "biopsy": base.parse_block_k(args.biopsy_k, "--biopsy-k"),
        "recipient_clinical": base.parse_block_k(args.recipient_k, "--recipient-k"),
        "oct": base.parse_block_k(args.oct_k, "--oct-k"),
    }
    selected_cases = base.parse_case_selection(args.cases)
    specs = base.filter_run_name(
        base.filter_experiment_specs(base.experiment_specs(), args.cohort),
        args.run_name,
    )
    required = {spec["dataset"] for spec in specs}
    datasets = {
        name: base.read_dataset(path, selected_cases=selected_cases)
        for name, path in base.DATASETS.items()
        if name in required
    }
    print(f"Case filter: {args.cases}")
    print(f"Normalization: {args.normalization}")
    print(f"Selector: corr_auc; corr_threshold={args.corr_threshold}; block_k={block_k}")
    print(f"ML method: {args.ml_method}; params={merged_model_params(args.ml_method, model_params)}")
    for name, df in datasets.items():
        print(
            f"{name}: {len(df)} cases, "
            f"label counts={df[base.LABEL_COL].value_counts().sort_index().to_dict()}"
        )

    rows = []
    if args.flat_run_dir == "true" and len(specs) != 1:
        raise ValueError("--flat-run-dir true requires --run-name to select exactly one experiment")
    print(f"Cohort filter: {args.cohort}; running {len(specs)} experiment specs")

    for exp in specs:
        rows.append(run_experiment(exp, datasets, args, block_k, model_params))

    summary = pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in rows])
    summary = summary.sort_values(["n_cases", "run_name"], ascending=[False, True])
    summary_path = base.update_summary(args.out_dir, summary)
    print("Updated summary:", summary_path)
    display_cols = [
        "run_name",
        "ml_method",
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

