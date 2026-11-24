"""LOOCV logistic quota experiment for the 18-case OCT spatial cohort."""

import argparse
import itertools
import json
import os
from collections import Counter
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler

from build_kidney_features_152to172_with_recipient import DONOR_EXTRA_COLS
from prediction_kidney_corr_auc_feature_selection import (
    CLINICAL_BASE_COLS,
    RECIPIENT_COLS,
    corr_prune_auc_select,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/kidney_features_merged_152to172_canonical_four_groups.csv"
DEFAULT_OUTPUT = ROOT / "results" / "spatial_12"
ID_COL = "case_num"
LABEL_COL = "IGF(0)/DGF(1)"
LEGACY_DONOR_COLS = [
    "age", "init_creatinine", "term_creatinine", "kdpi",
    "cold_ischemia_min", "sex_male", "donor_dcd", "race_code",
]
GROUPS = ("donor", "recipient", "morphology", "spatial")
COMBINATIONS = tuple(
    combination
    for size in range(1, len(GROUPS) + 1)
    for combination in itertools.combinations(GROUPS, size)
)
DEFAULT_QUOTAS = {
    "donor": (3, 4, 6, "all"),
    "recipient": (4, 6, "all"),
    "morphology": (1, 2, 3, 4, "all"),
    "spatial": (1, 2, 3, "all"),
}
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def feature_groups(df, exclude_recipient_rrt=False):
    canonical_present = [col for col in CLINICAL_BASE_COLS if col in df]
    if len(canonical_present) == len(CLINICAL_BASE_COLS):
        donor_cols = CLINICAL_BASE_COLS
    elif len(canonical_present) <= len(DONOR_EXTRA_COLS) + 1:
        donor_cols = LEGACY_DONOR_COLS + [col for col in DONOR_EXTRA_COLS if col in df]
    else:
        raise ValueError(f"Partial canonical donor group: {canonical_present}")
    groups = {
        "donor": donor_cols,
        "recipient": [col for col in RECIPIENT_COLS
                      if not (exclude_recipient_rrt and col == "recipient_rrt_modality")],
        "morphology": [c for c in df if c.startswith(("minor_", "ecc_", "sol_"))],
        "spatial": [c for c in df if c.startswith(("anterior_", "posterior_"))],
    }
    for group, cols in groups.items():
        missing = [c for c in cols if c not in df]
        if missing or not cols:
            raise ValueError(f"{group}: missing columns {missing}; found {len(cols)} columns")
        nonnumeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if nonnumeric:
            raise ValueError(f"{group}: nonnumeric columns {nonnumeric}")
        empty = [c for c in cols if df[c].notna().sum() == 0]
        if empty:
            raise ValueError(f"{group}: entirely missing columns {empty}")
    return groups


def prepare_folds(df, groups, y, corr_threshold):
    feature_names = [c for group in GROUPS for c in groups[group]]
    folds = []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(df), start=1):
        train = df.iloc[train_idx][feature_names]
        test = df.iloc[test_idx][feature_names]
        medians = train.median(numeric_only=True).fillna(0)
        x_train = train.fillna(medians).fillna(0)
        x_test = test.fillna(medians).fillna(0)
        scaler = StandardScaler()
        scaled_train = pd.DataFrame(scaler.fit_transform(x_train), columns=feature_names)
        scaled_test = pd.DataFrame(scaler.transform(x_test), columns=feature_names)
        rankings = {}
        for group in GROUPS:
            cols = groups[group]
            selected, _, _, _ = corr_prune_auc_select(
                scaled_train[cols].to_numpy(),
                y[train_idx],
                cols,
                block=group,
                k=None,
                corr_threshold=corr_threshold,
                feature_family_map={c: c for c in cols},
            )
            rankings[group] = selected
        folds.append((fold, train_idx, test_idx[0], scaled_train, scaled_test, rankings))
    return folds


def score_run(df, y, folds, combination, quotas, args, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.zeros(len(df), dtype=float)
    selections = []
    frequency = Counter()
    for fold, train_idx, test_idx, x_train, x_test, rankings in folds:
        selected = []
        for group in combination:
            quota = quotas[group]
            current = rankings[group] if quota == "all" else rankings[group][:quota]
            selected.extend(current)
            frequency.update(current)
            selections.append({
                "fold": fold,
                "heldout_case": df.iloc[test_idx][ID_COL],
                "group": group,
                "quota": quota,
                "n_selected": len(current),
                "selected_features": ";".join(current),
            })
        if not selected:
            raise ValueError(f"No features selected in fold {fold}")
        model = LogisticRegression(
            C=args.C,
            penalty=args.penalty,
            class_weight=args.class_weight,
            solver="liblinear",
            max_iter=5000,
            random_state=0,
        )
        model.fit(x_train[selected], y[train_idx])
        probabilities[test_idx] = model.predict_proba(x_test[selected])[0, 1]

    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    fpr, tpr, roc_threshold = roc_curve(y, probabilities)
    precision, recall, pr_threshold = precision_recall_curve(y, probabilities)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    metrics = {
        "combination": "+".join(combination),
        **{f"{group}_k": quotas[group] if group in combination else 0 for group in GROUPS},
        "n_cases": len(df),
        "n_dgf": int(y.sum()),
        "n_igf": int((y == 0).sum()),
        "C": args.C,
        "penalty": args.penalty,
        "class_weight": args.class_weight,
        "corr_threshold": args.corr_threshold,
        "threshold": 0.5,
        "auc_roc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predicted)),
        "sensitivity_label1": float(tp / (tp + fn)),
        "specificity_label0": float(tn / (tn + fp)),
        "max_f1": float(f1.max()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "run_dir": str(run_dir),
    }
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame({
        ID_COL: df[ID_COL], LABEL_COL: y,
        "p_label1_loocv": probabilities,
        "pred_loocv": predicted,
    }).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_threshold}).to_csv(
        run_dir / "roc_curve_points.csv", index=False
    )
    pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": np.r_[pr_threshold, np.nan],
        "f1": np.r_[f1, np.nan],
    }).to_csv(run_dir / "pr_curve_points.csv", index=False)
    pd.DataFrame(selections).to_csv(run_dir / "fold_selections.csv", index=False)
    pd.DataFrame([
        {"feature": feature, "selection_count": count, "selection_frequency": count / len(df)}
        for feature, count in frequency.most_common()
    ]).to_csv(run_dir / "feature_selection_frequency.csv", index=False)
    return metrics


def plot_best(best, out_dir):
    for _, row in best.iterrows():
        run_dir = Path(row["run_dir"])
        roc = pd.read_csv(run_dir / "roc_curve_points.csv")
        pr = pd.read_csv(run_dir / "pr_curve_points.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(roc.fpr, roc.tpr, label=f"AUC {row.auc_roc:.3f}")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="0.6")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[0].legend(loc="lower right")
        axes[1].plot(pr.recall, pr.precision, label=f"AP {row.average_precision:.3f}")
        axes[1].axhline(row.n_dgf / row.n_cases, linestyle="--", color="0.6")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall", xlim=(0, 1), ylim=(0, 1))
        axes[1].legend(loc="lower left")
        fig.suptitle(row.combination)
        fig.tight_layout()
        fig.savefig(out_dir / f"best_{row.combination.replace('+', '_')}_roc_pr.png", dpi=200)
        plt.close(fig)


def spatial_comparison(summary):
    rows = []
    for _, spatial in summary[summary.spatial_k != 0].iterrows():
        base_groups = spatial.combination.split("+")
        base_groups.remove("spatial")
        if not base_groups:
            continue
        base_name = "+".join(base_groups)
        matches = summary[summary.combination == base_name]
        matches = matches[matches.C == spatial.C]
        for group in ("donor", "recipient", "morphology"):
            matches = matches[matches[f"{group}_k"] == spatial[f"{group}_k"]]
        if len(matches) != 1:
            continue
        base = matches.iloc[0]
        rows.append({
            "base_combination": base_name,
            "spatial_combination": spatial.combination,
            "C": spatial.C,
            **{f"{group}_k": spatial[f"{group}_k"] for group in GROUPS},
            "base_auc_roc": base.auc_roc,
            "spatial_auc_roc": spatial.auc_roc,
            "delta_auc_roc": spatial.auc_roc - base.auc_roc,
            "base_average_precision": base.average_precision,
            "spatial_average_precision": spatial.average_precision,
            "delta_average_precision": spatial.average_precision - base.average_precision,
            "base_run_dir": base.run_dir,
            "spatial_run_dir": spatial.run_dir,
        })
    return pd.DataFrame(rows)


def parse_quotas(value):
    tokens = [token.strip().lower() for token in value.split(",")]
    quotas = []
    for token in tokens:
        if token == "all":
            quotas.append("all")
        elif token.isdigit() and int(token) > 0:
            quotas.append(int(token))
        else:
            raise argparse.ArgumentTypeError(f"Invalid quota: {token!r}")
    return tuple(dict.fromkeys(quotas))


def parse_c_grid(value):
    try:
        values = tuple(dict.fromkeys(float(token.strip()) for token in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("C-grid must be comma-separated positive numbers") from exc
    if not values or any(not np.isfinite(c) or c <= 0 for c in values):
        raise argparse.ArgumentTypeError("Every C value must be finite and positive")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    c_options = parser.add_mutually_exclusive_group()
    c_options.add_argument("--C", type=float, help="Run a single C instead of the grid")
    c_options.add_argument("--C-grid", type=parse_c_grid,
                           help="Comma-separated C values (default: 0.01,0.1,1,10)")
    parser.add_argument("--penalty", choices=("l1", "l2"), default="l2")
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument("--exclude-recipient-rrt", action="store_true")
    for group in GROUPS:
        parser.add_argument(f"--{group}-quotas", type=parse_quotas, default=DEFAULT_QUOTAS[group])
    parser.add_argument("--skip-best-plots", action="store_true")
    args = parser.parse_args()
    c_values = (args.C,) if args.C is not None else (
        args.C_grid if args.C_grid is not None else DEFAULT_C_GRID
    )
    args.C = c_values[0]
    if any(not np.isfinite(c) or c <= 0 for c in c_values) or not 0 < args.corr_threshold <= 1:
        parser.error("C must be positive and corr-threshold must be in (0, 1]")
    if not args.input_csv.exists():
        parser.error(f"Input CSV not found: {args.input_csv}; run the build script first")
    args.class_weight = None if args.class_weight == "none" else args.class_weight

    df = pd.read_csv(args.input_csv)
    if args.exclude_recipient_rrt and "recipient_rrt_modality" in df:
        raise ValueError("No-RRT input must not contain recipient_rrt_modality")
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate case IDs in input")
    y = df[LABEL_COL].to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Both 0 and 1 outcomes are required")
    groups = feature_groups(df, exclude_recipient_rrt=args.exclude_recipient_rrt)
    folds = prepare_folds(df, groups, y, args.corr_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"group": group, "feature": col} for group in GROUPS for col in groups[group]]).to_csv(
        args.output_dir / "feature_groups.csv", index=False
    )
    config = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "n_cases": len(df),
        "combinations": ["+".join(c) for c in COMBINATIONS],
        "quotas": {g: list(getattr(args, f"{g}_quotas")) for g in GROUPS},
        "selection": "fold-wise median imputation, z-score, abs Spearman pruning, direction-free AUC rank",
        "corr_threshold": args.corr_threshold,
        "logistic": {"C": args.C if len(c_values) == 1 else list(c_values), "penalty": args.penalty, "solver": "liblinear", "class_weight": args.class_weight, "max_iter": 5000},
        "C_grid": list(c_values),
        "best_plots_skipped": args.skip_best_plots,
        "fixed_threshold": 0.5,
        "excluded": "biopsy, nnd, recipient HLA/cPRA" +
                    (", recipient_rrt_modality" if args.exclude_recipient_rrt else ""),
        "recipient_rrt_removed": args.exclude_recipient_rrt,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    rows = []
    for combination in COMBINATIONS:
        ranges = [getattr(args, f"{group}_quotas") for group in combination]
        for C, values in itertools.product(c_values, itertools.product(*ranges)):
            args.C = C
            quotas = dict(zip(combination, values))
            name = "_".join(f"{group}-{quotas[group]}" for group in combination)
            combination_dir = args.output_dir / "+".join(combination)
            run_dir = (combination_dir / f"C-{C}" / name if len(c_values) > 1
                       else combination_dir / name)
            rows.append(score_run(df, y, folds, combination, quotas, args, run_dir))
        n_runs = len(c_values) * int(np.prod([len(r) for r in ranges]))
        print(f"Completed {'+'.join(combination)} ({n_runs} C/quota configurations)", flush=True)
    summary = pd.DataFrame(rows).sort_values(
        ["combination", "auc_roc", "average_precision"], ascending=[True, False, False]
    )
    summary.to_csv(args.output_dir / "quota_grid_summary.csv", index=False)
    best = summary.drop_duplicates("combination")
    best.to_csv(args.output_dir / "best_by_combination.csv", index=False)
    summary.drop_duplicates(["combination", "C"]).to_csv(
        args.output_dir / "best_by_combination_and_C.csv", index=False
    )
    spatial_comparison(summary).to_csv(args.output_dir / "spatial_increment_matched_quotas.csv", index=False)
    if not args.skip_best_plots:
        plot_best(best, args.output_dir)
    print(best[["combination", "C", "auc_roc", "average_precision", "donor_k", "recipient_k", "morphology_k", "spatial_k"]].to_string(index=False))
    print(f"Saved {len(summary)} configurations to {args.output_dir}")


if __name__ == "__main__":
    main()
