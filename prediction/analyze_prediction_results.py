"""Validate final ML searches and export manuscript-ready summary tables.

The parameter-space table is built from the PARAM_GRID constants in the model
runner scripts, so changes to an experimental grid propagate automatically.
"""

import argparse
import json
import os
import warnings
from pathlib import Path


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from prediction_kidney_corr_auc_quota_grid_RF import PARAM_GRID as RF_GRID
from prediction_kidney_corr_auc_quota_grid_catboost import PARAM_GRID as CATBOOST_GRID
from prediction_kidney_corr_auc_quota_grid_extra_trees import PARAM_GRID as EXTRA_TREES_GRID
from prediction_kidney_corr_auc_quota_grid_logistic import PARAM_GRID as LOGISTIC_GRID
from prediction_kidney_corr_auc_quota_grid_tabpfn_more import PARAM_GRID as TABPFN_GRID
from prediction_kidney_corr_auc_quota_grid_xgboost import PARAM_GRID as XGBOOST_GRID
from prediction_kidney_corr_auc_dim_reduction_quota_grid_hyper_logistic import (
    PCA_VARIANCE_GRID as LOGISTIC_PCA_VARIANCE_GRID,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "figures" / "ml_results_summary"

METHODS = ["logistic", "random_forest", "extra_trees", "xgboost", "catboost", "tabpfn"]
MODEL_LABELS = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
}
MODEL_TICK_LABELS = {
    "logistic": "Logistic\nregression",
    "random_forest": "Random\nforest",
    "extra_trees": "Extra\nTrees",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
}

GROUPS = [
    "clinical",
    "recipient",
    "oct",
    "clinical_recipient",
    "clinical_oct",
    "recipient_oct",
    "clinical_recipient_oct",
]
GROUP_LABELS = {
    "clinical": "Donor",
    "recipient": "Recipient",
    "oct": "OCT",
    "clinical_recipient": "Donor + recipient",
    "clinical_oct": "Donor + OCT",
    "recipient_oct": "Recipient + OCT",
    "clinical_recipient_oct": "Donor + recipient + OCT",
}

SUMMARY_FILES = {
    "logistic": ROOT
    / "results" / "models" / "52"
    / "logistic_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "random_forest": ROOT
    / "results" / "models" / "52"
    / "random_forest_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "extra_trees": ROOT
    / "results" / "models" / "52"
    / "extra_trees_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "xgboost": ROOT
    / "results" / "models" / "52"
    / "xgboost_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "catboost": ROOT
    / "results" / "models" / "52"
    / "catboost_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "tabpfn": ROOT
    / "results" / "models" / "52"
    / "tabpfn_corr_auc_quota_grid_more"
    / "quota_grid_summary.csv",
}

PARAM_GRIDS = {
    "logistic": LOGISTIC_GRID,
    "random_forest": RF_GRID,
    "extra_trees": EXTRA_TREES_GRID,
    "xgboost": XGBOOST_GRID,
    "catboost": CATBOOST_GRID,
    "tabpfn": TABPFN_GRID,
}

TEAL = "#1CA79A"
MAGENTA = "#A61C4E"
LIGHT_GREY = "#BEC7CF"
GRID_GREY = "#ECEFF2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize final ML grids and create manuscript-ready tables and figures."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-logistic-pca",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add the optional logistic-regression PCA retained-variance row "
            "(None, 85%%, 90%%, 95%%) to the hyperparameter table."
        ),
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip the combined ReportLab PDF while retaining all figures and CSV files.",
    )
    parser.add_argument(
        "--skip-prediction-audit",
        action="store_true",
        help="Do not recompute AUCs from the selected predictions.csv files.",
    )
    return parser.parse_args()


def load_results():
    frames = []
    audit_rows = []
    for method in METHODS:
        path = SUMMARY_FILES[method]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"status", "n_cases", "run_name", "feature_group", "auc_roc", "quota_run_dir"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{path} missing columns: {sorted(missing)}")
        if not frame["status"].eq("ok").all():
            raise ValueError(f"{path} contains non-ok rows")
        if not frame["n_cases"].eq(52).all():
            raise ValueError(f"{path} contains rows outside the 52-case cohort")
        frame = frame.copy()
        frame["ml_method"] = method
        frames.append(frame)
        audit_rows.append(
            {
                "Classifier": MODEL_LABELS[method],
                "Completed runs": len(frame),
                "Parameter sets": frame["param_run"].nunique(),
                "Label 1": int(frame["n_positive_label1"].iloc[0]),
                "Label 0": int(frame["n_negative_label0"].iloc[0]),
            }
        )
    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(audit_rows)


def select_best_rows(all_rows):
    ranked = all_rows.sort_values(
        ["auc_roc", "average_precision", "accuracy", "param_run", "quota_run"],
        ascending=[False, False, False, True, True],
    )
    best = ranked.groupby(["ml_method", "feature_group"], sort=False).head(1).copy()
    method_order = {method: index for index, method in enumerate(METHODS)}
    group_order = {group: index for index, group in enumerate(GROUPS)}
    best["display_order"] = (
        best["ml_method"].map(method_order) * len(GROUPS)
        + best["feature_group"].map(group_order)
    )
    return best.sort_values("display_order")


def verify_predictions(best):
    reference = None
    for _, row in best.iterrows():
        predictions_path = Path(row["quota_run_dir"]) / "predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(predictions_path)
        predictions = pd.read_csv(predictions_path)
        id_col, label_col = predictions.columns[:2]
        predictions = predictions.sort_values(id_col)
        signature = list(zip(predictions[id_col].astype(str), predictions[label_col].astype(int)))
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Different cases or labels in {predictions_path}")
        calculated = roc_auc_score(predictions[label_col], predictions["p_label1_loocv"])
        if abs(calculated - float(row["auc_roc"])) >= 1e-10:
            raise ValueError(f"AUC mismatch in {predictions_path}: {calculated} vs {row['auc_roc']}")


def best_auc_matrix(best):
    matrix = best.pivot(index="feature_group", columns="ml_method", values="auc_roc")
    matrix = matrix.reindex(index=GROUPS, columns=METHODS)
    matrix.index = [GROUP_LABELS[group] for group in GROUPS]
    matrix.columns = [MODEL_LABELS[method] for method in METHODS]
    return matrix


def save_result_tables(all_rows, best, audit, out_dir):
    matrix = best_auc_matrix(best)
    matrix.to_csv(out_dir / "best_roc_auc.csv", float_format="%.9f")

    columns = [
        "ml_method",
        "feature_group",
        "auc_roc",
        "average_precision",
        "accuracy",
        "donor_k",
        "recipient_k",
        "oct_k",
        "n_features_model",
        "model_params",
        "param_run",
        "quota_run",
        "quota_run_dir",
    ]
    best[[column for column in columns if column in best.columns]].to_csv(
        out_dir / "optimal_configurations.csv",
        index=False,
    )
    audit.to_csv(out_dir / "search_completion_audit.csv", index=False)

    common = all_rows.copy()
    for block, allowed in {
        "donor_k": ["6", "8", "all"],
        "recipient_k": ["6", "8", "all"],
        "oct_k": ["2", "3", "4", "all"],
    }.items():
        common = common[common[block].astype(str).isin(allowed)]
    common_matrix = (
        common.groupby(["feature_group", "ml_method"])["auc_roc"]
        .max()
        .unstack()
        .reindex(index=GROUPS, columns=METHODS)
    )
    common_matrix.index = [GROUP_LABELS[group] for group in GROUPS]
    common_matrix.columns = [MODEL_LABELS[method] for method in METHODS]
    common_matrix.to_csv(out_dir / "common_quota_roc_auc.csv", float_format="%.9f")
    return matrix


def display_value(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def search_space_rows(include_logistic_pca=False):
    rows = []
    for method in METHODS:
        for parameter, values in PARAM_GRIDS[method].items():
            rows.append(
                {
                    "Classifier": MODEL_LABELS[method],
                    "Parameter": parameter,
                    "Values": ", ".join(display_value(value) for value in values),
                }
            )
        if method == "logistic" and include_logistic_pca:
            retained_variance = ["None"] + [
                f"{value:.0%}" for value in LOGISTIC_PCA_VARIANCE_GRID
            ]
            rows.append(
                {
                    "Classifier": MODEL_LABELS[method],
                    "Parameter": "PCA retained variance",
                    "Values": ", ".join(retained_variance),
                }
            )
    return pd.DataFrame(rows)


def plot_search_space_table(search_space, out_dir):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    row_height = 0.36
    header_height = 0.48
    title_height = 0.75
    total_height = title_height + header_height + row_height * len(search_space) + 0.35
    fig, ax = plt.subplots(figsize=(13.5, total_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    left, right = 0.025, 0.975
    model_x, parameter_x, value_x = 0.15, 0.47, 0.79
    top = total_height - title_height
    ax.text(left, total_height - 0.18, "Hyperparameter search space", fontsize=23, fontweight="bold", va="top")
    ax.hlines(top, left, right, color="black", lw=1.2)
    header_y = top - header_height / 2
    ax.text(model_x, header_y, "Models", ha="center", va="center", fontweight="bold")
    ax.text(parameter_x, header_y, "Parameter", ha="center", va="center", fontweight="bold")
    ax.text(value_x, header_y, "Range", ha="center", va="center", fontweight="bold")
    current_y = top - header_height
    ax.hlines(current_y, left, right, color="black", lw=0.9)

    for method in METHODS:
        subset = search_space[search_space["Classifier"].eq(MODEL_LABELS[method])]
        group_top = current_y
        for _, row in subset.iterrows():
            y = current_y - row_height / 2
            ax.text(parameter_x, y, row["Parameter"], ha="center", va="center", fontstyle="italic")
            ax.text(value_x, y, row["Values"], ha="center", va="center")
            current_y -= row_height
        model_y = (group_top + current_y) / 2
        ax.text(model_x, model_y, MODEL_LABELS[method], ha="center", va="center", fontweight="bold")
        ax.hlines(current_y, left, right, color="black", lw=0.75)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            out_dir / f"hyperparameter_search_space.{extension}",
            dpi=450,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_oct_addition(best, out_dir):
    pairs = [
        ("clinical", "clinical_oct", "Donor"),
        ("recipient", "recipient_oct", "Recipient"),
        ("clinical_recipient", "clinical_recipient_oct", "Donor + recipient"),
    ]
    delta_rows = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, gridspec_kw={"wspace": 0.18})
    for ax, (baseline_group, oct_group, title), letter in zip(axes, pairs, "abc"):
        for index, method in enumerate(METHODS):
            baseline = float(
                best[
                    best["ml_method"].eq(method) & best["feature_group"].eq(baseline_group)
                ]["auc_roc"].iloc[0]
            )
            with_oct = float(
                best[
                    best["ml_method"].eq(method) & best["feature_group"].eq(oct_group)
                ]["auc_roc"].iloc[0]
            )
            delta_rows.append(
                {
                    "classifier": method,
                    "baseline": baseline_group,
                    "with_oct": oct_group,
                    "baseline_auc": baseline,
                    "oct_auc": with_oct,
                    "delta_auc": with_oct - baseline,
                }
            )
            ax.plot([index, index], [baseline, with_oct], color=LIGHT_GREY, lw=2, zorder=1)
            ax.scatter(index, baseline, color=TEAL, s=38, zorder=2)
            ax.scatter(index, with_oct, color=MAGENTA, s=38, zorder=3)
            ax.annotate(
                f"{baseline:.3f}",
                (index, baseline),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEAL,
            )
            ax.annotate(
                f"{with_oct:.3f}",
                (index, with_oct),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=MAGENTA,
            )
        ax.set_title(f"{letter}  {title} + OCT", loc="left", fontweight="bold", fontsize=12)
        ax.set_xticks(range(len(METHODS)), [MODEL_TICK_LABELS[method] for method in METHODS])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0.35, 0.95)
        ax.set_xlim(-0.5, len(METHODS) - 0.5)
        ax.grid(axis="y", color=GRID_GREY, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("LOOCV ROC-AUC")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=TEAL, label="Without OCT"),
            Line2D([], [], marker="o", ls="", color=MAGENTA, label="With OCT"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.22, wspace=0.18)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"oct_addition_comparison.{extension}", dpi=450, bbox_inches="tight")
    plt.close(fig)
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(out_dir / "oct_auc_changes.csv", index=False)
    return delta


def build_report(matrix, best, audit, search_space, out_dir):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise ImportError("reportlab is required for the combined PDF; use --skip-report to omit it") from exc

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Times-Roman", fontSize=8.5, leading=11)
    heading = ParagraphStyle("heading", parent=styles["Heading1"], fontName="Times-Bold", fontSize=17)

    def paragraph(text):
        return [Paragraph(text, body), Spacer(1, 7)]

    def make_table(rows, widths, repeat_rows=1):
        rendered = [[Paragraph(str(value), body) for value in row] for row in rows]
        table = Table(rendered, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    story = [Paragraph("Machine-learning prediction results", heading), Spacer(1, 8)]
    story += paragraph(
        "All summaries use the 52-case cohort (23 DGF and 29 IGF cases), corr-AUC feature selection, "
        "and pooled LOOCV predictions. Grid maxima are development-stage optimized estimates."
    )
    matrix_rows = [["Feature combination"] + list(matrix.columns)]
    matrix_rows += [
        [str(index)] + [f"{value:.3f}" for value in matrix.loc[index]]
        for index in matrix.index
    ]
    story.append(make_table(matrix_rows, [46 * mm] + [22 * mm] * len(METHODS)))
    story.append(PageBreak())
    story.append(Paragraph("Hyperparameter search space", heading))
    story.append(Spacer(1, 8))
    story.append(
        make_table(
            [["Classifier", "Parameter", "Range"]]
            + search_space[["Classifier", "Parameter", "Values"]].values.tolist(),
            [45 * mm, 55 * mm, 90 * mm],
        )
    )
    story.append(PageBreak())
    story.append(Paragraph("Search completion and OCT addition", heading))
    story.append(Spacer(1, 8))
    story.append(make_table([list(audit.columns)] + audit.values.tolist(), [45 * mm] + [28 * mm] * 4))
    story.append(Spacer(1, 12))
    story.append(Image(str(out_dir / "oct_addition_comparison.png"), width=245 * mm, height=78 * mm))

    for method in METHODS:
        story.append(PageBreak())
        story.append(Paragraph(f"Optimal configurations: {MODEL_LABELS[method]}", heading))
        story.append(Spacer(1, 8))
        rows = [["Features", "ROC-AUC", "AP", "Accuracy", "Quota D / R / O", "Model parameters"]]
        for _, row in best[best["ml_method"].eq(method)].iterrows():
            rows.append(
                [
                    GROUP_LABELS[row["feature_group"]],
                    f"{row['auc_roc']:.3f}",
                    f"{row['average_precision']:.3f}",
                    f"{row['accuracy']:.3f}",
                    f"{row['donor_k']} / {row['recipient_k']} / {row['oct_k']}",
                    row["model_params"],
                ]
            )
        story.append(make_table(rows, [39 * mm, 17 * mm, 15 * mm, 19 * mm, 29 * mm, 78 * mm]))

    report_path = out_dir / "ml_search_tables.pdf"
    SimpleDocTemplate(
        str(report_path),
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    ).build(story)
    return report_path


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, audit = load_results()
    best = select_best_rows(all_rows)
    if not args.skip_prediction_audit:
        verify_predictions(best)

    matrix = save_result_tables(all_rows, best, audit, args.out_dir)
    search_space = search_space_rows(args.include_logistic_pca)
    search_space.to_csv(args.out_dir / "hyperparameter_search_space.csv", index=False)
    plot_search_space_table(search_space, args.out_dir)
    plot_oct_addition(best, args.out_dir)

    report_path = None
    if not args.skip_report:
        try:
            report_path = build_report(matrix, best, audit, search_space, args.out_dir)
        except ImportError as exc:
            warnings.warn(str(exc), stacklevel=1)

    print(matrix.round(3).to_string())
    print(f"\nSaved outputs to: {args.out_dir}")
    if report_path is not None:
        print(f"Combined report: {report_path}")
    if not args.skip_prediction_audit:
        print("All 42 selected predictions were verified against their summary AUCs.")


if __name__ == "__main__":
    main()
