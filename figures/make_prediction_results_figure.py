import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "figures" / "prediction_results"

LOGISTIC_SUMMARY = (
    ROOT
    / "results" / "models" / "52"
    / "logistic_corr_auc_quota_grid"
    / "quota_grid_summary.csv"
)
UNIFIED_LOGISTIC_PARAM = (
    "params-C-0p01_class_weight-balanced_penalty-l2_solver-liblinear"
)

ML_SUMMARIES = {
    "Logistic\nregression": ROOT
    / "results" / "models" / "52"
    / "logistic_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "Random\nforest": ROOT
    / "results" / "models" / "52"
    / "random_forest_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "Extra\nTrees": ROOT
    / "results" / "models" / "52"
    / "extra_trees_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "XG\nBoost": ROOT
    / "results" / "models" / "52"
    / "xgboost_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "Cat\nBoost": ROOT
    / "results" / "models" / "52"
    / "catboost_corr_auc_quota_grid"
    / "quota_grid_summary.csv",
    "TabPFN": ROOT
    / "results" / "models" / "52"
    / "tabpfn_corr_auc_quota_grid_more"
    / "quota_grid_summary.csv",
}


RUN_LABELS = {
    "data_all_donor": "Donor",
    "data_all_recipient": "Recipient",
    "data_all_donor_recipient": "Donor + recipient",
    "data_all_oct": "OCT only",
    "data_all_donor_oct": "Donor + OCT",
    "data_all_recipient_oct": "Recipient + OCT",
    "data_all_donor_recipient_oct": "Donor + recipient + OCT",
}

PANEL_A_PAIRS = [
    ("data_all_donor", "data_all_donor_oct", "Donor"),
    ("data_all_recipient", "data_all_recipient_oct", "Recipient"),
    ("data_all_donor_recipient", "data_all_donor_recipient_oct", "Donor + recipient"),
]

PANEL_B_RUNS = [
    ("data_all_donor_recipient", "Donor + recipient"),
    ("data_all_oct", "OCT only"),
    ("data_all_donor_recipient_oct", "Donor + recipient + OCT"),
]

GREY = "#7D8994"
LIGHT_GREY = "#BEC7CF"
GRID = "#ECEFF2"
BLUE = "#1F77B4"
ORANGE = "#F28E2B"
TEAL = "#1CA79A"
MAGENTA = "#A61C4E"
PURPLE = "#8067B7"

RUN_COLORS = {
    "data_all_donor": BLUE,
    "data_all_donor_oct": BLUE,
    "data_all_recipient": ORANGE,
    "data_all_recipient_oct": ORANGE,
    "data_all_donor_recipient": TEAL,
    "data_all_donor_recipient_oct": MAGENTA,
    "data_all_oct": PURPLE,
}


def read_best_rows(summary_csv, param_run=None):
    df = pd.read_csv(summary_csv)
    df = df[df["status"].eq("ok")].copy()
    if param_run is not None:
        if "param_run" not in df.columns:
            raise KeyError(f"param_run not found in {summary_csv}")
        equivalent = {param_run, param_run.replace("penalty-l2", "l1_ratio-0p0")}
        df = df[df["param_run"].isin(equivalent)].copy()
        if df.empty:
            raise ValueError(f"No rows for {param_run} in {summary_csv}")
    idx = df.groupby("run_name")["auc_roc"].idxmax()
    return df.loc[idx].set_index("run_name")


def best_row(summary_csv, run_name):
    rows = read_best_rows(summary_csv)
    if run_name not in rows.index:
        raise KeyError(f"{run_name} not found in {summary_csv}")
    return rows.loc[run_name]


def read_curve(row):
    curve_csv = Path(row["quota_run_dir"]) / "roc_curve_points.csv"
    if not curve_csv.exists():
        raise FileNotFoundError(curve_csv)
    return pd.read_csv(curve_csv)


def annotate_auc(ax, x, y, auc, color, weight="normal", va="bottom", ha="center", dy=None):
    offset = dy if dy is not None else (0.010 if va == "bottom" else -0.012)
    ax.text(
        x,
        y + offset,
        f"{auc:.3f}",
        color=color,
        fontsize=8.5,
        fontweight=weight,
        ha=ha,
        va=va,
    )


def panel_a(ax, lg_rows, optimized_auc):
    x = [0, 1]
    offsets = [-0.035, 0.0, 0.035]
    text_nudges = {
        "data_all_donor": {"dy": 0.020, "ha": "center"},
        "data_all_donor_oct": {"dy": -0.012, "ha": "right", "va": "top"},
        "data_all_recipient": {"dy": 0.014, "ha": "center"},
        "data_all_recipient_oct": {"dy": -0.018, "ha": "left", "va": "top"},
        "data_all_donor_recipient": {"dy": -0.014, "ha": "left", "va": "top"},
        "data_all_donor_recipient_oct": {"dx": -0.015, "dy": 0.010, "ha": "center"},
    }
    for (without_run, with_run, label), dx in zip(PANEL_A_PAIRS, offsets):
        y0 = float(lg_rows.loc[without_run, "auc_roc"])
        y1 = float(lg_rows.loc[with_run, "auc_roc"])
        start_color = RUN_COLORS[without_run]
        end_color = RUN_COLORS[with_run]
        ax.plot(
            [x[0] + dx, x[1] + dx],
            [y0, y1],
            "-",
            color=start_color,
            lw=2.0,
            label=label,
            zorder=2,
        )
        ax.scatter([x[0] + dx], [y0], s=34, color=start_color, zorder=4)
        ax.scatter([x[1] + dx], [y1], s=34, color=end_color, zorder=4)
        n0 = text_nudges[without_run]
        n1 = text_nudges[with_run]
        annotate_auc(
            ax,
            x[0] + dx + n0.get("dx", 0.0),
            y0,
            y0,
            start_color,
            va=n0.get("va", "bottom"),
            ha=n0["ha"],
            dy=n0["dy"],
        )
        annotate_auc(
            ax,
            x[1] + dx + n1.get("dx", 0.0),
            y1,
            y1,
            end_color,
            va=n1.get("va", "bottom"),
            ha=n1["ha"],
            dy=n1["dy"],
        )

    oct_auc = float(lg_rows.loc["data_all_oct", "auc_roc"])
    ax.axhline(oct_auc, color=PURPLE, ls="--", lw=1.5, label=f"OCT only ({oct_auc:.3f})")

    primary_auc = float(lg_rows.loc["data_all_donor_recipient_oct", "auc_roc"])
    if abs(optimized_auc - primary_auc) > 1e-12:
        ax.scatter(
            [1.12],
            [optimized_auc],
            marker="*",
            s=105,
            facecolors="white",
            edgecolors=MAGENTA,
            linewidths=1.4,
            label=f"Optimized LR ({optimized_auc:.3f})",
            zorder=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Clinical only", "Clinical + OCT"])
    ax.set_xlim(-0.12, 1.18)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.50, 0.90)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("a", loc="left", fontweight="bold", fontsize=13)


def panel_b(ax, lg_rows):
    rows = [(run, label, lg_rows.loc[run]) for run, label in PANEL_B_RUNS]
    best_auc = max(float(row["auc_roc"]) for _, _, row in rows)
    for run, label, row in rows:
        auc = float(row["auc_roc"])
        curve = read_curve(row)
        is_best = auc == best_auc
        ax.step(
            curve["fpr"],
            curve["tpr"],
            where="post",
            color=RUN_COLORS[run],
            lw=3.0 if is_best else 1.8,
            alpha=1.0 if is_best else 0.95,
            label=f"{label} (LR, {auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#C8CDD2", ls=":", lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("b", loc="left", fontweight="bold", fontsize=13)


def ml_comparison_rows():
    rows = []
    for model, summary_csv in ML_SUMMARIES.items():
        best = read_best_rows(summary_csv)
        base = best.loc["data_all_donor_recipient"]
        oct_row = best.loc["data_all_donor_recipient_oct"]
        rows.append(
            {
                "model": model,
                "without_oct": float(base["auc_roc"]),
                "with_oct": float(oct_row["auc_roc"]),
            }
        )
    return pd.DataFrame(rows)


def panel_c(ax, highlight):
    df = ml_comparison_rows()
    x_shift = 0.12
    x = [i + x_shift for i in range(len(df))]
    best_idx = int(df["with_oct"].idxmax())

    for i, row in df.iterrows():
        xi = i + x_shift
        ax.plot([xi, xi], [row["without_oct"], row["with_oct"]], color=LIGHT_GREY, lw=2.0, zorder=1)

    ax.scatter(x, df["without_oct"], s=32, color=TEAL, label="Donor + recipient", zorder=3)
    ax.scatter(x, df["with_oct"], s=36, color=MAGENTA, label="Donor + recipient + OCT", zorder=4)

    for i, row in df.iterrows():
        xi = i + x_shift
        ax.text(xi, row["without_oct"] - 0.016, f"{row['without_oct']:.3f}", ha="center", va="top", color=TEAL, fontsize=8)
        ax.text(xi, row["with_oct"] + 0.012, f"{row['with_oct']:.3f}", ha="center", va="bottom", color=MAGENTA, fontsize=8)

    if highlight == "best":
        best_x = best_idx + x_shift
        ax.axvspan(best_x - 0.35, best_x + 0.35, color=MAGENTA, alpha=0.08, zorder=0)
        ax.scatter(
            [best_x],
            [df.loc[best_idx, "with_oct"]],
            marker="*",
            s=95,
            color="#333333",
            label="Highest AUC",
            zorder=5,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"])
    ax.set_xlim(-0.28, len(df) - 0.72)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.55, 0.91)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("c", loc="left", fontweight="bold", fontsize=13)


def style_axes(fig):
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)


def make_figure(highlight):
    lg_rows = read_best_rows(LOGISTIC_SUMMARY, param_run=UNIFIED_LOGISTIC_PARAM)
    optimized_auc = float(
        read_best_rows(LOGISTIC_SUMMARY).loc["data_all_donor_recipient_oct", "auc_roc"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 3.8), gridspec_kw={"wspace": 0.30})
    panel_a(axes[0], lg_rows, optimized_auc)
    panel_b(axes[1], lg_rows)
    panel_c(axes[2], highlight)
    style_axes(fig)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Make prediction result figure with OCT comparisons.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--highlight-c",
        choices=["none", "best"],
        default="none",
        help="Whether panel c should mark the best model.",
    )
    parser.add_argument("--prefix", default="prediction_results_oct")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure(args.highlight_c)
    suffix = "highlight_best" if args.highlight_c == "best" else "no_highlight"
    png = args.out_dir / f"{args.prefix}_{suffix}.png"
    pdf = args.out_dir / f"{args.prefix}_{suffix}.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

