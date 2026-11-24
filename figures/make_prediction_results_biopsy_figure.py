import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "models" / "33"
DEFAULT_OUT_DIR = ROOT / "results" / "figures" / "prediction_results_biopsy"

LOGISTIC_SUMMARY = RESULTS / "logistic_corr_auc_quota_grid" / "quota_grid_summary.csv"
LOGISTIC_PCA_SUMMARY = (
    RESULTS / "logistic_corr_auc_pca_quota_grid" / "quota_grid_summary.csv"
)
ML_SUMMARIES = {
    "Logistic\nregression": LOGISTIC_SUMMARY,
    "Random\nforest": RESULTS / "random_forest_corr_auc_quota_grid" / "quota_grid_summary.csv",
    "Extra\nTrees": RESULTS / "extra_trees_corr_auc_quota_grid" / "quota_grid_summary.csv",
    "XG\nBoost": RESULTS / "xgboost_corr_auc_quota_grid" / "quota_grid_summary.csv",
    "Cat\nBoost": RESULTS / "catboost_corr_auc_quota_grid" / "quota_grid_summary.csv",
    "TabPFN": RESULTS / "tabpfn_corr_auc_quota_grid_more" / "quota_grid_summary.csv",
}

OCT = "data_withBiopsy_oct"
OCT_BIOPSY = "data_withBiopsy_biopsy_oct"
DONOR_OCT = "data_withBiopsy_donor_oct"
DONOR_BIOPSY_OCT = "data_withBiopsy_donor_biopsy_oct"
DONOR_RECIPIENT_OCT = "data_withBiopsy_donor_recipient_oct"
DONOR_BIOPSY_RECIPIENT_OCT = "data_withBiopsy_donor_biopsy_recipient_oct"

BLUE = "#1F77B4"
TEAL = "#1CA79A"
ORANGE = "#F28E2B"
MAGENTA = "#A61C4E"
PURPLE = "#8067B7"
LIGHT_GREY = "#BEC7CF"
GRID = "#ECEFF2"

PANEL_A_PAIRS = [
    (OCT, OCT_BIOPSY, "OCT", PURPLE, -0.035),
    (DONOR_OCT, DONOR_BIOPSY_OCT, "Donor + OCT", BLUE, 0.0),
    (
        DONOR_RECIPIENT_OCT,
        DONOR_BIOPSY_RECIPIENT_OCT,
        "Donor + recipient + OCT",
        TEAL,
        0.035,
    ),
]


def read_summary(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[
        df["status"].eq("ok")
        & df["dataset"].eq("complete_biopsy")
        & df["n_cases"].eq(33)
    ].copy()
    if df.empty:
        raise ValueError(f"No successful 33-case biopsy runs in {path}")
    return df


def best_by_run(df):
    return df.loc[df.groupby("run_name")["auc_roc"].idxmax()].set_index("run_name")


def read_curve(row):
    run_dir = Path(row["quota_run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return pd.read_csv(run_dir / "roc_curve_points.csv")


def panel_a(ax, lg_rows, optimized_row):
    for without, with_biopsy, label, color, offset in PANEL_A_PAIRS:
        y0 = float(lg_rows.loc[without, "auc_roc"])
        y1 = float(lg_rows.loc[with_biopsy, "auc_roc"])
        ax.plot([offset, 1 + offset], [y0, y1], color=color, lw=2, label=label, zorder=2)
        ax.scatter([offset, 1 + offset], [y0, y1], color=color, s=36, zorder=3)

        left_offset = (6, 6) if without == DONOR_OCT else (-4, 9)
        ax.annotate(
            f"{y0:.3f}",
            (offset, y0),
            xytext=left_offset,
            textcoords="offset points",
            ha="left" if without == DONOR_OCT else "right",
            fontsize=11,
            color=color,
        )
        right_dy = -19 if without == OCT else 8
        ax.annotate(
            f"{y1:.3f}",
            (1 + offset, y1),
            xytext=(1, right_dy),
            textcoords="offset points",
            ha="left",
            fontsize=11,
            color=color,
        )

    star_run = optimized_row["run_name"]
    plotted_x = {
        run: side + offset
        for without, with_biopsy, _, _, offset in PANEL_A_PAIRS
        for side, run in [(0, without), (1, with_biopsy)]
    }
    if star_run not in plotted_x:
        raise ValueError(f"Optimized LR run is not shown in panel a: {star_run}")
    star_x = plotted_x[star_run] - 0.14 if plotted_x[star_run] < 0.5 else plotted_x[star_run] + 0.14
    star_auc = float(optimized_row["auc_roc"])
    reduction = "PCA" if optimized_row["dim_reduction"] == "pca" else "no PCA"
    ax.scatter(
        [star_x], [star_auc], marker="*", s=120, facecolors="white",
        edgecolors=MAGENTA, linewidths=1.5,
        label=f"Optimized LR, {reduction} ({star_auc:.3f})", zorder=5,
    )
    ax.annotate(
        f"{star_auc:.3f}", (star_x, star_auc), xytext=(0, 9),
        textcoords="offset points", ha="center", fontsize=11, color=MAGENTA,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Without biopsy", "With biopsy"])
    ax.set_xlim(-0.25, 1.18)
    ax.set_ylim(0.62, 0.94)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title("a", loc="left", fontweight="bold", fontsize=17)


def panel_b(ax, lg_rows):
    curves = [
        (OCT, "OCT only", PURPLE, 1.6),
        (DONOR_BIOPSY_OCT, "Donor + biopsy + OCT", MAGENTA, 1.9),
        (DONOR_OCT, "Donor + OCT", BLUE, 2.8),
    ]
    for run, label, color, width in curves:
        row = lg_rows.loc[run]
        curve = read_curve(row)
        auc = float(row["auc_roc"])
        ax.step(
            curve["fpr"], curve["tpr"], where="post", color=color,
            lw=width, label=f"{label} (LR, {auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#C8CDD2", ls=":", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 - specificity", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_title("b", loc="left", fontweight="bold", fontsize=17)


def ml_comparison_rows():
    rows = []
    for model, summary_csv in ML_SUMMARIES.items():
        best = best_by_run(read_summary(summary_csv))
        rows.append(
            {
                "model": model,
                "without_biopsy": float(best.loc[DONOR_OCT, "auc_roc"]),
                "with_biopsy": float(best.loc[DONOR_BIOPSY_OCT, "auc_roc"]),
            }
        )
    return pd.DataFrame(rows)


def panel_c(ax):
    df = ml_comparison_rows()
    x = range(len(df))
    for i, row in df.iterrows():
        ax.plot(
            [i, i], [row["without_biopsy"], row["with_biopsy"]],
            color=LIGHT_GREY, lw=2, zorder=1,
        )
    ax.scatter(x, df["without_biopsy"], s=34, color=BLUE, label="Donor + OCT", zorder=3)
    ax.scatter(
        x, df["with_biopsy"], s=38, color=MAGENTA,
        label="Donor + biopsy + OCT", zorder=4,
    )
    for i, row in df.iterrows():
        high = max(row["without_biopsy"], row["with_biopsy"])
        low = min(row["without_biopsy"], row["with_biopsy"])
        for key, color in [("without_biopsy", BLUE), ("with_biopsy", MAGENTA)]:
            value = row[key]
            dy = 9 if value == high else -17
            ax.annotate(
                f"{value:.3f}", (i, value), xytext=(0, dy),
                textcoords="offset points", ha="center", fontsize=10.5, color=color,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"])
    ax.set_xlim(-0.45, len(df) - 0.55)
    ax.set_ylim(0.66, 0.93)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title("c", loc="left", fontweight="bold", fontsize=17)


def make_figure():
    no_pca = read_summary(LOGISTIC_SUMMARY)
    unified_param = no_pca.loc[no_pca["auc_roc"].idxmax(), "param_run"]
    lg_rows = best_by_run(no_pca[no_pca["param_run"].eq(unified_param)])
    optimized = pd.concat([no_pca, read_summary(LOGISTIC_PCA_SUMMARY)], ignore_index=True)
    optimized_row = optimized.loc[optimized["auc_roc"].idxmax()]

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.0), gridspec_kw={"wspace": 0.31})
    panel_a(axes[0], lg_rows, optimized_row)
    panel_b(axes[1], lg_rows)
    panel_c(axes[2])
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.19, top=0.94)
    return fig, unified_param, optimized_row


def main():
    parser = argparse.ArgumentParser(description="Plot biopsy effects in the 33-case cohort.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="prediction_results_biopsy_33")
    args = parser.parse_args()

    fig, unified_param, optimized_row = make_figure()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for extension, options in [("png", {"dpi": 450}), ("pdf", {})]:
        output = args.out_dir / f"{args.prefix}.{extension}"
        fig.savefig(output, bbox_inches="tight", facecolor="white", **options)
        print(output)
    print(f"Shared no-PCA LR parameters: {unified_param}")
    print(
        "Optimized LR: "
        f"{optimized_row['run_name']}, AUC={optimized_row['auc_roc']:.3f}, "
        f"dim_reduction={optimized_row['dim_reduction']}"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

