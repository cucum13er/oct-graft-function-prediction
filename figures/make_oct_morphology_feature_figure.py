import argparse
import os
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
except ImportError:  # pragma: no cover
    mannwhitneyu = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "figures" / "oct_morphology"
SUMMARY_52_CSV = (
    ROOT
    / "results" / "models" / "52"
    / "logistic_corr_auc_quota_grid"
    / "quota_grid_summary.csv"
)
COHORTS = {
    "52": {
        "summary_csv": SUMMARY_52_CSV,
        "input_csv": ROOT / "data/processed" / "kidney_features_complete_clinical_oct_all_available.csv",
        "run_prefix": "data_all_",
        "n_cases": 52,
    },
    "33": {
        "summary_csv": ROOT / "results" / "models" / "33" / "logistic_corr_auc_quota_grid" / "quota_grid_summary.csv",
        "input_csv": ROOT / "data/processed" / "kidney_features_complete_clinical_oct_biopsy.csv",
        "run_prefix": "data_withBiopsy_",
        "n_cases": 33,
    },
}

OCT_FEATURES = [
    "minor_median",
    "minor_mean",
    "minor_std",
    "minor_p10",
    "minor_p90",
    "minor_iqr",
    "ecc_median",
    "ecc_mean",
    "ecc_std",
    "ecc_p10",
    "ecc_p90",
    "ecc_iqr",
    "sol_median",
    "sol_mean",
    "sol_std",
    "sol_p10",
    "sol_p90",
    "sol_iqr",
]

FEATURE_LABELS = {
    "minor": "Minor axis",
    "ecc": "Eccentricity",
    "sol": "Solidity",
    "median": "median",
    "mean": "mean",
    "std": "STD",
    "p10": "P10",
    "p90": "P90",
    "iqr": "IQR",
}

GRID = "#ECEFF2"
IGF_COLOR = "#1CA79A"
DGF_COLOR = "#A61C4E"


def pretty_feature(feature):
    family, stat = feature.split("_", 1)
    return f"{FEATURE_LABELS.get(family, family)}\n{FEATURE_LABELS.get(stat, stat)}"


def compact_feature(feature):
    family, stat = feature.split("_", 1)
    family_label = {"minor": "Minor", "ecc": "Ecc", "sol": "Sol"}.get(family, family)
    return f"{family_label} {FEATURE_LABELS.get(stat, stat)}"


def read_lg_summary(summary_csv, cohort):
    df = pd.read_csv(summary_csv)
    df = df[df["status"].eq("ok")].copy()
    df = df[
        df["run_name"].str.startswith(cohort["run_prefix"])
        & df["run_name"].str.endswith("_oct")
        & df["n_cases"].eq(cohort["n_cases"])
        & df["selector"].eq("corr_auc")
    ].copy()
    df["oct_k"] = pd.to_numeric(df["oct_k"], errors="coerce")
    df = df[df["oct_k"].isin([2, 3, 4])].copy()
    if df.empty:
        raise ValueError(f"No OCT-containing logistic rows with oct_k 2-4 in {summary_csv}")
    return df


def fold_selection_path(row):
    run_dir = Path(row["quota_run_dir"])
    direct = run_dir / "fold_selections.csv"
    if direct.exists():
        return direct
    return run_dir / row["run_name"] / "fold_selections.csv"


def build_frequency_tables(rows):
    records = []
    for _, row in rows.iterrows():
        fold_csv = fold_selection_path(row)
        if not fold_csv.exists():
            raise FileNotFoundError(fold_csv)
        folds = pd.read_csv(fold_csv)
        folds = folds[folds["block"].eq("oct")]
        if len(folds) != int(row["n_cases"]):
            raise ValueError(f"Unexpected OCT fold count in {fold_csv}")
        for fold in folds.itertuples():
            selected = frozenset(str(fold.selected_features).split(";"))
            unknown = selected.difference(OCT_FEATURES)
            if unknown:
                raise ValueError(f"Unknown OCT features in {fold_csv}: {unknown}")
            records.append(
                {
                    "param_run": row.get("param_run", "default"),
                    "run_name": row["run_name"],
                    "oct_k": int(row["oct_k"]),
                    "quota_run_dir": row["quota_run_dir"],
                    "heldout_case": str(fold.heldout_case),
                    "selected_features": ";".join(sorted(selected)),
                }
            )
    all_selections = pd.DataFrame(records)
    grouped = all_selections.groupby(["oct_k", "heldout_case"])["selected_features"]
    if (grouped.nunique() > 1).any():
        raise ValueError("OCT selections differ across model/combo runs for the same k and fold")
    unique_folds = all_selections.drop_duplicates(["oct_k", "heldout_case"]).copy()
    if unique_folds.groupby("oct_k").size().nunique() != 1:
        raise ValueError("Different numbers of unique held-out cases across OCT quotas")
    long = unique_folds.assign(feature=unique_folds["selected_features"].str.split(";")).explode("feature")
    counts = long.groupby(["oct_k", "feature"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=[2, 3, 4], columns=OCT_FEATURES, fill_value=0)
    n_folds = unique_folds.groupby("oct_k").size().reindex([2, 3, 4])
    heatmap = counts.div(n_folds, axis=0)
    overall = pd.DataFrame(
        {
            "selection_frequency": counts.sum(axis=0) / n_folds.sum(),
            "selected_unique_folds": counts.sum(axis=0),
            "total_unique_fold_opportunities": int(n_folds.sum()),
        }
    ).sort_values(["selection_frequency", "selected_unique_folds"], ascending=False)
    return all_selections, unique_folds, heatmap, overall


def direction_free_auc(y, values):
    if roc_auc_score is None or len(np.unique(y)) != 2:
        return np.nan, np.nan
    auc = float(roc_auc_score(y, values))
    return auc, max(auc, 1.0 - auc)


def feature_group_stats(data, top_features):
    rows = []
    y = data["IGF(0)/DGF(1)"].astype(int).to_numpy()
    for feature in top_features:
        values = pd.to_numeric(data[feature], errors="coerce")
        igf = values[data["IGF(0)/DGF(1)"].eq(0)].dropna()
        dgf = values[data["IGF(0)/DGF(1)"].eq(1)].dropna()
        auc, auc_direction_free = direction_free_auc(y[values.notna()], values.dropna())
        if mannwhitneyu is not None and len(igf) and len(dgf):
            p_value = float(mannwhitneyu(igf, dgf, alternative="two-sided").pvalue)
        else:
            p_value = np.nan
        rows.append(
            {
                "feature": feature,
                "igf_median": float(igf.median()),
                "dgf_median": float(dgf.median()),
                "median_difference_dgf_minus_igf": float(dgf.median() - igf.median()),
                "auc_dgf_high": auc,
                "auc_direction_free": auc_direction_free,
                "mann_whitney_p": p_value,
                "n_igf": int(len(igf)),
                "n_dgf": int(len(dgf)),
            }
        )
    return pd.DataFrame(rows)


def panel_heatmap(ax, heatmap, overall):
    shown = overall[overall["selected_unique_folds"].gt(0)].index.tolist()
    values = heatmap[shown].to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="RdPu", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(shown)))
    ax.set_xticklabels([pretty_feature(feature) for feature in shown])
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels([f"k = {k}" for k in heatmap.index])
    for y in range(len(heatmap.index)):
        for x in range(len(shown)):
            value = values[y, x]
            ax.text(x, y, f"{value:.0%}" if value in (0, 1) else f"{value:.1%}",
                    ha="center", va="center", fontsize=9,
                    color="white" if value > 0.65 else "#252B30")

    ax.set_xticks(np.arange(-0.5, len(shown), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heatmap.index), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Selection frequency")
    cbar.ax.tick_params(labelsize=8)
    ax.text(
        -0.045, 1.08, "a", transform=ax.transAxes,
        ha="left", va="top", fontsize=13, fontweight="bold",
    )


def group_dotplot(ax, values, labels, feature):
    igf = values[labels.eq(0)].dropna()
    dgf = values[labels.eq(1)].dropna()
    rng = np.random.default_rng(17)
    for x, group, color in [(0, igf, IGF_COLOR), (1, dgf, DGF_COLOR)]:
        jitter = rng.uniform(-0.16, 0.16, len(group))
        ax.scatter(x + jitter, group, s=22, color=color, alpha=0.75, lw=0, zorder=2)
        ax.plot([x - 0.23, x + 0.23], [group.median()] * 2,
                color="#20252A", lw=2.2, zorder=3)
    ax.set_title(compact_feature(feature), fontsize=10, fontweight="bold", pad=4)
    ax.set_xticks([0, 1], ["IGF", "DGF"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("Feature value")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def panel_distributions(axes, data, top_features):
    labels = data["IGF(0)/DGF(1)"].astype(int)
    for ax, feature in zip(axes, top_features):
        values = pd.to_numeric(data[feature], errors="coerce")
        group_dotplot(ax, values, labels, feature)
    axes[0].text(
        -0.23,
        1.08,
        "b",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


def style_axes(fig):
    for ax in fig.axes:
        if ax.images:
            continue
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8.5)


def make_figure(heatmap, overall, data, top_features):
    fig = plt.figure(figsize=(12.8, 7.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[0.85, 1.3], hspace=0.52, wspace=0.44)
    ax_heatmap = fig.add_subplot(grid[0, :])
    dist_axes = [fig.add_subplot(grid[1, i]) for i in range(4)]
    panel_heatmap(ax_heatmap, heatmap, overall)
    panel_distributions(dist_axes, data, top_features)
    style_axes(fig)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make OCT morphology feature-selection and IGF/DGF distribution figure."
    )
    parser.add_argument("--cohort", choices=["52", "33", "both"], default="both")
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="oct_morphology_feature_details")
    parser.add_argument("--top-n", type=int, default=4)
    return parser.parse_args()


def make_cohort_outputs(args, cohort_key):
    cohort = COHORTS[cohort_key]
    summary_csv = args.summary_csv or cohort["summary_csv"]
    input_csv = args.input_csv or cohort["input_csv"]
    prefix = args.prefix if cohort_key == "52" else f"{args.prefix}_33"

    summary = read_lg_summary(summary_csv, cohort)
    all_selections, unique_folds, heatmap, overall = build_frequency_tables(summary)
    case_ids = set(unique_folds.loc[unique_folds["oct_k"].eq(2), "heldout_case"])
    if len(case_ids) != cohort["n_cases"]:
        raise ValueError(f"Expected {cohort['n_cases']} unique held-out cases, found {len(case_ids)}")
    for k in [3, 4]:
        if set(unique_folds.loc[unique_folds["oct_k"].eq(k), "heldout_case"]) != case_ids:
            raise ValueError(f"Held-out cases differ between k=2 and k={k}")
    top_features = overall.head(args.top_n).index.tolist()

    data = pd.read_csv(input_csv)
    data = data[data["case_num"].astype(str).isin(case_ids)].copy()
    if len(data) != len(case_ids) or data["IGF(0)/DGF(1)"].isna().any():
        raise ValueError(f"Input data do not match the {cohort_key}-case LOOCV cases: {input_csv}")
    stats = feature_group_stats(data, top_features)

    summary.to_csv(args.out_dir / f"{prefix}_included_lg_rows.csv", index=False)
    all_selections.to_csv(args.out_dir / f"{prefix}_all_run_fold_selections.csv", index=False)
    unique_folds.to_csv(args.out_dir / f"{prefix}_unique_fold_selections.csv", index=False)
    frequency_long = heatmap.stack().rename("selection_frequency").reset_index()
    frequency_long["n_unique_folds"] = unique_folds.groupby("oct_k").size().reindex(
        frequency_long["oct_k"]
    ).to_numpy()
    frequency_long["selected_unique_folds"] = (
        frequency_long["selection_frequency"] * frequency_long["n_unique_folds"]
    ).round().astype(int)
    frequency_long.to_csv(args.out_dir / f"{prefix}_selection_frequency_long.csv", index=False)
    heatmap.to_csv(args.out_dir / f"{prefix}_selection_frequency_heatmap.csv")
    overall.to_csv(args.out_dir / f"{prefix}_selection_frequency_overall.csv")
    stats.to_csv(args.out_dir / f"{prefix}_igf_dgf_stats.csv", index=False)

    fig = make_figure(heatmap, overall, data, top_features)
    png = args.out_dir / f"{prefix}.png"
    pdf = args.out_dir / f"{prefix}.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)
    print(f"{cohort_key} cases: audited {len(summary)} runs and {len(all_selections)} run-fold selections; "
          f"{len(unique_folds)} unique quota-fold selections")
    print("Top features:", ", ".join(top_features))
    return overall


def main():
    args = parse_args()
    if not 1 <= args.top_n <= 4:
        raise ValueError("--top-n must be between 1 and 4")
    if args.cohort == "both" and (args.summary_csv or args.input_csv):
        raise ValueError("Custom --summary-csv and --input-csv require a single --cohort")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohorts = ["52", "33"] if args.cohort == "both" else [args.cohort]
    overall_by_cohort = {key: make_cohort_outputs(args, key) for key in cohorts}
    if args.cohort == "both":
        comparison = pd.DataFrame(index=OCT_FEATURES)
        comparison.index.name = "feature"
        for key, overall in overall_by_cohort.items():
            comparison[f"frequency_{key}"] = overall["selection_frequency"]
            comparison[f"selected_unique_folds_{key}"] = overall["selected_unique_folds"]
            comparison[f"total_unique_fold_opportunities_{key}"] = overall[
                "total_unique_fold_opportunities"
            ]
        comparison = comparison.sort_values(
            ["frequency_52", "frequency_33"], ascending=False
        )
        comparison.to_csv(args.out_dir / f"{args.prefix}_cohort_comparison.csv")


if __name__ == "__main__":
    main()
