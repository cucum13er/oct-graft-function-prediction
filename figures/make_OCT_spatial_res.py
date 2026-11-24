from pathlib import Path
import argparse
import json
import os


import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
from analyze_oct_spatial_donor_quota_grid import load_predictions


# ============================================================
# Configuration — modify column names here to match your files
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
ID_COL = "case_num"
OUTCOME_COL = "IGF(0)/DGF(1)"
MODEL_COMBINATIONS = {
    "donor": "donor",
    "donor_m": "donor+morphology",
    "donor_s": "donor+spatial",
    "donor_m_s": "donor+morphology+spatial",
}

# Columns in oof_predictions.csv
MODELS = {
    "donor": {
        "label": "Donor",
        "color": "#777777",
    },
    "donor_m": {
        "label": "Donor + M",
        "color": "#4477AA",
    },
    "donor_s": {
        "label": "Donor + S",
        "color": "#EEAA33",
    },
    "donor_m_s": {
        "label": "Donor + M + S",
        "color": "#228833",
    },
}

# Selected S features in the representative configuration
SPATIAL_COLS = [
    "anterior_curve_depth_p90_um",
    "anterior_density_count_gt_8",
]

SPATIAL_LABELS = [
    "Depth\nP90",
    "High-occurrence\narea fraction",
]

# Replace these names if the actual table uses different names.
STATS = ["mean", "median", "std", "p10", "p90", "iqr"]

M_GROUPS = {
    "Minor axis": [f"minor_{stat}" for stat in STATS],
    "Eccentricity": [f"ecc_{stat}" for stat in STATS],
    "Solidity": [f"sol_{stat}" for stat in STATS],
}

STAT_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "std": "SD",
    "p10": "P10",
    "p90": "P90",
    "iqr": "IQR",
}

# Continuous donor/preservation variables only.
# EDIT to match the actual feature table.
DONOR_COLS = {
    "donor_age": "Donor age",
    "donor_kdpi": "KDPI",
    "donor_term_creatinine": "Terminal creatinine",
    "donor_cold_ischemia_min": "Cold ischemia time",
}
DONOR_LABELS = {
    **DONOR_COLS,
    "donor_init_creatinine": "Initial creatinine",
    "donor_sex_male": "Male sex",
    "donor_dcd": "DCD",
    "donor_cause_of_death_stroke": "Cause of death: stroke",
    "donor_race_code": "Race",
    "donor_htn": "Hypertension",
}

CONFIG_TEXT = (
    "C = 0.01; feature quotas: donor = 4, M = 1, S = 1"
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# Input validation
# ============================================================

def encode_outcome(series):
    """Return binary outcome: IGF=0, DGF=1."""
    text = series.astype(str).str.strip().str.upper()
    mapping = {
        "IGF": 0,
        "DGF": 1,
        "0": 0,
        "1": 1,
        "0.0": 0,
        "1.0": 1,
    }
    encoded = text.map(mapping)

    if encoded.isna().any():
        bad = series[encoded.isna()].unique().tolist()
        raise ValueError(f"Unrecognized outcomes: {bad}")

    return encoded.astype(int)


def require_columns(df, columns, file_label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label}: missing columns {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def build_matched_predictions(results_dir, feature_csv):
    cohort = pd.read_csv(feature_csv)
    selected = pd.read_csv(results_dir / "analysis_no_recipient/shared_small_budget_models.csv")
    full = selected[selected.combination.eq(MODEL_COMBINATIONS["donor_m_s"])]
    if len(full) != 1:
        raise ValueError("Expected one selected six-feature-budget full model")
    setting = full.iloc[0]
    quotas = {group: int(setting[f"{group}_k"]) for group in ("donor", "morphology", "spatial")}
    if sum(quotas.values()) > 6 or any(value < 1 for value in quotas.values()):
        raise ValueError("Expected positive numeric quotas totaling at most six")
    grid = pd.read_csv(results_dir / "quota_grid_summary.csv", dtype={
        f"{group}_k": str for group in ("donor", "recipient", "morphology", "spatial")
    })
    predictions = cohort[[ID_COL, OUTCOME_COL]].copy()
    matched = []
    spatial_frequency = None
    for key, combination in MODEL_COMBINATIONS.items():
        active = combination.split("+")
        rows = grid[grid.combination.eq(combination) & grid.C.eq(setting.C)]
        for group in ("donor", "recipient", "morphology", "spatial"):
            quota = quotas[group] if group in active else 0
            rows = rows[rows[f"{group}_k"].eq(str(quota))]
        if len(rows) != 1:
            raise ValueError(f"Missing or ambiguous matched configuration: {combination}")
        row = rows.iloc[0]
        for field in ("penalty", "class_weight", "corr_threshold"):
            if row[field] != setting[field]:
                raise ValueError(f"Different {field} for {combination}")
        frame = load_predictions(row, cohort)
        probabilities = frame["p_label1_loocv"].to_numpy()
        if not np.isclose(roc_auc_score(frame[OUTCOME_COL], probabilities), row.auc_roc):
            raise ValueError(f"Prediction-derived AUC differs for {combination}")
        predictions[key] = probabilities
        matched.append(dict(row, plot_column=key))
        if key == "donor_m_s":
            frequency = pd.read_csv(Path(row.run_dir) / "feature_selection_frequency.csv")
            spatial_frequency = frequency[frequency.feature.isin(SPATIAL_COLS)].copy()
            spatial_selected = frequency[frequency.feature.str.startswith(("anterior_", "posterior_"))]
            if set(spatial_selected.feature) != set(SPATIAL_COLS):
                raise ValueError("Configured correlation features differ from spatial features selected across folds")
    config_text = (f"C = {setting.C:g}; feature quotas: donor = {quotas['donor']}, "
                   f"M = {quotas['morphology']}, S = {quotas['spatial']}")
    return predictions, pd.DataFrame(matched), spatial_frequency, config_text


def load_data(pred_path, feature_path, prediction_frame=None):
    pred = (pd.read_csv(pred_path, dtype={ID_COL: str}) if pred_path is not None
            else prediction_frame.copy())
    features = pd.read_csv(feature_path, dtype={ID_COL: str})

    require_columns(
        pred,
        [ID_COL, OUTCOME_COL, *MODELS],
        "Prediction table",
    )

    m_cols = [
        col
        for columns in M_GROUPS.values()
        for col in columns
    ]

    require_columns(
        features,
        [ID_COL, *SPATIAL_COLS, *m_cols, *DONOR_COLS],
        "Feature table",
    )

    for name, df in [
        ("Prediction table", pred),
        ("Feature table", features),
    ]:
        df[ID_COL] = df[ID_COL].str.strip()
        if df[ID_COL].isna().any() or df[ID_COL].eq("").any():
            raise ValueError(f"{name}: missing kidney IDs")
        if df[ID_COL].duplicated().any():
            raise ValueError(f"{name}: duplicate kidney IDs.")

    pred["y"] = encode_outcome(pred[OUTCOME_COL])

    if pred["y"].nunique() != 2:
        raise ValueError("Both IGF and DGF cases are required.")

    for col in MODELS:
        pred[col] = pd.to_numeric(pred[col], errors="raise")
        values = pred[col].to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(f"{col}: missing or infinite probabilities.")

        if not ((values >= 0) & (values <= 1)).all():
            raise ValueError(f"{col}: probabilities must be between 0 and 1.")

    # Restrict correlation analysis to exactly the prediction cohort.
    missing_ids = set(pred[ID_COL]) - set(features[ID_COL])
    if missing_ids:
        raise ValueError(
            f"Feature table lacks kidneys: {sorted(missing_ids)}"
        )

    features = (
        features.set_index(ID_COL)
        .loc[pred[ID_COL]]
        .copy()
    )

    # If both tables contain outcomes, check consistency.
    if OUTCOME_COL in features.columns:
        feature_y = encode_outcome(features[OUTCOME_COL]).to_numpy()
        if not np.array_equal(feature_y, pred["y"].to_numpy()):
            raise ValueError("Outcomes differ between the two tables.")

    print(
        f"Cohort: n={len(pred)}, "
        f"IGF={(pred['y'] == 0).sum()}, "
        f"DGF={(pred['y'] == 1).sum()}"
    )

    return pred, features


# ============================================================
# Figure 5a: ROC curves
# ============================================================

def plot_roc(pred, outdir, config_text=CONFIG_TEXT, positive_class="IGF"):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.96)
    fig.text(0.02, 0.98, "a", fontsize=12, fontweight="bold", va="top")

    metrics = []

    for col, style in MODELS.items():
        y = 1 - pred["y"] if positive_class == "IGF" else pred["y"]
        probability = 1 - pred[col] if positive_class == "IGF" else pred[col]
        fpr, tpr, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)

        ax.plot(
            fpr,
            tpr,
            color=style["color"],
            linewidth=2,
            label=f"{style['label']}  (AUC = {auc:.2f})",
        )

        metrics.append({
            "model": style["label"],
            "auc": auc,
            "positive_class": positive_class,
        })

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="#BBBBBB",
        linewidth=1,
        zorder=0,
    )

    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )

    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, outdir, "fig5a_roc")
    pd.DataFrame(metrics).to_csv(outdir / "fig5a_auc.csv", index=False)
    (outdir / "fig5a_caption.txt").write_text(
        f"(a) ROC curves from LOOCV predictions in the {len(pred)}-case cohort, with {positive_class} as the positive class. "
        f"The same shared setting is used across models: {config_text}; inactive feature groups are omitted. "
        "M and S denote OCT morphology and spatial features. Legend values are ROC-AUCs. "
        "Settings were selected on this same cohort, so performance is exploratory.\n", encoding="utf-8")


# ============================================================
# Figure 5b: held-out DGF probability for each kidney
# ============================================================

def plot_probabilities(pred, outdir, config_text=CONFIG_TEXT):
    trajectory_columns = ["donor", "donor_m", "donor_m_s"]
    table = pred.sort_values(
        ["y", "donor_m_s", ID_COL],
        kind="stable",
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.23, top=0.95)
    fig.text(0.02, 0.98, "b", fontsize=12, fontweight="bold", va="top")
    x = np.arange(len(trajectory_columns))
    for _, row in table[table.y.eq(1)].iterrows():
        ax.plot(x, row[trajectory_columns].to_numpy(dtype=float),
                color="#A4AAB2", alpha=0.7, lw=1.2, marker="^", markersize=4, zorder=2)
    igf_colors = ("#287F8E", "#B53B5B")
    igf_handles = []
    display_labels = {case: f"IGF {i + 1}"
                      for i, case in enumerate(sorted(table.loc[table.y.eq(0), ID_COL]))}
    for i, (_, row) in enumerate(table[table.y.eq(0)].iterrows()):
        color = igf_colors[i % len(igf_colors)]
        values = row[trajectory_columns].to_numpy(dtype=float)
        line, = ax.plot(x, values, color=color, lw=2.3, marker="o", markersize=6,
                        label=display_labels[row[ID_COL]], zorder=4)
        igf_handles.append(line)

    ax.axhline(
        0.5,
        color="#999999",
        linestyle="--",
        linewidth=1,
    )

    values = table[trajectory_columns].to_numpy()
    lo = max(0, min(float(values.min()), 0.5) - 0.025)
    hi = min(1, max(float(values.max()), 0.5) + 0.025)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.08, 2.08)
    ax.set_xticks(x, ["Donor", "Donor + M", "Donor + M + S"])

    ax.set(
        ylabel="P(DGF)",
    )
    ax.grid(axis="y", color="#ECEFF2", linewidth=0.7)
    ax.set_axisbelow(True)
    dgf_handle = Line2D([], [], marker="^", color="#A4AAB2", lw=1.2,
                        label="DGF")

    ax.legend(
        handles=[*igf_handles, dgf_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    ax.spines[["top", "right"]].set_visible(False)

    save_figure(fig, outdir, "fig5b_probabilities")
    table.to_csv(outdir / "fig5b_case_predictions.csv", index=False)
    pd.DataFrame([{ID_COL: case, "display_label": label} for case, label in display_labels.items()]).to_csv(
        outdir / "fig5b_display_labels.csv", index=False)
    (outdir / "fig5b_caption.txt").write_text(
        "(b) Each line connects a case's held-out DGF probabilities across Donor, Donor + M, and Donor + M + S models. "
        f"The two IGF cases are highlighted; the {int(table.y.eq(1).sum())} DGF cases are shown in gray. "
        "The dashed line indicates the fixed 0.5 classification threshold. The probability axis is zoomed and does not start at zero. "
        f"Shared settings: {config_text}. M and S denote OCT morphology and spatial features. "
        "Probabilities are not calibrated, and settings were selected on this same cohort.\n", encoding="utf-8")


# ============================================================
# Figure 5c: correlations with the two selected S features
# ============================================================

def correlation_block(features, row_cols, spatial_cols=None):
    # Use the same complete cases throughout both heatmap blocks.
    spatial_cols = SPATIAL_COLS if spatial_cols is None else spatial_cols
    all_cols = list(dict.fromkeys([*spatial_cols, *row_cols]))

    numeric = features[all_cols].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(
            "Feature table has missing/infinite values. "
            "Resolve missingness explicitly before plotting correlations."
        )

    # Constant features produce undefined correlations, shown in gray.
    corr = numeric.corr(method="spearman")
    return corr.loc[row_cols, spatial_cols]


def draw_heatmap(ax, matrix, row_labels, title, cmap, spatial_labels=None):
    values = matrix.to_numpy()

    ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    spatial_labels = SPATIAL_LABELS if spatial_labels is None else spatial_labels
    ax.set_xticks(range(len(matrix.columns)), spatial_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(axis="both", length=0, labelsize=9)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            color = (
                "white"
                if np.isfinite(value) and abs(value) >= 0.65
                else "#222222"
            )

            ax.text(
                j, i, label,
                ha="center", va="center",
                color=color, fontsize=9,
            )

    # White cell boundaries.
    ax.set_xticks(
        np.arange(-0.5, values.shape[1], 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, values.shape[0], 1),
        minor=True,
    )
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)


def parse_display_count(value):
    if value.lower() == "all":
        return "all"
    if value.isdigit() and int(value) > 0:
        return int(value)
    raise argparse.ArgumentTypeError("Use a positive integer or all")


def select_display_features(frequency, candidates, limit, ties, group):
    ranked = frequency[frequency.feature.isin(candidates) & frequency.selection_count.gt(0)].copy()
    ranked = ranked.sort_values(["selection_count", "feature"], ascending=[False, True], kind="stable")
    ranked.insert(0, "group", group)
    ranked["frequency_order"] = np.arange(1, len(ranked) + 1)
    if ranked.empty:
        raise ValueError(f"No selected features in {group}")
    if limit != "all" and limit > len(ranked):
        raise ValueError(f"{group}: requested {limit}, but only {len(ranked)} features were ever selected; use all")
    if limit == "all":
        chosen = ranked.copy()
    elif ties == "include":
        chosen = ranked[ranked.selection_count.ge(ranked.iloc[limit - 1].selection_count)].copy()
    else:
        chosen = ranked.head(limit).copy()
    return chosen, ranked


def plot_correlations(features, outdir, spatial_frequency=None, spatial_quota=1,
                      feature_frequency=None, donor_top=3, morphology_top=2,
                      spatial_top="all", ties="include", config_text=CONFIG_TEXT):
    if feature_frequency is None:
        raise ValueError("Fold-selection frequencies are required to select display features")
    m_candidates = [column for columns in M_GROUPS.values() for column in columns]
    donor, donor_ranking = select_display_features(feature_frequency, list(DONOR_LABELS), donor_top, ties, "donor")
    morphology, m_ranking = select_display_features(feature_frequency, m_candidates, morphology_top, ties, "morphology")
    spatial, s_ranking = select_display_features(feature_frequency, SPATIAL_COLS, spatial_top, ties, "spatial")
    spatial_cols = spatial.feature.tolist()
    label_map = dict(zip(SPATIAL_COLS, SPATIAL_LABELS))
    spatial_labels = [label_map[column] for column in spatial_cols]
    selected = pd.concat([donor, morphology, spatial], ignore_index=True)
    selected.to_csv(outdir / "fig5c_displayed_features.csv", index=False)
    pd.concat([donor_ranking, m_ranking, s_ranking], ignore_index=True).to_csv(
        outdir / "fig5c_frequency_ranking.csv", index=False)
    m_labels_map = {f"{prefix}_{stat}": f"{group}: {STAT_LABELS[stat]}"
                    for prefix, group in (("minor", "Minor axis"), ("ecc", "Eccentricity"), ("sol", "Solidity"))
                    for stat in STATS}
    m_cols = morphology.feature.tolist()
    m_labels = [m_labels_map[row.feature].replace(": ", " ")
                for _, row in morphology.iterrows()]
    donor_cols, donor_labels = [], []
    plot_features = features.copy()
    race_expanded = False
    for _, row in donor.iterrows():
        if row.feature == "donor_race_code":
            # Race is nominal: correlate category indicators, not arbitrary numeric ranks.
            values = pd.to_numeric(features[row.feature], errors="raise")
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError("Missing race codes; do not treat them as category indicators")
            for value in sorted(values.unique()):
                column = f"donor_race_category_{value:g}"
                plot_features[column] = values.eq(value).astype(int)
                donor_cols.append(column)
                donor_labels.append(f"Race category {value:g}")
            race_expanded = True
        else:
            donor_cols.append(row.feature)
            short_labels = {"donor_age": "Age", "donor_cause_of_death_stroke": "Stroke"}
            donor_labels.append(short_labels.get(row.feature, DONOR_LABELS[row.feature]))
    m_corr = correlation_block(plot_features, m_cols, spatial_cols)
    donor_corr = correlation_block(plot_features, donor_cols, spatial_cols)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#DDDDDD")

    max_rows = max(len(m_cols), len(donor_cols))
    fig = plt.figure(figsize=(8.4, max(3.3, 1.5 + max_rows * 0.32)))
    fig.text(0.02, 0.98, "c", fontsize=12, fontweight="bold", va="top")

    # Donor cells have approximately the same height as M cells.
    grid = fig.add_gridspec(
        nrows=max_rows,
        ncols=3,
        width_ratios=[1, 1, 0.07],
        left=0.23, right=0.93, bottom=0.28, top=0.95,
        wspace=0.90,
    )

    ax_m = fig.add_subplot(grid[:len(m_cols), 0])
    ax_d = fig.add_subplot(grid[:len(donor_cols), 1])
    cax = fig.add_subplot(grid[:, 2])

    draw_heatmap(
        ax_m, m_corr, m_labels,
        "OCT morphology", cmap, spatial_labels,
    )

    draw_heatmap(
        ax_d, donor_corr, donor_labels,
        "Donor", cmap, spatial_labels,
    )

    colorbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(-1, 1),
            cmap=cmap,
        ),
        cax=cax,
    )
    colorbar.set_label("ρ")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    selection_note = (f"Display subset: {len(donor)} donor, {len(morphology)} morphology, {len(spatial)} spatial features; "
                      f"ranked by fold-selection frequency ({'cutoff ties included' if ties == 'include' else 'ties broken alphabetically'}).")

    save_figure(fig, outdir, "fig5c_correlations")

    pd.concat(
        [m_corr, donor_corr],
        axis=0,
    ).to_csv(outdir / "fig5c_spearman.csv")
    caption = (
        f"(c) Spearman correlations between the displayed OCT spatial features and donor and OCT morphology features "
        f"in the {len(features)}-case cohort. Features were ranked within each group by their selection frequency across "
        f"LOOCV training folds of the donor + OCT morphology + OCT spatial model ({config_text}). "
        f"Requested display limits were donor={donor_top}, morphology={morphology_top}, spatial={spatial_top}; "
        f"the displayed subset contains {len(donor)} donor, {len(morphology)} morphology and {len(spatial)} spatial features. "
        + ("Features tied at the frequency cutoff were all retained. " if ties == "include" else
           "Frequency ties were resolved alphabetically by the internal feature name to retain the requested number. ")
        + "All means all features selected in at least one training fold, not all available input features. "
        "Selection counts are recorded in the companion feature tables rather than printed on the figure. "
        "Displayed spatial descriptors are anterior measurements; Stroke denotes donor cause of death due to stroke. "
        "Display limits do not change training quotas, predictions or feature-selection results. "
        + ("Nominal race was expanded into category indicators before correlation calculation; its row count may exceed the donor-feature count. "
           if race_expanded else "")
        + ("High-occurrence area fraction is the fraction of density-band pixels occupied by lumen in at least nine reviewed frames. "
           if SPATIAL_COLS[1] in spatial_cols else "")
        + "Correlations were calculated from raw features across the cohort, not averaged across training folds. "
        "This post-selection descriptive analysis does not establish independent predictive benefit."
    )
    (outdir / "fig5c_caption.txt").write_text(caption + "\n", encoding="utf-8")
    spatial_names = {SPATIAL_COLS[0]: "anterior depth P90", SPATIAL_COLS[1]: "anterior high-occurrence area fraction"}
    s_description = "; ".join(
        f"{spatial_names[row.feature]} was selected in {int(row.selection_count)}/{len(features)} folds"
        for _, row in spatial.iterrows())
    paper_caption = (
        f"(c) Spearman correlations between {len(spatial)} OCT spatial descriptors and the "
        f"{len(donor)} donor and {len(morphology)} morphology features displayed from the {len(features)}-case cohort. "
        "Features were ranked within each group by selection frequency across LOOCV training folds of the combined donor/M/S model. "
        + ("Cutoff ties were included. " if ties == "include" else "Ties were resolved alphabetically. ")
        + f"Displayed spatial descriptors are anterior measurements; {s_description}. "
        + "The left block shows morphology and the right block shows donor features; Stroke denotes donor cause of death due to stroke. "
        + ("High-occurrence area fraction denotes the fraction of band pixels occupied by lumen in at least nine reviewed frames. "
           if SPATIAL_COLS[1] in spatial_cols else "")
        + ("Race is represented by category indicators. " if race_expanded else "")
        + "The display subset does not alter the fitted models; these post-selection correlations are descriptive and do not establish independent predictive benefit."
    )
    (outdir / "fig5c_caption_paper.txt").write_text(paper_caption + "\n", encoding="utf-8")
    print(selection_note)


# ============================================================
# Export and execution
# ============================================================

def save_figure(fig, outdir, stem):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            outdir / f"{stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "spatial_12")
    parser.add_argument("--roc-positive", choices=("IGF", "DGF"), default="IGF")
    parser.add_argument("--c-donor-top", type=parse_display_count, default=3,
                        help="Display this many highest-frequency donor features, or all ever selected")
    parser.add_argument("--c-morphology-top", type=parse_display_count, default=3)
    parser.add_argument("--c-spatial-top", type=parse_display_count, default="all")
    parser.add_argument("--c-ties", choices=("include", "truncate"), default="include")
    parser.add_argument("--c-only", action="store_true", help="Render only panel c into a separate variant directory")

    parser.add_argument(
        "--pred",
        type=Path,
        help="Optional prediction CSV; must match the selected experiment predictions.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Raw feature table; default: the experiment input snapshot.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Output directory; default: <results-dir>/fig5_paper.",
    )

    args = parser.parse_args()
    default_output = (args.results_dir / "fig5c_paper_variants" /
                      f"donor-{args.c_donor_top}_M-{args.c_morphology_top}_S-{args.c_spatial_top}_ties-{args.c_ties}"
                      if args.c_only else args.results_dir / "fig5_paper")
    args.outdir = args.outdir or default_output
    args.features = args.features or args.results_dir / "input_snapshot/kidney_spatial_candidates.csv"
    args.outdir.mkdir(parents=True, exist_ok=True)

    expected, matched, spatial_frequency, config_text = build_matched_predictions(args.results_dir, args.features)
    pred, features = load_data(args.pred, args.features, expected)
    if set(pred[ID_COL]) != set(expected[ID_COL]):
        raise ValueError("Prediction cases differ from the selected experiment")
    ordered = pred.set_index(ID_COL).loc[expected[ID_COL]]
    if not np.array_equal(ordered.y, expected[OUTCOME_COL]) or not np.allclose(
            ordered[list(MODELS)], expected[list(MODELS)], atol=1e-12, rtol=0):
        raise ValueError("Prediction CSV does not match the fixed-C/quota experiment")
    pred.to_csv(args.outdir / "oof_predictions.csv", index=False)
    matched.to_csv(args.outdir / "matched_model_configs.csv", index=False)
    spatial_frequency.to_csv(args.outdir / "spatial_selection_frequency.csv", index=False)

    if not args.c_only:
        plot_roc(pred, args.outdir, config_text, args.roc_positive)
        plot_probabilities(pred, args.outdir, config_text)
    full = matched[matched.plot_column.eq("donor_m_s")].iloc[0]
    feature_frequency = pd.read_csv(Path(full.run_dir) / "feature_selection_frequency.csv")
    plot_correlations(features, args.outdir, spatial_frequency, int(full.spatial_k), feature_frequency,
                      args.c_donor_top, args.c_morphology_top, args.c_spatial_top, args.c_ties, config_text)
    (args.outdir / "figure_config.json").write_text(json.dumps({
        "results_dir": str(args.results_dir.resolve()), "features": str(args.features.resolve()),
        "configuration": config_text, "roc_positive_class": args.roc_positive,
        "spatial_internal_column": SPATIAL_COLS[1],
        "spatial_display_name": "High-occurrence area fraction",
        "spatial_definition": "Fraction of density-band pixels with lumen count >8 (at least 9 reviewed frames)",
        "predictions_are_dgf_probabilities": True,
        "panel_c_display_limits": {"donor": args.c_donor_top, "morphology": args.c_morphology_top,
                                   "spatial": args.c_spatial_top},
        "panel_c_tie_policy": args.c_ties, "panel_c_only": args.c_only,
        "panel_c_frequency_source": str(Path(full.run_dir) / "feature_selection_frequency.csv"),
        "figure_style": "paper_minimal",
    }, indent=2) + "\n", encoding="utf-8")

    print(config_text)
    print(f"Figures saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()

