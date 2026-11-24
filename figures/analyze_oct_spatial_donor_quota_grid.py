"""Audit a completed LR grid and compare three donor/OCT model lines."""

import argparse
import itertools
import json
import os
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "spatial_12"
GROUPS = ("donor", "recipient", "morphology", "spatial")
QUOTA_COLUMNS = [f"{group}_k" for group in GROUPS]
MODELS = ("donor", "donor+morphology", "donor+morphology+spatial")
LABELS = ("Donor", "Donor + OCT M", "Donor + OCT M + OCT S")
PREFIXES = ("donor", "donor_morphology", "donor_morphology_spatial")
COLORS = ("#197B70", "#386BB0", "#AD4266")
CASE = "case_num"
OUTCOME = "IGF(0)/DGF(1)"
PROBABILITY = "p_label1_loocv"
TOLERANCE = 1e-12


def workspace_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_predictions(row, cohort):
    path = workspace_path(row["run_dir"]) / "predictions.csv"
    frame = pd.read_csv(path)
    if len(frame) != len(cohort) or frame[CASE].duplicated().any():
        raise ValueError(f"Unexpected case count or duplicate IDs: {path}")
    if set(frame[CASE]) != set(cohort[CASE]):
        raise ValueError(f"Different cases: {path}")
    frame = frame.set_index(CASE).loc[cohort[CASE]].reset_index()
    if not np.array_equal(frame[OUTCOME], cohort[OUTCOME]):
        raise ValueError(f"Outcome mismatch: {path}")
    probabilities = frame[PROBABILITY].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"Invalid probabilities: {path}")
    return frame


def audit_grid(summary, config, cohort):
    keys = ["combination", "C", *QUOTA_COLUMNS]
    if summary.duplicated(keys).any() or summary.run_dir.duplicated().any():
        raise ValueError("Duplicate settings or reused output directories")
    c_grid = config.get("C_grid", [config["logistic"]["C"]])
    expected = set()
    for combination in config["combinations"]:
        active = combination.split("+")
        ranges = [config["quotas"][group] for group in active]
        for c, quotas in itertools.product(c_grid, itertools.product(*ranges)):
            mapping = dict(zip(active, quotas))
            expected.add((combination, float(c), *(str(mapping.get(group, 0)) for group in GROUPS)))
    observed = set(summary[keys].itertuples(index=False, name=None))
    if expected != observed:
        raise ValueError(f"Grid mismatch: {len(expected - observed)} missing, {len(observed - expected)} extra")
    y = cohort[OUTCOME].to_numpy()
    if len(cohort) != config["n_cases"]:
        raise ValueError("Cohort size differs from config")
    for i, (_, row) in enumerate(summary.iterrows(), start=1):
        predictions = load_predictions(row, cohort)
        metrics = pd.read_csv(workspace_path(row.run_dir) / "metrics.csv")
        if len(metrics) != 1:
            raise ValueError(f"Invalid per-run metrics: {row.run_dir}")
        for key in keys:
            actual = metrics.iloc[0][key]
            if key.endswith("_k"):
                equal = str(actual) == row[key]
            else:
                equal = actual == row[key]
            if not equal:
                raise ValueError(f"Per-run {key} differs from summary: {row.run_dir}")
        calculated = {
            "auc_roc": roc_auc_score(y, predictions[PROBABILITY]),
            "average_precision": average_precision_score(y, predictions[PROBABILITY]),
        }
        for metric, value in calculated.items():
            if not np.isclose(value, row[metric], atol=TOLERANCE, rtol=0):
                raise ValueError(f"Prediction-derived {metric} differs from summary: {row.run_dir}")
            if not np.isclose(value, metrics.iloc[0][metric], atol=TOLERANCE, rtol=0):
                raise ValueError(f"Prediction-derived {metric} differs from per-run metrics: {row.run_dir}")
        if i % 400 == 0:
            print(f"Audited {i}/{len(summary)} runs", flush=True)
    return {
        "expected_runs": len(expected),
        "verified_runs": len(summary),
        "n_cases": len(cohort),
        "n_dgf": int(y.sum()),
        "n_igf": int((y == 0).sum()),
        "prediction_cases_and_labels_match": True,
        "all_auc_and_ap_recomputed": True,
    }


def quota_budget(frame, columns):
    return frame[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=False)


def ranked(frame, auc_column, ap_column, budget_column):
    scored = frame.assign(
        _auc=frame[auc_column].round(12),
        _ap=frame[ap_column].round(12),
        _budget=frame[budget_column].fillna(np.inf),
    )
    scored = scored.sort_values(
        ["_auc", "_ap", "_budget", "C"],
        ascending=[False, False, True, True], kind="stable",
    )
    return scored.drop(columns=["_auc", "_ap", "_budget"]).reset_index(drop=True)


def build_shared_grid(summary):
    pieces = []
    for model, prefix in zip(MODELS, PREFIXES):
        quota_columns = [f"{group}_k" for group in model.split("+")]
        selected = summary[summary.combination.eq(model)][
            ["C", *quota_columns, "auc_roc", "average_precision", "run_dir"]
        ]
        pieces.append(selected.rename(columns={
            "auc_roc": f"{prefix}_auc", "average_precision": f"{prefix}_ap",
            "run_dir": f"{prefix}_run_dir",
        }))
    shared = pieces[2].merge(
        pieces[1], on=["C", "donor_k", "morphology_k"], validate="many_to_one",
    ).merge(pieces[0], on=["C", "donor_k"], validate="many_to_one")
    if len(shared) != len(pieces[2]):
        raise ValueError("Some full-model configurations lack matched simpler models")
    shared["mean_auc"] = shared[[f"{prefix}_auc" for prefix in PREFIXES]].mean(axis=1)
    shared["mean_ap"] = shared[[f"{prefix}_ap" for prefix in PREFIXES]].mean(axis=1)
    shared["delta_auc_morphology"] = shared.donor_morphology_auc - shared.donor_auc
    shared["delta_auc_spatial"] = shared.donor_morphology_spatial_auc - shared.donor_morphology_auc
    shared["delta_auc_total"] = shared.donor_morphology_spatial_auc - shared.donor_auc
    shared["max_requested_features"] = quota_budget(
        shared, ["donor_k", "morphology_k", "spatial_k"],
    )
    shared = ranked(shared, "mean_auc", "mean_ap", "max_requested_features")
    shared.insert(0, "rank", np.arange(1, len(shared) + 1))
    return shared


def shared_rows(setting, summary):
    rows = []
    for model, prefix in zip(MODELS, PREFIXES):
        selected = summary[summary.run_dir.eq(setting[f"{prefix}_run_dir"])]
        if len(selected) != 1 or selected.iloc[0].combination != model:
            raise ValueError(f"Cannot locate matched row for {model}")
        rows.append(selected.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def selection_counts(rows):
    counts = []
    for _, row in rows.iterrows():
        folds = pd.read_csv(workspace_path(row.run_dir) / "fold_selections.csv")
        total = folds.groupby("fold").n_selected.sum()
        if len(total) != int(row.n_cases):
            raise ValueError(f"Unexpected feature-selection fold count: {row.run_dir}")
        counts.append({
            "selected_features_min": int(total.min()),
            "selected_features_mean": float(total.mean()),
            "selected_features_max": int(total.max()),
        })
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(counts)], axis=1)


def bootstrap_auc(scores, y, n_bootstrap, seed):
    # Resample the same IGF/DGF case indices for all three models.
    negative, positive = scores[y == 0], scores[y == 1]
    differences = positive.T[:, None, :] - negative.T[:, :, None]
    concordance = (differences > 0).astype(float) + 0.5 * (differences == 0)
    rng = np.random.default_rng(seed)
    result = []
    for start in range(0, n_bootstrap, 2000):
        batch = min(2000, n_bootstrap - start)
        neg = rng.integers(len(negative), size=(batch, len(negative)))
        pos = rng.integers(len(positive), size=(batch, len(positive)))
        result.append(concordance[:, neg[:, :, None], pos[:, None, :]].mean(axis=(2, 3)).T)
    return np.concatenate(result, axis=0)


def prediction_analysis(rows, cohort, output_dir, name, n_bootstrap, seed):
    cases = cohort[[CASE, OUTCOME]].copy()
    scores = []
    for _, row in rows.iterrows():
        probabilities = load_predictions(row, cohort)[PROBABILITY].to_numpy()
        scores.append(probabilities)
        cases[row.combination] = probabilities
    cases.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    scores = np.column_stack(scores)
    y = cohort[OUTCOME].to_numpy()
    draws = bootstrap_auc(scores, y, n_bootstrap, seed)
    rows = selection_counts(rows)
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    rows["auc_ci_low"] = intervals[0]
    rows["auc_ci_high"] = intervals[1]
    comparisons = []
    for base, added, label in ((0, 1, "Morphology added to donor"),
                               (1, 2, "Spatial added to donor + morphology"),
                               (0, 2, "Morphology + spatial added to donor")):
        delta = draws[:, added] - draws[:, base]
        low, high = np.quantile(delta, [0.025, 0.975])
        comparisons.append({
            "comparison": label,
            "base_combination": MODELS[base], "added_combination": MODELS[added],
            "base_C": float(rows.iloc[base].C), "added_C": float(rows.iloc[added].C),
            "delta_auc": float(rows.iloc[added].auc_roc - rows.iloc[base].auc_roc),
            "delta_auc_ci_low": float(low), "delta_auc_ci_high": float(high),
            "delta_ap": float(rows.iloc[added].average_precision - rows.iloc[base].average_precision),
        })
    comparison = pd.DataFrame(comparisons)
    rows.to_csv(output_dir / f"{name}_models.csv", index=False)
    comparison.to_csv(output_dir / f"{name}_paired_differences.csv", index=False)
    return rows, comparison, cases


def plot_results(independent, independent_cases, shared, shared_cases, differences, setting, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.42})
    for ax, rows, cases, title in (
        (axes[0], independent, independent_cases, "a  Best observed per model"),
        (axes[1], shared, shared_cases, "b  Shared C and quotas"),
    ):
        for i, (_, row) in enumerate(rows.iterrows()):
            fpr, tpr, _ = roc_curve(cases[OUTCOME], cases[row.combination])
            ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                    label=f"{LABELS[i]}: {row.auc_roc:.3f}")
        ax.plot([0, 1], [0, 1], color="#A8ADB2", lw=1, ls="--")
        ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    ax = axes[2]
    for i, row in differences.iterrows():
        ax.plot([row.delta_auc_ci_low, row.delta_auc_ci_high], [i, i], color=COLORS[i], lw=2)
        ax.scatter(row.delta_auc, i, s=38, color=COLORS[i], zorder=3)
        ax.annotate(f"{row.delta_auc:+.3f}", (row.delta_auc, i),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="#A8ADB2", ls="--", lw=1)
    ax.set_yticks(range(3), ["+ Morphology", "+ Spatial after morphology", "+ Morphology + spatial"])
    ax.invert_yaxis()
    ax.set_ylim(2.5, -0.5)
    ax.set_xlabel("Paired ROC-AUC difference")
    ax.set_title("c  Shared-setting differences", loc="left", fontsize=11, fontweight="bold", pad=12)
    for ax in axes:
        ax.grid(color="#ECEFF2", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    n_dgf = int(shared_cases[OUTCOME].sum())
    n_igf = int(shared_cases[OUTCOME].eq(0).sum())
    footer = (f"{len(shared_cases)} cases ({n_dgf} DGF, {n_igf} IGF); recipient excluded. Shared: C={setting.C:g}, "
              f"donor={setting.donor_k}, morphology={setting.morphology_k}, spatial={setting.spatial_k}.")
    fig.text(0.5, 0.065, footer, ha="center", fontsize=9, color="#444444")
    fig.text(0.5, 0.025, "Intervals: exploratory 95% paired stratified bootstrap of selected LOOCV predictions; grid selection is not corrected.",
             ha="center", fontsize=8, color="#666666")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.22)
    for suffix, dpi in (("png", 350), ("pdf", None)):
        fig.savefig(output_dir / f"donor_oct_comparison.{suffix}", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_table(rows):
    lines = ["| Model | C | Donor k | Morphology k | Spatial k | AUC | AP | Actual features (min-max) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in rows.iterrows():
        lines.append(f"| {row.combination} | {row.C:g} | {row.donor_k} | {row.morphology_k} | "
                     f"{row.spatial_k} | {row.auc_roc:.6f} | {row.average_precision:.6f} | "
                     f"{row.selected_features_min}-{row.selected_features_max} |")
    return "\n".join(lines)


def write_report(output_dir, audit, independent, shared, differences, grid, small, config, n_bootstrap, seed):
    setting = grid.iloc[0]
    increasing = (grid.delta_auc_morphology > TOLERANCE) & (grid.delta_auc_spatial > TOLERANCE)
    improving = int((grid.delta_auc_spatial > TOLERANCE).sum())
    tying = int((grid.delta_auc_spatial.abs() <= TOLERANCE).sum())
    text = [
        "# Donor/OCT LR quota-grid analysis (recipient excluded)",
        "", "## Verification",
        f"All {audit['verified_runs']} expected configurations were checked against their per-run metrics and {audit['n_cases']} case-level predictions. "
        "Cases and labels matched the input cohort; ROC-AUC and AP were independently recomputed for every run.",
        "The three target model lines contain "
        f"{len(config['C_grid']) * len(config['quotas']['donor']) * (1 + len(config['quotas']['morphology']) * (1 + len(config['quotas']['spatial'])))} configurations. "
        f"There are {len(grid)} shared-parameter triplets. No models were retrained by this analysis.",
        "", "## Independently best observed configurations", model_table(independent),
        "Each model is selected by maximum observed AUC, then AP; remaining ties prefer fewer requested features and smaller C. "
        "These rows do not use common hyperparameters, so differences cannot isolate the effect of adding OCT.",
        "", "## Best common parameters",
        "Common settings are ranked by arithmetic mean of the three AUCs, then mean AP, then fewer total requested features, then smaller C. "
        "Only active groups are included in each model: donor-only ignores morphology/spatial quotas, and donor+morphology ignores spatial quota. "
        "Scores are rounded to 12 decimals only for ranking ties, to avoid floating-point noise deciding ties.",
        f"Selected C={setting.C:g}, donor={setting.donor_k}, morphology={setting.morphology_k}, spatial={setting.spatial_k}; "
        f"mean AUC={setting.mean_auc:.6f}, mean AP={setting.mean_ap:.6f}.",
        model_table(shared),
        "", "| Shared-setting comparison | Delta AUC | Conditional 95% interval | Delta AP |",
        "|---|---:|---:|---:|",
    ]
    for _, row in differences.iterrows():
        text.append(f"| {row.comparison} | {row.delta_auc:+.6f} | "
                    f"[{row.delta_auc_ci_low:+.6f}, {row.delta_auc_ci_high:+.6f}] | {row.delta_ap:+.6f} |")
    text += [
        "", "## Whole matched grid",
        f"Spatial improved AUC in {improving}/{len(grid)} matched triplets, tied in {tying}, and worsened it in "
        f"{len(grid) - improving - tying}. Both incremental steps improved AUC in {int(increasing.sum())}/{len(grid)} triplets. "
        "Configurations share cases and often predictions, so these counts are descriptive, not independent trials or evidence of statistical significance.",
        "", "## At most six requested features", model_table(small),
        "This additional comparison uses the same mean-AUC selection rule, restricted to numeric donor+morphology+spatial quotas summing to at most six. "
        "It is separate from the unrestricted answer. The all option has no numeric cap and is excluded from this subset.",
        "", "## Interpretation and limitations",
        f"The independently best full model minus the independently best donor model has an observed AUC difference of "
        f"{independent.iloc[2].auc_roc - independent.iloc[0].auc_roc:+.6f}. "
        f"At the best common setting, morphology changes AUC by {setting.delta_auc_morphology:+.6f}, "
        f"then spatial changes it by {setting.delta_auc_spatial:+.6f}; the total change relative to donor is {setting.delta_auc_total:+.6f}. "
        "These post-selection differences are exploratory and do not by themselves establish generalizable benefit. AP differences are reported separately.",
        "Actual selected feature counts are shown above. all retains every feature remaining after fold-wise correlation pruning, "
        "not a small fixed feature budget.",
        f"The correlation threshold stays at {config['corr_threshold']}; logistic settings are {config['logistic']}. "
        "Common quotas preserve a common selection procedure and donor features at each fold, while fitted coefficients can change when groups are added.",
        f"Intervals use {n_bootstrap} paired stratified bootstrap draws (seed {seed}), resampling the same {audit['n_igf']} IGF and {audit['n_dgf']} DGF case indices across models. "
        "They condition on the already fitted LOOCV prediction vectors; models, feature selection and grid search are not repeated. "
        "They do not account for selecting the best settings on these same outcomes or for dependence between overlapping LOOCV fits. "
        f"Treat them as exploratory prediction-level uncertainty, not a validated post-selection confidence claim. Only {audit['n_igf']} IGF cases are available.",
        "", "## Outputs",
        "- independent_best_models.csv: best observed settings and metrics for each of the three models.",
        "- shared_parameter_ranking.csv: all common settings ranked by mean AUC/AP.",
        "- shared_best_models.csv: three models at the best common setting.",
        "- shared_small_budget_models.csv: common settings with at most six requested features.",
        "- *_predictions.csv and *_paired_differences.csv: aligned predictions and paired AUC differences.",
        "- donor_oct_comparison.png/.pdf: one comparison figure.",
        "- audit.json: grid verification counts; best_no_recipient_combinations.csv: best results across all seven non-recipient combinations.",
    ]
    (output_dir / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=172)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    output_dir = args.output_dir or args.results_dir / "analysis_no_recipient"
    summary = pd.read_csv(args.results_dir / "quota_grid_summary.csv",
                          dtype={column: str for column in QUOTA_COLUMNS})
    config = json.loads((args.results_dir / "config.json").read_text())
    cohort = pd.read_csv(workspace_path(config["input_csv"]))
    audit = audit_grid(summary, config, cohort)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    summary["requested_features"] = quota_budget(summary, QUOTA_COLUMNS)
    best = ranked(summary, "auc_roc", "average_precision", "requested_features").drop_duplicates("combination")
    best[~best.combination.str.contains("recipient")].to_csv(output_dir / "best_no_recipient_combinations.csv", index=False)
    independent = best.set_index("combination", drop=False).loc[list(MODELS)].reset_index(drop=True)
    grid = build_shared_grid(summary)
    grid.to_csv(output_dir / "shared_parameter_ranking.csv", index=False)
    small_grid = grid[grid.max_requested_features <= 6]
    if small_grid.empty:
        raise ValueError("No matched configuration with at most six requested features")
    independent, _, independent_cases = prediction_analysis(
        independent, cohort, output_dir, "independent_best", args.bootstrap, args.seed,
    )
    shared, differences, shared_cases = prediction_analysis(
        shared_rows(grid.iloc[0], summary), cohort, output_dir, "shared_best", args.bootstrap, args.seed,
    )
    small, _, _ = prediction_analysis(
        shared_rows(small_grid.iloc[0], summary), cohort, output_dir, "shared_small_budget", args.bootstrap, args.seed,
    )
    plot_results(independent, independent_cases, shared, shared_cases, differences, grid.iloc[0], output_dir)
    write_report(output_dir, audit, independent, shared, differences, grid, small, config, args.bootstrap, args.seed)
    print("\nIndependently best:")
    print(model_table(independent))
    print("\nBest common setting:")
    print(model_table(shared))
    print("\nPaired shared-setting differences:")
    print(differences.to_string(index=False))
    print(f"\nSaved analysis to {output_dir}")


if __name__ == "__main__":
    main()
