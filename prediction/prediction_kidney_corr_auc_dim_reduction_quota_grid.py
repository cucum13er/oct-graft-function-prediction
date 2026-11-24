import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "corr_auc_pca_quota_grid"
)
DEFAULT_FEATURE_SCRIPT = (
    Path(__file__).resolve().parent / "prediction_kidney_corr_auc_dim_reduction_feature_selection.py"
)

EXPERIMENT_SPECS = [
    ("data_all_donor", "clinical", "all"),
    ("data_all_recipient", "recipient", "all"),
    ("data_all_oct", "oct", "all"),
    ("data_all_donor_recipient", "clinical_recipient", "all"),
    ("data_all_donor_oct", "clinical_oct", "all"),
    ("data_all_recipient_oct", "recipient_oct", "all"),
    ("data_all_donor_recipient_oct", "clinical_recipient_oct", "all"),
    ("data_withBiopsy_donor", "clinical", "biopsy"),
    ("data_withBiopsy_biopsy", "biopsy", "biopsy"),
    ("data_withBiopsy_recipient", "recipient", "biopsy"),
    ("data_withBiopsy_oct", "oct", "biopsy"),
    ("data_withBiopsy_donor_recipient", "clinical_recipient", "biopsy"),
    ("data_withBiopsy_donor_biopsy", "clinical_biopsy", "biopsy"),
    ("data_withBiopsy_donor_oct", "clinical_oct", "biopsy"),
    ("data_withBiopsy_biopsy_recipient", "biopsy_recipient", "biopsy"),
    ("data_withBiopsy_biopsy_oct", "biopsy_oct", "biopsy"),
    ("data_withBiopsy_recipient_oct", "recipient_oct", "biopsy"),
    ("data_withBiopsy_donor_biopsy_recipient", "clinical_biopsy_recipient", "biopsy"),
    ("data_withBiopsy_donor_biopsy_oct", "clinical_biopsy_oct", "biopsy"),
    ("data_withBiopsy_donor_recipient_oct", "clinical_recipient_oct", "biopsy"),
    ("data_withBiopsy_biopsy_recipient_oct", "biopsy_recipient_oct", "biopsy"),
    ("data_withBiopsy_donor_biopsy_recipient_oct", "clinical_biopsy_recipient_oct", "biopsy"),
]

FEATURE_GROUP_BLOCKS = {
    "clinical": ["donor"],
    "biopsy": ["biopsy"],
    "recipient": ["recipient"],
    "oct": ["oct"],
    "clinical_recipient": ["donor", "recipient"],
    "clinical_biopsy": ["donor", "biopsy"],
    "clinical_oct": ["donor", "oct"],
    "biopsy_recipient": ["biopsy", "recipient"],
    "biopsy_oct": ["biopsy", "oct"],
    "recipient_oct": ["recipient", "oct"],
    "clinical_biopsy_recipient": ["donor", "biopsy", "recipient"],
    "clinical_biopsy_oct": ["donor", "biopsy", "oct"],
    "clinical_recipient_oct": ["donor", "recipient", "oct"],
    "biopsy_recipient_oct": ["biopsy", "recipient", "oct"],
    "clinical_biopsy_recipient_oct": ["donor", "biopsy", "recipient", "oct"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a quota grid for prediction_kidney_corr_auc_dim_reduction_feature_selection.py "
            "and combine all run summaries. When --dim-reduction is not none, only "
            "feature-group combinations that meet --dim-reduction-min-groups are run."
        )
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--feature-script", default=str(DEFAULT_FEATURE_SCRIPT))
    parser.add_argument("--donor-k", default="4,6,8")
    parser.add_argument("--biopsy-k", default="4,6,8")
    parser.add_argument("--recipient-k", default="4,6,8")
    parser.add_argument("--oct-k", default="2,3")
    parser.add_argument("--cohort", default="all", choices=["both", "all", "biopsy"])
    parser.add_argument("--cases", default="1-172")
    parser.add_argument("--normalization", default="zscore", choices=["zscore", "minmax", "current"])
    parser.add_argument("--corr-threshold", default="0.80")
    parser.add_argument("--dim-reduction", default="pca", choices=["none", "pca"])
    parser.add_argument("--dim-reduction-min-groups", type=int, default=2)
    parser.add_argument(
        "--pca-variance",
        default="0.85, 0.90, 0.95",
        help=(
            "Comma-separated PCA variance values. Fractions or percentages are accepted, "
            "e.g. 0.85,0.90,0.95 or 85,90,95."
        ),
    )
    parser.add_argument("--one-hot", default="false", choices=["true", "false"])
    parser.add_argument("--threshold-mode", default="fixed", choices=["fixed", "youden"])
    parser.add_argument("--fixed-threshold", default="0.5")
    parser.add_argument("--max-iter", default="5000")
    parser.add_argument("--random-state", default="0")
    parser.add_argument("--skip-plots", default="true", choices=["true", "false"])
    parser.add_argument("--save-model-input", default="false", choices=["true", "false"])
    parser.add_argument(
        "--resume",
        default="false",
        choices=["true", "false"],
        help="Skip quota combinations that already have summary_metrics.csv.",
    )
    parser.add_argument(
        "--dry-run",
        default="false",
        choices=["true", "false"],
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--compact-grid",
        default="true",
        choices=["true", "false"],
        help="Run only k combinations relevant to each experiment's feature_group.",
    )
    return parser.parse_args()


def parse_k_values(value):
    values = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "all":
            values.append("all")
        else:
            k = int(token)
            if k < 0:
                raise ValueError("Quota values must be nonnegative integers or 'all'")
            values.append(str(k))
    if not values:
        raise ValueError("Each quota list must contain at least one value")
    return values


def parse_pca_variance_values(value):
    values = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        val = float(token)
        if val > 1.0:
            val = val / 100.0
        if not (0.0 < val < 1.0):
            raise ValueError("--pca-variance values must be in (0, 1) or percentages in (0, 100)")
        values.append(f"{val:.6g}")
    if not values:
        raise ValueError("--pca-variance must contain at least one value")
    return values


def safe_pca_label(value):
    pct = float(value) * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"pca-{int(round(pct))}"
    return f"pca-{str(value).replace('.', 'p')}"


def quota_run_name(donor_k, biopsy_k, recipient_k, oct_k, pca_variance):
    return (
        f"{safe_pca_label(pca_variance)}_donor-{donor_k}_biopsy-{biopsy_k}_"
        f"recipient-{recipient_k}_oct-{oct_k}"
    )


def experiment_dir_name(run_name):
    return run_name


def active_specs(cohort):
    if cohort == "both":
        return EXPERIMENT_SPECS
    return [spec for spec in EXPERIMENT_SPECS if spec[2] == cohort]


def should_run_spec_for_dim_reduction(spec, dim_reduction, min_groups):
    if dim_reduction == "none":
        return True
    _, feature_group, _ = spec
    return len(FEATURE_GROUP_BLOCKS[feature_group]) >= min_groups


def values_for_block(block, values, active_blocks, compact_grid):
    if compact_grid == "true" and block not in active_blocks:
        return ["all"]
    return values


def compact_combos_for_spec(spec, donor_values, biopsy_values, recipient_values, oct_values, compact_grid):
    _, feature_group, _ = spec
    active_blocks = set(FEATURE_GROUP_BLOCKS[feature_group])
    return itertools.product(
        values_for_block("donor", donor_values, active_blocks, compact_grid),
        values_for_block("biopsy", biopsy_values, active_blocks, compact_grid),
        values_for_block("recipient", recipient_values, active_blocks, compact_grid),
        values_for_block("oct", oct_values, active_blocks, compact_grid),
    )


def build_command(args, quota_dir, run_name, donor_k, biopsy_k, recipient_k, oct_k, pca_variance):
    cmd = [
        sys.executable,
        str(Path(args.feature_script)),
        "--out-dir",
        str(quota_dir),
        "--flat-run-dir",
        "true",
        "--donor-k",
        donor_k,
        "--biopsy-k",
        biopsy_k,
        "--recipient-k",
        recipient_k,
        "--oct-k",
        oct_k,
        "--cohort",
        "both",
        "--run-name",
        run_name,
        "--cases",
        args.cases,
        "--normalization",
        args.normalization,
        "--corr-threshold",
        str(args.corr_threshold),
        "--dim-reduction",
        args.dim_reduction,
        "--dim-reduction-min-groups",
        str(args.dim_reduction_min_groups),
        "--pca-variance",
        str(pca_variance),
        "--one-hot",
        args.one_hot,
        "--threshold-mode",
        args.threshold_mode,
        "--fixed-threshold",
        str(args.fixed_threshold),
        "--max-iter",
        str(args.max_iter),
        "--random-state",
        str(args.random_state),
    ]
    if args.skip_plots == "true":
        cmd.append("--skip-plots")
    if args.save_model_input == "false":
        cmd.append("--no-save-model-input")
    return cmd


def read_run_summary(run_metrics_path, quota_run, donor_k, biopsy_k, recipient_k, oct_k, pca_variance):
    if not run_metrics_path.exists():
        raise FileNotFoundError(f"Missing run metrics file: {run_metrics_path}")
    summary = pd.read_csv(run_metrics_path)
    summary.insert(0, "quota_run", quota_run)
    summary.insert(1, "quota_run_dir", str(run_metrics_path.parent))
    summary.insert(2, "donor_k_grid", donor_k)
    summary.insert(3, "biopsy_k_grid", biopsy_k)
    summary.insert(4, "recipient_k_grid", recipient_k)
    summary.insert(5, "oct_k_grid", oct_k)
    summary.insert(6, "pca_variance_grid", pca_variance)
    return summary


def write_combined_summary(out_dir, rows):
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True, sort=False)
    sort_cols = [
        col
        for col in [
            "auc_roc",
            "average_precision",
            "accuracy",
            "run_name",
            "quota_run",
        ]
        if col in combined.columns
    ]
    if sort_cols:
        ascending = [False if col in {"auc_roc", "average_precision", "accuracy"} else True for col in sort_cols]
        combined = combined.sort_values(sort_cols, ascending=ascending)
    out_path = out_dir / "quota_grid_summary.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def main():
    args = parse_args()
    if args.dim_reduction_min_groups < 1:
        raise ValueError("--dim-reduction-min-groups must be positive")
    pca_variance_values = parse_pca_variance_values(args.pca_variance)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    donor_values = parse_k_values(args.donor_k)
    biopsy_values = parse_k_values(args.biopsy_k)
    recipient_values = parse_k_values(args.recipient_k)
    oct_values = parse_k_values(args.oct_k)
    jobs = []
    for pca_variance, spec in itertools.product(pca_variance_values, active_specs(args.cohort)):
        if not should_run_spec_for_dim_reduction(
            spec,
            args.dim_reduction,
            args.dim_reduction_min_groups,
        ):
            continue
        run_name, _, _ = spec
        for donor_k, biopsy_k, recipient_k, oct_k in compact_combos_for_spec(
            spec,
            donor_values,
            biopsy_values,
            recipient_values,
            oct_values,
            args.compact_grid,
        ):
            quota_run = quota_run_name(donor_k, biopsy_k, recipient_k, oct_k, pca_variance)
            jobs.append((run_name, quota_run, donor_k, biopsy_k, recipient_k, oct_k, pca_variance))
    if args.max_combinations is not None:
        jobs = jobs[: args.max_combinations]

    print(f"Running {len(jobs)} experiment/quota/PCA-variance jobs", flush=True)
    print(f"Output directory: {out_dir}", flush=True)

    summaries = []
    for index, (run_name, quota_run, donor_k, biopsy_k, recipient_k, oct_k, pca_variance) in enumerate(jobs, start=1):
        experiment_dir = out_dir / experiment_dir_name(run_name)
        quota_dir = experiment_dir / quota_run
        run_metrics_path = quota_dir / "metrics.csv"
        cmd = build_command(args, quota_dir, run_name, donor_k, biopsy_k, recipient_k, oct_k, pca_variance)
        print(f"[{index}/{len(jobs)}] {quota_run}/{run_name}", flush=True)
        if args.resume == "true" and run_metrics_path.exists():
            print("  already complete; skipping", flush=True)
            summaries.append(read_run_summary(run_metrics_path, quota_run, donor_k, biopsy_k, recipient_k, oct_k, pca_variance))
            summary_path = write_combined_summary(out_dir, summaries)
            print(f"Updated combined summary: {summary_path}", flush=True)
            continue
        if args.dry_run == "true":
            print(" ".join(cmd), flush=True)
            continue
        subprocess.run(cmd, check=True)
        summaries.append(read_run_summary(run_metrics_path, quota_run, donor_k, biopsy_k, recipient_k, oct_k, pca_variance))
        summary_path = write_combined_summary(out_dir, summaries)
        print(f"Updated combined summary: {summary_path}", flush=True)

    if args.dry_run == "true":
        return
    summary_path = write_combined_summary(out_dir, summaries)
    print(f"Final combined summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
