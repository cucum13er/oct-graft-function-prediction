import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from prediction_kidney_corr_auc_dim_reduction_quota_grid import (
    active_specs,
    compact_combos_for_spec,
    parse_k_values,
    parse_pca_variance_values,
    quota_run_name,
    should_run_spec_for_dim_reduction,
)
from prediction_kidney_corr_auc_quota_grid_hyperparam import param_run_name


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "models" / "33" / "logistic_corr_auc_pca_quota_grid"
DEFAULT_FEATURE_SCRIPT = Path(__file__).resolve().parent / "prediction_kidney_corr_auc_dim_reduction_feature_selection.py"

PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
    "solver": ["liblinear"],
}
PCA_VARIANCE_GRID = [0.85, 0.90, 0.95]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the corr-AUC feature-selection + PCA + tuned logistic-regression grid."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--feature-script", default=str(DEFAULT_FEATURE_SCRIPT))
    parser.add_argument("--donor-k", default="6,8,all")
    parser.add_argument("--biopsy-k", default="4,all")
    parser.add_argument("--recipient-k", default="6,8,all")
    parser.add_argument("--oct-k", default="2,3,4")
    parser.add_argument(
        "--pca-variance",
        default=",".join(str(value) for value in PCA_VARIANCE_GRID),
    )
    parser.add_argument("--cohort", default="all", choices=["both", "all", "biopsy"])
    parser.add_argument("--cases", default="1-172")
    parser.add_argument("--normalization", default="zscore", choices=["zscore", "minmax", "current"])
    parser.add_argument("--corr-threshold", default="0.80")
    parser.add_argument("--dim-reduction-min-groups", type=int, default=2)
    parser.add_argument("--one-hot", default="false", choices=["true", "false"])
    parser.add_argument("--threshold-mode", default="fixed", choices=["fixed", "youden"])
    parser.add_argument("--fixed-threshold", default="0.5")
    parser.add_argument("--max-iter", default="5000")
    parser.add_argument("--random-state", default="0")
    parser.add_argument("--skip-plots", default="true", choices=["true", "false"])
    parser.add_argument("--save-model-input", default="false", choices=["true", "false"])
    parser.add_argument("--resume", default="true", choices=["true", "false"])
    parser.add_argument("--dry-run", default="false", choices=["true", "false"])
    parser.add_argument("--compact-grid", default="true", choices=["true", "false"])
    parser.add_argument("--max-param-combinations", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of independent experiment subprocesses to run concurrently.",
    )
    return parser.parse_args()


def parameter_sets():
    keys = list(PARAM_GRID)
    for values in itertools.product(*(PARAM_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def model_params_json(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def build_command(
    args,
    params,
    quota_dir,
    run_name,
    donor_k,
    biopsy_k,
    recipient_k,
    oct_k,
    pca_variance,
):
    cmd = [
        sys.executable,
        str(Path(args.feature_script)),
        "--out-dir",
        str(quota_dir),
        "--flat-run-dir",
        "true",
        "--model-params",
        model_params_json(params),
        "--donor-k",
        donor_k,
        "--biopsy-k",
        biopsy_k,
        "--recipient-k",
        recipient_k,
        "--oct-k",
        oct_k,
        "--cohort",
        args.cohort,
        "--run-name",
        run_name,
        "--cases",
        args.cases,
        "--normalization",
        args.normalization,
        "--corr-threshold",
        str(args.corr_threshold),
        "--dim-reduction",
        "pca",
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


def read_run_summary(
    metrics_path,
    param_run,
    params,
    quota_run,
    donor_k,
    biopsy_k,
    recipient_k,
    oct_k,
    pca_variance,
):
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing run metrics file: {metrics_path}")
    summary = pd.read_csv(metrics_path)
    metadata = [
        ("ml_method_grid", "logistic"),
        ("param_run", param_run),
        ("param_json_grid", model_params_json(params)),
        ("quota_run", quota_run),
        ("quota_run_dir", str(metrics_path.parent)),
        ("donor_k_grid", donor_k),
        ("biopsy_k_grid", biopsy_k),
        ("recipient_k_grid", recipient_k),
        ("oct_k_grid", oct_k),
        ("pca_variance_grid", pca_variance),
    ]
    for index, (column, value) in enumerate(metadata):
        summary.insert(index, column, value)
    return summary


def write_combined_summary(out_dir, rows):
    if not rows:
        return None
    new_rows = pd.concat(rows, ignore_index=True, sort=False)
    out_path = out_dir / "quota_grid_summary.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[~existing["quota_run_dir"].isin(new_rows["quota_run_dir"])].copy()
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows
    combined = combined.sort_values(
        ["auc_roc", "average_precision", "param_run", "run_name", "quota_run"],
        ascending=[False, False, True, True, True],
    )
    combined.to_csv(out_path, index=False)
    return out_path


def run_job(args, out_dir, job):
    params, pca_variance, run_name, quota_run, donor_k, biopsy_k, recipient_k, oct_k = job
    param_run = param_run_name(params)
    quota_dir = out_dir / param_run / run_name / quota_run
    metrics_path = quota_dir / "metrics.csv"
    if args.resume == "true" and metrics_path.exists():
        state = "already complete"
    elif args.dry_run == "true":
        command = build_command(
            args,
            params,
            quota_dir,
            run_name,
            donor_k,
            biopsy_k,
            recipient_k,
            oct_k,
            pca_variance,
        )
        return None, "dry run: " + " ".join(command)
    else:
        command = build_command(
            args,
            params,
            quota_dir,
            run_name,
            donor_k,
            biopsy_k,
            recipient_k,
            oct_k,
            pca_variance,
        )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Experiment failed ({completed.returncode}): {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        state = "complete"
    summary = read_run_summary(
        metrics_path,
        param_run,
        params,
        quota_run,
        donor_k,
        biopsy_k,
        recipient_k,
        oct_k,
        pca_variance,
    )
    return summary, state


def main():
    args = parse_args()
    if args.dim_reduction_min_groups < 1:
        raise ValueError("--dim-reduction-min-groups must be positive")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_list = list(parameter_sets())
    if args.max_param_combinations is not None:
        params_list = params_list[: args.max_param_combinations]
    pca_values = parse_pca_variance_values(args.pca_variance)
    donor_values = parse_k_values(args.donor_k)
    biopsy_values = parse_k_values(args.biopsy_k)
    recipient_values = parse_k_values(args.recipient_k)
    oct_values = parse_k_values(args.oct_k)

    jobs = []
    for params, pca_variance, spec in itertools.product(
        params_list,
        pca_values,
        active_specs(args.cohort),
    ):
        if not should_run_spec_for_dim_reduction(spec, "pca", args.dim_reduction_min_groups):
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
            jobs.append(
                (
                    params,
                    pca_variance,
                    run_name,
                    quota_run,
                    donor_k,
                    biopsy_k,
                    recipient_k,
                    oct_k,
                )
            )
    if args.max_combinations is not None:
        jobs = jobs[: args.max_combinations]

    print(f"Running {len(jobs)} PCA/logistic hyperparameter/quota jobs", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Parameter combinations: {len(params_list)}; PCA variances: {pca_values}", flush=True)

    summaries = []
    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        future_to_job = {
            executor.submit(run_job, args, out_dir, job): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            summary, state = future.result()
            completed_count += 1
            if summary is not None:
                summaries.append(summary)
            print(f"[{completed_count}/{len(jobs)}] {state}", flush=True)
            if summaries and len(summaries) % 25 == 0:
                print(f"Updated combined summary: {write_combined_summary(out_dir, summaries)}", flush=True)

    if args.dry_run == "false":
        print(f"Final combined summary: {write_combined_summary(out_dir, summaries)}", flush=True)


if __name__ == "__main__":
    main()

