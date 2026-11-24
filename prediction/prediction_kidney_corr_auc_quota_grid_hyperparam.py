import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ml_model_factory import load_local_env_file
from prediction_kidney_corr_auc_quota_grid import (
    FEATURE_GROUP_BLOCKS,
    active_specs,
    experiment_dir_name,
    parse_k_values,
    quota_run_name,
    values_for_block,
)


DEFAULT_FEATURE_SCRIPT = Path(__file__).resolve().parent / "prediction_kidney_corr_auc_feature_selection_ml.py"
DEFAULT_OUT_BASE = Path(__file__).resolve().parents[1] / "results" / "models" / "52"


def compact_combos_for_spec(spec, donor_values, biopsy_values, recipient_values, oct_values, compact_grid):
    _, feature_group, _ = spec
    active_blocks = set(FEATURE_GROUP_BLOCKS[feature_group])
    return itertools.product(
        values_for_block("donor", donor_values, active_blocks, compact_grid),
        values_for_block("biopsy", biopsy_values, active_blocks, compact_grid),
        values_for_block("recipient", recipient_values, active_blocks, compact_grid),
        values_for_block("oct", oct_values, active_blocks, compact_grid),
    )


def param_grid_combinations(param_grid):
    keys = list(param_grid)
    values = [param_grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def value_slug(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip().lower()
    text = text.replace(".", "p").replace("-", "m")
    text = "".join(ch if ch.isalnum() else "-" for ch in text)
    return "-".join(part for part in text.split("-") if part)


def param_run_name(params):
    if not params:
        return "params-default"
    parts = [f"{key}-{value_slug(params[key])}" for key in sorted(params)]
    return "params-" + "_".join(parts)


def model_params_json(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def build_command(args, method, params, quota_dir, run_name, donor_k, biopsy_k, recipient_k, oct_k):
    cmd = [
        sys.executable,
        str(Path(args.feature_script)),
        "--out-dir",
        str(quota_dir),
        "--flat-run-dir",
        "true",
        "--ml-method",
        method,
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


def read_run_summary(run_metrics_path, method, param_run, params, quota_run, donor_k, biopsy_k, recipient_k, oct_k):
    if not run_metrics_path.exists():
        raise FileNotFoundError(f"Missing run metrics file: {run_metrics_path}")
    summary = pd.read_csv(run_metrics_path)
    summary.insert(0, "ml_method_grid", method)
    summary.insert(1, "param_run", param_run)
    summary.insert(2, "param_json_grid", model_params_json(params))
    summary.insert(3, "quota_run", quota_run)
    summary.insert(4, "quota_run_dir", str(run_metrics_path.parent))
    summary.insert(5, "donor_k_grid", donor_k)
    summary.insert(6, "biopsy_k_grid", biopsy_k)
    summary.insert(7, "recipient_k_grid", recipient_k)
    summary.insert(8, "oct_k_grid", oct_k)
    return summary


def write_combined_summary(out_dir, rows):
    if not rows:
        return None
    new_rows = pd.concat(rows, ignore_index=True, sort=False)
    out_path = out_dir / "quota_grid_summary.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if "quota_run_dir" in existing.columns and "quota_run_dir" in new_rows.columns:
            existing = existing[~existing["quota_run_dir"].isin(new_rows["quota_run_dir"])].copy()
            combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
        else:
            combined = new_rows
    else:
        combined = new_rows

    sort_cols = [
        col
        for col in [
            "auc_roc",
            "average_precision",
            "accuracy",
            "param_run",
            "run_name",
            "quota_run",
        ]
        if col in combined.columns
    ]
    if sort_cols:
        ascending = [
            False if col in {"auc_roc", "average_precision", "accuracy"} else True
            for col in sort_cols
        ]
        combined = combined.sort_values(sort_cols, ascending=ascending)
    combined.to_csv(out_path, index=False)
    return out_path


def parse_args(method, out_dir, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-dir", default=str(out_dir))
    parser.add_argument("--feature-script", default=str(DEFAULT_FEATURE_SCRIPT))
    parser.add_argument("--donor-k", default="6,8,all")
    parser.add_argument("--biopsy-k", default="4,all")
    parser.add_argument("--recipient-k", default="6,8,all")
    parser.add_argument("--oct-k", default="2,3,4")
    parser.add_argument("--cohort", default="all", choices=["both", "all", "biopsy"])
    parser.add_argument("--cases", default="1-172")
    parser.add_argument("--normalization", default="zscore", choices=["zscore", "minmax", "current"])
    parser.add_argument("--corr-threshold", default="0.80")
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
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--extra-model-params",
        default="",
        help=(
            "Optional JSON merged into every hyperparameter combination. "
            "Useful for one-off smoke tests."
        ),
    )
    args = parser.parse_args()
    args.ml_method = method
    return args


def run_hyperparam_grid(method, param_grid, out_dir_name, description=None):
    default_out_dir = DEFAULT_OUT_BASE / out_dir_name
    args = parse_args(
        method,
        default_out_dir,
        description or f"Run corr_auc quota/hyperparameter grid for {method}.",
    )
    load_local_env_file()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")

    extra_params = json.loads(args.extra_model_params) if args.extra_model_params else {}
    param_sets = []
    for params in param_grid_combinations(param_grid):
        merged = dict(params)
        merged.update(extra_params)
        param_sets.append(merged)
    if args.max_param_combinations is not None:
        param_sets = param_sets[: args.max_param_combinations]

    donor_values = parse_k_values(args.donor_k)
    biopsy_values = parse_k_values(args.biopsy_k)
    recipient_values = parse_k_values(args.recipient_k)
    oct_values = parse_k_values(args.oct_k)

    jobs = []
    for params in param_sets:
        param_run = param_run_name(params)
        for spec in active_specs(args.cohort):
            run_name, _, _ = spec
            for donor_k, biopsy_k, recipient_k, oct_k in compact_combos_for_spec(
                spec,
                donor_values,
                biopsy_values,
                recipient_values,
                oct_values,
                args.compact_grid,
            ):
                quota_run = quota_run_name(donor_k, biopsy_k, recipient_k, oct_k)
                jobs.append((params, param_run, run_name, quota_run, donor_k, biopsy_k, recipient_k, oct_k))
    if args.max_combinations is not None:
        jobs = jobs[: args.max_combinations]

    print(f"Running {len(jobs)} {method} hyperparameter/quota jobs", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Parameter combinations: {len(param_sets)}", flush=True)
    for params in param_sets:
        print(f"  {param_run_name(params)}: {model_params_json(params)}", flush=True)

    def run_job(job):
        params, param_run, run_name, quota_run, donor_k, biopsy_k, recipient_k, oct_k = job
        quota_dir = out_dir / param_run / experiment_dir_name(run_name) / quota_run
        run_metrics_path = quota_dir / "metrics.csv"
        cmd = build_command(args, method, params, quota_dir, run_name, donor_k, biopsy_k, recipient_k, oct_k)
        if args.resume == "true" and run_metrics_path.exists():
            state = "already complete"
        elif args.dry_run == "true":
            return None, "dry run: " + " ".join(cmd)
        else:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Experiment failed ({completed.returncode}): {' '.join(cmd)}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            state = "complete"
        return (
            read_run_summary(
                run_metrics_path,
                method,
                param_run,
                params,
                quota_run,
                donor_k,
                biopsy_k,
                recipient_k,
                oct_k,
            ),
            state,
        )

    summaries = []
    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        future_to_job = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            summary, state = future.result()
            completed_count += 1
            if summary is not None:
                summaries.append(summary)
            print(f"[{completed_count}/{len(jobs)}] {state}", flush=True)
            if summaries and len(summaries) % 25 == 0:
                print(f"Updated combined summary: {write_combined_summary(out_dir, summaries)}", flush=True)

    if args.dry_run == "true":
        return
    summary_path = write_combined_summary(out_dir, summaries)
    print(f"Final combined summary: {summary_path}", flush=True)
