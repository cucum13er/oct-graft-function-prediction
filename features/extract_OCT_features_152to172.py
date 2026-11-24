import argparse
import glob
import os
import re

import pandas as pd

from extract_OCT_features import process_case, process_case_single


DEFAULT_ROOT = "data/raw/capsule_aligned"


def parse_case_num(case_dir):
    match = re.fullmatch(r"case(\d+)", os.path.basename(case_dir))
    if match is None:
        return None
    return int(match.group(1))


def discover_case_dirs(root_dir, case_min=152, case_max=172):
    case_dirs = []
    for path in glob.glob(os.path.join(root_dir, "case*")):
        if not os.path.isdir(path):
            continue
        case_num = parse_case_num(path)
        if case_num is None:
            continue
        if case_min <= case_num <= case_max:
            case_dirs.append((case_num, path))
    return sorted(case_dirs)


def get_lumen_mask_root(case_dir, case_num):
    if 152 <= case_num <= 156:
        return os.path.join(
            case_dir,
            "capsule_line_aligned_regional",
            "lumen_seg",
            "masks",
        )
    return os.path.join(case_dir, "OCT_review_top_lumen", "masks")


def collect_mask_paths(mask_root):
    exts = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    mask_paths = []
    for ext in exts:
        mask_paths.extend(glob.glob(os.path.join(mask_root, "**", ext), recursive=True))
    return sorted(mask_paths)


def resolve_output_path(out_dir, out_path):
    if os.path.isabs(out_path):
        return out_path
    return os.path.join(out_dir, out_path)


def build_features(args):
    rows_all = []
    rows_single = []
    summaries = []

    for case_num, case_dir in discover_case_dirs(args.root_dir, args.case_min, args.case_max):
        case_id = f"KTX{case_num}"
        mask_root = get_lumen_mask_root(case_dir, case_num)
        mask_paths = collect_mask_paths(mask_root)

        summary = {
            "case_num": case_id,
            "case_dir": case_dir,
            "mask_root": mask_root,
            "num_mask_files": len(mask_paths),
        }
        summaries.append(summary)

        if len(mask_paths) == 0:
            print(f"Skip {case_id}: no masks under {mask_root}")
            continue

        print(f"Processing {case_id}: {len(mask_paths)} masks")
        rows_all.append(
            process_case(
                case_id,
                mask_paths,
                dx_um=args.dx_um,
                dz_um=args.dz_um,
                min_area_px=args.min_area_px,
                nn_border_margin_um=args.nn_border_margin_um,
            )
        )
        rows_single.append(
            process_case_single(
                case_id,
                mask_paths,
                dx_um=args.dx_um,
                dz_um=args.dz_um,
                min_area_px=args.min_area_px,
                nn_border_margin_um=args.nn_border_margin_um,
            )
        )

    return pd.DataFrame(rows_all), pd.DataFrame(rows_single), pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract OCT lumen features from Georgetown real cases with mixed "
            "capsule-aligned and top-lumen mask layouts."
        )
    )
    parser.add_argument("--root-dir", default=DEFAULT_ROOT)
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help=(
            "Directory for output CSVs. Defaults to --root-dir. "
            "Ignored for output paths given as absolute file paths."
        ),
    )
    parser.add_argument("--case-min", type=int, default=152)
    parser.add_argument("--case-max", type=int, default=172)
    parser.add_argument("--dx-um", type=float, default=3.5)
    parser.add_argument("--dz-um", type=float, default=3.5)
    parser.add_argument("--min-area-px", type=int, default=20)
    parser.add_argument("--nn-border-margin-um", type=float, default=0.0)
    parser.add_argument(
        "--out-all",
        default="georgetown_real_case_features_all.csv",
        help=(
            "CSV with pooled lumen measurements across all masks per case. "
            "Relative paths are saved under --out-dir."
        ),
    )
    parser.add_argument(
        "--out-single",
        default="georgetown_real_case_features_single.csv",
        help=(
            "CSV with features from the single largest-lumen-area mask per case. "
            "Relative paths are saved under --out-dir."
        ),
    )
    parser.add_argument(
        "--out-summary",
        default="georgetown_real_case_feature_input_summary.csv",
        help=(
            "CSV recording mask roots and mask counts used for each case. "
            "Relative paths are saved under --out-dir."
        ),
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.root_dir
    os.makedirs(out_dir, exist_ok=True)
    args.out_all = resolve_output_path(out_dir, args.out_all)
    args.out_single = resolve_output_path(out_dir, args.out_single)
    args.out_summary = resolve_output_path(out_dir, args.out_summary)

    df_all, df_single, df_summary = build_features(args)

    df_all.to_csv(args.out_all, index=False)
    df_single.to_csv(args.out_single, index=False)
    df_summary.to_csv(args.out_summary, index=False)

    print("Saved:", args.out_all, "shape:", df_all.shape)
    print("Saved:", args.out_single, "shape:", df_single.shape)
    print("Saved:", args.out_summary, "shape:", df_summary.shape)


if __name__ == "__main__":
    main()
