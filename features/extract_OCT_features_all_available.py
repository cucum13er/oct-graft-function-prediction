import argparse
import os

import pandas as pd

import extract_OCT_features as octf
from extract_OCT_features_biopsy import (
    DEFAULT_MASK_ROOT,
    DEFAULT_OUT_DIR,
    case_mask_reader,
    case_num_from_id,
    collect_mask_paths,
    normalize_case_id,
    resolve_output_path,
)


def discover_available_cases(mask_root, case_min=None, case_max=None):
    cases = []
    for name in os.listdir(mask_root):
        path = os.path.join(mask_root, name)
        if not os.path.isdir(path):
            continue
        case_id = normalize_case_id(name)
        if case_id is None:
            continue
        case_num = case_num_from_id(case_id)
        if case_num is None:
            continue
        if case_min is not None and case_num < case_min:
            continue
        if case_max is not None and case_num > case_max:
            continue
        cases.append((case_num, case_id))
    return [case_id for _, case_id in sorted(cases)]


def build_features(args):
    rows_all = []
    rows_single = []
    summaries = []

    for case_id in discover_available_cases(args.mask_root, args.case_min, args.case_max):
        case_num = case_num_from_id(case_id)
        mask_paths, summary = collect_mask_paths(
            args.mask_root,
            case_id,
            resize_width_after_156=args.resize_width_after_156,
        )
        summaries.append(summary)

        if len(mask_paths) == 0:
            print(f"Skip {case_id}: no masks found")
            continue

        resize_width = args.resize_width_after_156 if case_num is not None and case_num > 156 else None
        print(f"Processing {case_id}: {len(mask_paths)} masks")
        with case_mask_reader(resize_width=resize_width):
            rows_all.append(
                octf.process_case(
                    case_id,
                    mask_paths,
                    dx_um=args.dx_um,
                    dz_um=args.dz_um,
                    min_area_px=args.min_area_px,
                    nn_border_margin_um=args.nn_border_margin_um,
                )
            )
            rows_single.append(
                octf.process_case_single(
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
            "Extract OCT lumen features for every available KTX mask folder, "
            "without filtering on biopsy or clinical metadata."
        )
    )
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--case-min", type=int, default=None)
    parser.add_argument("--case-max", type=int, default=None)
    parser.add_argument("--dx-um", type=float, default=3.5)
    parser.add_argument("--dz-um", type=float, default=3.5)
    parser.add_argument("--min-area-px", type=int, default=20)
    parser.add_argument("--nn-border-margin-um", type=float, default=0.0)
    parser.add_argument(
        "--resize-width-after-156",
        type=int,
        default=1536,
        help="Resize masks for cases after KTX156 to this width before feature extraction.",
    )
    parser.add_argument("--out-all", default="oct_features_all_available_all.csv")
    parser.add_argument("--out-single", default="oct_features_all_available_single.csv")
    parser.add_argument("--out-summary", default="oct_features_all_available_input_summary.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.out_all = resolve_output_path(args.out_dir, args.out_all)
    args.out_single = resolve_output_path(args.out_dir, args.out_single)
    args.out_summary = resolve_output_path(args.out_dir, args.out_summary)

    df_all, df_single, df_summary = build_features(args)
    df_all.to_csv(args.out_all, index=False)
    df_single.to_csv(args.out_single, index=False)
    df_summary.to_csv(args.out_summary, index=False)

    print("Saved:", args.out_all, "shape:", df_all.shape)
    print("Saved:", args.out_single, "shape:", df_single.shape)
    print("Saved:", args.out_summary, "shape:", df_summary.shape)


if __name__ == "__main__":
    main()
