import argparse
import glob
import os
import re
from contextlib import contextmanager


import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize

import extract_OCT_features as octf


DEFAULT_CLINICAL_CSV = "data/raw/biopsy_source.csv"
DEFAULT_MASK_ROOT = "data/raw/lumen_masks"
DEFAULT_OUT_DIR = "data/processed"

CASE_COL = "Unnamed: 0"
BIOPSY_COLUMNS = [
    "Number of Glomeruli",
    "# of globally sclerosed glomeruli",
    "% of globally scleorosed glomeruli",
    "Interstitial fibrosis (and tubular atrophy) = IFTA",
    "Vascular disease",
    "Arterioral hylinosis",
    "Cortical necrosis",
    "Fibrin thrombi",
    "Number of arteries",
]
NO_BIOPSY_PATTERN = re.compile(r"\bno\s*(bx|biopsy)\s*taken\b", re.IGNORECASE)
REGION_NAMES = {
    "PUL",
    "PUM",
    "PM1M",
    "PM1L",
    "PM2M",
    "PM2L",
    "PLM",
    "PLL",
    "AUL",
    "AUM",
    "AM1L",
    "AM1M",
    "AM2L",
    "AM2M",
    "ALL",
    "ALM",
}
IMAGE_EXTENSIONS = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")


def normalize_case_id(value):
    if pd.isna(value):
        return None
    match = re.search(r"KTX\s*0*(\d+)", str(value), re.IGNORECASE)
    if match is None:
        return None
    return f"KTX{int(match.group(1))}"


def case_num_from_id(case_id):
    match = re.fullmatch(r"KTX(\d+)", case_id)
    if match is None:
        return None
    return int(match.group(1))


def case_dir_candidates(mask_root, case_id):
    case_num = case_num_from_id(case_id)
    candidates = [os.path.join(mask_root, case_id)]
    if case_num is not None:
        candidates.append(os.path.join(mask_root, f"KTX{case_num:03d}"))
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def row_has_no_biopsy(row):
    for col in BIOPSY_COLUMNS:
        value = row.get(col)
        if pd.isna(value):
            continue
        if NO_BIOPSY_PATTERN.search(str(value).strip()):
            return True
    return False


def row_has_any_biopsy_value(row):
    for col in BIOPSY_COLUMNS:
        value = row.get(col)
        if pd.isna(value):
            continue
        if str(value).strip():
            return True
    return False


def load_biopsy_clinical_rows(clinical_csv):
    df = pd.read_csv(clinical_csv)
    missing = [col for col in [CASE_COL] + BIOPSY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {missing}")

    df = df.copy()
    df["case_num"] = df[CASE_COL].map(normalize_case_id)
    keep = (
        df["case_num"].notna()
        & df.apply(row_has_any_biopsy_value, axis=1)
        & ~df.apply(row_has_no_biopsy, axis=1)
    )
    return df.loc[keep].copy()


def glob_images(root, recursive=False):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(root, "**", ext) if recursive else os.path.join(root, ext)
        paths.extend(glob.glob(pattern, recursive=recursive))
    return sorted(paths)


def collect_region_subfolder_masks(case_dir):
    paths = []
    present_regions = []
    ignored_dirs = []
    for name in sorted(os.listdir(case_dir)):
        path = os.path.join(case_dir, name)
        if not os.path.isdir(path):
            continue
        if name.upper() in REGION_NAMES:
            present_regions.append(name.upper())
            paths.extend(glob_images(path, recursive=False))
        else:
            ignored_dirs.append(name)
    return sorted(paths), present_regions, ignored_dirs


def collect_post156_masks(case_dir):
    masks_dir = os.path.join(case_dir, "masks")
    if os.path.isdir(masks_dir):
        return glob_images(masks_dir, recursive=True), masks_dir, "masks_subdir"

    root_images = glob_images(case_dir, recursive=False)
    mask_paths = [
        path for path in root_images if "mask" in os.path.splitext(os.path.basename(path))[0].lower()
    ]
    return sorted(mask_paths), case_dir, "root_mask_name_filter"


def collect_mask_paths(mask_root, case_id, resize_width_after_156=1536):
    case_num = case_num_from_id(case_id)
    candidates = case_dir_candidates(mask_root, case_id)
    case_dir = next((path for path in candidates if os.path.isdir(path)), candidates[0])
    summary = {
        "case_num": case_id,
        "case_dir": case_dir,
        "case_dir_candidates": ";".join(candidates),
        "mask_root_used": "",
        "mask_layout": "",
        "num_mask_files": 0,
        "resize_to_width": "",
        "regions_found": "",
        "ignored_subdirs": "",
    }
    if case_num is None or not os.path.isdir(case_dir):
        summary["mask_layout"] = "missing_case_dir"
        return [], summary

    if 152 <= case_num <= 156:
        mask_paths, regions, ignored = collect_region_subfolder_masks(case_dir)
        summary["mask_root_used"] = case_dir
        summary["mask_layout"] = "region_subdirs"
        summary["regions_found"] = ";".join(regions)
        summary["ignored_subdirs"] = ";".join(ignored)
    elif case_num > 156:
        mask_paths, root_used, layout = collect_post156_masks(case_dir)
        summary["mask_root_used"] = root_used
        summary["mask_layout"] = layout
        summary["resize_to_width"] = str(resize_width_after_156)
    else:
        mask_paths = glob_images(case_dir, recursive=False)
        summary["mask_root_used"] = case_dir
        summary["mask_layout"] = "case_dir_pngs"

    summary["num_mask_files"] = len(mask_paths)
    return mask_paths, summary


def read_binary_mask_resized(path, resize_width=None):
    mask = imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)

    if resize_width is not None and mask.shape[1] != resize_width:
        height = int(round(mask.shape[0] * (resize_width / mask.shape[1])))
        mask = resize(
            mask,
            (height, resize_width),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        mask = (mask > 0).astype(np.uint8)
    return mask


@contextmanager
def case_mask_reader(resize_width=None):
    original_reader = octf.read_binary_mask
    octf.read_binary_mask = lambda path: read_binary_mask_resized(path, resize_width=resize_width)
    try:
        yield
    finally:
        octf.read_binary_mask = original_reader


def resolve_output_path(out_dir, out_path):
    if os.path.isabs(out_path):
        return out_path
    return os.path.join(out_dir, out_path)


def build_features(args):
    clinical = load_biopsy_clinical_rows(args.clinical_csv)
    rows_all = []
    rows_single = []
    summaries = []

    for _, clinical_row in clinical.iterrows():
        case_id = clinical_row["case_num"]
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
            feature_all = octf.process_case(
                case_id,
                mask_paths,
                dx_um=args.dx_um,
                dz_um=args.dz_um,
                min_area_px=args.min_area_px,
                nn_border_margin_um=args.nn_border_margin_um,
            )
            feature_single = octf.process_case_single(
                case_id,
                mask_paths,
                dx_um=args.dx_um,
                dz_um=args.dz_um,
                min_area_px=args.min_area_px,
                nn_border_margin_um=args.nn_border_margin_um,
            )

        rows_all.append(feature_all)
        rows_single.append(feature_single)

    return (
        pd.DataFrame(rows_all),
        pd.DataFrame(rows_single),
        pd.DataFrame(summaries),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract OCT lumen features for Georgetown cases with biopsy data, "
            "excluding empty and 'No Bx Taken' biopsy rows."
        )
    )
    parser.add_argument("--clinical-csv", default=DEFAULT_CLINICAL_CSV)
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
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
    parser.add_argument("--out-all", default="biopsy_oct_features_all.csv")
    parser.add_argument("--out-single", default="biopsy_oct_features_single.csv")
    parser.add_argument("--out-summary", default="biopsy_oct_feature_input_summary.csv")
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

