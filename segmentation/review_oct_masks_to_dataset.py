import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CASES = "162-172"
DEFAULT_REVIEW_FOLDER = "OCT_review_top_lumen"
DEFAULT_MASK_SUFFIX = "_mask"
DEFAULT_ALPHA = 0.35
DEFAULT_DISPLAY_WIDTH = 1800
DEFAULT_OVERWRITE = False
DEFAULT_DRY_RUN = False
DEFAULT_START_INDEX = 0
WINDOW_NAME = "OCT image/mask review"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_cases(values):
    cases = []
    for value in values:
        value = str(value).strip()
        if value.startswith("cases="):
            value = value.split("=", 1)[1]
        value = value.strip("[](){}")
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                step = 1 if end >= start else -1
                cases.extend(range(start, end + step, step))
            else:
                cases.append(int(item))
    return list(dict.fromkeys(cases))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively review OCT images and matching masks, then copy kept pairs "
            "into a case-wise target dataset folder."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[DEFAULT_CASES],
        help=(
            "Cases to review, for example: --cases 161-172, "
            "--cases 161,162,170, or --cases '[161-172]'. "
            f"Default: {DEFAULT_CASES}."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing case folders named exactly case{number}.",
    )
    parser.add_argument(
        "--review-folder",
        default=DEFAULT_REVIEW_FOLDER,
        help="Review subfolder inside each case directory.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Output root. Each kept case is copied into target-root/case{number}.",
    )
    parser.add_argument(
        "--mask-suffix",
        default=DEFAULT_MASK_SUFFIX,
        help="Mask filename suffix inserted before the extension.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Mask overlay opacity.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=DEFAULT_DISPLAY_WIDTH,
        help="Maximum display width for the combined review figure.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=DEFAULT_OVERWRITE,
        help="Overwrite existing copied files in the target folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help="Review images without copying kept files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help="Start reviewing from this zero-based index within each case.",
    )
    args = parser.parse_args()
    args.cases = parse_cases(args.cases)
    if not args.cases:
        raise ValueError("No valid cases were provided.")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.display_width < 300:
        raise ValueError("--display-width must be at least 300.")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    return args


def collect_pairs(case_dir, mask_suffix):
    image_dir = case_dir / "images"
    mask_dir = case_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        return []

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_name = f"{image_path.stem}{mask_suffix}{image_path.suffix}"
        mask_path = mask_dir / mask_name
        if not mask_path.exists():
            alt_matches = sorted(mask_dir.glob(f"{image_path.stem}{mask_suffix}.*"))
            if alt_matches:
                mask_path = alt_matches[0]
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            print(f"[missing mask] {image_path.name}")
    return pairs


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def normalize_mask(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8) * 255


def resize_to_match(mask, shape):
    height, width = shape[:2]
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def make_label(text, width, height=36):
    label = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        label,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return label


def build_review_canvas(image_path, mask_path, case_num, index, total, alpha, display_width):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    image_bgr = to_bgr(image)
    mask_binary = resize_to_match(normalize_mask(mask), image_bgr.shape)

    mask_color = np.zeros_like(image_bgr)
    mask_color[:, :, 2] = mask_binary
    overlay = cv2.addWeighted(image_bgr, 1.0, mask_color, alpha, 0)
    mask_vis = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)

    panels = [
        np.vstack([make_label("image", image_bgr.shape[1]), image_bgr]),
        np.vstack([make_label("mask", mask_vis.shape[1]), mask_vis]),
        np.vstack([make_label("overlay", overlay.shape[1]), overlay]),
    ]
    canvas = np.hstack(panels)

    header_text = (
        f"case{case_num}  {index + 1}/{total}  {image_path.name}     "
        "keys: k/space=keep, s=skip, q/esc=quit"
    )
    header = make_label(header_text, canvas.shape[1], height=42)
    canvas = np.vstack([header, canvas])

    if canvas.shape[1] > display_width:
        scale = display_width / canvas.shape[1]
        new_size = (display_width, max(1, int(canvas.shape[0] * scale)))
        canvas = cv2.resize(canvas, new_size, interpolation=cv2.INTER_AREA)
    return canvas


def copy_pair(image_path, mask_path, target_case_dir, overwrite):
    target_case_dir.mkdir(parents=True, exist_ok=True)
    image_target = target_case_dir / image_path.name
    mask_target = target_case_dir / mask_path.name

    if not overwrite and (image_target.exists() or mask_target.exists()):
        return False, "exists"

    shutil.copy2(image_path, image_target)
    shutil.copy2(mask_path, mask_target)
    return True, "copied"


def append_log(log_path, row):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "action",
                "image_path",
                "mask_path",
                "target_image",
                "target_mask",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def review_case(case_num, args):
    case_dir = args.source_root / f"case{case_num}" / args.review_folder
    pairs = collect_pairs(case_dir, args.mask_suffix)
    if not pairs:
        print(f"[skip case{case_num}] no valid image/mask pairs at {case_dir}")
        return 0, 0

    target_case_dir = args.target_root / f"case{case_num}"
    root_log_path = args.target_root / "review_log.csv"
    case_log_path = target_case_dir / "review_log.csv"
    kept = 0
    reviewed = 0
    print(f"[case{case_num}] reviewing {len(pairs)} pairs from {case_dir}")

    for index, (image_path, mask_path) in enumerate(pairs[args.start_index :], start=args.start_index):
        canvas = build_review_canvas(
            image_path=image_path,
            mask_path=mask_path,
            case_num=case_num,
            index=index,
            total=len(pairs),
            alpha=args.alpha,
            display_width=args.display_width,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in {ord("q"), 27}:
            print("[quit] stopped by user")
            return kept, reviewed
        if key in {ord("k"), ord(" "), 13}:
            reviewed += 1
            if args.dry_run:
                copied, status = True, "dry-run"
            else:
                copied, status = copy_pair(image_path, mask_path, target_case_dir, args.overwrite)
            if copied:
                kept += 1
            row = {
                "case": f"case{case_num}",
                "action": f"keep:{status}",
                "image_path": image_path,
                "mask_path": mask_path,
                "target_image": target_case_dir / image_path.name,
                "target_mask": target_case_dir / mask_path.name,
            }
            append_log(root_log_path, row)
            if copied and not args.dry_run:
                append_log(case_log_path, row)
            print(f"[keep:{status}] case{case_num} {image_path.name}")
        else:
            reviewed += 1
            append_log(
                root_log_path,
                {
                    "case": f"case{case_num}",
                    "action": "skip",
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "target_image": "",
                    "target_mask": "",
                },
            )
            print(f"[skip] case{case_num} {image_path.name}")

    return kept, reviewed


def main():
    args = parse_args()
    total_kept = 0
    total_reviewed = 0
    try:
        for case_num in args.cases:
            kept, reviewed = review_case(case_num, args)
            total_kept += kept
            total_reviewed += reviewed
    finally:
        cv2.destroyAllWindows()
    print(f"Done. Reviewed {total_reviewed} pairs, kept {total_kept}.")


if __name__ == "__main__":
    main()
