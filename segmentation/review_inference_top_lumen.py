import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from inference import load_and_preprocess, pred_to_mask
from model import Attention_ResUNet
from review_inference import (
    RECT_STATE,
    WINDOW_NAME,
    apply_rectangles_to_mask,
    build_full_resized_mask,
    collect_image_paths,
    create_review_image,
    prepare_working_image,
    project_mask_to_original,
    rectangle_mouse_callback,
    render_rectangles,
    review_key,
    save_acceptance,
    split_resized_image_into_patches,
    stitch_patch_masks,
)


MANIFEST_FIELDS = [
    "image_path",
    "cache_key",
    "config_signature",
    "source_mtime_ns",
    "source_size",
    "lumen_pixels",
    "resized_mask_path",
    "original_mask_path",
] 
DEFAULT_REFERENCE_FOV_WIDTH_MM = 5.52
DEFAULT_REFERENCE_WIDTH_PX = 1587
DEFAULT_SOURCE_FOV_WIDTH_MM = 8.0
DEFAULT_CROP_HEIGHT = 256

def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Attention_ResUNet on all frames, cache predicted masks, rank them by "
            "predicted lumen area, and only review/save the top fraction."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder of images to review",
    )
    parser.add_argument(
        "--image-list",
        type=Path,
        help="Text file containing one image path per line",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional explicit image paths",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to Attention_ResUNet weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Accepted images, masks, cache, and logs are saved here",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("W", "H"),
        help="Patch size fed into the model",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help=(
            "Cropped review region size before patch inference. If omitted, width "
            "uses --target-width and height uses --crop-height."
        ),
    )
    parser.add_argument(
        "--row-range",
        type=int,
        nargs=2,
        default=(100, 1000),
        metavar=("START", "END"),
        help="Use only rows [START:END] of the input before resizing/cropping",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help=(
            "Resize width before finding the brightest crop region. If omitted, it "
            "is calculated from the physical FOV arguments."
        ),
    )
    parser.add_argument(
        "--reference-fov-width-mm",
        type=float,
        default=DEFAULT_REFERENCE_FOV_WIDTH_MM,
        help="Old/reference physical FOV width in mm.",
    )
    parser.add_argument(
        "--reference-width-px",
        type=int,
        default=DEFAULT_REFERENCE_WIDTH_PX,
        help="Old/reference image width in pixels.",
    )
    parser.add_argument(
        "--source-fov-width-mm",
        type=float,
        default=DEFAULT_SOURCE_FOV_WIDTH_MM,
        help="Current/source physical FOV width in mm.",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=DEFAULT_CROP_HEIGHT,
        help="Crop height used when --crop-size is omitted.",
    )
    parser.add_argument(
        "--trim-black-threshold",
        type=int,
        default=0,
        help=(
            "Trim pure-black side columns before resizing by keeping columns whose max "
            "intensity is greater than this value. Use -1 to disable."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--batch-size", type=int, default=24, help="Prediction batch size")
    parser.add_argument("--min-area", type=int, default=40, help="Ignore tiny connected regions")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Mask overlay alpha for the review window",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1400,
        help="Resize the review window if it is wider than this",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.05,
        help="Review only this top fraction of remaining images ranked by lumen pixels.",
    )
    parser.add_argument(
        "--min-review-count",
        type=int,
        default=1,
        help="Always review at least this many images when inputs are non-empty.",
    )
    parser.add_argument(
        "--include-saved",
        type=parse_bool,
        default=False,
        help="Include images previously marked as save in the candidate pool. Use true or false.",
    )
    parser.add_argument(
        "--include-reviewed",
        type=parse_bool,
        default=False,
        help="Include images already present in review_log.csv in the candidate pool. Use true or false.",
    )
    parser.add_argument(
        "--cache-only",
        type=parse_bool,
        default=False,
        help=(
            "Build/update mask cache and ranking CSV, then exit without opening review "
            "windows. Use true or false."
        ),
    )
    parser.add_argument(
        "--max-same-region",
        type=int,
        default=6,
        help=(
            "Maximum saved/selected images allowed per region. Once a region has this "
            "many saved images in review_log.csv, later candidates from that region "
            "are skipped. Use 0 to disable."
        ),
    )
    explicit_target_width = any(
        arg == "--target-width" or arg.startswith("--target-width=")
        for arg in sys.argv[1:]
    )
    explicit_crop_size = any(
        arg == "--crop-size" or arg.startswith("--crop-size=")
        for arg in sys.argv[1:]
    )
    args = parser.parse_args()
    resolve_physical_width_args(args, explicit_target_width, explicit_crop_size)
    args.trim_black_threshold = max(-1, min(255, args.trim_black_threshold))
    if not (0 < args.top_fraction <= 1):
        raise ValueError("--top-fraction must be in (0, 1].")
    if args.min_review_count < 1:
        raise ValueError("--min-review-count must be >= 1.")
    if args.max_same_region < 0:
        raise ValueError("--max-same-region must be >= 0.")
    return args


def physical_target_width(reference_fov_width_mm, reference_width_px, source_fov_width_mm):
    if reference_fov_width_mm <= 0:
        raise ValueError("--reference-fov-width-mm must be > 0.")
    if reference_width_px <= 0:
        raise ValueError("--reference-width-px must be > 0.")
    if source_fov_width_mm <= 0:
        raise ValueError("--source-fov-width-mm must be > 0.")

    reference_mm_per_px = reference_fov_width_mm / reference_width_px
    return int(round(source_fov_width_mm / reference_mm_per_px))


def resolve_physical_width_args(args, explicit_target_width, explicit_crop_size):
    calculated_width = physical_target_width(
        args.reference_fov_width_mm,
        args.reference_width_px,
        args.source_fov_width_mm,
    )
    if not explicit_target_width:
        args.target_width = calculated_width
    if args.crop_height <= 0:
        raise ValueError("--crop-height must be > 0.")
    if not explicit_crop_size:
        args.crop_size = (args.target_width, args.crop_height)


def predict_patch_masks(model, patches, img_size, threshold, batch_size):
    patch_inputs = [
        load_and_preprocess(
            img_path=None,
            img_array=patch,
            img_size=tuple(img_size),
            use_clahe=True,
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        for patch in patches
    ]

    patch_masks = []
    for start in range(0, len(patch_inputs), batch_size):
        batch = np.stack(patch_inputs[start : start + batch_size], axis=0)
        preds = model.predict(batch, verbose=0)
        patch_masks.extend(pred_to_mask(pred, threshold=threshold) for pred in preds)
    return patch_masks


def infer_resized_mask(model, img_path: Path, args):
    original_gray, _, resized_top, geometry = prepare_working_image(
        img_path,
        row_range=tuple(args.row_range),
        target_width=args.target_width,
        trim_black_threshold=args.trim_black_threshold,
    )
    patches, _, crop_start = split_resized_image_into_patches(
        resized_top,
        crop_size=tuple(args.crop_size),
        patch_size=tuple(args.img_size),
    )
    patch_masks = predict_patch_masks(
        model,
        patches,
        img_size=args.img_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    combined_mask = stitch_patch_masks(
        patch_masks,
        crop_size=tuple(args.crop_size),
        patch_size=tuple(args.img_size),
    )
    full_resized_mask = build_full_resized_mask(
        combined_mask,
        resized_shape=resized_top.shape,
        crop_start=crop_start,
    )
    return original_gray, resized_top, geometry, full_resized_mask


def make_cache_key(img_path: Path):
    return hashlib.sha1(str(img_path.resolve()).encode("utf-8")).hexdigest()


def load_manifest(manifest_path: Path):
    manifest = {}
    if not manifest_path.exists():
        return manifest

    with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            manifest[row["image_path"]] = row
    return manifest


def write_manifest(manifest_path: Path, manifest):
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for image_path in sorted(manifest):
            writer.writerow(manifest[image_path])


def read_reviewed_paths(decision_log: Path, exclude_saved=True):
    reviewed_paths = set()
    if not decision_log.exists():
        return reviewed_paths

    excluded_decisions = {"skip", "quit"}
    if exclude_saved:
        excluded_decisions.add("save")

    with decision_log.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            decision = row.get("decision")
            image_path = row.get("image_path")
            if decision in excluded_decisions and image_path:
                reviewed_paths.add(str(Path(image_path).resolve()))
    return reviewed_paths


def read_reviewed_saved_paths(decision_log: Path):
    saved_paths = set()
    if not decision_log.exists():
        return saved_paths

    with decision_log.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("decision") == "save" and row.get("image_path"):
                saved_paths.add(str(Path(row["image_path"]).resolve()))
    return saved_paths


def infer_region_from_path(img_path: Path):
    match = re.search(r"(?:^|_)([AP][A-Z0-9]{1,3}[LM])_Mode2D_", img_path.stem)
    if match:
        return match.group(1)

    parts = img_path.stem.split("_")
    if len(parts) >= 3 and parts[-2] == "Mode2D":
        return parts[-3]
    return ""


def count_saved_regions(decision_log: Path):
    saved_regions = Counter()
    if not decision_log.exists():
        return saved_regions

    with decision_log.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("decision") != "save" or not row.get("image_path"):
                continue
            region = infer_region_from_path(Path(row["image_path"]))
            if region:
                saved_regions[region] += 1
    return saved_regions


def ensure_decision_log(decision_log: Path):
    if decision_log.exists():
        return
    with decision_log.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "lumen_pixels", "decision", "saved_image", "saved_mask"])


def get_source_signature(img_path: Path):
    stat = img_path.stat()
    return str(stat.st_mtime_ns), str(stat.st_size)


def get_config_signature(args):
    weights_path = args.weights.resolve()
    weights_stat = weights_path.stat() if weights_path.exists() else None
    config_text = "|".join(
        [
            str(weights_path),
            str(weights_stat.st_mtime_ns) if weights_stat is not None else "missing",
            str(weights_stat.st_size) if weights_stat is not None else "missing",
            str(tuple(args.img_size)),
            str(tuple(args.crop_size)),
            str(tuple(args.row_range)),
            str(args.target_width),
            str(args.reference_fov_width_mm),
            str(args.reference_width_px),
            str(args.source_fov_width_mm),
            str(args.crop_height),
            str(args.trim_black_threshold),
            str(args.threshold),
        ]
    )
    return hashlib.sha1(config_text.encode("utf-8")).hexdigest()


def load_cached_masks(manifest_entry):
    resized_mask = cv2.imread(manifest_entry["resized_mask_path"], cv2.IMREAD_GRAYSCALE)
    original_mask = cv2.imread(manifest_entry["original_mask_path"], cv2.IMREAD_GRAYSCALE)
    if resized_mask is None or original_mask is None:
        return None, None
    return resized_mask, original_mask


def write_cache_entry(
    cache_dirs, img_path: Path, original_mask, full_resized_mask, lumen_pixels, config_signature
):
    cache_key = make_cache_key(img_path)
    resized_mask_path = cache_dirs["resized"] / f"{cache_key}.png"
    original_mask_path = cache_dirs["original"] / f"{cache_key}.png"
    cv2.imwrite(str(resized_mask_path), full_resized_mask.astype(np.uint8))
    cv2.imwrite(str(original_mask_path), original_mask.astype(np.uint8))
    source_mtime_ns, source_size = get_source_signature(img_path)
    return {
        "image_path": str(img_path.resolve()),
        "cache_key": cache_key,
        "config_signature": config_signature,
        "source_mtime_ns": source_mtime_ns,
        "source_size": source_size,
        "lumen_pixels": str(int(lumen_pixels)),
        "resized_mask_path": str(resized_mask_path),
        "original_mask_path": str(original_mask_path),
    }


def get_cached_or_infer_result(model, img_path: Path, args, manifest, cache_dirs):
    image_key = str(img_path.resolve())
    region = infer_region_from_path(img_path)
    manifest_entry = manifest.get(image_key)
    source_mtime_ns, source_size = get_source_signature(img_path)
    config_signature = get_config_signature(args)

    if (
        manifest_entry is not None
        and manifest_entry.get("config_signature") == config_signature
        and manifest_entry.get("source_mtime_ns") == source_mtime_ns
        and manifest_entry.get("source_size") == source_size
        and Path(manifest_entry["resized_mask_path"]).exists()
        and Path(manifest_entry["original_mask_path"]).exists()
    ):
        full_resized_mask, original_mask = load_cached_masks(manifest_entry)
        if full_resized_mask is not None and original_mask is not None:
            return {
                "image_path": img_path,
                "region": region,
                "lumen_pixels": int(manifest_entry["lumen_pixels"]),
                "full_resized_mask": full_resized_mask,
                "original_mask": original_mask,
                "cache_status": "hit",
            }

    original_gray, _, geometry, full_resized_mask = infer_resized_mask(model, img_path, args)
    original_mask = project_mask_to_original(
        full_resized_mask,
        original_shape=original_gray.shape,
        geometry=geometry,
    )
    lumen_pixels = int(np.count_nonzero(full_resized_mask))
    manifest[image_key] = write_cache_entry(
        cache_dirs=cache_dirs,
        img_path=img_path,
        original_mask=original_mask,
        full_resized_mask=full_resized_mask,
        lumen_pixels=lumen_pixels,
        config_signature=config_signature,
    )
    return {
        "image_path": img_path,
        "region": region,
        "lumen_pixels": lumen_pixels,
        "full_resized_mask": full_resized_mask,
        "original_mask": original_mask,
        "cache_status": "miss",
    }


def build_ranked_results(model, image_paths, args, manifest, cache_dirs):
    ranked = []
    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        result = get_cached_or_infer_result(model, img_path, args, manifest, cache_dirs)
        ranked.append(result)
        print(
            f"[rank {idx}/{total}] {img_path.name}: "
            f"lumen_pixels={result['lumen_pixels']} cache={result['cache_status']}"
        )
    ranked.sort(key=lambda item: (-item["lumen_pixels"], str(item["image_path"])))
    return ranked


def choose_top_images(ranked_results, top_fraction, min_review_count, max_same_region, saved_regions):
    total = len(ranked_results)
    review_count = max(min_review_count, int(np.ceil(total * top_fraction)))
    review_count = min(total, review_count)
    if max_same_region == 0:
        return ranked_results[:review_count], set()

    selected = []
    skipped_region_cap = set()
    region_counts = Counter(saved_regions)
    for result in ranked_results:
        region = result.get("region", "")
        if region and region_counts[region] >= max_same_region:
            skipped_region_cap.add(str(result["image_path"].resolve()))
            continue

        selected.append(result)
        if region:
            region_counts[region] += 1
        if len(selected) >= review_count:
            break
    return selected, skipped_region_cap


def reset_rect_state():
    RECT_STATE["drawing"] = False
    RECT_STATE["start"] = None
    RECT_STATE["current"] = None
    RECT_STATE["rectangles"] = []
    RECT_STATE["display_scale"] = 1.0
    RECT_STATE["mode"] = "erase"


def main():
    args = parse_args()
    if args.image_list is None and args.input_dir is None and not args.images:
        raise SystemExit("Provide --input-dir, --image-list, or image paths.")
    image_paths = collect_image_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "cache"
    cache_dirs = {
        "resized": cache_root / "resized_masks",
        "original": cache_root / "original_masks",
    }
    for cache_dir in cache_dirs.values():
        cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "cache_manifest.csv"
    ranking_log = args.output_dir / "top_lumen_ranking.csv"
    decision_log = args.output_dir / "review_log.csv"
    ensure_decision_log(decision_log)

    manifest = load_manifest(manifest_path)

    model = Attention_ResUNet((args.img_size[1], args.img_size[0], 1))
    model.load_weights(str(args.weights))

    ranked = build_ranked_results(model, image_paths, args, manifest, cache_dirs)
    write_manifest(manifest_path, manifest)

    reviewed_paths = (
        set()
        if args.include_reviewed
        else read_reviewed_paths(decision_log, exclude_saved=not args.include_saved)
    )
    saved_paths = read_reviewed_saved_paths(decision_log)
    saved_regions = count_saved_regions(decision_log)
    candidate_results = [
        result
        for result in ranked
        if str(result["image_path"].resolve()) not in reviewed_paths
        and (args.include_saved or str(result["image_path"].resolve()) not in saved_paths)
    ]
    selected, skipped_region_cap = choose_top_images(
        candidate_results,
        args.top_fraction,
        args.min_review_count,
        args.max_same_region,
        saved_regions,
    )
    selected_paths = {str(result["image_path"].resolve()) for result in selected}

    with ranking_log.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_path",
                "region",
                "lumen_pixels",
                "cache_status",
                "excluded_reviewed",
                "excluded_saved",
                "excluded_region_cap",
                "selected_for_review",
            ]
        )
        for result in ranked:
            resolved = str(result["image_path"].resolve())
            writer.writerow(
                [
                    str(result["image_path"]),
                    result["region"],
                    result["lumen_pixels"],
                    result["cache_status"],
                    resolved in reviewed_paths,
                    not args.include_saved and resolved in saved_paths,
                    resolved in skipped_region_cap,
                    resolved in selected_paths,
                ]
            )

    print(
        f"Candidate pool: {len(candidate_results)}/{len(ranked)} "
        f"(excluded_reviewed={len(reviewed_paths) if not args.include_reviewed else 0}, "
        f"excluded_saved={len(saved_paths) if not args.include_saved else 0})."
    )
    if args.max_same_region:
        capped_regions = [
            region
            for region, count in sorted(saved_regions.items())
            if count >= args.max_same_region
        ]
        print(
            f"Region cap: max_same_region={args.max_same_region}, "
            f"skipped_by_cap={len(skipped_region_cap)}, capped_regions={capped_regions}"
        )
    print(
        f"Selected {len(selected)}/{len(candidate_results)} images for review "
        f"(top_fraction={args.top_fraction:.4f})."
    )

    if args.cache_only:
        cv2.destroyAllWindows()
        print(f"Cache-only mode complete. Cached {len(ranked)} frames in {args.output_dir}")
        return

    if not selected:
        cv2.destroyAllWindows()
        print("No images left to review.")
        return

    with decision_log.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        saved_count = 0
        total_selected = len(selected)

        for idx, result in enumerate(selected, start=1):
            img_path = result["image_path"]
            lumen_pixels = result["lumen_pixels"]
            reset_rect_state()

            original_gray, _, resized_top, geometry = prepare_working_image(
                img_path,
                row_range=tuple(args.row_range),
                target_width=args.target_width,
                trim_black_threshold=args.trim_black_threshold,
            )
            base_full_resized_mask = result["full_resized_mask"]

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW_NAME, rectangle_mouse_callback)

            while True:
                edited_full_resized_mask = apply_rectangles_to_mask(
                    base_full_resized_mask,
                    RECT_STATE["rectangles"],
                )
                review_image, scale = create_review_image(
                    resized_top=resized_top,
                    full_resized_mask=edited_full_resized_mask,
                    alpha=args.alpha,
                    min_area=args.min_area,
                    display_width=args.display_width,
                    index=idx,
                    total=total_selected,
                    image_name=f"{img_path.name} | lumen_pixels={lumen_pixels}",
                )
                RECT_STATE["display_scale"] = scale
                rendered = render_rectangles(review_image)
                cv2.imshow(WINDOW_NAME, rendered)
                cv2.resizeWindow(WINDOW_NAME, rendered.shape[1], rendered.shape[0])
                decision = review_key()
                if decision is None:
                    continue
                if decision == "undo":
                    if RECT_STATE["rectangles"]:
                        RECT_STATE["rectangles"].pop()
                    continue
                if decision == "clear":
                    RECT_STATE["rectangles"].clear()
                    continue
                if decision == "erase_mode":
                    RECT_STATE["mode"] = "erase"
                    continue
                if decision == "restore_mode":
                    RECT_STATE["mode"] = "restore"
                    continue
                break

            if decision == "quit":
                writer.writerow([str(img_path), lumen_pixels, "quit", "", ""])
                break

            if decision == "save":
                edited_full_resized_mask = apply_rectangles_to_mask(
                    base_full_resized_mask,
                    RECT_STATE["rectangles"],
                )
                original_mask = project_mask_to_original(
                    edited_full_resized_mask,
                    original_shape=original_gray.shape,
                    geometry=geometry,
                )
                image_out, mask_out = save_acceptance(
                    img_path,
                    original_mask,
                    output_dir=args.output_dir,
                    input_dir=args.input_dir,
                )
                saved_count += 1
                writer.writerow([str(img_path), lumen_pixels, "save", str(image_out), str(mask_out)])
            else:
                writer.writerow([str(img_path), lumen_pixels, "skip", "", ""])

    cv2.destroyAllWindows()
    print(f"Review finished. Saved {saved_count} accepted frames to {args.output_dir}")


if __name__ == "__main__":
    main()
