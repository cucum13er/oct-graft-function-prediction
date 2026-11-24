import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from inference import load_and_preprocess, pred_to_mask
from model import Attention_ResUNet


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
WINDOW_NAME = "Attention_ResUNet Review"


RECT_STATE = {
    "drawing": False,
    "start": None,
    "current": None,
    "rectangles": [],
    "display_scale": 1.0,
    "mode": "erase",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Attention_ResUNet on a folder or image list, review each frame with "
            "circled segmented regions, and selectively save accepted images and masks."
        )
    )
    parser.add_argument("--input-dir", type=Path, 
                        default=None, 
                        help="Folder of images to review")
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
        # required=True,
        required=True,
        help="Accepted images and masks are saved here",
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
        default=(1536, 256),
        metavar=("W", "H"),
        help="Cropped review region size before patch inference",
    )
    parser.add_argument(
        "--row-range",
        type=int,
        nargs=2,
        default=(200, 700),
        metavar=("START", "END"),
        help="Use only rows [START:END] of the input before resizing/cropping",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1536,
        help="Resize width before finding the brightest crop region",
    )
    parser.add_argument(
        "--trim-black-threshold",
        type=int,
        default=0,
        help=(
            "Trim pure-black side columns before resizing by keeping columns whose max "
            "intensity is greater than this value. Use -1 to disable, 0 to trim only "
            "fully black side padding."
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
        default=2200,
        help="Resize the review window if it is wider than this",
    )
    args = parser.parse_args()
    args.trim_black_threshold = max(-1, min(255, args.trim_black_threshold))
    return args


def collect_image_paths(args):
    image_paths = []

    if args.input_dir is not None:
        image_paths.extend(
            sorted(
                path
                for path in args.input_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMG_EXTS
            )
        )

    if args.image_list is not None:
        with args.image_list.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    image_paths.append(Path(line))

    if args.images:
        image_paths.extend(Path(path) for path in args.images)

    deduped = []
    seen = set()
    for path in image_paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)

    if not deduped:
        raise ValueError("No input images found. Use --input-dir, --image-list, or explicit image paths.")

    missing = [path for path in deduped if not path.exists()]
    if missing:
        raise FileNotFoundError(f"These input images do not exist: {missing[:5]}")

    return deduped


def find_content_x_range(gray_image: np.ndarray, threshold: int):
    if threshold < 0:
        return (0, gray_image.shape[1])

    content_cols = np.flatnonzero(np.max(gray_image, axis=0) > threshold)
    if content_cols.size == 0:
        return (0, gray_image.shape[1])
    return int(content_cols[0]), int(content_cols[-1] + 1)


def prepare_working_image(img_path: Path, row_range, target_width: int, trim_black_threshold: int):
    original_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise ValueError(f"Failed to read image: {img_path}")

    row_start, row_end = row_range
    row_start = max(0, min(row_start, original_gray.shape[0]))
    row_end = max(row_start, min(row_end, original_gray.shape[0]))
    working_gray = original_gray[row_start:row_end, :]
    col_start, col_end = find_content_x_range(working_gray, trim_black_threshold)
    cropped_working = working_gray[:, col_start:col_end]
    if cropped_working.size == 0:
        cropped_working = working_gray
        col_start, col_end = 0, working_gray.shape[1]

    working_h, working_w = cropped_working.shape
    target_h = int(working_h * target_width / working_w)
    resized_working = cv2.resize(cropped_working, (target_width, target_h), interpolation=cv2.INTER_CUBIC)
    geometry = {
        "row_range": (row_start, row_end),
        "col_range": (col_start, col_end),
    }
    return original_gray, cropped_working, resized_working, geometry


def patch_axis_starts(length, patch_length):
    if length < patch_length:
        raise ValueError(
            f"Crop width {length} is smaller than patch width {patch_length}."
        )

    last_start = length - patch_length
    starts = list(range(0, last_start + 1, patch_length))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def split_resized_image_into_patches(resized_gray, crop_size, patch_size):
    crop_w, crop_h = crop_size
    patch_w, patch_h = patch_size

    if resized_gray.shape[1] != crop_w:
        raise ValueError(
            f"Resized working image width {resized_gray.shape[1]} must match crop width {crop_w}."
        )
    if crop_h > 2 * patch_h:
        raise ValueError("crop_size height greater than two patch heights is not supported.")
    if resized_gray.shape[0] < crop_h:
        raise ValueError(
            f"Resized working image height {resized_gray.shape[0]} is smaller than crop height {crop_h}."
        )

    mean_intensity = np.mean(resized_gray, axis=1)
    window_means = np.convolve(mean_intensity, np.ones(crop_h, dtype=np.float32), mode="valid") / crop_h
    crop_start = int(np.argmax(window_means))
    cropped = resized_gray[crop_start : crop_start + crop_h, :]

    col_starts = patch_axis_starts(crop_w, patch_w)
    patches = []
    if crop_h <= patch_h:
        for x0 in col_starts:
            patches.append(cropped[:, x0 : x0 + patch_w])
    else:
        for x0 in col_starts:
            patches.append(cropped[:patch_h, x0 : x0 + patch_w])
        for x0 in col_starts:
            patches.append(cropped[-patch_h:, x0 : x0 + patch_w])

    return patches, cropped, crop_start


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


def stitch_patch_masks(patch_masks, crop_size, patch_size):
    crop_w, crop_h = crop_size
    patch_w, patch_h = patch_size
    col_starts = patch_axis_starts(crop_w, patch_w)
    num_cols = len(col_starts)

    if crop_h <= patch_h:
        if len(patch_masks) != num_cols:
            raise ValueError("Unexpected patch count for single-row crop.")
        stitched = np.zeros((crop_h, crop_w), dtype=np.uint8)
        for mask, x0 in zip(patch_masks, col_starts):
            stitched[:, x0 : x0 + patch_w] = np.maximum(
                stitched[:, x0 : x0 + patch_w],
                mask[:crop_h, :patch_w],
            )
        return stitched

    if crop_h <= 2 * patch_h:
        if len(patch_masks) != 2 * num_cols:
            raise ValueError("Unexpected patch count for two-row crop.")
        stitched = np.zeros((crop_h, crop_w), dtype=np.uint8)
        for mask, x0 in zip(patch_masks[:num_cols], col_starts):
            stitched[:patch_h, x0 : x0 + patch_w] = np.maximum(
                stitched[:patch_h, x0 : x0 + patch_w],
                mask[:patch_h, :patch_w],
            )
        for mask, x0 in zip(patch_masks[num_cols:], col_starts):
            stitched[-patch_h:, x0 : x0 + patch_w] = np.maximum(
                stitched[-patch_h:, x0 : x0 + patch_w],
                mask[:patch_h, :patch_w],
            )
        return stitched

    raise ValueError("crop height greater than two patch heights is not supported.")


def build_full_resized_mask(combined_mask, resized_shape, crop_start):
    full_mask = np.zeros(resized_shape, dtype=np.uint8)
    crop_end = min(crop_start + combined_mask.shape[0], resized_shape[0])
    insert_h = crop_end - crop_start
    full_mask[crop_start:crop_end, :] = combined_mask[:insert_h, :]
    return full_mask


def project_mask_to_original(full_resized_mask, original_shape, geometry):
    original_h, original_w = original_shape
    row_start, row_end = geometry["row_range"]
    col_start, col_end = geometry["col_range"]
    row_start = max(0, min(row_start, original_h))
    row_end = max(row_start, min(row_end, original_h))
    col_start = max(0, min(col_start, original_w))
    col_end = max(col_start, min(col_end, original_w))
    working_h = row_end - row_start
    working_w = col_end - col_start
    resized_back = cv2.resize(
        full_resized_mask,
        (working_w, working_h),
        interpolation=cv2.INTER_NEAREST,
    )
    full_original_mask = np.zeros((original_h, original_w), dtype=np.uint8)
    full_original_mask[row_start:row_end, col_start:col_end] = resized_back
    return full_original_mask


def overlay_mask(gray_image, mask, alpha):
    canvas = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    color_layer = canvas.copy()
    color_layer[mask > 0] = (0, 0, 255)
    if np.any(mask > 0):
        canvas[mask > 0] = cv2.addWeighted(
            color_layer[mask > 0],
            alpha,
            canvas[mask > 0],
            1.0 - alpha,
            0,
        )
    return canvas


def annotate_regions(canvas, mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        kept += 1
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        center = (int(round(cx)), int(round(cy)))
        radius = max(10, int(round(radius)))
        cv2.circle(canvas, center, radius, (0, 255, 255), 2, cv2.LINE_AA)
    return kept


def create_review_image(resized_top, full_resized_mask, alpha, min_area, display_width, index, total, image_name):
    annotated = overlay_mask(resized_top, full_resized_mask, alpha=alpha)
    region_count = annotate_regions(annotated, full_resized_mask, min_area=min_area)

    header = f"[{index}/{total}] {image_name}"
    footer = "left-drag: apply mode   e: erase   r: restore   u: undo   c: clear   s: save   k: skip   q: quit"
    cv2.putText(annotated, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"regions >= {min_area}px: {region_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        footer,
        (20, annotated.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"mode: {RECT_STATE['mode']}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if annotated.shape[1] > display_width:
        scale = display_width / float(annotated.shape[1])
        annotated = cv2.resize(
            annotated,
            (display_width, int(round(annotated.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
    return annotated, scale


def get_output_stem(img_path: Path, input_dir: Path | None):
    if input_dir is not None:
        try:
            rel = img_path.relative_to(input_dir.resolve())
            return rel.parent / rel.stem
        except ValueError:
            pass
    return Path(img_path.stem)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_acceptance(img_path, original_mask, output_dir, input_dir):
    stem = get_output_stem(img_path, input_dir)
    image_out = output_dir / "images" / f"{stem}{img_path.suffix.lower()}"
    mask_out = output_dir / "masks" / f"{stem}_mask.png"

    ensure_parent(image_out)
    ensure_parent(mask_out)

    shutil.copy2(img_path, image_out)
    cv2.imwrite(str(mask_out), original_mask)
    return image_out, mask_out


def normalize_rect(start, end):
    x0, y0 = start
    x1, y1 = end
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def rectangle_mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        RECT_STATE["drawing"] = True
        RECT_STATE["start"] = (x, y)
        RECT_STATE["current"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and RECT_STATE["drawing"]:
        RECT_STATE["current"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and RECT_STATE["drawing"]:
        RECT_STATE["drawing"] = False
        RECT_STATE["current"] = (x, y)
        rect = normalize_rect(RECT_STATE["start"], RECT_STATE["current"])
        if rect[2] - rect[0] > 3 and rect[3] - rect[1] > 3:
            scale = max(RECT_STATE["display_scale"], 1e-6)
            mx0 = int(np.floor(rect[0] / scale))
            my0 = int(np.floor(rect[1] / scale))
            mx1 = int(np.ceil(rect[2] / scale))
            my1 = int(np.ceil(rect[3] / scale))
            RECT_STATE["rectangles"].append((mx0, my0, mx1, my1, RECT_STATE["mode"]))


def render_rectangles(image):
    rendered = image.copy()
    for rect in RECT_STATE["rectangles"]:
        x0, y0, x1, y1, mode = rect
        color = (0, 255, 0) if mode == "erase" else (255, 0, 0)
        dx0 = int(round(x0 * RECT_STATE["display_scale"]))
        dy0 = int(round(y0 * RECT_STATE["display_scale"]))
        dx1 = int(round(x1 * RECT_STATE["display_scale"]))
        dy1 = int(round(y1 * RECT_STATE["display_scale"]))
        cv2.rectangle(rendered, (dx0, dy0), (dx1, dy1), color, 2, cv2.LINE_AA)
    if RECT_STATE["drawing"] and RECT_STATE["start"] is not None and RECT_STATE["current"] is not None:
        x0, y0, x1, y1 = normalize_rect(RECT_STATE["start"], RECT_STATE["current"])
        color = (0, 255, 255) if RECT_STATE["mode"] == "erase" else (255, 0, 255)
        cv2.rectangle(rendered, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    return rendered


def apply_rectangles_to_mask(base_mask, rectangles):
    edited = base_mask.copy()
    for rect in rectangles:
        x0, y0, x1, y1, mode = rect
        mx0 = max(0, min(base_mask.shape[1], x0))
        my0 = max(0, min(base_mask.shape[0], y0))
        mx1 = max(0, min(base_mask.shape[1], x1))
        my1 = max(0, min(base_mask.shape[0], y1))
        if mode == "erase":
            edited[my0:my1, mx0:mx1] = 0
        else:
            edited[my0:my1, mx0:mx1] = base_mask[my0:my1, mx0:mx1]
    return edited


def review_key(delay_ms=30):
    key = cv2.waitKeyEx(delay_ms)
    if key in (-1, 255):
        return None
    if key in (13, ord("s"), ord("S")):
        return "save"
    if key in (32, ord("k"), ord("K"), ord("n"), ord("N")):
        return "skip"
    if key in (27, ord("q"), ord("Q")):
        return "quit"
    if key in (ord("u"), ord("U")):
        return "undo"
    if key in (ord("c"), ord("C")):
        return "clear"
    if key in (ord("e"), ord("E")):
        return "erase_mode"
    if key in (ord("r"), ord("R")):
        return "restore_mode"
    return None


def main():
    args = parse_args()
    if args.image_list is None and args.input_dir is None:
        raise SystemExit("Provide --input-dir or --image-list.")
    image_paths = collect_image_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_log = args.output_dir / "review_log.csv"

    model = Attention_ResUNet((args.img_size[1], args.img_size[0], 1))
    model.load_weights(str(args.weights))

    with decision_log.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "decision", "saved_image", "saved_mask"])

        total = len(image_paths)
        saved_count = 0

        for idx, img_path in enumerate(image_paths, start=1):
            RECT_STATE["drawing"] = False
            RECT_STATE["start"] = None
            RECT_STATE["current"] = None
            RECT_STATE["rectangles"] = []
            RECT_STATE["display_scale"] = 1.0
            RECT_STATE["mode"] = "erase"

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
            base_full_resized_mask = build_full_resized_mask(
                combined_mask,
                resized_shape=resized_top.shape,
                crop_start=crop_start,
            )

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
                    total=total,
                    image_name=img_path.name,
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
                writer.writerow([str(img_path), "quit", "", ""])
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
                writer.writerow([str(img_path), "save", str(image_out), str(mask_out)])
            else:
                writer.writerow([str(img_path), "skip", "", ""])

    cv2.destroyAllWindows()
    print(f"Review finished. Saved {saved_count} accepted frames to {args.output_dir}")


if __name__ == "__main__":
    main()
