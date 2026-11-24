import argparse
import csv
import json
import os
import posixpath
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is required to run inference. Install it in your environment and rerun."
    ) from exc


WINDOW_NAME = "Capsule Segmentation Review"
RECT_STATE = {
    "drawing": False,
    "start": None,
    "current": None,
    "rectangles": [],
    "display_scale": 1.0,
    "mode": "erase",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run capsule-region segmentation inference, review each prediction visually, "
            "and save only accepted masks, source images, and metadata."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to best_model.keras or last_model.keras.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing OCT images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where accepted images, masks, and metadata will be written.",
    )
    parser.add_argument("--img-width", type=int, default=640, help="Model input width.")
    parser.add_argument("--img-height", type=int, default=416, help="Model input height.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to sigmoid probabilities.",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Mask overlay alpha in the review window.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1200,
        help="Resize the review window if it is wider than this.",
    )
    parser.add_argument(
        "--save-probability",
        action="store_true",
        help="Also save grayscale probability maps for accepted images.",
    )
    return parser.parse_args()


def load_image(image_path: Path, target_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    image = Image.open(image_path).convert("L")
    original_size = image.size
    image = image.resize(target_size, resample=Image.BILINEAR)
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    image_arr = image_arr[..., np.newaxis]
    return image_arr, original_size


def predict_original_size_outputs(
    model: tf.keras.Model,
    image_path: Path,
    target_size: tuple[int, int],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image_arr, original_size = load_image(image_path, target_size)
    pred = model.predict(image_arr[np.newaxis, ...], verbose=0)[0, ..., 0]

    pred_img = Image.fromarray((pred * 255.0).astype(np.uint8), mode="L")
    pred_img = pred_img.resize(original_size, resample=Image.BILINEAR)
    prob_arr = np.asarray(pred_img, dtype=np.uint8)
    binary_mask = ((prob_arr / 255.0) >= threshold).astype(np.uint8) * 255
    return prob_arr, binary_mask, original_size


def overlay_mask(gray_image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
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


def create_review_image(
    image_path: Path,
    gray_image: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    display_width: int,
    index: int,
    total: int,
    threshold: float,
) -> np.ndarray:
    annotated = overlay_mask(gray_image, mask, alpha=alpha)
    positive_pixels = int(np.count_nonzero(mask))

    cv2.putText(
        annotated,
        f"[{index}/{total}] {image_path.name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"threshold={threshold:.2f}   positive_pixels={positive_pixels}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "left-drag: apply mode   e: erase   r: restore   u: undo   c: clear   s: save   k: skip   q: quit",
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
    RECT_STATE["display_scale"] = scale
    return annotated


def review_key(delay_ms: int = 30) -> str | None:
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


def normalize_rect(start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0 = start
    x1, y1 = end
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def rectangle_mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    del flags, param
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


def render_rectangles(image: np.ndarray) -> np.ndarray:
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


def apply_rectangles_to_mask(base_mask: np.ndarray, rectangles: list[tuple[int, int, int, int, str]]) -> np.ndarray:
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


def get_output_stem(image_path: Path, input_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    return relative.parent / relative.stem


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_acceptance(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    binary_mask: np.ndarray,
    prob_arr: np.ndarray,
    threshold: float,
    save_probability: bool,
) -> tuple[Path, Path, Path, Path | None]:
    stem = get_output_stem(image_path, input_dir)
    image_out = output_dir / "images" / f"{stem}{image_path.suffix.lower()}"
    mask_out = output_dir / "masks" / f"{stem}_mask.png"
    meta_out = output_dir / "meta" / f"{stem}_meta.json"
    prob_out = output_dir / "probability" / f"{stem}_prob.png" if save_probability else None

    ensure_parent(image_out)
    ensure_parent(mask_out)
    ensure_parent(meta_out)
    if prob_out is not None:
        ensure_parent(prob_out)

    shutil.copy2(image_path, image_out)
    cv2.imwrite(str(mask_out), binary_mask)
    if prob_out is not None:
        cv2.imwrite(str(prob_out), prob_arr)

    meta = {
        "source_image": str(image_path),
        "saved_image": str(image_out),
        "saved_mask": str(mask_out),
        "saved_probability": str(prob_out) if prob_out is not None else None,
        "threshold": threshold,
        "image_height": int(binary_mask.shape[0]),
        "image_width": int(binary_mask.shape[1]),
        "positive_pixels": int(np.count_nonzero(binary_mask)),
        "mask_area_fraction": float(np.count_nonzero(binary_mask) / binary_mask.size),
    }
    with meta_out.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return image_out, mask_out, meta_out, prob_out


def load_capsule_model(path):
    if os.name != "nt" or path.suffix.lower() != ".keras":
        return tf.keras.models.load_model(path, compile=False)

    # Keras 2.15 uses the OS path separator for HDF5 groups in .keras archives.
    gfile = tf.compat.v2.io.gfile
    original_join = gfile.join
    try:
        gfile.join = posixpath.join
        return tf.keras.models.load_model(path, compile=False)
    finally:
        gfile.join = original_join


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_capsule_model(args.model)

    exts = {ext.lower() for ext in args.exts}
    image_paths = sorted(
        path for path in args.input_dir.rglob("*") if path.is_file() and path.suffix.lower() in exts
    )

    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    target_size = (args.img_width, args.img_height)
    decision_log = args.output_dir / "review_log.csv"
    saved_count = 0

    with decision_log.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_path",
                "decision",
                "saved_image",
                "saved_mask",
                "saved_meta",
                "saved_probability",
            ]
        )

        total = len(image_paths)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, rectangle_mouse_callback)

        for index, image_path in enumerate(image_paths, start=1):
            RECT_STATE["drawing"] = False
            RECT_STATE["start"] = None
            RECT_STATE["current"] = None
            RECT_STATE["rectangles"] = []
            RECT_STATE["display_scale"] = 1.0
            RECT_STATE["mode"] = "erase"

            original_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if original_gray is None:
                raise ValueError(f"Failed to read image: {image_path}")

            prob_arr, base_binary_mask, _ = predict_original_size_outputs(
                model=model,
                image_path=image_path,
                target_size=target_size,
                threshold=args.threshold,
            )

            while True:
                edited_binary_mask = apply_rectangles_to_mask(
                    base_binary_mask,
                    RECT_STATE["rectangles"],
                )
                review_image = create_review_image(
                    image_path=image_path,
                    gray_image=original_gray,
                    mask=edited_binary_mask,
                    alpha=args.alpha,
                    display_width=args.display_width,
                    index=index,
                    total=total,
                    threshold=args.threshold,
                )
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
                writer.writerow([str(image_path), "quit", "", "", "", ""])
                break

            if decision == "save":
                edited_binary_mask = apply_rectangles_to_mask(
                    base_binary_mask,
                    RECT_STATE["rectangles"],
                )
                image_out, mask_out, meta_out, prob_out = save_acceptance(
                    image_path=image_path,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    binary_mask=edited_binary_mask,
                    prob_arr=prob_arr,
                    threshold=args.threshold,
                    save_probability=args.save_probability,
                )
                saved_count += 1
                writer.writerow(
                    [
                        str(image_path),
                        "save",
                        str(image_out),
                        str(mask_out),
                        str(meta_out),
                        str(prob_out) if prob_out is not None else "",
                    ]
                )
            else:
                writer.writerow([str(image_path), "skip", "", "", "", ""])

    cv2.destroyAllWindows()
    print(f"Review finished. Saved {saved_count} accepted frames to {args.output_dir}")


if __name__ == "__main__":
    main()
