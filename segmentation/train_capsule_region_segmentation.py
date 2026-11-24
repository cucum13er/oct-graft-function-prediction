import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is required to train this model. Install it in your environment and rerun."
    ) from exc


@dataclass(frozen=True)
class Sample:
    image_id: int
    image_path: Path
    width: int
    height: int
    segmentations: Sequence[Sequence[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small U-Net for OCT capsule-region segmentation from COCO polygons."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing OCT images and _annotations.coco_capsule.json.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional explicit COCO annotation path. Defaults to <data-dir>/_annotations.coco_capsule.json.",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=2,
        help="COCO category id to train as foreground.",
    )
    parser.add_argument("--img-width", type=int, default=640, help="Resized training width.")
    parser.add_argument("--img-height", type=int, default=416, help="Resized training height.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of labeled images used for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, logs, and validation predictions.",
    )
    parser.add_argument(
        "--export-val-preds",
        action="store_true",
        help="Save predicted masks for the validation set after training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_coco_samples(annotation_path: Path, image_root: Path, category_id: int) -> List[Sample]:
    with annotation_path.open("r", encoding="utf-8") as fh:
        coco = json.load(fh)

    image_by_id: Dict[int, dict] = {img["id"]: img for img in coco["images"]}
    anns_by_image: Dict[int, List[Sequence[float]]] = {}

    for ann in coco["annotations"]:
        if ann.get("category_id") != category_id:
            continue
        segmentations = ann.get("segmentation", [])
        if not isinstance(segmentations, list):
            continue
        anns_by_image.setdefault(ann["image_id"], []).extend(segmentations)

    samples: List[Sample] = []
    for image_id, image_info in image_by_id.items():
        polygons = anns_by_image.get(image_id, [])
        if not polygons:
            continue
        image_path = image_root / image_info["file_name"]
        if not image_path.exists():
            original_name = image_info.get("extra", {}).get("name")
            if original_name:
                fallback_path = image_root / original_name
                if fallback_path.exists():
                    image_path = fallback_path
        if not image_path.exists():
            raise FileNotFoundError(
                "Missing image referenced by COCO file: "
                f"{image_info['file_name']} (also checked extra.name={image_info.get('extra', {}).get('name')})"
            )
        samples.append(
            Sample(
                image_id=image_id,
                image_path=image_path,
                width=int(image_info["width"]),
                height=int(image_info["height"]),
                segmentations=polygons,
            )
        )

    if not samples:
        raise ValueError(f"No samples found for category_id={category_id} in {annotation_path}")

    return sorted(samples, key=lambda sample: sample.image_path.name)


def polygon_to_mask(width: int, height: int, polygons: Sequence[Sequence[float]]) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        xy = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        drawer.polygon(xy, outline=1, fill=1)
    return np.asarray(mask, dtype=np.float32)


def load_image_mask(sample: Sample, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(sample.image_path).convert("L")
    mask = polygon_to_mask(sample.width, sample.height, sample.segmentations)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    image = image.resize(target_size, resample=Image.BILINEAR)
    mask_img = mask_img.resize(target_size, resample=Image.NEAREST)

    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    mask_arr = (np.asarray(mask_img, dtype=np.float32) > 127).astype(np.float32)

    image_arr = image_arr[..., np.newaxis]
    mask_arr = mask_arr[..., np.newaxis]
    return image_arr, mask_arr


def split_samples(samples: Sequence[Sample], val_fraction: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, int(round(len(samples) * val_fraction)))
    val_indices = set(indices[:val_count])

    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_samples, val_samples


def build_arrays(samples: Sequence[Sample], target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    images, masks = zip(*(load_image_mask(sample, target_size) for sample in samples))
    return np.stack(images, axis=0), np.stack(masks, axis=0)


def augment_batch(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask


def make_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    if training:
        dataset = dataset.shuffle(len(images), reshuffle_each_iteration=True)
        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_small_unet(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    bottleneck = conv_block(p3, 128)

    u4 = tf.keras.layers.UpSampling2D()(bottleneck)
    u4 = tf.keras.layers.Concatenate()([u4, c3])
    c4 = conv_block(u4, 64)

    u5 = tf.keras.layers.UpSampling2D()(c4)
    u5 = tf.keras.layers.Concatenate()([u5, c2])
    c5 = conv_block(u5, 32)

    u6 = tf.keras.layers.UpSampling2D()(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c1])
    c6 = conv_block(u6, 16)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c6)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="small_unet_capsule_region")


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def iou_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + (1.0 - dice_coef(y_true, y_pred))


def export_validation_predictions(
    model: tf.keras.Model,
    samples: Sequence[Sample],
    target_size: Tuple[int, int],
    output_dir: Path,
) -> None:
    pred_dir = output_dir / "val_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        image_arr, _ = load_image_mask(sample, target_size)
        pred = model.predict(image_arr[np.newaxis, ...], verbose=0)[0, ..., 0]
        pred_mask = (pred >= 0.5).astype(np.uint8) * 255
        Image.fromarray(pred_mask, mode="L").save(pred_dir / sample.image_path.name)


def main() -> None:
    args = parse_args()
    annotation_path = args.annotations or (args.data_dir / "_annotations.coco_capsule.json")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    samples = load_coco_samples(annotation_path, args.data_dir, args.category_id)
    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    target_size = (args.img_width, args.img_height)

    train_images, train_masks = build_arrays(train_samples, target_size)
    val_images, val_masks = build_arrays(val_samples, target_size)

    train_ds = make_dataset(train_images, train_masks, args.batch_size, training=True)
    val_ds = make_dataset(val_images, val_masks, args.batch_size, training=False)

    model = build_small_unet((args.img_height, args.img_width, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=bce_dice_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            dice_coef,
            iou_coef,
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(output_dir / "last_model.keras")

    summary = {
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "img_width": args.img_width,
        "img_height": args.img_height,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.history["loss"]),
        "best_val_dice": float(max(history.history["val_dice_coef"])),
        "best_val_iou": float(max(history.history["val_iou_coef"])),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    split_manifest = {
        "train": [sample.image_path.name for sample in train_samples],
        "val": [sample.image_path.name for sample in val_samples],
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)

    if args.export_val_preds:
        export_validation_predictions(model, val_samples, target_size, output_dir)

    print(json.dumps(summary, indent=2))
    print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
