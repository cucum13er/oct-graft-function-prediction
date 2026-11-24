# from tensorflow.keras.optimizers import Adam
import glob
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import cv2
from model import Attention_ResUNet

def load_and_preprocess(
    img_path: str,
    img_array: np.ndarray = None, # input as array (H,W) or (H,W,1)
    img_size=(256, 256),
    use_clahe: bool = True,
    clipLimit: float = 2.0,
    tileGridSize=(8, 8),
) -> np.ndarray:
    """
    读取图片 -> 转灰度(1通道) -> resize到(256,256) -> (可选)CLAHE -> 归一化到[0,1]
    return: (H,W,1) float32
    """
    if img_array is None:
        img = Image.open(img_path).convert("L")
        img = img.resize(img_size, resample=Image.BILINEAR)
        arr_u8 = np.asarray(img, dtype=np.uint8)
    else:
        if img_array.ndim == 2:
            arr_u8 = cv2.resize(img_array.astype(np.uint8), img_size[::-1], interpolation=cv2.INTER_LINEAR)
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            arr_u8 = cv2.resize(img_array[..., 0].astype(np.uint8), img_size[::-1], interpolation=cv2.INTER_LINEAR)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            img = Image.fromarray(img_array.astype(np.uint8)).convert("L")
            img = img.resize(img_size, resample=Image.BILINEAR)
            arr_u8 = np.asarray(img, dtype=np.uint8)
        else:
            raise ValueError("img_array must be (H,W), (H,W,1), or (H,W,3)")

    # # 2) 转成 uint8 (CLAHE 需要 8-bit 最常见)
    # arr_u8 = np.asarray(img, dtype=np.uint8)  # (H,W) in [0,255]

    # 3) CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        arr_u8 = clahe.apply(arr_u8)

    # 4) 归一化 + 加通道维
    arr = arr_u8.astype(np.float32) / 255.0  # (H,W) float32
    arr = np.expand_dims(arr, axis=-1)       # (H,W,1)
    return arr


def pred_to_mask(pred: np.ndarray, threshold=0.5) -> np.ndarray:
    """
    pred: (H,W,1) or (H,W)
    return: uint8 mask (H,W) in {0,255}
    """
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    mask = (pred >= threshold).astype(np.uint8) * 255
    return mask


def make_output_path(input_path: Path, input_root: Path, output_root: Path, save_same_ext=False) -> Path:
    """
    保留相对路径 + 保留文件名
    """
    rel = input_path.relative_to(input_root)
    out_path = output_root / rel
    if not save_same_ext:
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def split_image_into_patches(img_path, 
                             crop_size=(1536,384),
                             patch_size=(256,256),
                             row_range=(500, 900),
                             ):
    patches = []
    names = []
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    row_start, row_end = row_range
    row_start = max(0, min(row_start, img.shape[0]))
    row_end = max(row_start, min(row_end, img.shape[0]))
    img = img[row_start:row_end, :]
    target_width = 1536
    img_h, img_w = img.shape
    target_h = int(img_h * target_width / img_w)
    img = cv2.resize(img, (target_width, target_h), interpolation=cv2.INTER_CUBIC)
    # crop based on mean intensity as a whole image
    mean_intensity = np.mean(img, axis=1)
    max_mean = 0
    max_start = 0
    for j in range(0, len(mean_intensity)-crop_size[1]+1):
        window_mean = np.mean(mean_intensity[j:j+crop_size[1]])
        if window_mean > max_mean:
            max_mean = window_mean
            max_start = j
    cropped = img[max_start:max_start+crop_size[1], :]
    # calculate how many patches can be extracted
    if crop_size[0] % patch_size[0] !=0:
        raise ValueError("crop_size width must be multiple of patch_size width")
    if crop_size[1] / patch_size[1] >2:
        raise ValueError("crop_size height must smaller than 2 times of patch_size height")
    elif crop_size[1] / patch_size[1] <=2 and crop_size[1] / patch_size[1] >1:
        num_patches = cropped.shape[1] // patch_size[1] * 2
        # in this case, we extract patches in two rows, first row from 0 to patch_size[1], 
        # second row from end - patch_size[1] to end
        for j in range(2):
            for i in range(num_patches//2):
                if j==0:
                    crop_img = cropped[:patch_size[1], i*patch_size[0]:(i+1)*patch_size[0]]
                else:
                    crop_img = cropped[-patch_size[1]:, i*patch_size[0]:(i+1)*patch_size[0]]
                patches.append(crop_img)
                names.append(f"{img_path.stem}_part{(j*num_patches//2+i):02d}{img_path.suffix}")
    else:
        num_patches = cropped.shape[1] // patch_size[1]
        for i in range(num_patches):
            crop_img = cropped[:, i*patch_size[0]:(i+1)*patch_size[0]]
            patches.append(crop_img)
            names.append(f"{img_path.stem}_part{i:02d}{img_path.suffix}")

    return patches, names, cropped, max_start, crop_size
