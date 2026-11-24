# extract_case_features.py
from logging import root
import os, glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import shutil
def read_binary_mask(path):
    m = imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)

def lumen_props_from_mask(mask, dx_um=1.0, dz_um=1.0, min_area_px=20):
    """
    Returns a dict of lumen-level arrays.
    """
    lab = label(mask, connectivity=2)
    props = regionprops(lab)

    areas = []
    perims = []
    major = []
    minor = []
    ecc = []
    sol = []
    centroids = []

    for p in props:
        if p.area < min_area_px:
            continue

        # Convert pixel-based measures to microns where applicable
        # Note: skimage perimeter is approximate; ok for relative descriptors.
        area_um2 = p.area * dx_um * dz_um
        perimeter_um = p.perimeter * np.sqrt(dx_um * dz_um)  # rough approx

        major_um = p.major_axis_length * dx_um
        minor_um = p.minor_axis_length * dx_um  # assuming x pixel corresponds to lateral size

        circularity = (4 * np.pi * area_um2) / (perimeter_um**2 + 1e-9)

        areas.append(area_um2)
        perims.append(perimeter_um)
        major.append(major_um)
        minor.append(minor_um)
        ecc.append(p.eccentricity)
        sol.append(p.solidity)
        centroids.append(p.centroid)  # (row, col) in px

    out = {
        "area": np.array(areas, float),
        "perimeter": np.array(perims, float),
        "major": np.array(major, float),
        "minor": np.array(minor, float),
        "ecc": np.array(ecc, float),
        "solidity": np.array(sol, float),
        "centroid_rc": np.array(centroids, float),
    }
    return out

def filter_centroids_by_border(centroid_rc, H, W, margin_um=0.0, dx_um=1.0, dz_um=1.0):
    """
    Keep only centroids at least margin_um away from image border.
    This reduces edge-effect bias in cropped patches.
    """
    if centroid_rc is None or len(centroid_rc) == 0:
        return centroid_rc

    mr = margin_um / dz_um
    mc = margin_um / dx_um
    r = centroid_rc[:, 0]
    c = centroid_rc[:, 1]
    keep = (r >= mr) & (r < H - mr) & (c >= mc) & (c < W - mc)
    return centroid_rc[keep]

def nearest_neighbor_distances(centroid_rc, dx_um=1.0, dz_um=1.0):
    """
    centroid_rc: Nx2 (row, col) in pixels
    Return N distances (um) to nearest neighbor (centroid distance).
    """
    if centroid_rc.shape[0] < 2:
        return np.array([], float)

    # convert to um coordinates
    r_um = centroid_rc[:, 0] * dz_um
    c_um = centroid_rc[:, 1] * dx_um
    pts = np.stack([r_um, c_um], axis=1)

    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=2)  # first neighbor is itself
    return dists[:, 1]

# def robust_stats(x):
#     """
#     Return robust summary stats for 1D array.
#     """
#     if x is None or len(x) == 0:
#         return dict(median=np.nan, mean=np.nan, std=np.nan, p10=np.nan, p90=np.nan, iqr=np.nan)

#     x = np.asarray(x, float)
#     return {
#         "median": float(np.nanmedian(x)),
#         "mean": float(np.nanmean(x)),
#         "std": float(np.nanstd(x)),
#         "p10": float(np.nanpercentile(x, 10)),
#         "p90": float(np.nanpercentile(x, 90)),
#         "iqr": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
#     }
def robust_stats(x, stats=("median", "mean", "std", "p10", "p90", "iqr")):
    """
    Return selected robust summary stats for 1D array.
    stats: tuple of keys among ("median","mean","std","p10","p90","iqr")
    """
    keys_all = ("median", "mean", "std", "p10", "p90", "iqr")

    if x is None or len(x) == 0:
        return {k: np.nan for k in stats}

    x = np.asarray(x, float)

    out = {}
    if "median" in stats:
        out["median"] = float(np.nanmedian(x))
    if "mean" in stats:
        out["mean"] = float(np.nanmean(x))
    if "std" in stats:
        out["std"] = float(np.nanstd(x))
    if "p10" in stats:
        out["p10"] = float(np.nanpercentile(x, 10))
    if "p90" in stats:
        out["p90"] = float(np.nanpercentile(x, 90))
    if "iqr" in stats:
        out["iqr"] = float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
    return out


def bscan_features(mask, dx_um=1.0, dz_um=1.0, min_area_px=20, tissue_mask=None):
    """
    Compute per-Bscan features from lumen mask.
    tissue_mask: optional binary mask of tissue region for density normalization.
    """
    lp = lumen_props_from_mask(mask, dx_um, dz_um, min_area_px)
    # n = len(lp["minor"])

    # nn = nearest_neighbor_distances(lp["centroid_rc"], dx_um, dz_um)

    # # Density definition (pick one)
    # if tissue_mask is None:
    #     tissue_area_um2 = mask.size * dx_um * dz_um
    # else:
    #     tissue_area_um2 = tissue_mask.sum() * dx_um * dz_um

    # lumen_area_frac = (np.nansum(lp["area"]) / (tissue_area_um2 + 1e-9))
    # lumen_count_density = n / (tissue_area_um2 + 1e-9)  # per um^2 (might be tiny); rescale later if desired

    feats = {}    
    # feats["lumen_count"] = n
    # feats["lumen_area_frac"] = lumen_area_frac
    # feats["lumen_count_density"] = lumen_count_density

    for name, arr in [("minor", lp["minor"]), 
                      ("ecc", lp["ecc"]), 
                      ("sol", lp["solidity"]), 
                    #   ("nn", nn),
                      ]:
        s = robust_stats(arr)
        for k, v in s.items():
            feats[f"{name}_{k}"] = v

    return feats

def aggregate_case(bscan_feat_list):
    """
    bscan_feat_list: list of dicts (each dict is per-bscan)
    Return one dict: case-level features
    """
    df = pd.DataFrame(bscan_feat_list)
    out = {}

    for col in df.columns:
        x = df[col].values.astype(float)

        # across-bscan aggregation
        out[f"{col}__mean"] = float(np.nanmean(x))
        out[f"{col}__std"]  = float(np.nanstd(x))
        out[f"{col}__p10"]  = float(np.nanpercentile(x[~np.isnan(x)], 10)) if np.any(~np.isnan(x)) else np.nan
        out[f"{col}__p90"]  = float(np.nanpercentile(x[~np.isnan(x)], 90)) if np.any(~np.isnan(x)) else np.nan

    out["num_bscans"] = len(df)
    return out

def pick_largest_area_mask(mask_paths, dx_um=1.0, dz_um=1.0, min_area_px=20):
    """
    Pick the mask whose *total lumen area* (sum of region areas after filtering) is largest.
    Returns: (best_path, best_total_area_um2)
    """
    best_path = None
    best_area = -1.0

    for mp in mask_paths:
        try:
            m = read_binary_mask(mp)
        except Exception:
            continue
        if m is None or m.size == 0 or m.sum() == 0:
            continue

        lp = lumen_props_from_mask(m, dx_um=dx_um, dz_um=dz_um, min_area_px=min_area_px)
        total_area = float(np.nansum(lp["area"])) if len(lp["area"]) else 0.0

        if total_area > best_area:
            best_area = total_area
            best_path = mp

    return best_path, best_area


def process_case_single(case_id, mask_paths, dx_um=1.0, dz_um=1.0, min_area_px=20,
                        stats=("median", "mean", "std", "p10", "p90", "iqr"),
                        nn_border_margin_um=0.0):
    """
    Use ONE mask (the one with the largest total lumen area) to represent the case.
    Compute the same features as process_case, but from only that selected mask.
    """
    best_path, best_total_area_um2 = pick_largest_area_mask(
        mask_paths, dx_um=dx_um, dz_um=dz_um, min_area_px=min_area_px
    )

    out = {
        "case_num": case_id,
        "num_bscans_total": int(len(mask_paths)),
        "picked_mask_path": best_path if best_path is not None else "",
        "picked_lumen_area_um2": float(best_total_area_um2) if best_total_area_um2 >= 0 else np.nan,
    }

    if best_path is None:
        # no valid masks
        for name in ("minor", "ecc", "sol", "nnd"):
            for k in stats:
                out[f"{name}_{k}"] = np.nan
        out["num_bscans_used"] = 0
        out["num_lumens_total"] = 0
        out["num_nnd_total"] = 0
        return out

    # compute lumen props on the chosen mask only
    try:
        m = read_binary_mask(best_path)
    except Exception:
        # fail-safe
        for name in ("minor", "ecc", "sol", "nnd"):
            for k in stats:
                out[f"{name}_{k}"] = np.nan
        out["num_bscans_used"] = 0
        out["num_lumens_total"] = 0
        out["num_nnd_total"] = 0
        return out

    lp = lumen_props_from_mask(m, dx_um=dx_um, dz_um=dz_um, min_area_px=min_area_px)

    all_minor = lp["minor"]
    all_ecc   = lp["ecc"]
    all_sol   = lp["solidity"]

    # NND within the chosen mask
    all_nn = np.array([], float)
    c_rc = lp["centroid_rc"]
    if c_rc is not None and len(c_rc) >= 2:
        H, W = m.shape[:2]
        c_rc2 = filter_centroids_by_border(
            c_rc, H, W,
            margin_um=nn_border_margin_um,
            dx_um=dx_um, dz_um=dz_um
        )
        all_nn = nearest_neighbor_distances(c_rc2, dx_um=dx_um, dz_um=dz_um)

    for name, arr in [("minor", all_minor),
                      ("ecc", all_ecc),
                      ("sol", all_sol),
                      ("nnd", all_nn)]:
        s = robust_stats(arr, stats=stats)
        for k, v in s.items():
            out[f"{name}_{k}"] = v

    out["num_bscans_used"] = 1
    out["num_lumens_total"] = int(len(all_minor)) if all_minor is not None else 0
    out["num_nnd_total"] = int(len(all_nn)) if all_nn is not None else 0
    return out




def process_case(case_id, mask_paths, dx_um=1.0, dz_um=1.0, min_area_px=20,
                 stats=("median", "mean", "std", "p10", "p90", "iqr"),
                 nn_border_margin_um=0.0):
    """
    Pool lumen measurements across masks (case-level).
    Adds NND (centroid nearest-neighbor distance) pooled across masks.
    nn_border_margin_um: exclude centroids within this distance to border to reduce crop edge bias.
    """
    all_minor = []
    all_ecc = []
    all_sol = []
    all_nn = []

    num_total = len(mask_paths)
    num_used = 0

    for mp in mask_paths:
        try:
            m = read_binary_mask(mp)
        except Exception:
            continue

        if m is None or m.size == 0 or m.sum() == 0:
            continue

        lp = lumen_props_from_mask(m, dx_um=dx_um, dz_um=dz_um, min_area_px=min_area_px)

        if len(lp["minor"]) == 0:
            continue

        # --- morphology pooled across masks ---
        all_minor.append(lp["minor"])
        all_ecc.append(lp["ecc"])
        all_sol.append(lp["solidity"])

        # --- NND pooled across masks (compute within each mask) ---
        c_rc = lp["centroid_rc"]
        if c_rc is not None and len(c_rc) >= 2:
            H, W = m.shape[:2]
            c_rc2 = filter_centroids_by_border(
                c_rc, H, W,
                margin_um=nn_border_margin_um,
                dx_um=dx_um, dz_um=dz_um
            )
            nn = nearest_neighbor_distances(c_rc2, dx_um=dx_um, dz_um=dz_um)
            if len(nn):
                all_nn.append(nn)

        num_used += 1

    # concatenate pooled lumen arrays
    all_minor = np.concatenate(all_minor) if len(all_minor) else np.array([], float)
    all_ecc   = np.concatenate(all_ecc)   if len(all_ecc)   else np.array([], float)
    all_sol   = np.concatenate(all_sol)   if len(all_sol)   else np.array([], float)
    all_nn    = np.concatenate(all_nn)    if len(all_nn)    else np.array([], float)

    out = {}

    for name, arr in [("minor", all_minor),
                      ("ecc", all_ecc),
                      ("sol", all_sol),
                      ("nnd", all_nn)]:
        s = robust_stats(arr, stats=stats)
        for k, v in s.items():
            out[f"{name}_{k}"] = v

    # meta info
    out["case_num"] = case_id
    out["num_bscans_total"] = int(num_total)
    out["num_bscans_used"] = int(num_used)
    out["num_lumens_total"] = int(len(all_minor))
    out["num_nnd_total"] = int(len(all_nn))

    return out


# def process_case(case_id, mask_paths, dx_um=1.0, dz_um=1.0, min_area_px=20):
#     """
#     Pool ALL lumen measurements across all selected masks in this case.
#     No image-level (B-scan-level) aggregation.
#     Output: one row per case with robust stats over pooled lumens.
#     """
#     all_minor = []
#     all_ecc = []
#     all_sol = []

#     num_total = len(mask_paths)
#     num_used = 0

#     for mp in mask_paths:
#         try:
#             m = read_binary_mask(mp)
#         except Exception:
#             continue

#         if m is None or m.size == 0 or m.sum() == 0:
#             continue

#         lp = lumen_props_from_mask(m, dx_um=dx_um, dz_um=dz_um, min_area_px=min_area_px)

#         # If a picked mask still has zero detected lumens, skip it
#         if len(lp["minor"]) == 0:
#             continue

#         all_minor.append(lp["minor"])
#         all_ecc.append(lp["ecc"])
#         all_sol.append(lp["solidity"])
#         num_used += 1

#     # concatenate pooled lumen arrays
#     all_minor = np.concatenate(all_minor) if len(all_minor) else np.array([], float)
#     all_ecc   = np.concatenate(all_ecc)   if len(all_ecc)   else np.array([], float)
#     all_sol   = np.concatenate(all_sol)   if len(all_sol)   else np.array([], float)

#     out = {}
#     for name, arr in [("minor", all_minor), ("ecc", all_ecc), ("sol", all_sol)]:
#         s = robust_stats(arr)
#         for k, v in s.items():
#             out[f"{name}_{k}"] = v

#     # meta info (useful for debugging / confidence)
#     out["case_num"] = case_id
#     out["num_bscans_total"] = int(num_total)
#     out["num_bscans_used"] = int(num_used)
#     out["num_lumens_total"] = int(len(all_minor))

#     return out
