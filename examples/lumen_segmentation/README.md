# Lumen segmentation demo

Each example case folder contains one original grayscale OCT B-scan. The case folders are demonstration labels. To try another scan, use an unannotated grayscale PNG or TIFF.

```text
case_01/*.png
case_02/*.png
case_03/*.png
```

The review script searches these folders recursively. Follow the [lumen demo command](../../README.md#1-segment-oct-b-scans) from the repository root. The default crop is suited to the study image dimensions. Use `--row-range START END` to select the rows containing the tissue if the scan is framed differently.

Press **S** or **Enter** to accept and save the mask, **Space** to skip, or **Q** to quit. Saved masks are binary PNGs under `results/lumen_demo/masks/`. The corresponding images are saved under `results/lumen_demo/images/`.
