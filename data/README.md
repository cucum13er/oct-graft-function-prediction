# Data and model inputs

## Prediction inputs

Place the three analysis-ready CSVs in `data/processed/`. Each table contains one row per kidney with the features and outcome used by the prediction scripts.

| Filename | Analysis |
| --- | --- |
| `kidney_features_complete_clinical_oct_all_available.csv` | Clinical and OCT morphology analysis in 52 kidneys |
| `kidney_features_complete_clinical_oct_biopsy.csv` | Clinical, biopsy, and OCT morphology analysis. The script applies the `donor_num_glom > 35` criterion to obtain the 33-kidney cohort. |
| `kidney_spatial_candidates.csv` | Exploratory OCT spatial analysis in 12 kidneys |

The join key is `case_num`. The outcome column `IGF(0)/DGF(1)` encodes immediate graft function as 0 and delayed graft function as 1. Keep the original column names when running the study analyses.

## Segmentation demo inputs

Save the two [downloadable model weights](../README.md#1-segment-oct-b-scans) in `data/weights/`. Place example B-scans in [`examples/lumen_segmentation/`](../examples/lumen_segmentation/README.md). The review script writes the accepted B-scans and generated binary masks to `results/lumen_demo/`.

Prediction tables, model weights, and generated results are ignored by Git.
