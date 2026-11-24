from prediction_kidney_corr_auc_quota_grid_hyperparam import run_hyperparam_grid


PARAM_GRID = {
    "n_estimators": [500],
    "max_depth": [2, 3, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt"],
}


if __name__ == "__main__":
    run_hyperparam_grid(
        "random_forest",
        PARAM_GRID,
        "random_forest_corr_auc_quota_grid",
        "Run random forest hyperparameter/quota grid for corr_auc feature selection.",
    )
