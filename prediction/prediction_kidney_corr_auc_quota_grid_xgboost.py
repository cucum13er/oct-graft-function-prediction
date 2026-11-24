from prediction_kidney_corr_auc_quota_grid_hyperparam import run_hyperparam_grid


PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


if __name__ == "__main__":
    run_hyperparam_grid(
        "xgboost",
        PARAM_GRID,
        "xgboost_corr_auc_quota_grid",
        "Run XGBoost hyperparameter/quota grid for corr_auc feature selection.",
    )
