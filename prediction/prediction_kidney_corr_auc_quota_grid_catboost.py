from prediction_kidney_corr_auc_quota_grid_hyperparam import run_hyperparam_grid


PARAM_GRID = {
    "iterations": [50, 100],
    "depth": [1, 2, 3],
    "learning_rate": [0.03, 0.1],
}


if __name__ == "__main__":
    run_hyperparam_grid(
        "catboost",
        PARAM_GRID,
        "catboost_corr_auc_quota_grid",
        "Run CatBoost hyperparameter/quota grid for corr_auc feature selection.",
    )
