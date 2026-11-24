from prediction_kidney_corr_auc_quota_grid_hyperparam import run_hyperparam_grid


PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
    "solver": ["liblinear"],
}


if __name__ == "__main__":
    run_hyperparam_grid(
        "logistic",
        PARAM_GRID,
        "logistic_corr_auc_quota_grid",
        "Run the effective L1/L2 logistic-regression hyperparameter/quota grid for corr_auc feature selection.",
    )
