from prediction_kidney_corr_auc_quota_grid_hyperparam import run_hyperparam_grid


PARAM_GRID = {
    "n_estimators": [4, 8, 16, 32],
    "softmax_temperature": [0.7, 0.9, 1.0],
    "balance_probabilities": [False],
}


if __name__ == "__main__":
    run_hyperparam_grid(
        "tabpfn",
        PARAM_GRID,
        "tabpfn_corr_auc_quota_grid_more",
        "Run TabPFN hyperparameter/quota grid for corr_auc feature selection.",
    )
