import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

SUPPORTED_ML_METHODS = [
    "logistic",
    "random_forest",
    "extra_trees",
    "svm_rbf",
    "xgboost",
    "catboost",
    "tabpfn",
]


DEFAULT_MODEL_PARAMS = {
    "logistic": {
        "solver": "liblinear",
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 500,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "svm_rbf": {
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "n_jobs": 1,
    },
    "catboost": {
        "iterations": 100,
        "depth": 2,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "verbose": False,
        "allow_writing_files": False,
    },
    "tabpfn": {},
}


def load_local_env_file(filename=".env.local"):
    """Load simple KEY=VALUE lines without requiring python-dotenv."""
    here = Path(__file__).resolve().parent
    filename_path = Path(filename)
    if filename_path.is_absolute():
        candidates = [filename_path]
    else:
        candidates = [
            Path.cwd() / filename,
            Path.cwd().parent / filename,
            here / filename,
            here.parent / filename,
        ]
    candidates.extend(
        [
            Path.cwd() / ".env.local",
            Path.cwd().parent / ".env.local",
            here / ".env.local",
            here.parent / ".env.local",
        ]
    )
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_model_params(value):
    if value is None or str(value).strip() == "":
        return {}
    try:
        params = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--model-params must be valid JSON, for example '{\"max_depth\": 2}'") from exc
    if not isinstance(params, dict):
        raise ValueError("--model-params must decode to a JSON object")
    return params


def merged_model_params(method, override_params=None):
    if method not in SUPPORTED_ML_METHODS:
        raise ValueError(f"Unsupported ml_method={method}. Choices: {', '.join(SUPPORTED_ML_METHODS)}")
    params = dict(DEFAULT_MODEL_PARAMS.get(method, {}))
    params.update(override_params or {})
    return params


def class_balance_scale_pos_weight(y):
    y = np.asarray(y)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def create_model(method, random_state=0, max_iter=5000, y_train=None, override_params=None):
    params = merged_model_params(method, override_params)

    if method == "logistic":
        params.setdefault("max_iter", max_iter)
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)

    if method in {"random_forest", "extra_trees"}:
        params.setdefault("random_state", random_state)
        klass = RandomForestClassifier if method == "random_forest" else ExtraTreesClassifier
        return klass(**params)

    if method == "svm_rbf":
        params.setdefault("random_state", random_state)
        return SVC(kernel="rbf", **params)

    if method == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Install it before using --ml-method xgboost.") from exc
        params.setdefault("random_state", random_state)
        if y_train is not None:
            params.setdefault("scale_pos_weight", class_balance_scale_pos_weight(y_train))
        return XGBClassifier(**params)

    if method == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is not installed. Install it before using --ml-method catboost.") from exc
        params.setdefault("random_seed", random_state)
        if y_train is not None:
            scale = class_balance_scale_pos_weight(y_train)
            params.setdefault("class_weights", [1.0, scale])
        return CatBoostClassifier(**params)

    if method == "tabpfn":
        load_local_env_file()
        if not os.environ.get("TABPFN_TOKEN"):
            raise RuntimeError(
                "TABPFN_TOKEN is required for --ml-method tabpfn. "
                "Create .env.local with TABPFN_TOKEN=your_api_key, or export it in your shell."
            )
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError("tabpfn is not installed. Install it before using --ml-method tabpfn.") from exc
        try:
            return TabPFNClassifier(random_state=random_state, **params)
        except TypeError:
            return TabPFNClassifier(**params)

    raise ValueError(f"Unsupported ml_method={method}")


def predict_positive_probability(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return np.asarray(proba).reshape(-1)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-scores))

    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    return pred
