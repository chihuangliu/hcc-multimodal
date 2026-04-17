"""Shared configuration for baseline experiments."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

MODELS: dict[str, object] = {
    "LR": LogisticRegression(
        solver="saga", l1_ratio=1.0, max_iter=1000, random_state=RANDOM_STATE
    ),
    "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

PARAM_GRIDS: dict[str, dict] = {
    "LR": {
        "model__C": [1.0, 10.0, 100.0],
        "model__l1_ratio": [0.5, 1.0],  # 0=L2, 0.5=ElasticNet, 1=L1
    },
    "RF": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_leaf": [1, 3, 5],
    },
}
