"""Shared configuration for baseline experiments."""

from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

MODELS: dict[str, object] = {
    "LR": LogisticRegression(
        solver="saga", l1_ratio=1.0, max_iter=5000, random_state=RANDOM_STATE
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

MODELS_LR: dict[str, LogisticRegression] = {
    f"LR_C={c}": LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=c,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    for c in [0.001, 0.01, 0.1, 1]
}

MODELS_RF: dict[str, RandomForestClassifier] = {
    f"RF_max_depth={d}_min_samples_leaf={m}": RandomForestClassifier(
        max_depth=d, min_samples_leaf=m, random_state=RANDOM_STATE
    )
    for d, m in product([2, 4], [5, 10, 15])
}
