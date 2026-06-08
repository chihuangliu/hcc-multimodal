"""Shared utilities for downstream evaluation pipelines."""

from pathlib import Path

import hcc_multimodal
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hcc_multimodal.train.config import RANDOM_STATE

PROJECT_ROOT: Path = Path(hcc_multimodal.__file__).resolve().parents[1]

DOWNSTREAM_MODELS = {
    "lr": LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=1.0,
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
    "rf": RandomForestClassifier(
        n_estimators=100,
        max_depth=2,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    ),
}


def build_pipeline(model, select_k: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=min(select_k, 9999))),
            ("model", clone(model)),
        ]
    )
