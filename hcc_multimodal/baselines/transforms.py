"""Reusable sklearn preprocessing pipelines for HCC multimodal baselines."""

from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


class CPMTransformer(BaseEstimator, TransformerMixin):
    """Convert raw counts to log2(CPM + pseudocount), sample-wise.

    Library size is computed per sample from whatever columns are present in the
    input, so this step must receive the full count matrix (not a post-selection
    subset) to get accurate library sizes.
    """

    def __init__(self, pseudocount: float = 1.0):
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.array([f"x{i}" for i in range(self.n_features_in_)])

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            lib = np.asarray(X.sum(axis=1), dtype=float)
            lib = np.where(lib <= 0, 1.0, lib)
            return np.log2(X.div(lib, axis=0) * 1e6 + self.pseudocount)
        lib = X.sum(axis=1).astype(float)
        lib = np.where(lib <= 0, 1.0, lib)
        return np.log2(X / lib[:, None] * 1e6 + self.pseudocount)


class DataType(StrEnum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


CLINICAL_X_COLUMNS: dict[str, DataType] = {
    "Age": DataType.CONTINUOUS,
    "Sex": DataType.CATEGORICAL,
    "BCLC Stage": DataType.ORDINAL,
    "Max. Diameter of Largest Lesion": DataType.CONTINUOUS,
    "HCC Etiology: Hepatitis B": DataType.CATEGORICAL,
    "HCC Etiology: Hepatitis C": DataType.CATEGORICAL,
    "HCC Etiology: Alcohol": DataType.CATEGORICAL,
    "HCC Etiology: NASH": DataType.CATEGORICAL,
    "Child-Pugh": DataType.CATEGORICAL,
    "vascular_invasion_label": DataType.CATEGORICAL,
    "grading": DataType.CATEGORICAL,
}


def build_preprocessor(
    x_columns: dict[str, DataType],
    rna_cpm: bool = False,
) -> Pipeline | ColumnTransformer:
    """Return a preprocessor for the given column-type mapping.

    Parameters
    ----------
    x_columns:
        Mapping of column name → DataType.
    rna_cpm:
        When True, all columns are treated as raw RNA-seq counts and the
        preprocessor is ``CPMTransformer → StandardScaler`` (no imputation
        needed for filtered count matrices). Use this for the SelectKBest
        path so that features entering the model are log2(CPM)-normalised,
        matching the DeseqCPMSelector path.
    """
    if rna_cpm:
        return Pipeline([
            ("cpm", CPMTransformer()),
            ("scaler", StandardScaler()),
        ])

    continuous_cols = [col for col, t in x_columns.items() if t == DataType.CONTINUOUS]
    categorical_cols = [col for col, t in x_columns.items() if t == DataType.CATEGORICAL]
    ordinal_cols = [col for col, t in x_columns.items() if t == DataType.ORDINAL]

    continuous_transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "to_str",
                FunctionTransformer(
                    lambda X: X.astype(str), feature_names_out="one-to-one"
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_transform, continuous_cols),
            ("categorical", categorical_transform, categorical_cols),
            ("ordinal", ordinal_transform, ordinal_cols),
        ]
    )
