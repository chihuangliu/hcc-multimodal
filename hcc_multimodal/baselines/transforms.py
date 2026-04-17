"""Reusable sklearn preprocessing pipelines for HCC multimodal baselines."""

from enum import StrEnum

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


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


def build_preprocessor(x_columns: dict[str, DataType]) -> ColumnTransformer:
    """Return a ColumnTransformer for the given column-type mapping.

    Parameters
    ----------
    x_columns:
        Mapping of column name → DataType.
    """
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
