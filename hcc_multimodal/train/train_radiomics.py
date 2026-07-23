"""Train an arterial radiomic baseline model for HCC RFS prediction.

Loads arterial-phase radiomic features from the resection cohort
(radiomic_cluster.csv, _art columns only), applies SelectKBest(f_classif, k=100)
on the full labelled set, fits LR and RF, then saves one joblib pipeline per
model type to --output-dir (default: models/radiomics/).

Usage
-----
    python -m hcc_multimodal.train.train_radiomics
    python -m hcc_multimodal.train.train_radiomics --target rfs_1year
    python -m hcc_multimodal.train.train_radiomics --output-dir /path/to/output
"""

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.train.config import (
    LR_C, LR_L1_RATIO, LR_MAX_ITER, LR_SOLVER,
    RANDOM_STATE, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS, SELECT_K,
)
from hcc_multimodal.utils.data import (
    CLINICAL_CSV,
    RADIOMICS_FEATURES,
    RADIOMIC_CLUSTER_CSV as RADIOMIC_CSV,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "lr": LogisticRegression(solver=LR_SOLVER, C=LR_C, l1_ratio=LR_L1_RATIO, max_iter=LR_MAX_ITER, random_state=RANDOM_STATE),
    "rf": RandomForestClassifier(
        max_depth=RF_MAX_DEPTH, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE,
    ),
}


def load_data() -> pd.DataFrame:
    clinical = pd.read_csv(CLINICAL_CSV).dropna(how="all")
    clinical = add_rfs_columns(clinical)

    radiomic = pd.read_csv(RADIOMIC_CSV).dropna(how="all")
    art_cols = [f"{f}_art" for f in RADIOMICS_FEATURES]
    radiomic = radiomic[["SID"] + art_cols].copy()
    radiomic.columns = ["SID"] + RADIOMICS_FEATURES

    merged = radiomic.merge(clinical[["SID", "rfs_1year", "rfs_2year"]], on="SID")
    return merged.set_index("SID").sort_index()


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=SELECT_K)),
        ("model", clone(model)),
    ])


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main(target: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    X = data.drop(columns=["rfs_1year", "rfs_2year"])

    y = data[target]
    mask = ~y.isna()
    X_fit, y_fit = X.loc[mask], y[mask]
    print(f"Target: {target}  n={len(y_fit)}  positives={int(y_fit.sum())}  features={X_fit.shape[1]}")

    saved_models = []
    for name, model in MODELS.items():
        pipe = build_pipeline(model)
        pipe.fit(X_fit, y_fit)
        path = output_dir / f"radiomic_{target}_{name}.joblib"
        joblib.dump(pipe, path)
        saved_models.append(path.name)
        print(f"  Saved {path}")

    metadata = {
        "git_commit": git_commit(),
        "args": {
            "target": target,
            "output_dir": str(output_dir),
        },
        "hyperparameters": {
            "select_k": SELECT_K,
            "random_state": RANDOM_STATE,
            "lr": {"solver": LR_SOLVER, "C": LR_C, "l1_ratio": LR_L1_RATIO, "max_iter": LR_MAX_ITER},
            "rf": {
                "max_depth": RF_MAX_DEPTH,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                "n_estimators": RF_N_ESTIMATORS,
            },
        },
        "data": {
            "n_samples": int(len(y_fit)),
            "n_positives": int(y_fit.sum()),
            "n_features_input": int(X_fit.shape[1]),
            "n_features_selected": SELECT_K,
        },
        "saved_models": saved_models,
    }
    meta_path = output_dir / f"radiomic_{target}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Saved {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="rfs_2year",
        choices=["rfs_1year", "rfs_2year"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "radiomics",
    )
    args = parser.parse_args()
    main(args.target, args.output_dir)
