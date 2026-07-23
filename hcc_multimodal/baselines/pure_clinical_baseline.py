"""Pure clinical baseline: Logistic Regression and Random Forest on 11 clinical features."""

import argparse
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from hcc_multimodal.baselines.config import PARAM_GRIDS, RANDOM_STATE
from hcc_multimodal.baselines.evaluation import run_cv_experiment, plot_cv_results
from hcc_multimodal.baselines.transforms import CLINICAL_X_COLUMNS, DataType, build_preprocessor
from hcc_multimodal.utils.data import CLINICAL_CSV

ROOT = Path(__file__).resolve().parent.parent.parent


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "baseline" / "pure_clinical"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(CLINICAL_CSV).dropna(how="all")

    x_columns = CLINICAL_X_COLUMNS
    preprocessor = build_preprocessor(x_columns)
    y_col = args.outcome
    X = data[list(x_columns.keys())]
    y = data[y_col]

    # Without grid search
    cv_no_gs, fold_no_gs = run_cv_experiment(
        X, y, x_columns,
        label=f"Clinical → {y_col}",
        n_splits=args.cv_folds,
    )
    print("\n=== Without grid search ===")
    print(cv_no_gs.to_string(index=False))
    plot_cv_results(
        fold_no_gs,
        title=f"CV AUC — clinical features → {y_col} (no grid search)",
        save_path=output_dir / "cv_no_grid_search.png",
    )

    # With grid search
    cv_gs, fold_gs = run_cv_experiment(
        X, y, x_columns,
        label=f"Clinical → {y_col} (grid search)",
        param_grids=PARAM_GRIDS,
        n_splits=args.cv_folds,
    )
    print("\n=== With grid search ===")
    print(fold_gs[["model", "fold", "best_params"]].to_string(index=False))
    print(cv_gs.to_string(index=False))
    plot_cv_results(
        fold_gs,
        title=f"CV AUC — clinical features → {y_col} (grid search)",
        save_path=output_dir / "cv_grid_search.png",
    )

    pd.concat([cv_no_gs, cv_gs], ignore_index=True).to_csv(output_dir / "summary.csv", index=False)
    pd.concat([fold_no_gs, fold_gs], ignore_index=True).drop(
        columns=["selected_features", "lr_nonzero_features"], errors="ignore"
    ).to_csv(output_dir / "fold_records.csv", index=False)

    # Feature importance by Random Forest trained on full dataset
    continuous_cols = [col for col, t in x_columns.items() if t == DataType.CONTINUOUS]
    categorical_cols = [col for col, t in x_columns.items() if t == DataType.CATEGORICAL]
    ordinal_cols = [col for col, t in x_columns.items() if t == DataType.ORDINAL]

    mask = ~y.isna()
    X_clean, y_clean = data[list(x_columns.keys())][mask], y[mask]

    full_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    full_pipe.fit(X_clean, y_clean)

    ohe = (
        full_pipe.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    feature_names = continuous_cols + cat_feature_names + ordinal_cols

    importances = full_pipe.named_steps["model"].feature_importances_
    indices = np.argsort(importances)
    top_indices = indices[-5:]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([feature_names[i] for i in top_indices], importances[top_indices], color="steelblue")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest — Feature Importances")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "feature_importance.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure clinical baseline for HCC outcome prediction.")
    parser.add_argument("--outcome", default="OS_central_event",
                        help="Target column name (default: OS_central_event)")
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/baseline/pure_clinical)")
    main(parser.parse_args())
