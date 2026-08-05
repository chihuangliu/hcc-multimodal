"""Clinical characteristics of the three cohorts, restricted to the patients that
actually enter a downstream evaluation.

The thesis clinical tables describe the *full* cohorts (60 / 104 / 68). The AUROC is
computed on the intersection of "has a cached image embedding" and "has a label for the
target endpoint" (54 / 57 / 66 for ``rfs_2year``), which is a different and smaller set.
This script tabulates that set.

Note on encodings: the three CRF exports mostly share a scheme (1 = yes / A, 2 = no / B),
but ``BCLC Stage`` does not — Soramic and Lausanne use ``1=0, 2=A, 3=B, 4=C`` while the
resection export is offset by one (``1=A, 2=B, 3=C``). ``BCLC_OFFSET`` encodes that.

Usage:
  python -m hcc_multimodal.eval.cohort_characteristics --model-id d7085bf5 --input raw
"""

import argparse
from pathlib import Path

import pandas as pd

from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.eval.data import (
    RESECTION_CLINICAL_CSV,
    TRAINING_ROOT,
    get_ablation_config,
    load_ablation_outcomes,
    load_resection_outcomes,
)
from hcc_multimodal.eval.eval_utils import PROJECT_ROOT

COHORTS = ("resection", "soramic", "lausanne")
_COHORT_FILENAME = {"soramic": "soramic", "lausanne": "lusanne"}

# BCLC raw code -> stage. The resection export is shifted one down from the other two.
BCLC_OFFSET = {"resection": {1: "A", 2: "B", 3: "C"},
               "soramic":   {1: "0", 2: "A", 3: "B", 4: "C"},
               "lausanne":  {1: "0", 2: "A", 3: "B", 4: "C"}}


def _emb_path(model_id: str, suffix: str, cohort: str) -> Path:
    base = TRAINING_ROOT / model_id / "cached_embeddings"
    if cohort == "resection":
        return base / "resection_img_emb.parquet"
    return base / f"ablation_{_COHORT_FILENAME[cohort]}_img_emb_{suffix}.parquet"


def _clinical(cohort: str) -> pd.DataFrame:
    if cohort == "resection":
        df = pd.read_csv(RESECTION_CLINICAL_CSV).dropna(how="all")
    else:
        cfg = get_ablation_config(_COHORT_FILENAME[cohort])
        df = cfg.read_clinical(cfg.clinical_path).dropna(how="all")
    df = add_rfs_columns(df)
    df["SID"] = df["SID"].astype(int)
    return df.set_index("SID")


def _outcomes(cohort: str, target: str) -> pd.Series:
    if cohort == "resection":
        return load_resection_outcomes(target)
    return load_ablation_outcomes(_COHORT_FILENAME[cohort], target)


def evaluation_subsets(model_id: str, suffix: str, target: str) -> dict[str, pd.DataFrame]:
    """Clinical rows for the patients with both a cached embedding and a label."""
    subsets = {}
    for cohort in COHORTS:
        emb = pd.read_parquet(_emb_path(model_id, suffix, cohort))
        idx = emb.index.intersection(_outcomes(cohort, target).index)
        clinical = _clinical(cohort)
        subsets[cohort] = clinical.loc[clinical.index.intersection(idx)]
    return subsets


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _median_iqr(df: pd.DataFrame, col: str, fmt: str = "%.1f") -> str:
    s = _num(df, col).dropna()
    if s.empty:
        return "—"
    return f"{fmt % s.median()} [{fmt % s.quantile(.25)}–{fmt % s.quantile(.75)}]"


def _frac(mask: pd.Series, denom: int) -> str:
    if not denom:
        return "—"
    k = int(mask.fillna(False).sum())
    return f"{100 * k / denom:.1f}% ({k}/{denom})"


def _recorded(df: pd.DataFrame, col: str) -> int:
    return int(_num(df, col).notna().sum())


def build_table(subsets: dict[str, pd.DataFrame], target: str) -> pd.DataFrame:
    rows: list[dict] = []

    def add(name, fn):
        rows.append({"variable": name, **{c: fn(c, d) for c, d in subsets.items()}})

    add("n entering evaluation", lambda c, d: str(len(d)))
    add(f"{target} positive", lambda c, d: _frac(_num(d, target) == 1, len(d)))
    add("RFS_central, months", lambda c, d: _median_iqr(d, "RFS_central"))
    add("Child-Pugh points", lambda c, d: _median_iqr(d, "Child-Pugh Points", "%.0f"))
    add("Child-Pugh points ≥ 7",
        lambda c, d: _frac(_num(d, "Child-Pugh Points") >= 7, _recorded(d, "Child-Pugh Points")))
    add("Child-Pugh class B",
        lambda c, d: _frac(_num(d, "Child-Pugh") == 2, _recorded(d, "Child-Pugh")))
    add("BCLC beyond stage A", lambda c, d: _frac(
        _num(d, "BCLC Stage").map(BCLC_OFFSET[c]).isin(["B", "C"]), _recorded(d, "BCLC Stage")))
    add("Max lesion diameter, mm",
        lambda c, d: _median_iqr(d, "Max. Diameter of Largest Lesion"))
    add("Number of lesions",
        lambda c, d: _median_iqr(d, "Number of Lesions (99 = Diffuse Disease)", "%.0f"))
    add("Age, years", lambda c, d: _median_iqr(d, "Age"))
    add("Male", lambda c, d: _frac(_num(d, "Sex") == 1, len(d)))
    # ablation exports code aetiology as 1 = yes / blank = no; resection codes 1 / 0
    for label, col in (("alcohol", "HCC Etiology: Alcohol"),
                       ("hepatitis B", "HCC Etiology: Hepatitis B"),
                       ("hepatitis C", "HCC Etiology: Hepatitis C")):
        add(f"Aetiology: {label}", lambda c, d, col=col: _frac(_num(d, col) == 1, len(d)))
    add("Liver cirrhosis",
        lambda c, d: _frac(_num(d, "Liver Cirrhosis") == 1, _recorded(d, "Liver Cirrhosis")))
    add("Distinct recruiting sites", lambda c, d: str(d["Site"].nunique(dropna=True) or "—"))
    add("Distinct countries", lambda c, d: str(d["Country"].nunique(dropna=True) or "—"))

    return pd.DataFrame(rows)[["variable", *COHORTS]]


def label_availability(model_id: str, suffix: str, target: str) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        emb = pd.read_parquet(_emb_path(model_id, suffix, cohort))
        y = _outcomes(cohort, target)
        full = _clinical(cohort)
        rows.append({
            "cohort": cohort,
            "cached_embeddings": len(emb),
            f"{target}_labelled": len(y),
            "entering_evaluation": len(emb.index.intersection(y.index)),
            "median_RFS_central_whole_file":
                round(float(pd.to_numeric(full["RFS_central"], errors="coerce").median()), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True,
                        help="Encoder whose cached embeddings define the evaluable set.")
    parser.add_argument("--input", choices=["raw", "bbox"], default="raw")
    parser.add_argument("--target", default="rfs_2year")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "results" / "eval" / "cohort_characteristics.csv")
    args = parser.parse_args()

    subsets = evaluation_subsets(args.model_id, args.input, args.target)
    table = build_table(subsets, args.target)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Saved → {args.out}\n")
    print(table.to_string(index=False))
    print("\nLabel availability")
    print(label_availability(args.model_id, args.input, args.target).to_string(index=False))


if __name__ == "__main__":
    main()
