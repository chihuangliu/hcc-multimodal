"""Clinical data helpers for HCC multimodal baselines."""

import warnings

import pandas as pd

from hcc_multimodal.utils.data import GENE_SETS_DIR, RNA_SEQ_CSV


_GENE_NAME_MAP: dict[str, str] = {
    "AARS1": "AARS",
    "BPNT2": "IMPAD1",
    "CCN1": "CYR61",
    "CCN2": "CTGF",
    "CILK1": "ICK",
    "DYNC2I2": "WDR34",
    "EPRS1": "EPRS",
    "G6PC1": "G6PC",
    "GBA1": "GBA",
    "H1-0": "H1F0",
    "H1-6": "HIST1H1E",
    "H2BC7": "HIST1H2BG",
    "H2BC15": "HIST1H2BN",
    "H2BC21": "HIST2H2BF",
    "H4C1": "HIST1H4A",
    "H4C3": "HIST1H4C",
    "H4C14": "HIST2H4B",
    "HJV": "HFE2",
    "IARS1": "IARS",
    "ILRUN": "C6orf106",
    "ITPRID2": "SSFA2",
    "MACROH2A2": "H2AFY2",
    "MMUT": "MUT",
    "NARS1": "NARS",
    "NHERF1": "SLC9A3R1",
    "NHERF2": "SLC9A3R2",
    "NIBAN1": "FAM129A",
    "NTAQ1": "NTAN1",
    "PHB1": "PHB",
    "PLAAT3": "PLA2G16",
}


def get_hcc_genes() -> list[str]:
    """Return deduplicated gene names from all HCC gene sets that exist in the RNA-seq matrix.

    Genes present in gene_sets_combined but absent from Matrix_output_radiology_only.csv
    are reported via a warning and excluded from the returned list.
    """
    all_genes: set[str] = set()
    for path in GENE_SETS_DIR.glob("*.txt"):
        df = pd.read_csv(path, sep="\t")
        all_genes.update(df["GeneName"].tolist())

    all_genes = {_GENE_NAME_MAP.get(g, g) for g in all_genes}

    matrix_genes: set[str] = set(
        pd.read_csv(RNA_SEQ_CSV, usecols=["Gene Symbol"])["Gene Symbol"]
    )

    unmatched = all_genes - matrix_genes
    if unmatched:
        warnings.warn(
            f"{len(unmatched)} gene(s) not found in RNA-seq matrix and will be excluded: "
            + ", ".join(sorted(unmatched)),
            stacklevel=2,
        )

    return sorted(all_genes & matrix_genes)


def rfs(
    rfs_central: float,
    rfs_central_event: float,
    months: int,
    tolerance_months: int = 0,
) -> int | None:
    if rfs_central_event == 1:
        if rfs_central <= months:
            return 1
        else:
            return 0
    else:
        # Censored: drop only if follow-up is shorter than (threshold - tolerance).
        # e.g. threshold=24, tolerance=3: a patient last seen at 21 months with no
        # event is retained as negative because 21 >= 24 - 3.
        if rfs_central < (months - tolerance_months):
            return None
        else:
            return 0


def add_rfs_columns(
    clinical_data: pd.DataFrame,
    tolerance_months: int = 0,
) -> pd.DataFrame:
    """Return a copy of *clinical_data* with ``rfs_1year`` and ``rfs_2year`` columns."""
    clinical_data = clinical_data.copy()
    clinical_data["rfs_1year"] = clinical_data.apply(
        lambda row: rfs(row["RFS_central"], row["RFS_central_event"], 1 * 12, tolerance_months), axis=1
    )
    clinical_data["rfs_2year"] = clinical_data.apply(
        lambda row: rfs(row["RFS_central"], row["RFS_central_event"], 2 * 12, tolerance_months), axis=1
    )
    return clinical_data
