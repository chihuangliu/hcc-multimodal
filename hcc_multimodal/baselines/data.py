"""Clinical data helpers for HCC multimodal baselines."""

import pandas as pd


def rfs(rfs_central: float, rfs_central_event: float, months: int) -> int | None:
    if rfs_central_event == 1:
        if rfs_central <= months:
            return 1
        else:
            return 0
    else:
        if rfs_central < months:
            return None
        else:
            return 0


def add_rfs_columns(clinical_data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *clinical_data* with ``rfs_1year`` and ``rfs_2year`` columns."""
    clinical_data = clinical_data.copy()
    clinical_data["rfs_1year"] = clinical_data.apply(
        lambda row: rfs(row["RFS_central"], row["RFS_central_event"], 1 * 12), axis=1
    )
    clinical_data["rfs_2year"] = clinical_data.apply(
        lambda row: rfs(row["RFS_central"], row["RFS_central_event"], 2 * 12), axis=1
    )
    return clinical_data
