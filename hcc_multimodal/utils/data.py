from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
# Repository data root: …/hcc-multimodal/data
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
RESECTION_ROOT = DATA_ROOT / "Resection"

# Resection cohort inputs (canonical locations under data/Resection/)
CLINICAL_CSV = RESECTION_ROOT / "Clinical" / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
RNA_SEQ_DIR = RESECTION_ROOT / "RNA_seq"
RNA_SEQ_CSV = RNA_SEQ_DIR / "Matrix_output_radiology_only.csv"
GENE_SETS_DIR = RNA_SEQ_DIR / "gene_sets_combined"
RADIOMIC_CLUSTER_CSV = RESECTION_ROOT / "Images" / "Radiomics" / "radiomic_cluster.csv"
# Per-lesion arterial radiomic TSVs: arterial/{SID}/{SID}.nii-results.tsv
# (same layout as the ablation cohorts — read via load_resection_arterial_radiomics).
MRI_ARTERIAL_ROOT = RESECTION_ROOT / "Images" / "Radiomics" / "arterial"


# Radiomic features used during training (from radiomic_cluster.csv, _art phase only).
# In the ablation TSV files the phase suffix is absent, so these names map directly
# to TSV column headers.
RADIOMICS_FEATURES = [
    "FOS_Imode",
    "FOS_RMS",
    "FD_min_4gl",
    "FD_max_8gl",
    "FD_max_16gl",
    "FOS_CV_LLL",
    "GLSZM_ZoneHiGl_LLL_4gl",
    "GLSZM_SzoneLoGl_LLL_4gl",
    "FD_max_LLL_4gl",
    "FD_min_LLL_4gl",
    "FD_max_LLL_8gl",
    "FD_min_LLL_8gl",
    "FD_max_LLL_16gl",
    "NGTDM_Complex_LLL_256gl",
    "FD_lacunarity_LLL_256gl",
    "GLCM_sumEnt_LLL_256gl",
    "FOS_CV_LLH",
    "FOS_Imedian_LLH",
    "FOS_RMS_LLH",
    "GLSZM_ZoneLoGl_LLH_4gl",
    "GLSZM_ZoneHiGl_LLH_4gl",
    "GLSZM_SzoneLoGl_LLH_4gl",
    "FD_max_LLH_4gl",
    "FD_min_LLH_4gl",
    "FD_max_LLH_8gl",
    "FD_min_LLH_8gl",
    "FD_mean_LLH_16gl",
    "FD_max_LLH_16gl",
    "FD_min_LLH_16gl",
    "GLSZM_GlVarianc_LLH_32gl",
    "FD_lacunarity_LLH_32gl",
    "FD_min_LLH_32gl",
    "FD_min_LLH_64gl",
    "FD_mean_LLH_256gl",
    "FD_lacunarity_LLH_256gl",
    "FD_max_LLH_256gl",
    "GLCM_Correl_LLH_256gl",
    "GLCM_ClShad_LLH_256gl",
    "GLCM_AutoCorrel_LLH_256gl",
    "FOS_Skew_LHL",
    "GLSZM_ZoneHiGl_LHL_4gl",
    "FD_min_LHL_4gl",
    "FD_max_LHL_8gl",
    "FD_min_LHL_8gl",
    "FD_min_LHL_16gl",
    "GLRLM_LRLGLE_LHL_32gl",
    "FD_min_LHL_32gl",
    "FD_min_LHL_256gl",
    "GLCM_ClShad_LHL_256gl",
    "GLCM_AutoCorrel_LHL_256gl",
    "FOS_CV_LHH",
    "FOS_Skew_LHH",
    "GLSZM_ZoneLoGl_LHH_4gl",
    "GLSZM_ZoneHiGl_LHH_4gl",
    "FD_max_LHH_4gl",
    "FD_min_LHH_4gl",
    "FD_min_LHH_8gl",
    "FD_min_LHH_16gl",
    "GLRLM_LRLGLE_LHH_32gl",
    "FD_lacunarity_LHH_32gl",
    "FD_lacunarity_LHH_256gl",
    "FD_min_LHH_256gl",
    "GLCM_Correl_LHH_256gl",
    "GLCM_AutoCorrel_LHH_256gl",
    "FOS_CV_HLL",
    "FOS_Imedian_HLL",
    "GLSZM_ZoneLoGl_HLL_4gl",
    "GLSZM_ZoneHiGl_HLL_4gl",
    "FD_min_HLL_4gl",
    "FD_max_HLL_8gl",
    "FD_min_HLL_8gl",
    "FD_min_HLL_16gl",
    "GLCM_invVar_HLL_16gl",
    "FD_min_HLL_32gl",
    "FD_lacunarity_HLL_256gl",
    "FD_min_HLL_256gl",
    "GLCM_AutoCorrel_HLL_256gl",
    "FOS_Imean_HLH",
    "FOS_Imedian_HLH",
    "FOS_Skew_HLH",
    "GLSZM_ZoneLoGl_HLH_4gl",
    "GLSZM_ZoneHiGl_HLH_4gl",
    "GLSZM_SzVarianc_HLH_4gl",
    "FD_max_HLH_4gl",
    "FD_min_HLH_4gl",
    "GLCM_ClShad_HLH_4gl",
    "FD_max_HLH_8gl",
    "FD_min_HLH_8gl",
    "FD_mean_HLH_16gl",
    "FD_max_HLH_16gl",
    "FD_min_HLH_16gl",
    "GLRLM_LRLGLE_HLH_32gl",
    "GLSZM_SzVarianc_HLH_64gl",
    "FD_lacunarity_HLH_256gl",
    "GLCM_Correl_HLH_256gl",
    "GLCM_ClShad_HLH_256gl",
    "GLCM_AutoCorrel_HLH_256gl",
    "FOS_CV_HHL",
    "FOS_Imedian_HHL",
    "GLSZM_ZoneLoGl_HHL_4gl",
    "GLSZM_ZoneHiGl_HHL_4gl",
    "FD_max_HHL_4gl",
    "FD_min_HHL_4gl",
    "FD_max_HHL_8gl",
    "FD_min_HHL_8gl",
    "FD_min_HHL_16gl",
    "GLCM_invVar_HHL_16gl",
    "GLSZM_LzoneHiGl_HHL_256gl",
    "GLCM_Correl_HHL_256gl",
    "GLCM_ClShad_HHL_256gl",
    "GLCM_AutoCorrel_HHL_256gl",
    "FOS_CV_HHH",
    "FOS_Imedian_HHH",
    "FOS_RMS_HHH",
    "GLSZM_ZoneHiGl_HHH_4gl",
    "FD_max_HHH_4gl",
    "FD_min_HHH_4gl",
    "FD_max_HHH_8gl",
    "FD_min_HHH_8gl",
    "GLCM_ClShad_HHH_8gl",
    "GLCM_invVar_HHH_16gl",
    "FD_lacunarity_HHH_256gl",
    "FD_max_HHH_256gl",
    "GLCM_Correl_HHH_256gl",
    "GLCM_ClShad_HHH_256gl",
    "GLCM_Entrop_HHH_256gl",
    "GLCM_AutoCorrel_HHH_256gl",
    "GLCM_invVar_HHH_256gl",
    "MeanIntensity3D",
    "TenthIntensityPercentile3D",
    "CoefficientOfVariation3D",
    "QuartileCoefficientOfDispersion3D",
    "MinimumDiscretisedIntensity3D",
    "MaximumHistogramGradientIntensity3D",
    "TenPercentIntensityFraction3D",
    "ClusterShadeMerged3D",
    "ClusterProminenceMerged3D",
    "HighGrayLevelZoneDistanceEmphasis3D",
    "SmallDistanceHighGrayLevelEmphasis3D",
    "GrayLevelDistanceVariance3D",
    "Strength3D",
    "HighGrayLevelCountEmphasis3D",
    "LowDependenceLowGrayLevelEmphasis3D",
    "HighDependenceLowGrayLevelEmphasis3D",
    "NormalisedGrayLevelDependenceNonUniformity3D",
    "DependenceCountNonUniformity3D",
    "DependenceCountPercentage3D",
    "GrayLevelDependenceVariance3D",
    "DependenceCountEnergy3D",
]


def load_resection_arterial_radiomics(root: Path = MRI_ARTERIAL_ROOT) -> pd.DataFrame:
    """Read per-lesion arterial radiomic TSVs into one wide DataFrame.

    Mirrors the ablation cohorts' layout: each subdirectory ``{name}/`` under
    ``root`` holds a single-row export ``{name}.nii-results.tsv``. For the
    resection cohort ``name`` is the patient ``SID``; ``Scan name`` inside each
    TSV is ``{SID}.nii.gz``. Rows are concatenated to reproduce the (now-absent)
    ``arterial_radiomics.csv`` — metadata columns (``Scan name``/``VOI name``/…)
    included — so callers derive ``SID`` and select features exactly as before.
    """
    frames = [
        pd.read_csv(tsv, sep="\t", nrows=1)
        for lesion_dir in sorted(p for p in root.iterdir() if p.is_dir())
        if (tsv := lesion_dir / f"{lesion_dir.name}.nii-results.tsv").exists()
    ]
    if not frames:
        raise FileNotFoundError(f"No '*.nii-results.tsv' found under {root}")
    return pd.concat(frames, ignore_index=True)
