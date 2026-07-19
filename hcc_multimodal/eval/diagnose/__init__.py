"""Label-free transfer-diagnosis scripts.

Reproduces ``reports/0720/0720_best_2_diagnose.md``: given only embeddings (no
target labels), diagnose whether a contrastive model will transfer to an external
cohort. Four angles, one script each:

  collapse  §3  representational collapse of the training-cohort embeddings
  support   §4  fraction of target cells outside the training support + centroid shift
  pca       §5  PCA projection of cohorts onto training-fit axes
  ks        §6  per-dimension KS drift (+ drift↔transfer correlation across models)

All four reuse the cohort loader and KS routine from
``hcc_multimodal.eval.embedding_drift`` via ``diagnose.common``.
"""
