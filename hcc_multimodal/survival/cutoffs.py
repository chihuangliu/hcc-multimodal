"""High/low-risk cutoff strategies.

Each strategy maps a continuous risk score to a high/low grouping of the test
cohort. They differ in what information they use:

  median_within  (PRIMARY)     - split at the test cohort's own median score.
                                  Leakage-free (no outcome); balanced ~50/50;
                                  tests within-cohort rank-ordering.
  median_frozen  (SENSITIVITY) - split at the resection out-of-fold median.
                                  Leakage-free; tests absolute-score transfer;
                                  sensitive to base-rate shift.
  kmeans_within                - KMeans(k=2) on the 1-D test scores. Leakage-free,
                                  data-driven; adapts to the cohort distribution.
  youden_frozen                - threshold maximizing Youden's J (sens+spec-1) on
                                  resection vs the rfs_2year label, then frozen.
                                  Uses the *training* outcome only, so it is
                                  leakage-free for the held-out test cohort.

Higher score == higher risk for both routes (Route A probability, Route B
partial hazard), so "high" is always ``score >= threshold``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

from hcc_multimodal.train.config import RANDOM_STATE

PRIMARY_METHOD = "median_within"
SENSITIVITY_METHOD = "median_frozen"


def _groups(scores: pd.Series, thr: float) -> pd.Series:
    return pd.Series(
        np.where(scores.values >= thr, "high", "low"), index=scores.index, name="group"
    )


def median_within(train_scores, train_labels, test_scores):
    thr = float(np.median(test_scores.values))
    return _groups(test_scores, thr), {"scope": "within_cohort", "threshold": thr}


def median_frozen(train_scores, train_labels, test_scores):
    thr = float(np.median(train_scores.values))
    return _groups(test_scores, thr), {"scope": "frozen_resection", "threshold": thr}


def kmeans_within(train_scores, train_labels, test_scores):
    x = test_scores.values.reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(x)
    centers = km.cluster_centers_.ravel()
    high_cluster = int(np.argmax(centers))
    groups = pd.Series(
        np.where(km.labels_ == high_cluster, "high", "low"),
        index=test_scores.index, name="group",
    )
    thr = float(centers.mean())  # boundary between the two 1-D centroids
    return groups, {"scope": "within_cohort_kmeans", "threshold": thr, "centers": sorted(centers.tolist())}


def kmeans_log_within(train_scores, train_labels, test_scores):
    """KMeans(k=2) on log-transformed test scores.

    The Cox partial hazard (Route B) is heavy-tailed, so squared-distance k-means
    on the raw scale isolates a single outlier. Clustering in log space compresses
    the tail so the split reflects the bulk of the distribution.
    """
    s = test_scores.to_numpy(dtype=float)
    x = np.log(np.clip(s, 1e-12, None)).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(x)
    centers = km.cluster_centers_.ravel()
    high_cluster = int(np.argmax(centers))
    groups = pd.Series(
        np.where(km.labels_ == high_cluster, "high", "low"),
        index=test_scores.index, name="group",
    )
    thr = float(np.exp(centers.mean()))
    return groups, {
        "scope": "within_cohort_kmeans_log",
        "threshold": thr,
        "centers": sorted(float(c) for c in np.exp(centers)),
    }


def youden_frozen(train_scores, train_labels, test_scores):
    mask = train_labels.notna()
    y = train_labels[mask].astype(int).to_numpy()
    s = train_scores.loc[train_labels[mask].index].to_numpy()
    if len(np.unique(y)) < 2:
        thr = float(np.median(train_scores.values))
    else:
        fpr, tpr, thresholds = roc_curve(y, s)
        thr = float(thresholds[np.argmax(tpr - fpr)])
    return _groups(test_scores, thr), {"scope": "frozen_resection_youden", "threshold": thr}


CUTOFF_METHODS = {
    "median_within": median_within,
    "median_frozen": median_frozen,
    "kmeans_within": kmeans_within,
    "kmeans_log_within": kmeans_log_within,
    "youden_frozen": youden_frozen,
}
