"""Survival analysis for contrastive embeddings.

Turns the two best-per-cohort embedding models from the 0608 ablation report into
risk-stratification models: a high/low-risk cutoff is learned on the resection
(training) cohort, frozen, and applied to the Soramic / Lausanne ablation cohorts.

Two risk-score routes are compared:
  - Route A: reuse the 2-year RFS classifier probability (SelectKBest + LR/RF head).
  - Route B: penalized Cox proportional hazards on PCA-reduced embeddings.

Run order:
  python -m hcc_multimodal.survival.compute_risk_scores
  python -m hcc_multimodal.survival.run_analysis
  python -m hcc_multimodal.survival.make_figures
(or python -m hcc_multimodal.survival.run_all)
"""

MODELS = {"9109a6c2": "lr", "1361bef2": "rf"}  # model_id -> Route-A report-best head
COHORTS = ("soramic", "lusanne")
ROUTES = ("a", "b")

# All 17 contrastive models from the 0608 ablation report, in report order.
ALL_MODELS = (
    "6a1a1bdf", "1361bef2", "982a6fa2", "a6f970d6",
    "12e4ba6a", "34e6806f", "5d04e6ba", "9109a6c2",
    "dc7e1d10", "5e3f71a0", "a64b245f", "06c598c0",
    "050d401d", "f8aabb75", "e12b0592", "8715461c", "92b9afed",
)
