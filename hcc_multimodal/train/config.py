"""Best-performing hyperparameters for the arterial radiomic RFS baseline.

Values taken from reports/0504/0504_rfs_baselines_v2.md (before-CV, SelectKBest
F-score k=100):
  - RF  (1y 0.821 ± 0.068, 2y 0.781 ± 0.069): RF_max_depth=2_min_samples_leaf=10
  - LR  (1y 0.780 ± 0.103, 2y 0.752 ± 0.092): LR_C=1
"""

RANDOM_STATE = 42
SELECT_K = 100

RF_MAX_DEPTH = 2
RF_MIN_SAMPLES_LEAF = 10
RF_N_ESTIMATORS = 100

LR_C = 1.0
LR_SOLVER = "saga"
LR_L1_RATIO = 1.0
LR_MAX_ITER = 1000
