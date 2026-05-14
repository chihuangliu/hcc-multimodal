
## Table of Contents

- [1. Task](#1-task)
- [2. Key findings](#2-key-findings)
- [3. Methodology](#3-methodology)
  - [3.1 Contrastive pre-training](#31-contrastive-pre-training)
  - [3.2 Embedding extraction](#32-embedding-extraction)
- [3.3 Downstream CV pipeline](#33-downstream-cv-pipeline)
- [4. In-CV results](#4-in-cv-results)
- [5. Before-CV results](#5-before-cv-results)

# 1. Task

A contrastive model (`997dc829`) is pre-trained to align arterial-phase MRI single-axes slices (ViT-B/32 → 128-dim) and RNA-seq expression (3-layer MLP → 128-dim) per patient using NT-Xent loss with outcome regularisation. The resulting 256-dim joint embeddings are then evaluated in a 3-fold CV on 2-year RFS prediction, comparing four feature strategies.

| Task | Feature matrix | Dim |
|------|---------------|-----|
| radiomics | Arterial radiomics, SelectKBest F-score k=100 | 4132 → 100 |
| embeddings | Contrastive img+gene embeddings (128+128), SelectKBest k=100 | 256 → 100 |
| concat | Single model of Embeddings + arterial radiomics, SelectKBest k=100 | 4388 → 100 |
| ensemble | Embeddings model + radiomics model, probabilities averaged | — |

Each task is run with SelectKBest placed **in-CV** and **before-CV**.  
Results: `results/multimodal_prediction/<task>_997dc829_rfs_2year_{selector}_k100_bc0d6e7/`

---

# 2. Key findings

- **Pre-training:** train loss 3.15 → 1.39, val loss 1.65 → 1.47.
- **In-CV:** embeddings RF (0.706 ± 0.093) outperforms radiomics RF baseline (0.569 ± 0.133).
- **Before-CV** Concatenating (RF,0.873 ± 0.083) and ensemble (LR, 0.859 ± 0.130) surpass the radiomics baseline.

---

# 3. Methodology

## 3.1 Contrastive pre-training
**Data**: `/Resection/Images/Radiomics/arterial`

The contrastive model learns a joint embedding space aligning arterial-phase MRI slices and RNA-seq expression per patient.

**Encoders**

| Encoder | Architecture | Output dim |
|---------|-------------|------------|
| Image | ViT-B/32 (ImageNet1K pretrained, unfrozen) → 2-layer MLP (768 → 128 → 128, ReLU) | 128 |
| Gene | 3-layer MLP (gene_dim → 256 → 128 → 128, ReLU) | 128 |

**augmentation**: Images are  during training with random horizontal and vertical flips before ViT normalisation.

**Gene set** (`GENE_SET`, 40 genes): union of 20 genes selected by DESeq2 before-CV for 2-year RFS and 20 genes from the predefined HCC gene set that passed in-CV. See `hcc_multimodal/contrastive/config.py`.

**Loss**

$$\mathcal{L} = \mathcal{L}_{\text{NT-Xent}} + \lambda \cdot \mathcal{L}_{\text{reg}}$$

- **NT-Xent** (τ=0.07): for a batch of N patients, L2-normalised image and gene embeddings are concatenated into a 2N×128 matrix. Cosine similarities are computed across all 2N×2N pairs; each embedding must identify its cross-modal pair against 2N−1 negatives using cross-entropy.

- **Outcome regularisation** (λ=0.1, `reg_mode=per_modality`): applied independently to each modality, pulls together same-outcome patients within the embedding space:

  $$\mathcal{L}_{\text{reg}} = \frac{\sum_{i \neq j} \mathbf{1}[y_i = y_j]\,(1 - \cos(z_i, z_j))}{\sum_{i \neq j} \mathbf{1}[y_i = y_j]}$$

  Summed over image and gene embeddings separately.

**Training hyperparameters (run `997dc829`)**

| Parameter | Value |
|-----------|-------|
| Backbone | ViT-B/32 (unfrozen) |
| Embed dim | 128 per modality |
| Gene hidden dim | 256 |
| Temperature τ | 0.07 |
| λ (reg weight) | 0.1 |
| Slices per patient | 10 (sagittal axis=0) |
| Image size | 224 × 224 |
| Epochs | 50 |
| Batch size | 32 |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) |
| LR schedule | Cosine annealing (T_max=50) |
| Val split | 10% stratified hold-out |
| Checkpoint | Best validation loss |
| Seed | 42 |

Training loss: 3.15 → 1.39 (train), 1.65 → 1.47 (val) over 50 epochs.

---

## 3.2 Embedding extraction

At inference the encoders are frozen. For each patient:
1. All 10 sagittal slices pass through the image encoder → 10 × 128 vectors, **mean-pooled** → 128-dim image embedding.
2. The gene expression vector passes through the gene encoder → 128-dim gene embedding.
3. The two are concatenated → **256-dim joint embedding** per patient.

54 patients have paired MRI, RNA-seq, and clinical outcome data.

---

## 3.3 Downstream CV pipeline

**3-fold stratified CV** (`StratifiedKFold`, `random_state=42`) across all tasks.

**Feature selection:** `SelectKBest(f_classif, k=100)`, placed either:
- **in-CV** (default): fitted on the training fold only.
- **before-CV** (`--selector_before_cv`): fitted on all 54 patients before splitting.

For the ensemble task, SelectKBest is applied independently to each modality's pipeline.

**Classifiers** (embeddings / concat / ensemble):

| Model | Configuration |
|-------|--------------|
| LR | `LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=1.0, C=1, max_iter=1000)` |
| RF | `RandomForestClassifier(n_estimators=100)` |

For the standalone **radiomics** task, a fixed grid is evaluated: LR (C ∈ {0.001, 0.01, 0.1, 1}) × RF (max_depth ∈ {2, 4}, min_samples_leaf ∈ {5, 10, 15}), and the best model is selected.

**Preprocessing:** `StandardScaler` applied to all continuous features inside the pipeline, after feature selection.

**Metric:** ROC-AUC, reported as mean ± std across 3 folds.

---

# 4. In-CV results

SelectKBest fitted on the training fold only. ROC-AUC mean ± std across 3 folds.

| Task | LR AUC ± std | RF AUC ± std |
|------|-------------|-------------|
| Radiomics | 0.496 ± 0.081 | 0.569 ± 0.133 |
| Embeddings | 0.673 ± 0.149 | **0.706 ± 0.093** |
| Concat | 0.615 ± 0.146 | 0.609 ± 0.170 |
| Ensemble | 0.594 ± 0.104 | 0.605 ± 0.111 |


---

# 5. Before-CV results

SelectKBest fitted on all 54 patients before splitting.

| Task | LR AUC ± std | RF AUC ± std |
|------|-------------|-------------|
| Radiomics | 0.752 ± 0.092 | 0.781 ± 0.069 |
| Embeddings | 0.627 ± 0.139 | 0.696 ± 0.067 |
| Concat | 0.834 ± 0.083 | **0.873 ± 0.083** |
| Ensemble | **0.859 ± 0.130** | 0.840 ± 0.085 |

