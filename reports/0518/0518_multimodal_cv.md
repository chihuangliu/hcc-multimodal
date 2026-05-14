
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

Two contrastive models are pre-trained to align arterial-phase MRI single-axes slices (ViT-B/32 → 128-dim) and RNA-seq expression (3-layer MLP → 128-dim) per patient using NT-Xent loss, varying the outcome regularisation weight λ: `997dc829` (λ=0.1) and `9d48c320` (λ=0). The resulting 256-dim joint embeddings are then evaluated in a 3-fold CV on 2-year RFS prediction, comparing four feature strategies.

| Task | Feature matrix | Dim |
|------|---------------|-----|
| radiomics | Arterial radiomics, SelectKBest F-score k=100 | 4132 → 100 |
| embeddings | Contrastive img+gene embeddings (128+128), SelectKBest k=100 | 256 → 100 |
| concat | Single model of Embeddings + arterial radiomics, SelectKBest k=100 | 4388 → 100 |
| ensemble | Embeddings model + radiomics model, probabilities averaged | — |

Each task is run with SelectKBest placed **in-CV** and **before-CV**.  
Results:
- λ=0.1: `results/multimodal_prediction/<task>_997dc829_rfs_2year_{selector}_k100_bc0d6e7/`
- λ=0: `results/multimodal_prediction/<task>_9d48c320_rfs_2year_{selector}_k100_eb63546/`

---

# 2. Key findings

Best AUC across all tasks and models:

| | In-CV | Before-CV |
|---|---|---|
| Radiomics baseline | RF: 0.569 ± 0.133 | RF: 0.781 ± 0.069 |
| λ=0.1 | Embeddings, RF: 0.706 ± 0.093 | Concat, RF: 0.873 ± 0.083 |
| λ=0   | Embeddings, RF: 0.752 ± 0.111 | Concat, RF: 0.867 ± 0.076 |

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

**Training hyperparameters** (all other parameters shared across runs)

| Parameter | `997dc829` (λ=0.1) | `9d48c320` (λ=0) |
|-----------|-------------------|-----------------|
| Backbone | ViT-B/32 (unfrozen) | ViT-B/32 (unfrozen) |
| Embed dim | 128 per modality | 128 per modality |
| Gene hidden dim | 256 | 256 |
| Temperature τ | 0.07 | 0.07 |
| λ (reg weight) | 0.1 | 0.0 |
| Slices per patient | 10 (sagittal axis=0) | 10 (sagittal axis=0) |
| Image size | 224 × 224 | 224 × 224 |
| Epochs | 50 | 50 |
| Batch size | 32 | 32 |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) |
| LR schedule | Cosine annealing (T_max=50) | Cosine annealing (T_max=50) |
| Val split | 10% stratified hold-out | 10% stratified hold-out |
| Checkpoint | Best validation loss | Best validation loss |
| Seed | 42 | 42 |
| Train loss (final) | 1.39 | 2.90 |
| Val loss (best) | 1.47 | 4.00 (epoch 13) |

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

| Task | Model | λ=0.1 AUC ± std | λ=0 AUC ± std |
|------|-------|----------------|--------------|
| Radiomics | LR | 0.496 ± 0.081 | 0.496 ± 0.081 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | 0.673 ± 0.149 | 0.586 ± 0.144 |
| Embeddings | RF | **0.706 ± 0.093** | **0.752 ± 0.111** |
| Concat | LR | 0.615 ± 0.146 | 0.508 ± 0.079 |
| Concat | RF | 0.609 ± 0.170 | 0.545 ± 0.133 |
| Ensemble | LR | 0.594 ± 0.104 | 0.599 ± 0.109 |
| Ensemble | RF | 0.605 ± 0.111 | 0.601 ± 0.108 |


---

# 5. Before-CV results

SelectKBest fitted on all 54 patients before splitting.

| Task | Model | λ=0.1 AUC ± std | λ=0 AUC ± std |
|------|-------|----------------|--------------|
| Radiomics | LR | 0.752 ± 0.092 | 0.752 ± 0.092 |
| Radiomics | RF | 0.781 ± 0.069 | 0.781 ± 0.069 |
| Embeddings | LR | 0.627 ± 0.139 | 0.652 ± 0.139 |
| Embeddings | RF | 0.696 ± 0.067 | 0.690 ± 0.028 |
| Concat | LR | 0.834 ± 0.083 | 0.797 ± 0.107 |
| Concat | RF | **0.873 ± 0.083** | **0.867 ± 0.076** |
| Ensemble | LR | **0.859 ± 0.130** | 0.751 ± 0.163 |
| Ensemble | RF | 0.840 ± 0.085 | 0.845 ± 0.081 |

