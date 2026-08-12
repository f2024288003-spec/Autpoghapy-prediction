# PSCV + PLM Pipeline — Colab Edition (13 classifiers, step-by-step)

Four notebooks. Run them in order, one cell at a time.

## Inputs you provide
- `positive.fasta` — protein sequences in your positive class
- `negative.fasta` — protein sequences in your negative class

**No other parameters. No test FASTA.** The 70/30 split, 5-fold CV, and 10-fold CV are derived automatically.

## What you edit
Only the two paths in **Step 1, Cell 3**:
```python
POSITIVE_FASTA = '/content/drive/MyDrive/pscv_data/positive.fasta'
NEGATIVE_FASTA = '/content/drive/MyDrive/pscv_data/negative.fasta'
WORK_DIR       = '/content/drive/MyDrive/pscv_work'
```

## Notebooks

| # | File | What it does |
|---|---|---|
| 1 | `01_setup_and_partition.ipynb` | Install, mount Drive, load FASTAs, 70/30 split + 5/10-fold CV indices |
| 2 | `02_feature_extraction.ipynb` | Extract PSCV (153-d) + ESM-2 (1280-d), cache to `features.npz` |
| 3 | `03_classification.ipynb` | **All 13 classifiers** × 4 protocols. Graceful skip if optional deps missing. Resumable. |
| 4 | `04_analysis.ipynb` | Combined ROC (filterable), UMAP boundary, SHAP with per-classifier dispatch |

## All 13 classifiers in Notebook 3

| # | Classifier | Family | Optional dep |
|---|---|---|---|
| 1 | MLP | Dense baseline | — |
| 2 | CNN | 1-D conv | — |
| 3 | LSTM | Recurrent | — |
| 4 | BiLSTM | Bidirectional recurrent | — |
| 5 | CNN-BiLSTM | Hybrid | — |
| 6 | Transformer | Attention | — |
| 7 | XGBoost | Boosted trees | `xgboost` |
| 8 | TabNet | Attentive tabular | `pytorch-tabnet` |
| 9 | FT-Transformer | Tabular transformer | `rtdl_revisiting_models` |
| 10 | TabResNet | Residual MLP | — |
| 11 | BlockMoE | Block-wise mixture-of-experts | — |
| 12 | WideDeep | Linear PSCV + MLP PLM | — |
| 13 | BayesianMLP | MC-Dropout uncertainty | — |

If `xgboost`, `pytorch-tabnet`, or `rtdl_revisiting_models` fails to install on your Colab, the corresponding classifier is **automatically skipped** with a warning — the other 12 still run.

Notebook 3 **checkpoints after every classifier**. Disconnects and re-runs pick up where they left off.

## Choosing classifiers featured in the ROC plot (Notebook 4)

In **Step 4, Cell 1**, set `ROC_MODELS`:
```python
ROC_MODELS = None                                     # all available (default)
ROC_MODELS = ['MLP', 'CNN', 'XGBoost']                # just these 3
ROC_MODELS = ['XGBoost', 'TabNet', 'FT-Transformer']  # only tabular
ROC_MODELS = ['BlockMoE', 'WideDeep', 'BayesianMLP']  # only block-aware
```

A separate `ANALYSIS_MODELS` controls which classifiers get decision-boundary + SHAP plots (those are slower; you may want fewer).

## SHAP dispatch in Notebook 4

| Classifier type | SHAP explainer used |
|---|---|
| `XGBoostClassifier` | `shap.TreeExplainer` (fast, exact) |
| `TabNetClassifier_` | Built-in attentive feature importance, broadcast to per-sample shape |
| All PyTorch models | `shap.GradientExplainer` |

Two views per model: **block-level** (PSCV vs ESM-2) and **PSCV per-feature** (top-30 in the interpretable 153-d block).

## Output structure
```
WORK_DIR/
├── train.csv                                # 70% training
├── independent_test.csv                     # 30% held-out
├── folds_5.json / folds_10.json             # stratified CV indices
├── features.npz                             # PSCV + ESM-2 features
├── results.pkl                              # all predictions & metrics
├── summary.csv                              # mean±std metrics table
├── clf_<NAME>_independent.pkl               # pickled classifier+combiner per model
└── plots/
    ├── roc/roc_<protocol>.png               # combined ROC, filtered by ROC_MODELS
    ├── boundary/boundary_<model>.png        # UMAP decision landscape
    └── shap/
        ├── shap_blocks_<model>.png          # PSCV-vs-PLM block importance
        └── shap_pscv_<model>.png            # top-30 PSCV features
```

## Validated
All 13 classifiers smoke-tested with realistic input dim (1433 = 153 + 1280):
all instantiate, all forward-pass to correct output shape, parameter counts
range from 220K (LSTM) to 1.55M (CNN). Graceful skip confirmed when
`rtdl_revisiting_models` is uninstalled.
