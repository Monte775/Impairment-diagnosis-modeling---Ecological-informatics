# river-impairment-mlp

MLP-based binary classification model for diagnosing river ecosystem impairment using biological and physicochemical indicators.

This repository provides the source code to reproduce the results presented in the paper:

> Explainable Machine Learning for Diagnosing Ecological Impairment in Streams Using Reach-Scale Biological and Environmental Data(On the Review)
> Ecological Informatics, 2026

## Overview

The model classifies monitoring sites as *impaired* or *non-impaired* based on three biological assessment indices:

| Target | Description |
|--------|-------------|
| **TDI** | Trophic Diatom Index |
| **BMI** | Benthic Macroinvertebrate Index |
| **FAI** | Fish Assessment Index |

Key features:

- **Narrowing MLP architecture** with configurable depth, width shrinkage ratio, and activation functions.
- **Hyperparameter optimisation** via TPE (Tree-structured Parzen Estimator) with stratified K-fold cross-validation.
- **SHAP explainability** through DeepExplainer (summary and waterfall plots).

## Project Structure

```
river-impairment-mlp/
├── configs/
│   └── default.yaml              # All hyperparameters and paths
├── river_impairment/
│   ├── __init__.py
│   ├── model.py                  # MLPImpairment model definition
│   ├── data.py                   # Data loading, preprocessing, splitting
│   ├── trainer.py                # TPE optimisation & training loop
│   ├── metrics.py                # Classification metrics
│   └── explainer.py              # SHAP analysis utilities
├── train.py                      # CLI: train models
├── evaluate.py                   # CLI: evaluate saved models
├── explain.py                    # CLI: generate SHAP plots
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Data will be made available on request.

## Usage

### 1. Training

Run hyperparameter optimisation and train the best MLP model for each target:

```bash
python train.py --config configs/default.yaml
```

Results are saved to `results/`:
- `results/models/{TARGET}_best.pkl` — best model weights, scaler, and metrics.
- `results/{TARGET}_trials.csv` — all trial records from TPE optimisation.
- `results/performance.csv` — aggregated train/test performance.

### 2. Evaluation

Re-evaluate saved models on the test set:

```bash
python evaluate.py --config configs/default.yaml
```

### 3. SHAP Explanation

Generate SHAP summary and waterfall plots:

```bash
# All targets
python explain.py --config configs/default.yaml

# Single target
python explain.py --config configs/default.yaml --target BMI_훼손
```

Plots are saved under `results/shap/`.

## Configuration

All parameters are managed through `configs/default.yaml`:

| Section | Key parameters |
|---------|---------------|
| `data` | `data_path`, `encoding`, `site_river_path` |
| `variables` | `predictors`, `predictor_labels`, `targets` |
| `split` | `train_years`, `test_years` |
| `training` | `epochs`, `n_folds`, `max_evals` |
| `output` | `results_dir`, `model_dir`, `shap_dir` |

## Model Architecture

```
Input (13 features)
  → Linear(13, hidden_dim) → Dropout(0.2)
  → Linear(hidden_dim, hidden_dim × ratio) → Activation
  → Linear(…, … × ratio) → Activation
  → ...
  → Linear(last_dim, 2) → Softmax
```

The hidden-layer width shrinks by `ratio` at each layer, creating a narrowing (funnel) architecture. The activation function and all architectural hyperparameters are optimised via TPE.
## License

[Choose a license, e.g., MIT, Apache 2.0]
