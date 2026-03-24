# Housing Price Regression with FFNN

Predicting housing prices using feedforward neural networks.

## Dataset

[Kaggle Housing Prices dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) with 545 samples and 12 features (1 numerical: `area`, 11 binary/categorical). Target variable is `price`.

## Project Structure

```
├── data/
│   ├── raw/                  # Original Housing.csv
│   └── processed/            # Train/val/test splits (scaled)
├── models/                   # Saved scalers, grid search results, final model
├── notebooks/
│   └── preprocessing.ipynb   # EDA, cleaning, encoding, scaling, splitting
└── src/
    ├── train.py              # Model definitions, grid search, training pipeline
    └── utils.py              # Data loading, evaluation, plotting
```

## Preprocessing

- Removed duplicate rows and area outliers using IQR 
- One-hot encoded multi-class categorical features
- Scaled `area` with `StandardScaler` (fit on train only)
- Scaled target `price` with a separate `StandardScaler` (fit on train only)
- Split: 72% train / 13% val / 15% test

## Architectures

Three architectures of FFNN are searched. Each hidden layer uses `Linear → BatchNorm → Activation → Dropout`.

| Name | Hidden layers |
|---|---|
| `two_layer` | [N1, N2] |
| `three_layer` | [N1, N2, N3] |
| `four_layer` | [N1, N2, N3, N4] |

## Hyperparameter Search

Grid search over all architecture + hyperparameter combinations:

| Hyperparameter | Values                         |
|---|--------------------------------|
| neurons per layer | [64, 128] [32, 64] [16, 32] [16] |
| activation | relu, tanh            |
| learning rate | 1e-3, 5e-4, 1e-4               |
| dropout | 0.1, 0.3                       |
| batch size | 32, 64                         |
| weight decay | 1e-3, 1e-4, 1e-5               |

Each config is trained with **early stopping** (patience=15) and **ReduceLROnPlateau** scheduling. Best config is selected by lowest validation RMSE.

## Final Model

The best configuration is retrained from scratch on train+val data and evaluated once on the held-out test set.

## Usage

```bash
# 1. Processed data and scalers are available in the repository.
#    To reproduce preprocessing from scratch:
jupyter notebook notebooks/preprocessing.ipynb

# 2. Run training
cd src
python train.py
```