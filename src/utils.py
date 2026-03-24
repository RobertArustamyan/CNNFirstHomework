import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error, r2_score


def load_data(data_dir):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv").values.astype(np.float32)
    X_val = pd.read_csv(f"{data_dir}/X_val.csv").values.astype(np.float32)
    X_test = pd.read_csv(f"{data_dir}/X_test.csv").values.astype(np.float32)
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.astype(np.float32).ravel()
    y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.astype(np.float32).ravel()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.astype(np.float32).ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_loader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def evaluate(model, X, y, target_scaler=None):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X)).numpy()
    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        y = target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    return rmse, r2


def plot_losses(train_losses, val_losses, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved loss plot -> {save_path}")