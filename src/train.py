import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from itertools import product

from utils import load_data, make_loader, evaluate, plot_losses

DATA_DIR = "../data/processed"
MODELS_DIR = "../models"

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)


def build_two_layer(input_dim, hyperparams):
    return _build_ffnn(input_dim, [
        hyperparams["neurons_1"],
        hyperparams["neurons_2"],
    ], hyperparams)


def build_three_layer(input_dim, hyperparams):
    return _build_ffnn(input_dim, [
        hyperparams["neurons_1"],
        hyperparams["neurons_2"],
        hyperparams["neurons_3"],
    ], hyperparams)


def build_four_layer(input_dim, hyperparams):
    return _build_ffnn(input_dim, [
        hyperparams["neurons_1"],
        hyperparams["neurons_2"],
        hyperparams["neurons_3"],
        hyperparams["neurons_4"],
    ], hyperparams)


ARCHITECTURES = {
    "two_layer": build_two_layer,
    "three_layer": build_three_layer,
    "four_layer": build_four_layer,
}

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def _build_ffnn(input_dim, hidden_layers, hyperparams):
    act_fn = ACTIVATION_MAP[hyperparams["activation"]]
    layers = []
    in_dim = input_dim
    for out_dim in hidden_layers:
        layers += [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            act_fn(),
            nn.Dropout(hyperparams["dropout"]),
        ]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))
    net = nn.Sequential(*layers)

    class FFNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x).squeeze(1)

    return FFNN()


def get_hyperparam_grid(architecture_name):
    common = {
        "activation": ["relu", "tanh"],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "dropout": [0.1, 0.3],
        "batch_size": [32, 64],
        "weight_decay": [1e-3, 1e-4, 1e-5],
    }

    if architecture_name == "two_layer":
        grid = {
            **common,
            "neurons_1": [64, 128],
            "neurons_2": [32, 64],
        }
    elif architecture_name == "three_layer":
        grid = {
            **common,
            "neurons_1": [64, 128],
            "neurons_2": [32, 64],
            "neurons_3": [16, 32],
        }
    elif architecture_name == "four_layer":
        grid = {
            **common,
            "neurons_1": [64, 128],
            "neurons_2": [32, 64],
            "neurons_3": [16, 32],
            "neurons_4": [16],
        }
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    return [dict(zip(keys, combo)) for combo in combos]


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        model.load_state_dict(self.best_state)


def train_model(model, train_loader, val_loader, hyperparams, max_epochs=60, patience=15):
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    stopper = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        model.eval()
        with torch.no_grad():
            batch_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                batch_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if stopper.step(val_loss, model):
            print(f"    Early stop at epoch {epoch}  |  best val MSE: {stopper.best_loss:.4f}")
            break

    stopper.restore(model)
    return train_losses, val_losses


def run_grid_search(X_train, y_train, X_val, y_val, device, input_dim):
    results = []
    best_val_rmse = float("inf")
    best_config = None
    best_train_hist = None
    best_val_hist = None

    all_jobs = [
        (arch_name, hyperparams)
        for arch_name in ARCHITECTURES
        for hyperparams in get_hyperparam_grid(arch_name)
    ]

    total = len(all_jobs)
    print(f"Total configs to search: {total}\n")

    for run, (arch_name, hyperparams) in enumerate(all_jobs, 1):
        print(f"[{run}/{total}] {arch_name} | {hyperparams}")

        train_loader = make_loader(X_train, y_train, hyperparams["batch_size"], shuffle=True)
        val_loader = make_loader(X_val, y_val, hyperparams["batch_size"], shuffle=False)

        build_fn = ARCHITECTURES[arch_name]
        model = build_fn(input_dim, hyperparams).to(device)

        train_hist, val_hist = train_model(model, train_loader, val_loader, hyperparams)

        val_rmse, val_r2 = evaluate(model, X_val, y_val)
        print(f"    Val RMSE: {val_rmse:.4f}  |  Val R2: {val_r2:.4f}\n")

        results.append({
            "architecture": arch_name,
            **hyperparams,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_config = {**results[-1], "hyperparams": hyperparams, "arch_name": arch_name}
            best_train_hist = train_hist
            best_val_hist = val_hist

    return results, best_config, best_train_hist, best_val_hist


def train_final_model(best_config, X_train, y_train, X_val, y_val, input_dim, device):
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    # 10% pseudo-val from train+val only to drive early stopping
    split = int(0.9 * len(X_trainval))
    X_tv_tr, X_tv_val = X_trainval[:split], X_trainval[split:]
    y_tv_tr, y_tv_val = y_trainval[:split], y_trainval[split:]

    hyperparams = best_config["hyperparams"]

    tv_train_loader = make_loader(X_tv_tr, y_tv_tr, hyperparams["batch_size"], shuffle=True)
    tv_val_loader = make_loader(X_tv_val, y_tv_val, hyperparams["batch_size"], shuffle=False)

    build_fn = ARCHITECTURES[best_config["arch_name"]]
    final_model = build_fn(input_dim, hyperparams).to(device)

    train_hist, val_hist = train_model(final_model, tv_train_loader, tv_val_loader, hyperparams)
    return final_model, train_hist, val_hist


if __name__ == "__main__":
    device = torch.device("cpu")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATA_DIR)
    input_dim = X_train.shape[1]
    target_scaler = joblib.load(f"{MODELS_DIR}/target_scaler.pkl")

    results, best_config, best_train_hist, best_val_hist = run_grid_search(
        X_train, y_train, X_val, y_val, device, input_dim
    )

    results_df = pd.DataFrame(results).sort_values("val_rmse")
    print("\nTop 5 configurations (by Val RMSE):")
    print(results_df.head(5).to_string(index=False))
    results_df.to_csv(f"{MODELS_DIR}/grid_search_results.csv", index=False)

    print(f"\nBest config:")
    print(f"Architecture : {best_config['arch_name']}")
    print(f"Hyperparams  : {best_config['hyperparams']}")
    print(f"Val RMSE     : {best_config['val_rmse']:.4f}")
    print(f"Val R2       : {best_config['val_r2']:.4f}")

    plot_losses(
        best_train_hist, best_val_hist,
        title=f"Best candidate - {best_config['arch_name']}",
        save_path=f"{MODELS_DIR}/best_candidate_loss.png",
    )

    print("\nRetraining best architecture on train + val...")
    final_model, final_train_hist, final_val_hist = train_final_model(
        best_config, X_train, y_train, X_val, y_val, input_dim, device
    )

    plot_losses(
        final_train_hist, final_val_hist,
        title="Final model (train+val) - loss curve",
        save_path=f"{MODELS_DIR}/final_model_loss.png",
    )

    test_rmse, test_r2 = evaluate(final_model, X_test, y_test, target_scaler=target_scaler)
    print(f"\nTest set results:")
    print(f"RMSE: {test_rmse:,.2f}")
    print(f"R2: {test_r2:.4f}")

    torch.save(final_model.state_dict(), f"{MODELS_DIR}/final_model.pth")
