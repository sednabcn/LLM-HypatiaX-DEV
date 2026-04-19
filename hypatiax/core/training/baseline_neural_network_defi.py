# hypatiax/core/generation/baseline_neural_network_defi.py
"""
Neural network baseline for DeFi and Risk Management (updated)
- Adds train_get_model() returning (model, metrics, scaler_X, scaler_y)
- Adds get_predictions() to provide predictions for arbitrary X arrays
- Keeps train_and_evaluate() for backward compatibility and reporting
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the experiment protocol (adjust import path if needed in your repo)
from hypatiax.protocols.experiment_protocol_defi import DeFiExperimentProtocol


class SimpleNN(nn.Module):
    """Simple 3-layer MLP for regression."""

    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetworkBaseline:
    """Neural network baseline for formula discovery in DeFi and Risk Management."""

    def __init__(
        self,
        hidden_dims=[64, 32],
        learning_rate=0.001,
        epochs=200,
        device: Optional[str] = None,
    ):
        """
        Initialize Neural Network baseline.

        Args:
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for Adam optimizer
            epochs: Number of training epochs
            device: 'cpu' or 'cuda' (auto-detected if None)
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.results = []
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _fit_model(
        self,
        X_train_s: np.ndarray,
        y_train_s: np.ndarray,
        input_dim: int,
        verbose: bool = False,
    ) -> nn.Module:
        """Create and fit PyTorch model on scaled data; returns trained model."""
        model = SimpleNN(input_dim, hidden_dims=self.hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_train_t = torch.FloatTensor(X_train_s).to(self.device)
        y_train_t = torch.FloatTensor(y_train_s.reshape(-1, 1)).to(self.device)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

            # optional verbose logging
            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                print(f"    Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.6f}")

        return model

    def train_get_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Optional[Dict] = None,
        is_extrapolation: bool = False,
        epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[nn.Module, Dict, StandardScaler, StandardScaler]:
        """
        Train NN and return (model, metrics, scaler_X, scaler_y).
        This method is intended for integration with hybrid system which expects this shape.
        """
        epochs = epochs or self.epochs

        # Split
        test_size = 0.3 if is_extrapolation else 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Fit
        model = self._fit_model(X_train_s, y_train_s, X.shape[1], verbose=verbose)

        # Eval
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_s).to(self.device)
            y_pred_s = model(X_test_t).cpu().numpy().flatten()
            try:
                y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
            except Exception:
                # fallback if scaler_y wasn't fit
                y_pred = y_pred_s

            mse = float(np.mean((y_test - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_test - y_pred)))
            rmse = float(np.sqrt(mse))
            ss_res = float(np.sum((y_test - y_pred) ** 2))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "success": r2 > 0.0,
            }

        return model, metrics, scaler_X, scaler_y

    def get_predictions(
        self,
        model: nn.Module,
        scaler_X: StandardScaler,
        scaler_y: StandardScaler,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return predictions for full dataset X using trained model and scalers.
        This is useful for ensembles in the hybrid system.
        """
        model.eval()
        X_s = scaler_X.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        with torch.no_grad():
            y_pred_s = model(X_t).cpu().numpy().flatten()
        try:
            y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        except Exception:
            y_pred = y_pred_s
        return y_pred

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        description: str,
        metadata: Optional[Dict] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train and evaluate NN and return a comprehensive result dict (for reporting).
        Keeps backward compatibility with previous API.
        """
        model, metrics, scaler_X, scaler_y = self.train_get_model(
            X,
            y,
            metadata=metadata,
            is_extrapolation=bool(metadata and metadata.get("extrapolation_test")),
            epochs=self.epochs,
            verbose=verbose,
        )

        # compute extrapolation note if requested
        extrap_results = None
        if metadata and metadata.get("extrapolation_test", False):
            X_max = X.max(axis=0)
            X_min = X.min(axis=0)
            X_range = X_max - X_min
            X_extrap = X + X_range  # simple shift
            y_extrap_pred = self.get_predictions(model, scaler_X, scaler_y, X_extrap)
            extrap_results = {
                "mean_prediction": float(np.mean(y_extrap_pred)),
                "std_prediction": float(np.std(y_extrap_pred)),
            }

        result = {
            "method": "neural_network",
            "architecture": f"{X.shape[1]}-{'-'.join(map(str, self.hidden_dims))}-1",
            "description": description,
            "metadata": metadata,
            "epochs": self.epochs,
            "metrics": metrics,
            "evaluation": metrics,  # legacy compat
            "extrapolation": extrap_results,
            "timestamp": datetime.now().isoformat(),
        }

        # store for later inspection
        self.results.append(result)
        return result

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to: {filepath}")


if __name__ == "__main__":
    # quick manual test
    protocol = DeFiExperimentProtocol()
    desc, X, y, var_names, meta = protocol.load_test_data("liquidity", num_samples=100)[
        0
    ]
    nn = NeuralNetworkBaseline(epochs=50)
    result = nn.train_and_evaluate(X, y, description=desc, metadata=meta, verbose=True)
    print(result["metrics"])
    # Save results
    os.makedirs("hypatiax/data/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hypatiax/data/results/baseline_nn_defi_{timestamp}.json"
    report_file = f"hypatiax/data/results/report_nn_defi_{timestamp}.json"

    nn.save_results(results_file)
