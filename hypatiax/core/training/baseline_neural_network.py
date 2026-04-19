import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_and_evaluate(X, y, description, domain, metadata=None, epochs=200):
    """
    Train and evaluate a simple neural network with proper data scaling.
    """
    # CRITICAL: Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = SimpleNN(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)

    # Train
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        y_pred_scaled = model(X_test_t).numpy().flatten()

        # Transform predictions back to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Calculate metrics on original scale
        mse = np.mean((y_test_original - y_pred) ** 2)
        mae = np.mean(np.abs(y_test_original - y_pred))
        rmse = np.sqrt(mse)

        # R² score
        ss_res = np.sum((y_test_original - y_pred) ** 2)
        ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    assert model is not None

    return {
        "method": "neural_network",
        "description": description,
        "domain": domain,
        # 🔑 REQUIRED for extrapolation
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "evaluation": {  # Wrap metrics in "evaluation" to match Pure LLM format
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
            "success": True,
        },
        "timestamp": datetime.now().isoformat(),
    }


def nn_predict(model, scaler_X, scaler_y, X_new):
    """
    Predict using trained NN with proper scaling.
    """
    model.eval()

    X_scaled = scaler_X.transform(X_new)
    X_t = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        y_scaled = model(X_t).numpy().reshape(-1, 1)

    return scaler_y.inverse_transform(y_scaled).flatten()


def run_nn_baseline(domains=None, save_dir="hypatiax/data/results"):
    """
    Run neural network baseline across domains.
    """
    from hypatiax.protocols.experiment_protocol import ExperimentProtocol

    if domains is None:
        domains = ["materials", "fluids"]

    print("=" * 70)
    print("Neural Network Baseline Evaluation".center(70))
    print("=" * 70)
    print(f"\nTesting {len(domains)} domains\n")

    all_results = []

    for domain in domains:
        print(f"\n{'=' * 70}")
        print(f"Domain: {domain.upper()}".center(70))
        print(f"{ExperimentProtocol.get_domain_description(domain)}".center(70))
        print("=" * 70)

        test_cases = ExperimentProtocol.load_test_data(domain, num_samples=100)

        for i, (description, X, y, var_names, metadata) in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {description}")

            result = train_and_evaluate(X, y, description, domain, metadata, epochs=200)

            # Add extrapolation test
            model = result["model"]
            scaler_X = result["scaler_X"]
            scaler_y = result["scaler_y"]

            # Create extrapolation point
            X_extrap = X.max(axis=0).reshape(1, -1) * 1.2

            assert callable(nn_predict)

            y_pred = nn_predict(model, scaler_X, scaler_y, X_extrap)

            # Validate prediction
            assert y_pred is not None, "Prediction should not be None"
            assert len(y_pred) == 1, f"Expected 1 prediction, got {len(y_pred)}"
            assert not np.isnan(y_pred[0]), "Prediction should not be NaN"
            assert not np.isinf(y_pred[0]), "Prediction should not be Inf"

            print(f"  🔮 Extrapolation prediction: {y_pred[0]:.6f}")

            # Fix: Access metrics from the 'evaluation' dictionary
            print(f"  ✓ R² Score: {result['evaluation']['r2']:.4f}")
            print(f"  ✓ RMSE: {result['evaluation']['rmse']:.6f}")

            all_results.append(result)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{save_dir}/baseline_neural_network_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    from hypatiax.protocols.experiment_protocol import ExperimentProtocol

    report = ExperimentProtocol.generate_experiment_report(all_results)
    report_file = f"{save_dir}/nn_experiment_report_{timestamp}.json"

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY".center(70))
    print("=" * 70)

    print(f"\n📊 Overall Results:")
    print(f"   Total test cases: {report['overall']['total_cases']}")
    print(
        f"   Successfully evaluated: {report['overall']['successful']}/{report['overall']['total_cases']}"
    )

    if report["overall"].get("mean_r2"):
        print(f"\n📈 R² Score Statistics:")
        print(f"   Mean:   {report['overall']['mean_r2']:.4f}")
        print(f"   Median: {report['overall']['median_r2']:.4f}")

    print(f"\n🎯 Performance by Domain:")
    for domain, stats in report["by_domain"].items():
        r2_str = f"R²={stats['mean_r2']:.3f}" if stats["mean_r2"] else "N/A"
        print(
            f"   {domain:12s}: {stats['successful']}/{stats['total']} ({100 * stats['success_rate']:5.1f}%)  {r2_str}"
        )

    print(f"\n💾 Results saved to:")
    print(f"   {results_file}")
    print(f"   {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            run_nn_baseline(
                domains=[
                    "materials",
                    "fluids",
                    "thermodynamics",
                    "mechanics",
                    "chemistry",
                ]
            )
        elif sys.argv[1] == "--domain":
            domains = sys.argv[2].split(",")
            run_nn_baseline(domains=domains)
        elif sys.argv[1] == "--quick":
            run_nn_baseline(domains=["materials", "fluids"])
        else:
            print("Usage:")
            print("  python baseline_neural_network.py --all")
            print("  python baseline_neural_network.py --domain materials,fluids")
            print("  python baseline_neural_network.py --quick")
    else:
        run_nn_baseline(domains=["materials", "fluids"])
