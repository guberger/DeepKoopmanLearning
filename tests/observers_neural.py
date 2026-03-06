from __future__ import annotations

import argparse
import numpy as np

from src.observers import NeuralObserver

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

print("Start tests neural_obs:")

rng = np.random.default_rng(0)

# -------------------------
# Example 1 (input = 3, output = 2)
# -------------------------
N = 200
input_dim = 3
output_dim = 2

X = rng.normal(size=(N, input_dim))

# target: ``V = [1, 1] x0^2 + [0.5, -0.5] * x1 * x2 + [noise, noise]``
phi = np.column_stack([
    X[:, 0] ** 2,
    X[:, 1] * X[:, 2],
])
W = np.array([
    [1.0, 1.0],
    [0.5, -0.5],
])
V = phi @ W + 0.1 * rng.normal(size=(N, output_dim))

obs = NeuralObserver(
    input_dim,
    output_dim,
    hidden_dims=(64, 64),
    activation="tanh",
    lr=1e-3,
    weight_decay=0.0,
    batch_size=128,
    epochs=400,
    device=None,  # auto: "cuda" if available else "cpu"
    dtype=np.float32,
)
obs.fit(X, V)
V_hat = obs.eval(X)

rmse = np.sqrt(np.mean((V_hat - V) ** 2, axis=0))
print("Example 1 RMSE per component:", rmse)
assert np.all(rmse < 0.2)

# -------------------------
# Example 2 (input = 1, output = 2) + plotting
# -------------------------
N = 200
input_dim = 1
output_dim = 2

X = rng.normal(size=(N, input_dim))

# target: ``V = [1, 1] x^2 + [0.5, -0.5] * x + [noise, noise]``
phi = np.column_stack([
    X[:, 0] ** 2,
    X[:, 0],
])
W = np.array([
    [1.0, 1.0],
    [0.5, -0.5],
])
V = phi @ W + 0.1 * rng.normal(size=(N, output_dim))

obs = NeuralObserver(
    input_dim,
    output_dim,
    hidden_dims=(64, 64),
    activation="tanh",
    lr=1e-3,
    weight_decay=0.0,
    batch_size=128,
    epochs=400,
    device=None,
    dtype=np.float32,
)
obs.fit(X, V)
V_hat = obs.eval(X)

rmse = np.sqrt(np.mean((V_hat - V) ** 2, axis=0))
print("Example 2 RMSE per component:", rmse)
assert np.all(rmse < 0.2)

print("Tests ok!")

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    # Smooth grid for plotting the learned function
    X_grid = np.linspace(X.min(), X.max(), 400)[:, None]
    V_pred = obs.eval(X_grid)

    fig, axes = plt.subplots(1, output_dim, figsize=(10, 4), sharex=True)

    if output_dim == 1:
        axes = [axes]

    for j in range(output_dim):
        ax = axes[j]

        # training samples
        ax.scatter(X[:, 0], V[:, j], s=20, alpha=0.5, label="data")

        # learned regressor
        ax.plot(X_grid[:, 0], V_pred[:, j], linestyle="--", linewidth=2, label="NN")

        ax.set_title(f"Output {j}")
        ax.set_xlabel("x")
        ax.set_ylabel("value")
        ax.legend()

    plt.tight_layout()
    plt.show()