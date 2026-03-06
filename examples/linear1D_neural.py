from __future__ import annotations

import argparse
import numpy as np

from src.systems import DiscreteMapSystem
from src.observers import NeuralObserver
from src.koopman import koopman_modes, koopman_operator

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# -------------------------
# System definition
# -------------------------
# Define system dimension
state_dim = 1

# Define linear dynamics
A = np.array([[0.9]])
b = np.array([0.1])

def f(X: np.ndarray) -> np.ndarray:
    return X @ A.T + b

# Create system
sys = DiscreteMapSystem(f, state_dim, seed=1234)

# -------------------------
# Observer definition
# -------------------------
# Define observer dimensions
input_dim = state_dim
output_dim = 3

# Create observer
obs = NeuralObserver(
    input_dim,
    output_dim,
    hidden_dims=(8, 8),
    activation="tanh",
    lr=1e-3,
    epochs=800,
    dtype=np.float32,
)

# Initialize observer
rng = np.random.default_rng(1)
N = 200
X = sys.sample(N)

# target: ``V[:, k] = cos(alpha * k * X[:, 0] + phi) + noise``
X_ang = X @ (np.array([range(output_dim)]) * 1.5) + 1
V = np.cos(X_ang) + 0.1 * rng.normal(size=(N, output_dim))
Q, _ = np.linalg.qr(V, mode="reduced")
V = Q * np.sqrt(N)

obs.fit(X, V)

# -------------------------
# Koopman iterations
# -------------------------
N = 500
max_iter = 50

X = koopman_modes(sys, obs, N, max_iter)

Kop, Vop, Vop_next = koopman_operator(sys, obs, N)
print(Kop)
print((Vop.T @ Vop) / N)
print(np.linalg.norm(Vop_next - Vop @ Kop, axis=0) / np.sqrt(N))

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    V = obs.eval(X)

    # Smooth grid for plotting the learned function
    X_grid = np.linspace(X.min(), X.max(), 400)[:, None]
    V_grid = obs.eval(X_grid)

    fig, axes = plt.subplots(1, output_dim, figsize=(10, 4), sharex=True)

    if output_dim == 1:
        axes = [axes]

    for j in range(output_dim):

        ax = axes[j]

        # training samples
        ax.scatter(X[:, 0], V[:, j], s=30, alpha=0.5)

        # learned Koopman modes
        ax.plot(X_grid[:, 0], V_grid[:, j], linestyle="--", linewidth=2)

        ax.set_title(f"Output {j}")
        ax.set_xlabel("x")
        ax.set_ylabel("value")
        ax.set_ylim([np.min(V), np.max(V)])

    plt.tight_layout()
    plt.show()