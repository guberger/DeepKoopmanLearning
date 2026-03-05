import argparse
import numpy as np

from src.linear_sys import LinearSystem
from src.neural_obs import NeuralObserver
from src.koopman import data_koopman_eigen

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

# Create system
sys = LinearSystem(A, b)

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
    hidden_dims=(64, 64),
    activation="tanh",
    lr=1e-3,
    weight_decay=0.0,
    batch_size=128,
    epochs=100,
    device=None,
    dtype=np.float32,
    seed=1,
)

# Initialize observer
rng = np.random.default_rng(1)
N = 200
X = sys.sample(N)

# target: V[:, k] = cos(2 * pi * (k + 1) * X[:, 0] + phi) + noise
X_ang = X @ (np.array([range(1, output_dim + 1)]) * np.pi) + np.pi / 5
V = np.cos(X_ang) + 1 + 0.1 * rng.normal(size=(N, output_dim))
Q, _ = np.linalg.qr(V, mode="reduced")
V = Q * np.sqrt(N)

obs.fit(X, V)

# -------------------------
# Koopman iterations
# -------------------------
N = 500
max_iter = 100

X, V_trace = data_koopman_eigen(sys, obs, N, max_iter, trace=5)

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    # Smooth grid for plotting the learned function
    X_grid = np.linspace(X.min(), X.max(), 400)[:, None]
    V_grid = obs.eval(X_grid)
    V = obs.eval(X)

    fig, axes = plt.subplots(1, output_dim, figsize=(10, 4), sharex=True)

    for j in range(output_dim):

        ax = axes[j]

        # training samples
        ax.scatter(X[:, 0], V[:, j], s=30, alpha=0.5)

        for W in V_trace[:-1]:
            ax.scatter(X[:, 0], W[:, j], s=20, alpha=0.5, edgecolors="none")

        # learned Koopman modes
        ax.plot(X_grid[:, 0], V_grid[:, j], linestyle="--", linewidth=2)

        ax.set_title(f"Output {j}")
        ax.set_xlabel("x")
        ax.set_ylabel("value")
        ax.set_ylim([np.min(V), np.max(V)])

    plt.tight_layout()
    plt.show()