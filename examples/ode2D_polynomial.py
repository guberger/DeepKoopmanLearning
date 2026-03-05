from __future__ import annotations

import argparse
import numpy as np

from src.systems import ODEDiscretizedSystem
from src.observers import PolynomialObserver
from src.koopman import data_koopman_eigen

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# -------------------------
# System definition
# -------------------------
# Define system dimension
state_dim = 2

# Define dynamics
def f(X: np.ndarray) -> np.ndarray:
    X0 = X[:, 0]
    X1 = X[:, 1]
    dX0 = X1
    dX1 = 0.1 * X1 - X0 - 0.5 * X1**3
    return np.column_stack([dX0, dX1])

# Create system
sys = ODEDiscretizedSystem(
    f=f,
    state_dim=2,
    T=1.0,     # discrete map horizon
    dt=0.01,   # RK4 integration step
)

# -------------------------
# Observer definition
# -------------------------
# Define observer dimensions
input_dim = state_dim
output_dim = 3

# Create observer
obs = PolynomialObserver(input_dim, output_dim, degree=3, alpha=1e-4)

# Initialize observer
rng = np.random.default_rng(1)
N = 200
X = sys.sample(N)

# target: V[:, k] =
#     cos(alpha * k * X[:, 0] + phi0)
#     + sin(alpha * k * X[:, 1] + phi1)
#     + noise
X0_ang = X[:, [0]] @ (np.array([range(output_dim)]) * 1.5) + 1
X1_ang = X[:, [1]] @ (np.array([range(output_dim)]) * 1.5) - 1
V = np.cos(X0_ang) + np.sin(X1_ang) + 0.1 * rng.normal(size=(N, output_dim))
Q, _ = np.linalg.qr(V, mode="reduced")
V = Q * np.sqrt(N)

obs.fit(X, V)

# -------------------------
# Koopman iterations
# -------------------------
N = 500
max_iter = 50

X, V_rec = data_koopman_eigen(sys, obs, N, max_iter, rec=5)

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    V = obs.eval(X)

    # Smooth grid for plotting the learned function
    n_grid = 100
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    V_grid = obs.eval(X_grid)

    fig, axes = plt.subplots(
        1, output_dim,
        figsize=(12, 5),
        subplot_kw={"projection": "3d"},
    )

    if output_dim == 1:
        axes = [axes]

    vmin = np.min(V)
    vmax = np.max(V)

    for j in range(output_dim):

        ax = axes[j]

        # reshape grid evaluation
        Z = V_grid[:, j].reshape(n_grid, n_grid)

        # contour plot of learned function
        ax.plot_surface(X1, X2, Z, linewidth=0, antialiased=True, alpha=0.85)

        # training samples
        ax.scatter(
            X[:, 0],
            X[:, 1],
            V[:, j],
            s=20,
            alpha=0.5,
            edgecolors="none",
        )

        for W in V_rec[:-1]:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                W[:, j],
                s=10,
                alpha=0.2,
                edgecolors="none",
            )

        ax.set_title(f"Output {j}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("value")
        ax.set_zlim([vmin, vmax])

    plt.tight_layout()
    plt.show()