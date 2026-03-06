import argparse
import numpy as np

from src.observers import PolynomialObserver, NeuralObserver

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

print("Start tests observers:")

rng = np.random.default_rng(0)

N = 200
input_dim = 2
output_dim = 2

X = rng.normal(size=(N, input_dim))

# target: ``V = [1, 1] x0^2 + [0, -1] x1^2 + [0.5, -0.5] x1 + [noise, noise]``
phi = np.column_stack([
    X[:, 0] ** 2,
    X[:, 1] ** 2,
    X[:, 1],
])
W = np.array([
    [1.0, 1.0],
    [0.0, -1.0],
    [0.5, -0.5],
])
V = phi @ W + 0.1 * rng.normal(size=(N, 2))

# Create observers
obs0 = PolynomialObserver(input_dim, output_dim, degree=2, alpha=1e-4)
obs1 = NeuralObserver(
    input_dim,
    output_dim,
    hidden_dims=(64, 64),
    activation="tanh",
    lr=1e-3,
    epochs=800,
    dtype="float64",
)

# Fit observers
obs0.fit(X, V)
obs1.fit(X, V)

V0_hat = obs0.eval(X)
rmse = np.sqrt(np.mean((V0_hat - V) ** 2, axis=0))
print("Obs 0 RMSE per component:", rmse)

V1_hat = obs1.eval(X)
rmse = np.sqrt(np.mean((V1_hat - V) ** 2, axis=0))
print("Obs 1 RMSE per component:", rmse)

print("Tests ok!")

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    n_grid = 100
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    V0_grid = obs0.eval(X_grid)
    V1_grid = obs1.eval(X_grid)
    Vs_grid = [V0_grid, V1_grid]

    fig, axes = plt.subplots(
        2, output_dim,
        figsize=(12, 7),
        subplot_kw={"projection": "3d"},
    )

    if output_dim == 1:
        axes = axes[:, None]

    vmin = np.min(V)
    vmax = np.max(V)

    for i in range(2):
        for j in range(output_dim):

            ax = axes[i, j]

            # reshape grid evaluation
            Z = Vs_grid[i][:, j].reshape(n_grid, n_grid)

            # surface plot of learned function
            ax.plot_surface(
                X1, X2, Z,
                linewidth=0, antialiased=True, alpha=0.85
            )

            # training samples
            ax.scatter(
                X[:, 0], X[:, 1], V[:, j],
                s=20, alpha=0.5, edgecolors="none",
            )

            if i == 0:
                ax.set_title(f"Output {j}")

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("value")
            ax.set_zlim([vmin, vmax])

    plt.tight_layout()
    plt.show()