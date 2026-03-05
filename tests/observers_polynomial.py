import argparse
import numpy as np

from src.observers import PolynomialObserver

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

print("Start tests poly_obs:")

# -------------------------
# Example 1 (input = 3, output = 2)
# -------------------------
rng = np.random.default_rng(0)

N = 200
input_dim = 3
output_dim = 2

X = rng.normal(size=(N, input_dim))

# target: V = [1, 1] x0^2 + [0.5, -0.5] * x1 * x2 + [noise, noise]
phi = np.column_stack([
    X[:, 0] ** 2,
    X[:, 1] * X[:, 2],
])
W = np.array([
    [1.0, 1.0],
    [0.5, -0.5],
])
V = phi @ W + 0.1 * rng.normal(size=(N, 2))

obs = PolynomialObserver(input_dim, output_dim, degree=2, alpha=1e-4)
obs.fit(X, V)
V_hat = obs.eval(X)

rmse = np.sqrt(np.mean((V_hat - V) ** 2, axis=0))
print("Example 1 RMSE per component:", rmse)
assert np.all(np.abs(rmse - np.array([0.09259954, 0.09082429])) < 1e-8)

# -------------------------
# Example 2 (input = 1, output = 2) + plotting
# -------------------------
rng = np.random.default_rng(1)

N = 200
input_dim = 1
output_dim = 2

X = rng.normal(size=(N, input_dim))

# target: V = [1, 1] x^2 + [0.5, -0.5] * x + [noise, noise]
phi = np.column_stack([
    X[:, 0] ** 2,
    X[:, 0],
])
W = np.array([
    [1.0, 1.0],
    [0.5, -0.5],
])
V = phi @ W + 0.1 * rng.normal(size=(N, output_dim))

obs = PolynomialObserver(input_dim, output_dim, degree=2, alpha=1e-4)
obs.fit(X, V)
V_hat = obs.eval(X)

rmse = np.sqrt(np.mean((V_hat - V) ** 2, axis=0))
print("Example 2 RMSE per component:", rmse)
assert np.all(np.abs(rmse - np.array([0.09487042, 0.09631185])) < 1e-8)

print("Tests ok!")

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    # Smooth grid for plotting the learned function
    X_grid = np.linspace(X.min(), X.max(), 400)[:, None]
    V_pred = obs.eval(X_grid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for j in range(output_dim):

        ax = axes[j]

        # training samples
        ax.scatter(X[:, 0], V[:, j], s=20, alpha=0.5)

        # learned regressor
        ax.plot(X_grid[:, 0], V_pred[:, j], linestyle="--", linewidth=2)

        ax.set_title(f"Output {j}")
        ax.set_xlabel("x")
        ax.set_ylabel("value")

    plt.tight_layout()
    plt.show()