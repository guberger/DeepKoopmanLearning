import numpy as np
from matplotlib import pyplot as plt

from src.observers import AbstractObserver

def plot_learned_function(obs: AbstractObserver, X: np.ndarray):
    if obs.input_dim == 1:
        _plot_function_1d(obs, X)
    elif obs.input_dim == 2:
        _plot_function_2d(obs, X)
    else:
        raise NotImplementedError

def _plot_function_1d(obs: AbstractObserver, X: np.ndarray):
    V = obs.eval(X)

    # Smooth grid for plotting the learned function
    X_grid = np.linspace(X.min(), X.max(), 400)[:, None]
    V_grid = obs.eval(X_grid)

    _, axes = plt.subplots(1, obs.output_dim, figsize=(10, 4), sharex=True)

    if obs.output_dim == 1:
        axes = [axes]

    for j in range(obs.output_dim):
        ax = axes[j]

        # training samples
        ax.scatter(X[:, 0], V[:, j], s=30, alpha=0.5)

        # learned function
        ax.plot(X_grid[:, 0], V_grid[:, j], linestyle="--", linewidth=2)

        ax.set_title(f"Output {j}")
        ax.set_xlabel("x")
        ax.set_ylabel("value")
        ax.set_ylim([np.min(V), np.max(V)])

    plt.tight_layout()
    plt.show()

def _plot_function_2d(obs: AbstractObserver, X: np.ndarray):
    V = obs.eval(X)

    # Smooth grid for plotting the learned function
    n_grid = 100
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    V_grid = obs.eval(X_grid)

    _, axes = plt.subplots(
        1,
        obs.output_dim,
        figsize=(12, 5),
        subplot_kw={"projection": "3d"},
    )

    if obs.output_dim == 1:
        axes = [axes]

    vmin = np.min(V)
    vmax = np.max(V)

    for j in range(obs.output_dim):
        ax = axes[j]

        # reshape grid evaluation
        Z = V_grid[:, j].reshape(n_grid, n_grid)

        # surface plot of learned function
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

        ax.set_title(f"Output {j}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("value")
        ax.set_zlim([vmin, vmax])

    plt.tight_layout()
    plt.show()