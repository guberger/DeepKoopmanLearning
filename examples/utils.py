import numpy as np
from matplotlib import pyplot as plt

from src.systems import AbstractSystem
from src.observers import AbstractObserver

def plot_learned_function(
    sys: AbstractSystem,
    obs: AbstractObserver,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    X: np.ndarray,
):
    if obs.input_dim == 1:
        return _plot_function_1d(sys, obs, eigvals, eigvecs, X)
    elif obs.input_dim == 2:
        return _plot_function_2d(sys, obs, eigvals, eigvecs, X)
    else:
        raise NotImplementedError

def _plot_function_1d(
    sys: AbstractSystem,
    obs: AbstractObserver,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    X: np.ndarray,
):
    V = obs.eval(X)

    # Smooth grid for plotting the learned function
    X_grid = np.linspace(X.min(), X.max(), 400)[:, None]
    X_grid_next = sys.next(X_grid)
    V_grid = obs.eval(X_grid)
    V_grid_next = obs.eval(X_grid_next)
    n_mode = len(eigvals)

    fig, axes = plt.subplots(1, n_mode, figsize=(10, 4), sharex=True)

    if n_mode == 1:
        axes = [axes]

    for j in range(len(eigvals)):
        ax = axes[j]
        eigvec = eigvecs[:, j]
        eigval = eigvals[j]
        W = V_grid @ eigvec
        W_next = V_grid_next @ (eigvec / eigval)

        # ax.scatter(X[:, 0], V @ eigvec, marker="x", c="green", s=10, alpha=0.2)

        # learned function and Koopman image
        ax.plot(X_grid[:, 0], W, lw=2)
        ax.plot(X_grid[:, 0], W_next, ls="--", lw=2)

        ax.set_title(f"Mode {j}")
        ax.set_xlabel("x")
        if j == 0:
            ax.set_ylabel("value")

    fig.tight_layout()

    return fig

def _plot_function_2d(
    sys: AbstractSystem,
    obs: AbstractObserver,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    X: np.ndarray,
):
    V = obs.eval(X)

    # Smooth grid for plotting the learned function
    n_grid = 100
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    X_grid_next = sys.next(X_grid)
    V_grid = obs.eval(X_grid)
    V_grid_next = obs.eval(X_grid_next)
    n_mode = len(eigvals)

    fig, axes = plt.subplots(
        1,
        n_mode,
        figsize=(12, 5),
        # subplot_kw={"projection": "3d"},
    )

    if n_mode == 1:
        axes = [axes]

    vmin = np.min(V)
    vmax = np.max(V)

    for j in range(n_mode):
        ax = axes[j]
        eigvec = eigvecs[:, j]
        eigval = eigvals[j]
        W = V_grid @ eigvec
        W_next = V_grid_next @ (eigvec / eigval)

        # ax.scatter(
        #     X[:, 0],
        #     X[:, 1],
        #     V[:, j],
        #     s=20,
        #     alpha=0.5,
        #     edgecolors="none",
        # )

        # reshape grid evaluation
        Z = W.reshape(n_grid, n_grid)
        Z_next = W_next.reshape(n_grid, n_grid)

        # surface plot of learned function
        # ax.plot_surface(X1, X2, Z, lw=0, antialiased=True, alpha=0.85)
        # ax.plot_surface(X1, X2, Z_next, lw=0, antialiased=True, alpha=0.85)
        ax.contourf(X1, X2, Z)

        ax.set_title(f"Mode {j}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if j == 0:
            # ax.set_zlabel("value")
            pass

    fig.tight_layout()

    return fig