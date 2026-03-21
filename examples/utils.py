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
    *,
    plot_sample: bool = True,
    V_min: np.ndarray | None = None,
    V_max: np.ndarray | None = None,
):
    if obs.input_dim == 1:
        return _plot_function_1d(
            sys, obs, eigvals, eigvecs, X,
            plot_sample=plot_sample,
            V_min=V_min,
            V_max=V_max,
        )
    elif obs.input_dim == 2:
        return _plot_function_2d(
            sys, obs, eigvals, eigvecs, X,
            plot_sample=plot_sample,
            V_min=V_min,
            V_max=V_max,
        )
    else:
        raise NotImplementedError

def _plot_function_1d(
    sys: AbstractSystem,
    obs: AbstractObserver,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    X: np.ndarray,
    *,
    plot_sample: bool = True,
    V_min: np.ndarray | None = None,
    V_max: np.ndarray | None = None,
):
    # Smooth grid for plotting the learned function
    X_grid = np.linspace(X.min(), X.max(), 400)[:, None]
    X_grid_next = sys.next(X_grid)
    V_grid = obs.eval(X_grid)
    V_grid_next = obs.eval(X_grid_next)
    n_mode = len(eigvals)

    if V_min is not None:
        V_grid = np.maximum(V_grid, V_min)
        V_grid_next = np.maximum(V_grid_next, V_min)
    
    if V_max is not None:
        V_grid = np.minimum(V_grid, V_max)
        V_grid_next = np.minimum(V_grid_next, V_max)

    fig, axes = plt.subplots(1, n_mode, figsize=(10, 4), sharex=True)

    if n_mode == 1:
        axes = [axes]

    for j in range(len(eigvals)):
        ax = axes[j]
        eigvec = eigvecs[:, j]
        eigval = eigvals[j]
        G = V_grid @ eigvec
        G_next = V_grid_next @ (eigvec / eigval)

        if plot_sample:
            ax.scatter(
                X[:, 0], np.zeros(X.shape[0]),
                c="green", s=10, alpha=0.2,
            )

        # learned function and Koopman image
        ax.plot(X_grid[:, 0], G, lw=2)
        ax.plot(X_grid[:, 0], G_next, ls="--", lw=2)

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
    *,
    plot_sample: bool = True,
    V_min: np.ndarray | None = None,
    V_max: np.ndarray | None = None,
):
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

    if V_min is not None:
        V_grid = np.maximum(V_grid, V_min)
        V_grid_next = np.maximum(V_grid_next, V_min)
    
    if V_max is not None:
        V_grid = np.minimum(V_grid, V_max)
        V_grid_next = np.minimum(V_grid_next, V_max)


    fig, axes = plt.subplots(2, n_mode, figsize=(12, 7), sharex=True)

    if n_mode == 1:
        axes = axes[:, None]

    for j in range(n_mode):
        ax = axes[0, j]
        ax_next = axes[1, j]
        eigvec = eigvecs[:, j]
        eigval = eigvals[j]
        G = V_grid @ eigvec
        G_next = V_grid_next @ (eigvec / eigval)
        # reshape grid evaluation
        Z = G.reshape(n_grid, n_grid)
        Z_next = G_next.reshape(n_grid, n_grid)

        cf = ax.contourf(X1, X2, Z)
        plt.colorbar(cf, ax=ax)
        cf_next = ax_next.contourf(X1, X2, Z_next)
        plt.colorbar(cf_next, ax=ax_next)

        if plot_sample:
            ax.scatter(
                X[:, 0], X[:, 1],
                c="black", s=5, alpha=0.2,
            )

        ax.set_title(f"Mode {j}")
        ax_next.set_xlabel("x1")
        if j == 0:
            ax.set_ylabel("x2")
            ax_next.set_ylabel("x2")

    fig.tight_layout()

    return fig