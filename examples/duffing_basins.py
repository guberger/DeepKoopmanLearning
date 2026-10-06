import numpy as np
import matplotlib.pyplot as plt

from src.systems import ODEDiscretizedSystem

# Define dynamics
def f(X: np.ndarray) -> np.ndarray:
    X0 = X[:, 0]
    X1 = X[:, 1]
    dX0 = X1
    dX1 = -0.5 * X1 + X0 - X0**3
    return np.column_stack([dX0, dX1])

# Create system
sys = ODEDiscretizedSystem(f, 2, T=1.0, dt=0.01)

def integrate_grid(sys, X0, n_steps):
    X = X0.copy()
    for _ in range(n_steps):
        X = sys.next(X)
    return X

def classify_basins(XT):
    eq_left = np.array([-1.0, 0.0])
    eq_right = np.array([1.0, 0.0])

    d_left = np.linalg.norm(XT - eq_left, axis=1)
    d_right = np.linalg.norm(XT - eq_right, axis=1)

    labels = np.empty(XT.shape[0], dtype=int)

    nearest = np.argmin(np.column_stack([d_left, d_right]), axis=1)

    labels[nearest == 0] = -1
    labels[nearest == 1] = +1

    return labels

def plot_basins(sys, x_min, x_max, y_min, y_max, nx, ny, n_steps):
    # Build grid of initial conditions
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(x, y)

    X0 = np.column_stack([XX.ravel(), YY.ravel()])

    # Integrate all initial conditions
    XT = integrate_grid(sys, X0, n_steps=n_steps)

    # Classify by attractor
    labels = classify_basins(XT)
    Z = labels.reshape(ny, nx)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        Z,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )

    # Overlay equilibria
    ax.plot(
        [-1, 1], [0, 0],
        "wo", markersize=8, markeredgecolor="k",
        label="stable equilibria"
    )
    ax.plot(
        0, 0,
        "w*", markersize=10, markeredgecolor="k",
        label="saddle"
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

fig = plot_basins(sys, -4.0, 4.0, -4.0, 4.0, 500, 500, 10)
plt.show()
fig.savefig("figures/duffing_basins.png", dpi=300)