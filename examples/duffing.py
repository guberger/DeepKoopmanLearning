from __future__ import annotations

import argparse
import numpy as np

from src.domains import UniformDomain
from src.systems import ODEDiscretizedSystem
from src.observers import MonomialObserver, PolynomialObserver, NeuralObserver
from src.koopman import koopman_modes, koopman_operator

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
parser.add_argument("--polynomial", action="store_true")
parser.add_argument("--neural", action="store_true")
parser.add_argument("--edmd", action="store_true")
args = parser.parse_args()

# -------------------------
# System definition
# -------------------------
# Create domain
dom = UniformDomain(
    2, np.array([-4, -4]), np.array([4, 4]), seed=1234,
)

# Define dynamics
def f(X: np.ndarray) -> np.ndarray:
    X0 = X[:, 0]
    X1 = X[:, 1]
    dX0 = X1
    dX1 = -0.5 * X1 + X0 - X0**3
    return np.column_stack([dX0, dX1])

# Create system
sys = ODEDiscretizedSystem(f, dom.state_dim, T=1.0, dt=0.01)

# -------------------------
# Observer definition
# -------------------------
# Define observer dimensions
output_dim = 2

# Create observer
if args.polynomial:
    obs = PolynomialObserver(
        dom.state_dim,
        output_dim,
        degree=6,
        alpha=1e-4,
    )
elif args.neural:
    obs = NeuralObserver(
        dom.state_dim,
        output_dim,
        hidden_dims=(64, 64),
        activation="tanh",
        lr=1e-3,
        epochs=800,
    )
elif args.edmd:
    obs = MonomialObserver(
        dom.state_dim,
        degree=6,
    )
else:
    raise ValueError("No observer defined")

# Initialize observer
if not args.edmd:
    N = 10_000
    X = dom.sample(N)

    # initialize if polynomial
    if args.polynomial:
        rng = np.random.default_rng(1)
        V_init = rng.normal(size=X.shape)
        obs.fit(X, V_init)

    V = obs.eval(X)
    V = V - np.mean(V, axis=0)
    Q, _ = np.linalg.qr(V, mode="reduced")
    V = Q * np.sqrt(N)
    obs.fit(X, V)

# -------------------------
# Koopman iterations #1
# -------------------------
N = 10_000
max_iter = 50

if not args.edmd:
    koopman_modes(dom, sys, obs, N, max_iter)

print("Koopman regression:")
Kop, Vop, Vop_next = koopman_operator(dom, sys, obs, N)
print((Vop.T @ Vop) / N)
print(np.linalg.norm(Vop_next - Vop @ Kop, axis=0) / np.sqrt(N))
print("Koopman modes error:")
eigvals, eigvecs = np.linalg.eig(Kop)
idx = np.argsort(np.abs(eigvals))[::-1][0:output_dim]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
print(eigvals)
X = dom.sample(N)
X_next = sys.next(X)
V = obs.eval(X)
V_next = obs.eval(X_next)
num = np.linalg.norm(V_next @ (eigvecs / eigvals) - V @ eigvecs, axis=0)
den = np.linalg.norm(V @ eigvecs, axis=0)
print(num / den)

# ---- plotting ----

if args.plot:
    from examples.utils import plot_learned_function, plt
    fig = plot_learned_function(sys, obs, eigvals, eigvecs, X)
    plt.show()
    fig.savefig("figures/duffing.png", dpi=300)