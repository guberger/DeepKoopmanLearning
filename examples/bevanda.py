from __future__ import annotations

import argparse
import numpy as np

from src.domains import GaussianDomain
from src.systems import DiscreteMapSystem
from src.observers import PolynomialObserver, NeuralObserver
from src.koopman import koopman_modes, koopman_operator

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
parser.add_argument("--polynomial", action="store_true")
parser.add_argument("--neural", action="store_true")
args = parser.parse_args()

# -------------------------
# System definition
# -------------------------
# Create domain
dom = GaussianDomain(2, seed=1234)

# Define dynamics
def f(X: np.ndarray) -> np.ndarray:
    X0 = X[:, 0]
    X1 = X[:, 1]
    X0_next = 0.9 * X0
    X1_next = 0.8 * X1 + (0.8 - 0.9**2) * X0**2
    return np.column_stack([X0_next, X1_next])

# Create system
sys = DiscreteMapSystem(f, dom.state_dim)

# -------------------------
# Observer definition
# -------------------------
# Define observer dimensions
output_dim = 4

# Create observer
if args.polynomial:
    obs = PolynomialObserver(
        dom.state_dim,
        output_dim,
        degree=3,
        alpha=1e-4
    )
elif args.neural:
    obs = NeuralObserver(
        dom.state_dim,
        output_dim,
        hidden_dims=(16, 16),
        activation="tanh",
        lr=1e-3,
        epochs=800,
    )
else:
    raise ValueError("No observer defined")

# Initialize observer
rng = np.random.default_rng(1)
N = 2000
X = dom.sample(N)

# target: ``V[:, k] =
#     cos(alpha * k * X[:, 0] + phi0) +
#     sin(alpha * k * X[:, 1] + phi1) + noise``
X0_ang = X[:, [0]] @ (np.array([range(output_dim)]) * 1.5) + 1
X1_ang = X[:, [1]] @ (np.array([range(output_dim)]) * 1.5) - 1
V = np.cos(X0_ang) + np.sin(X1_ang) + 0.1 * rng.normal(size=(N, output_dim))
Q, _ = np.linalg.qr(V, mode="reduced")
V = Q * np.sqrt(N)

obs.fit(X, V)

# -------------------------
# Koopman iterations
# -------------------------
N = 2500
max_iter = 50

koopman_modes(dom, sys, obs, N, max_iter)

print("Koopman regression:")
Kop, Vop, Vop_next = koopman_operator(dom, sys, obs, N)
print((Vop.T @ Vop) / N)
print(np.linalg.norm(Vop_next - Vop @ Kop, axis=0) / np.sqrt(N))
print("Koopman modes error:")
evals, EVECS = np.linalg.eig(Kop)
print(evals)
X = dom.sample(N)
X_next = sys.next(X)
V = obs.eval(X)
V_next = obs.eval(X_next)
num = np.linalg.norm(V_next @ EVECS - V @ EVECS * evals, axis=0)
den = np.linalg.norm(V @ EVECS, axis=0)
print(num / den)

# ---- plotting ----

if args.plot:
    from examples.utils import plot_learned_function
    plot_learned_function(obs, X)