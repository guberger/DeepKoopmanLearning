from __future__ import annotations

import argparse
import numpy as np

from src.systems import DiscreteMapSystem

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

print("Start tests neural_obs:")

# -------------------------
# Example 1 (state = 2)
# -------------------------

# Define system dimension
state_dim = 2

# Define dynamics
def f(X: np.ndarray) -> np.ndarray:
    X0 = X[:, 0]
    X1 = X[:, 1]
    X0_next = 0.9 * X0
    X1_next = 0.8 * X1 + (0.8 - 0.9**2) * X0**2
    return np.column_stack([X0_next, X1_next])

# Create system
sys = DiscreteMapSystem(f, state_dim)

# Sample initial states
X0 = sys.sample(N=5, seed=0)

print("Initial states:")
print(X0)

# Compute successor states
X1 = sys.next(X0)

print("\nNext states:")
print(X1)

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    # Simulate trajectory
    T = 30
    X = sys.sample(1, seed=1)   # shape (1, state_dim)

    traj = [X[0]]
    for _ in range(T):
        X = sys.next(X)
        traj.append(X[0])

    traj = np.array(traj)

    # Plot trajectory
    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], marker="o")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Trajectory of the Linear System")
    plt.show()