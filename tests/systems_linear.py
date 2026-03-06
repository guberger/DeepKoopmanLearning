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

# Define linear dynamics
A = np.array([
    [0.9, 0.2],
    [-0.3, 0.9],
])
b = np.array([0.1, -0.05])

def f(X: np.ndarray) -> np.ndarray:
    return X @ A.T + b

# Create system
sys = DiscreteMapSystem(f, state_dim, seed=0)

# Sample initial states
X0 = sys.sample(N=5)

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
    X = sys.sample(1)   # shape (1, state_dim)

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