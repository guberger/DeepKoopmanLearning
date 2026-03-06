from __future__ import annotations

import argparse
import numpy as np

from src.systems import ODEDiscretizedSystem

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
    dX0 = X1
    dX1 = 0.1 * X1 - X0 - 0.5 * X1**3
    return np.column_stack([dX0, dX1])

# Create system
sys = ODEDiscretizedSystem(
    f=f,
    state_dim=2,
    T=1.0,     # discrete map horizon
    dt=0.01,   # RK4 integration step
    seed=0,
)

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
    T = 100
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