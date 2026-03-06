from __future__ import annotations

from typing import Final
import argparse
import numpy as np

from src.systems import DiscreteMapSystem, ODEDiscretizedSystem

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

print("Start tests systems:")

# Define system dimension
state_dim = 2

# Define linear dynamics
x_eq = np.array([1.0, 1.0])
ANG: Final = np.pi / 3
RHO: Final = 0.9
A0 = RHO * np.array([
    [np.cos(ANG), -np.sin(ANG)],
    [np.sin(ANG), np.cos(ANG)],
])
b0 = A0 @ x_eq - x_eq

def f0(X: np.ndarray) -> np.ndarray:
    return X @ A0.T - b0

MU: Final = np.log(RHO)
A1 = np.array([
    [MU, -1.0],
    [1.0, MU],
])
b1 = A1 @ x_eq

print(A1 @ x_eq - b1)

def f1(X: np.ndarray) -> np.ndarray:
    return X @ A1.T - b1

# Create systems
sys0 = DiscreteMapSystem(f0, state_dim)
sys1 = ODEDiscretizedSystem(f1, state_dim, T=1.0, dt=0.01)

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    # Simulate trajectory
    T = 30
    X = np.zeros((1, state_dim))

    traj0 = [X[0]]
    traj1 = [X[0]]

    for _ in range(T):
        X = sys0.next(traj0[-1][None, :])
        traj0.append(X[0])
        X = sys1.next(traj1[-1][None, :])
        traj1.append(X[0])

    traj0 = np.array(traj0)
    traj1 = np.array(traj1)

    # Plot trajectory
    plt.figure()
    plt.plot(traj0[:, 0], traj0[:, 1], marker="o")
    plt.plot(traj1[:, 0], traj1[:, 1], marker="x")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()