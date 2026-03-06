from __future__ import annotations

import argparse
import numpy as np

from src.domains import GaussianDomain, UniformDomain

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

print("Start tests domains:")

# Define system dimension
state_dim = 2

# Create domains
dom0 = GaussianDomain(state_dim, init_mean=0.5, seed=0)
dom1 = UniformDomain(state_dim, low=-2, high=np.array([2, 3]), seed=0)

# Sample states
X0 = dom0.sample(N=20)
X1 = dom1.sample(N=20)

# ---- plotting ----

if args.plot:
    import matplotlib.pyplot as plt

    # Plot trajectory
    plt.figure()
    plt.plot(X0[:, 0], X0[:, 1], linestyle="none", marker="o")
    plt.plot(X1[:, 0], X1[:, 1], linestyle="none", marker="x")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()