from __future__ import annotations

import time
import numpy as np

from src.domains import AbstractDomain
from src.systems import AbstractSystem
from src.observers import AbstractObserver


def koopman_modes(
    dom: AbstractDomain,
    sys: AbstractSystem,
    obs: AbstractObserver,
    N: int,
    max_iter: int,
) -> np.ndarray:
    """
    Estimate Koopman eigenfunctions from data using a power-iteration scheme.

    Parameters
    ----------
    dom : AbstractDomain
        Samplable domain.
    sys : AbstractSystem
        Black-box dynamical system.
    obs : AbstractObserver
        Observer to be trained.
    N : int
        Number of sampled states.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    X : ndarray of shape (N, sys.state_dim)
        Batch of sampled states used throughout the iterations.
    """

    if dom.state_dim != sys.state_dim:
        raise ValueError("Domaim and system state_dim must match.")

    if obs.input_dim != sys.state_dim:
        raise ValueError("Observer input_dim must match system state_dim.")

    for k in range(max_iter):
        print(f"Iter {k}:")
        start_all = time.perf_counter()

        X = dom.sample(N)
        start = time.perf_counter()
        X_next = sys.next(X)
        end = time.perf_counter()
        print(f"  Map time: {end - start:.3f} seconds")

        V = obs.eval(X_next)

        # Orthonormalize columns of ``V``
        Q, _ = np.linalg.qr(V, mode="reduced")
        V = Q * np.sqrt(N)

        start = time.perf_counter()
        obs.fit(X, V)
        end = time.perf_counter()
        print(f"  Training time: {end - start:.3f} seconds")

        end_all = time.perf_counter()
        print(f"  Total time: {end_all - start_all:.3f} seconds")

    return X


def koopman_operator(
    dom: AbstractDomain,
    sys: AbstractSystem,
    obs: AbstractObserver,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the Koopman operator ``K`` using least squares.

    Given sampled states ``X`` and their successors ``X_next``, this solves

        ``V_next ≈ V @ K``

    in the least-squares sense, where

    - ``V = obs.eval(X)``
    - ``V_next = obs.eval(X_next)``

    Parameters
    ----------
    dom : AbstractDomain
        Samplable domain.
    sys : AbstractSystem
        Black-box dynamical system.
    obs : AbstractObserver
        Observer mapping states to features.
    N : int
        Number of sampled states.

    Returns
    -------
    K : ndarray of shape (obs.output_dim, obs.output_dim)
        Estimated Koopman operator.
    V : ndarray of shape (N, obs.output_dim)
        Features at sampled states.
    V_next : ndarray of shape (N, obs.output_dim)
        Features at successors of sampled states.
    """

    if dom.state_dim != sys.state_dim:
        raise ValueError("Domaim and system state_dim must match.")

    if obs.input_dim != sys.state_dim:
        raise ValueError("Observer input_dim must match system state_dim.")

    X = dom.sample(N)
    X_next = sys.next(X)
    
    V = obs.eval(X)
    V_next = obs.eval(X_next)

    # Solve ``V_next ≈ V @ K``
    K, _, _, _ = np.linalg.lstsq(V, V_next, rcond=None)

    return K, V, V_next