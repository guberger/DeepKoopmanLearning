from __future__ import annotations

import numpy as np

from src.systems import AbstractSystem
from src.observers import AbstractObserver


def data_koopman_eigen(
    sys: AbstractSystem,
    obs: AbstractObserver,
    N: int,
    max_iter: int,
    seed: int | None = None,
    rec: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Estimate Koopman eigenfunctions from data using a power-iteration scheme.

    Parameters
    ----------
    sys : AbstractSystem
        Generative model of the dynamical system.
    obs : AbstractObserver
        Observer to be trained.
    N : int
        Number of sampled states.
    max_iter : int
        Maximum number of iterations.
    seed : int or None, default=None
        If not ``None``, seed used for state sampling.
    rec : int or None, default=None
        If not ``None``, record the value of ``obs`` at ``X`` every
        ``rec`` iterations.

    Returns
    -------
    X : ndarray of shape (N, sys.state_dim)
        Batch of sampled states used throughout the iterations.
    V_rec : list of ndarray
        Sequence of observer evaluations on ``X`` recorded during the
        iterations. Each element has shape ``(N, obs.output_dim)``.
    """

    if obs.input_dim != sys.state_dim:
        raise ValueError(f"Observer input_dim ({obs.input_dim}) must match system state_dim ({sys.state_dim}).")
    
    X = sys.sample(N, seed=seed)
    X_next = sys.next(X)

    V_rec = []

    for k in range(max_iter):
        if rec is not None and k % rec == 0:
            V = obs.eval(X)
            V_rec.append(V)

        V = obs.eval(X_next)

        # Orthonormalize columns of V
        Q, _ = np.linalg.qr(V, mode="reduced")
        V = Q * np.sqrt(N)

        obs.fit(X, V)

    return X, V_rec