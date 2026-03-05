from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class AbstractSystem(ABC):
    """
    Abstract base class for a discrete-time dynamical system.

    Notes
    -----
    This library uses a *batch-first, row-vector* convention for states:

    - A batch of states is an array of shape ``(N, state_dim)``, where each row
      is one state vector.

    Attributes
    ----------
    state_dim : int
        Dimension of the state space.
    dtype : numpy.dtype
        Floating dtype used for states returned by this system.
    """

    state_dim: int
    dtype: np.dtype

    @abstractmethod
    def sample(self, N: int, seed: int | None = None) -> np.ndarray:
        """
        Sample a batch of (initial) states.

        Parameters
        ----------
        N : int
            Number of states to sample.
        seed : int or None, default=None
            If not ``None``, seed used for sampling.

        Returns
        -------
        X : ndarray of shape (N, state_dim)
            Sampled states.
        """
        raise NotImplementedError

    @abstractmethod
    def next(self, X: np.ndarray) -> np.ndarray:
        """
        Compute successor states for a batch of states.

        Parameters
        ----------
        X : ndarray of shape (N, state_dim)
            Batch of states.

        Returns
        -------
        Y : ndarray of shape (N, state_dim)
            Successor states.
        """
        raise NotImplementedError


class AbstractObserver(ABC):
    """
    Abstract base class for vector-valued observable functions.

    An observer represents a function

        ``f : R^{input_dim} -> R^{output_dim}``.

    Notes
    -----
    This library uses a *batch-first* convention for inputs and outputs:

    - Inputs are arrays of shape ``(N, input_dim)``.
    - Outputs are arrays of shape ``(N, output_dim)``.

    Attributes
    ----------
    input_dim : int
        Dimension of the input.
    output_dim : int
        Dimension of the output.
    input_dtype : numpy.dtype
        Floating dtype used for inputs.
    output_dtype : numpy.dtype
        Floating dtype used for outputs.
    """

    input_dim: int
    output_dim: int
    input_dtype: np.dtype
    output_dtype: np.dtype

    @abstractmethod
    def fit(self, X: np.ndarray, V: np.ndarray) -> None:
        """
        Fit the observer so that ``eval(X)`` approximates ``V``.

        Parameters
        ----------
        X : ndarray of shape (N, input_dim)
            Input points at which the observer is fit.
        V : ndarray of shape (N, output_dim)
            Target values of the observer at the points ``X``.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the observer.

        Parameters
        ----------
        X : ndarray of shape (N, input_dim)
            Input points.

        Returns
        -------
        V : ndarray of shape (N, output_dim)
            Observer values at the input points.
        """
        raise NotImplementedError


def data_koopman_eigen(
    sys: AbstractSystem,
    obs: AbstractObserver,
    N: int,
    max_iter: int,
    seed: int | None = None,
    trace: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Iterative Koopman eigenfunction estimation from data (power iteration style).

    This routine updates an observer so that its values evolve approximately
    linearly under the system dynamics.

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
    trace : int or None, default=None
        If not ``None``, record the value of ``obs`` at ``X`` every ``trace``
        iteration.

    Returns
    -------
    X : ndarray of shape (N, state_dim)
        Batch of sampled states used throughout the iterations.
    V_trace : list of ndarray
        Sequence of observer evaluations on ``X`` across iterations.
        ``V_trace[k]`` has shape ``(N, output_dim)`` and corresponds to
        the observer output on ``X`` after iteration ``k - 1``.

    Notes
    -----
    Each iteration performs:

    1. Sample states ``X`` (once at the beginning in this implementation).
    2. Compute successors ``Y = sys.next(X)``.
    3. Evaluate ``V = obs.eval(Y)``.
    4. Orthonormalize columns of ``V`` with QR, yielding ``Q``.
    5. Fit ``obs`` so that ``obs.eval(X) ≈ Q``.

    If ``V`` is rank-deficient, QR may return fewer than ``output_dim``
    orthonormal columns (i.e., ``Q`` can have shape ``(N, r)`` with ``r <= output_dim``).
    In that case, the observer must be able to fit that output shape or you must
    choose an orthonormalization strategy that preserves ``output_dim``.
    """
    X = sys.sample(N, seed=seed)

    V_trace = []

    for k in range(max_iter):
        if trace is not None and k % trace == 0:
            V = obs.eval(X)
            V_trace.append(V)

        Y = sys.next(X)
        V = obs.eval(Y)

        # Orthonormalize columns of V
        Q, _ = np.linalg.qr(V, mode="reduced")
        V = Q * np.sqrt(N)

        obs.fit(X, V)

    return X, V_trace