from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class AbstractDomain(ABC):
    """
    Abstract base class for a discrete-time dynamical system.

    Attributes
    ----------
    state_dim : int
        Dimension of the state space.
    """

    state_dim: int

    @abstractmethod
    def sample(self, N: int) -> np.ndarray:
        """
        Sample a batch of (initial) states.

        Parameters
        ----------
        N : int
            Number of states to sample.

        Returns
        -------
        X : ndarray of shape (N, state_dim)
            Sampled states.
        """
        raise NotImplementedError

class GaussianDomain(AbstractDomain):
    """
    Gaussian domain in ``R^{state_dim}``.

    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    init_mean : float or ndarray of shape (state_dim,), optional
        Mean of the Gaussian distribution.
        If not provided, zeros are used.
    init_std : float, default=1.0
        Standard deviation of the Gaussian distribution.
    seed : int or None, default=None
        If not ``None``, seed used for sampling.
    """

    def __init__(
        self,
        state_dim: int,
        init_mean: np.ndarray | None = None,
        init_std: float = 1.0,
        seed: int | None = None,
    ) -> None:

        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        if init_mean is None:
            init_mean = 0.0

        self.init_mean = np.broadcast_to(
            np.asarray(init_mean), (state_dim,)
        )

        if init_std < 0.0:
            raise ValueError("init_std must be >= 0.")
        self.init_std = float(init_std)

    def sample(self, N: int) -> np.ndarray:
        return self.rng.normal(self.init_mean, self.init_std, size=(N, self.state_dim))
    
class UniformDomain(AbstractDomain):
    """
    Uniform rectangular domain in ``R^{state_dim}``.

    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    low : float or ndarray of shape (state_dim,), optional
        Lower bounds of the domain.
        If not provided, -1 is used for all dimensions.
    high : float or ndarray of shape (state_dim,), optional
        Upper bounds of the domain.
        If not provided, +1 is used for all dimensions.
    seed : int or None, default=None
        If not ``None``, seed used for sampling.
    """

    def __init__(
        self,
        state_dim: int,
        low: float | np.ndarray | None = None,
        high: float | np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:

        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        if low is None:
            low = -1.0
        if high is None:
            high = 1.0

        self.low = np.broadcast_to(
            np.asarray(low), (state_dim,)
        )
        self.high = np.broadcast_to(
            np.asarray(high), (state_dim,)
        )

        if np.any(self.high <= self.low):
            raise ValueError("Each element of high must be greater than low.")

    def sample(self, N: int) -> np.ndarray:
        return self.rng.uniform(self.low, self.high, size=(N, self.state_dim))