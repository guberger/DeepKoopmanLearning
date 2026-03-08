from __future__ import annotations

from typing import Literal
from abc import ABC, abstractmethod
import numpy as np


class AbstractDomain(ABC):
    """
    Abstract base class for a discrete-time dynamical system.

    Attributes
    ----------
    state_dim : int
        Dimension of the state space.
    dtype : {'float32', 'float64'}, default='float64'
        Floating dtype used for numpy arrays.
    """

    state_dim: int
    dtype: np.dtype

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
    dtype : {'float32', 'float64'}, default='float64'
        Floating dtype used for numpy arrays.
    """

    def __init__(
        self,
        state_dim: int,
        init_mean: np.ndarray | None = None,
        init_std: float = 1.0,
        *,
        seed: int | None = None,
        dtype: Literal["float32", "float64"] = "float64",
    ) -> None:

        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        if dtype == "float64":
            self.dtype = np.float64
        elif dtype == "float32":
            self.dtype = np.float32
        else:
            raise ValueError("dtype must be 'float32' or 'float64'.")

        if init_mean is None:
            init_mean = 0.0

        self.init_mean = np.broadcast_to(
            np.asarray(init_mean), (state_dim,)
        )

        if init_std < 0.0:
            raise ValueError("init_std must be >= 0.")
        self.init_std = float(init_std)

    def sample(self, N: int) -> np.ndarray:
        return self.rng.normal(
            self.init_mean, self.init_std, size=(N, self.state_dim)
        ).astype(self.dtype, copy=False)
    
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
    dtype : {'float32', 'float64'}, default='float64'
        Floating dtype used for numpy arrays.
    """

    def __init__(
        self,
        state_dim: int,
        low: float | np.ndarray | None = None,
        high: float | np.ndarray | None = None,
        *,
        seed: int | None = None,
        dtype: Literal["float32", "float64"] = "float64",
    ) -> None:

        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        if dtype == "float64":
            self.dtype = np.float64
        elif dtype == "float32":
            self.dtype = np.float32
        else:
            raise ValueError("dtype must be 'float32' or 'float64'.")

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
        return self.rng.uniform(
            self.low, self.high, size=(N, self.state_dim)
        ).astype(self.dtype, copy=False)