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
    mean : ndarray of shape (state_dim,)
        Mean of the Gaussian distribution.
    cov : ndarray of shape (state_dim, state_dim)
        Covariance matrix.
    seed : int, optional
        If provided, seed used for sampling.
    dtype : {'float32', 'float64'}, default='float64'
        Floating dtype used for numpy arrays.
    """

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        *,
        seed: int | None = None,
        dtype: Literal["float32", "float64"] = "float64",
    ) -> None:

        self.rng = np.random.default_rng(seed)

        if dtype == "float64":
            self.dtype = np.float64
        elif dtype == "float32":
            self.dtype = np.float32
        else:
            raise ValueError("dtype must be 'float32' or 'float64'.")

        mean = np.asarray(mean, dtype=self.dtype)
        if mean.ndim != 1:
            raise ValueError("mean must be a 1D array.")
        self.mean = mean
        self.state_dim = mean.shape[0]

        cov = np.asarray(cov, dtype=self.dtype)
        if cov.shape != (self.state_dim, self.state_dim):
            raise ValueError("cov must have shape (state_dim, state_dim).")
        self.chol = np.linalg.cholesky(cov)

    def sample(self, N: int) -> np.ndarray:
        Z = self.rng.normal(
            size=(N, self.state_dim)
        ).astype(self.dtype, copy=False)
        return Z @ self.chol.T + self.mean
    
class UniformDomain(AbstractDomain):
    """
    Uniform rectangular domain in ``R^{state_dim}``.

    Parameters
    ----------
    low : ndarray of shape (state_dim,)
        Lower bounds of the domain.
    high : ndarray of shape (state_dim,)
        Upper bounds of the domain.
    seed : int, optional
        If provided, seed used for sampling.
    dtype : {'float32', 'float64'}, default='float64'
        Floating dtype used for numpy arrays.
    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        *,
        seed: int | None = None,
        dtype: Literal["float32", "float64"] = "float64",
    ) -> None:

        self.rng = np.random.default_rng(seed)

        if dtype == "float64":
            self.dtype = np.float64
        elif dtype == "float32":
            self.dtype = np.float32
        else:
            raise ValueError("dtype must be 'float32' or 'float64'.")

        low = np.asarray(low, dtype=self.dtype)
        if low.ndim != 1:
            raise ValueError("low must be a 1D array.")
        self.low = low
        self.state_dim = low.shape[0]

        high = np.asarray(high, dtype=self.dtype)
        if high.shape != (self.state_dim,):
            raise ValueError("high must have shape (state_dim,).")
        self.high = high

        if np.any(self.high <= self.low):
            raise ValueError("Each element of high must be greater than low.")

    def sample(self, N: int) -> np.ndarray:
        return self.rng.uniform(
            self.low, self.high, size=(N, self.state_dim)
        ).astype(self.dtype, copy=False)