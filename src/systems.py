from __future__ import annotations

from typing import Callable, Literal
from abc import ABC, abstractmethod
import numpy as np


class AbstractSystem(ABC):
    """
    Abstract base class for a discrete-time dynamical system.

    Attributes
    ----------
    state_dim : int
        Dimension of the state space.
    """

    state_dim: int

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
    

class DiscreteMapSystem(AbstractSystem):
    """
    Discrete-time dynamical system defined by a map.

    The system evolves according to a function

        ``f : R^{state_dim} -> R^{state_dim}``

    applied to each state.

    Parameters
    ----------
    f : callable
        State transition map. Must accept an array ``X`` of shape
        ``(N, state_dim)`` and return an array of shape ``(N, state_dim)``.
    state_dim : int
        Dimension of the state space.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        state_dim: int,
        dtype: np.dtype = np.float64,
    ) -> None:

        self.f = f
        self.state_dim = state_dim
        self.dtype = np.dtype(dtype)

    def next(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] != self.state_dim:
            raise ValueError(f"X must have shape (N, {self.state_dim}), got {X.shape}.")
        
        return self.f(X)


class ODEDiscretizedSystem(AbstractSystem):
    """
    Discrete-time dynamical system obtained by integrating an ODE.

    The underlying continuous-time dynamics are defined by a vector field
    ``f``. The discrete-time map advances the state by integrating the ODE
    over a fixed time horizon ``T``.

    Parameters
    ----------
    f : callable
        Vector field. Must accept an array ``X`` of shape ``(N, state_dim)``
        and return an array of shape ``(N, state_dim)`` representing the
        time derivatives of the states.
    state_dim : int
        Dimension of the state space.
    T : float
        Time horizon of the discrete-time map.
    dt : float
        Internal integration step used by the numerical method.
    method : {"euler", "rk4"}, default="rk4"
        Numerical integration method used to advance the state.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        state_dim: int,
        T: float,
        dt: float,
        method: Literal["euler", "rk4"] = "rk4",
    ) -> None:

        self.f = f
        self.state_dim = state_dim
        self.T = float(T)
        self.dt = float(dt)
        self.method = method

        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        self.n_steps = int(np.ceil(self.T / self.dt))
        self.dt = self.T / self.n_steps  # adjust so we land exactly at T

    def next(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] != self.state_dim:
            raise ValueError(f"X must have shape (N, {self.state_dim}), got {X.shape}.")

        for _ in range(self.n_steps):
            if self.method == "euler":
                X = X + self.dt * self.f(X)
            elif self.method == "rk4":
                k1 = self.f(X)
                k2 = self.f(X + 0.5 * self.dt * k1)
                k3 = self.f(X + 0.5 * self.dt * k2)
                k4 = self.f(X + self.dt * k3)

                X = X + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError("method must be 'euler' or 'rk4'")

        return X