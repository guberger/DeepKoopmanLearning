import numpy as np

from src.koopman import AbstractSystem


class LinearSystem(AbstractSystem):
    """
    Linear discrete-time system in ``R^{state_dim}``.

    The dynamics are

        ``x_next = x @ A + b``

    where ``A`` is a square matrix and ``b`` is a vector.

    Parameters
    ----------
    A : ndarray of shape (state_dim, state_dim)
        State transition matrix.
    b : ndarray of shape (state_dim,), optional
        Bias / drift term. If not provided, uses zeros.
    init_mean : ndarray of shape (state_dim,), optional
        Mean of the Gaussian used by ``sample``. If not provided, uses zeros.
    init_std : float, default=1.0
        Standard deviation (scalar) of the Gaussian used by ``sample``.
    dtype : numpy.dtype, default=np.float64
        Floating dtype used for arrays returned by ``sample`` and ``next``.

    Attributes
    ----------
    state_dim : int
        Dimension of the state space.
    dtype : numpy.dtype
        Floating dtype used for arrays returned by this system.
    A : ndarray of shape (state_dim, state_dim)
        State transition matrix.
    b : ndarray of shape (state_dim,)
        Bias / drift term.
    init_mean : ndarray of shape (state_dim,)
        Mean of the Gaussian used by ``sample``.
    init_std : float
        Standard deviation of the Gaussian used by ``sample``.
    """

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray | None = None,
        init_mean: np.ndarray | None = None,
        init_std: float = 1.0,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.dtype = np.dtype(dtype)

        A = np.asarray(A, dtype=self.dtype)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}.")
        self.A = A
        self.state_dim = A.shape[0]

        if b is None:
            self.b = np.zeros(self.state_dim, dtype=self.dtype)
        else:
            b = np.asarray(b, dtype=self.dtype)
            if b.shape != (self.state_dim,):
                raise ValueError(f"b must have shape ({self.state_dim},), got {b.shape}.")
            self.b = b

        if init_mean is None:
            self.init_mean = np.zeros(self.state_dim, dtype=self.dtype)
        else:
            init_mean = np.asarray(init_mean, dtype=self.dtype)
            if init_mean.shape != (self.state_dim,):
                raise ValueError(
                    f"init_mean must have shape ({self.state_dim},), got {init_mean.shape}."
                )
            self.init_mean = init_mean

        if init_std < 0.0:
            raise ValueError("init_std must be >= 0.")
        self.init_std = float(init_std)

    def sample(self, N: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        X = rng.normal(loc=self.init_mean, scale=self.init_std, size=(N, self.state_dim))
        return X.astype(self.dtype, copy=False)

    def next(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] != self.state_dim:
            raise ValueError(f"X must have shape (N, {self.state_dim}), got {X.shape}.")

        Y = X @ self.A + self.b
        return Y.astype(self.dtype, copy=False)