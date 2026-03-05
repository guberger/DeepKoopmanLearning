from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AbstractObserver(ABC):
    """
    Abstract base class for vector-valued observable functions.

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


class PolynomialObserver(AbstractObserver):
    """
    Polynomial observer implemented using scikit-learn.

    Notes
    -----
    The model internally computes

        ``V ≈ Φ(X) @ W + b``

    where ``Φ(X)`` are polynomial features of ``X``.

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    output_dim : int
        Dimension of the observable output.
    degree : int, default=2
        Maximum degree of the polynomial feature expansion.
    alpha : float, default=1e-6
        Ridge regularization strength. Set to ``0.0`` for (near)
        ordinary least squares.
    dtype : numpy.dtype, default=np.float64
        Floating dtype used for inputs and outputs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int = 2,
        *,
        alpha: float = 1e-6,
        dtype: np.dtype = np.float64,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_dtype = np.dtype(dtype)
        self.output_dtype = np.dtype(dtype)

        self.degree = int(degree)
        self.alpha = float(alpha)

        self.model = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("reg", Ridge(alpha=self.alpha, fit_intercept=True)),
            ]
        )

    def fit(self, X: np.ndarray, V: np.ndarray) -> None:
        X = np.asarray(X, dtype=self.input_dtype)
        V = np.asarray(V, dtype=self.output_dtype)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (n_samples, {self.input_dim}), got {X.shape}."
            )

        if V.ndim != 2 or V.shape[1] != self.output_dim:
            raise ValueError(
                f"V must have shape (n_samples, {self.output_dim}), got {V.shape}."
            )

        self.model.fit(X, V)

    def eval(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.input_dtype)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (n_samples, {self.input_dim}), got {X.shape}."
            )

        return np.asarray(self.model.predict(X), dtype=self.output_dtype)
    
    
class _MLP(nn.Module):
    """
    Simple fully-connected MLP used by ``NeuralObserver``.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    hidden_dims : tuple of int
        Sizes of hidden layers.
    activation : {'tanh', 'relu', 'gelu'}
        Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str
    ):
        super().__init__()

        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError("activation must be one of {'tanh', 'relu', 'gelu'}.")

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (N, input_dim)
            Input batch.

        Returns
        -------
        y : torch.Tensor of shape (N, output_dim)
            Output batch.
        """
        return self.net(x)


class NeuralObserver(AbstractObserver):
    """
    Neural-network observer implemented using PyTorch.

    Notes
    -----
    The method ``fit`` performs gradient-based optimization for a fixed
    number of epochs. The method ``eval`` runs the model in evaluation mode
    and returns NumPy arrays on CPU.

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    output_dim : int
        Dimension of the observable output.
    hidden_dims : tuple of int, default=(64, 64)
        Hidden layer sizes of the MLP.
    activation : {'tanh', 'relu', 'gelu'}, default='tanh'
        Activation function used between linear layers.
    lr : float, default=1e-3
        Learning rate for the optimizer.
    weight_decay : float, default=0.0
        Weight decay (L2 regularization) used by AdamW.
    batch_size : int, default=256
        Training batch size.
    epochs : int, default=200
        Number of training epochs per call to :meth:`fit`.
    device : {'cpu', 'cuda'} or None, default=None
        Device to use. If ``None``, selects ``'cuda'`` when available, else ``'cpu'``.
    dtype : numpy.dtype, default=np.float32
        Floating dtype used for inputs and outputs. Must be ``np.float32`` or ``np.float64``.
    seed : int or None, default=0
        Random seed for reproducibility. If ``None``, no seeding is performed.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 200,
        device: str | None = None,
        dtype: np.dtype = np.float32,
        seed: int | None = 0,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_dtype = np.dtype(dtype)
        self.output_dtype = np.dtype(dtype)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Match torch dtype to numpy dtype
        if self.input_dtype == np.dtype(np.float64):
            self.torch_dtype = torch.float64
        elif self.input_dtype == np.dtype(np.float32):
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"dtype must be {np.float32} or {np.float64}, got {dtype}.")

        self.model = _MLP(
            self.input_dim, self.output_dim, hidden_dims, activation
        ).to(
            device=self.device, dtype=self.torch_dtype
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.batch_size = int(batch_size)
        self.epochs = int(epochs)

    def fit(self, X: np.ndarray, V: np.ndarray) -> None:
        X = np.asarray(X, dtype=self.input_dtype)
        V = np.asarray(V, dtype=self.output_dtype)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n_samples, {self.input_dim}), got {X.shape}.")

        if V.ndim != 2 or V.shape[1] != self.output_dim:
            raise ValueError(f"V must have shape (n_samples, {self.output_dim}), got {V.shape}.")

        X_t = torch.as_tensor(X, dtype=self.torch_dtype, device=self.device)
        V_t = torch.as_tensor(V, dtype=self.torch_dtype, device=self.device)

        loader = DataLoader(
            TensorDataset(X_t, V_t),
            batch_size=min(self.batch_size, X.shape[0]),
            shuffle=True,
            drop_last=False,
        )

        self.model.train()
        for _ in range(self.epochs):
            for xb, vb in loader:
                pred = self.model(xb)
                loss = self.loss_fn(pred, vb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def eval(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.input_dtype)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n_samples, {self.input_dim}), got {X.shape}.")

        self.model.eval()
        X_t = torch.as_tensor(X, dtype=self.torch_dtype, device=self.device)
        Y_t = self.model(X_t)

        return Y_t.detach().cpu().numpy().astype(self.output_dtype, copy=False)