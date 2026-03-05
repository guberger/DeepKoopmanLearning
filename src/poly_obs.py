import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from src.koopman import AbstractObserver


class PolynomialObserver(AbstractObserver):
    """
    Polynomial observer implemented using scikit-learn.

    The observer models a vector-valued function

        ``f : R^{input_dim} -> R^{output_dim}``

    by expanding the inputs with polynomial features and fitting a
    linear regression model.

    Notes
    -----
    This class follows the batch-first convention:

    - Inputs ``X`` have shape ``(N, input_dim)``.
    - Outputs ``V`` have shape ``(N, output_dim)``.

    The model internally computes

    ``V ≈ Φ(X) W + b``

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
    degree : int
        Maximum polynomial degree.
    alpha : float
        Ridge regularization parameter.
    model : sklearn.pipeline.Pipeline
        Scikit-learn pipeline consisting of polynomial feature expansion
        followed by ridge regression.
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
        """
        Fit the polynomial observer.

        The model is trained so that ``eval(X)`` approximates ``V``.

        Parameters
        ----------
        X : ndarray of shape (N, input_dim)
            Input points where the observer is fit.
        V : ndarray of shape (N, output_dim)
            Target observable values at the input points.
        """
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
        """
        Evaluate the observer.

        Parameters
        ----------
        X : ndarray of shape (N, input_dim)
            Input points.

        Returns
        -------
        V : ndarray of shape (N, output_dim)
            Predicted observable values.
        """
        X = np.asarray(X, dtype=self.input_dtype)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (n_samples, {self.input_dim}), got {X.shape}."
            )

        return np.asarray(self.model.predict(X), dtype=self.output_dtype)