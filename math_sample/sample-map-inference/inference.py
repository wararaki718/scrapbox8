from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fit_map_linear_regression(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lambda_reg: float,
) -> tuple[NDArray[np.float64], float]:
    """Estimate parameters with MAP for linear regression with Gaussian prior.

    The MAP solution is equivalent to ridge regression where the bias term is not
    regularized.
    """
    n_samples = x.shape[0]
    ones = np.ones((n_samples, 1), dtype=np.float64)
    x_aug = np.hstack((ones, x))

    n_params = x_aug.shape[1]
    reg = np.eye(n_params, dtype=np.float64) * lambda_reg
    reg[0, 0] = 0.0

    a = x_aug.T @ x_aug + reg
    b = x_aug.T @ y

    try:
        theta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(a) @ b

    bias = float(theta[0])
    weights = theta[1:]
    return weights, bias


def fit_map_linear_regression_gd(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lambda_reg: float,
    learning_rate: float = 0.05,
    n_iterations: int = 5000,
    tol: float = 1e-8,
) -> tuple[NDArray[np.float64], float]:
    """Estimate MAP parameters with gradient descent.

    Objective:
        J(w, b) = (1 / (2n)) * ||y - (Xw + b)||^2 + (lambda_reg / (2n)) * ||w||^2
    """
    n_samples, n_features = x.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    prev_loss = np.inf

    for _ in range(n_iterations):
        y_pred = x @ weights + bias
        residual = y_pred - y

        grad_w = (x.T @ residual) / n_samples + (lambda_reg / n_samples) * weights
        grad_b = float(np.mean(residual))

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        y_pred_updated = x @ weights + bias
        residual_updated = y_pred_updated - y
        loss = float(
            0.5 * np.mean(residual_updated**2)
            + 0.5 * (lambda_reg / n_samples) * np.sum(weights**2)
        )

        if abs(prev_loss - loss) < tol:
            break

        prev_loss = loss

    return weights, bias


def predict_linear_regression(
    x: NDArray[np.float64],
    weights: NDArray[np.float64],
    bias: float,
) -> NDArray[np.float64]:
    """Predict target values for input features."""
    return x @ weights + bias


def mean_squared_error(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))
