from __future__ import annotations

import numpy as np

from data_generator import generate_linear_regression_data
from inference import fit_map_linear_regression_gd, mean_squared_error, predict_linear_regression


def main() -> None:
    n_samples = 300
    n_features = 5
    noise_std = 0.3
    lambda_reg = 1.0
    seed = 42

    x, y, true_w, true_b = generate_linear_regression_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        seed=seed,
    )

    estimated_w, estimated_b = fit_map_linear_regression_gd(
        x=x,
        y=y,
        lambda_reg=lambda_reg,
        learning_rate=0.05,
        n_iterations=5000,
    )

    y_pred = predict_linear_regression(
        x=x,
        weights=estimated_w,
        bias=estimated_b,
    )

    mse = mean_squared_error(y_true=y, y_pred=y_pred)
    weight_l2_error = float(np.linalg.norm(estimated_w - true_w))
    bias_abs_error = abs(estimated_b - true_b)

    np.set_printoptions(precision=4, suppress=True)

    print("=== MAP Inference for Linear Regression ===")
    print(f"n_samples={n_samples}, n_features={n_features}")
    print(f"noise_std={noise_std}, lambda_reg={lambda_reg}, seed={seed}")
    print()
    print("True weights:     ", true_w)
    print("Estimated weights:", estimated_w)
    print(f"True bias: {true_b:.4f}, Estimated bias: {estimated_b:.4f}")
    print()
    print(f"Training MSE: {mse:.6f}")
    print(f"Weight L2 error: {weight_l2_error:.6f}")
    print(f"Bias absolute error: {bias_abs_error:.6f}")


if __name__ == "__main__":
    main()
