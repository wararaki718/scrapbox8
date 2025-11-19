import numpy as np


def main() -> None:
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])

    L = np.linalg.cholesky(cov)

    num_samples = 1000
    z = np.random.normal(size=(num_samples, 2))
    samples = mean + z @ L.T

    print("Generated samples:")
    print(samples)

if __name__ == "__main__":
    main()
