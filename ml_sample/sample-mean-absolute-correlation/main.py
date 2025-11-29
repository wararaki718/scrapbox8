import torch

from metrics import compute_mean_absolute_correlation


def main() -> None:
    high_correlation_embeddings = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.9, 1.9, 2.9],
    ])
    low_correlation_embeddings = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    mac = compute_mean_absolute_correlation(high_correlation_embeddings)
    print(f"Mean Absolute Correlation (high): {mac}")

    mac = compute_mean_absolute_correlation(low_correlation_embeddings)
    print(f"Mean Absolute Correlation (low): {mac}")

if __name__ == "__main__":
    main()
