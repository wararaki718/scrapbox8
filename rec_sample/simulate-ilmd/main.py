import numpy as np

from metrics import ilmd_metric


def main() -> None:
    embeddings = np.random.rand(100, 128)  # Simulate 100 samples with 128-dimensional embeddings
    ilmd_value = ilmd_metric(embeddings)
    print(f"ILMD Metric Value: {ilmd_value:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
