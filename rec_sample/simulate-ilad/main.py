import numpy as np

from metrics import ilad_metric


def main() -> None:
    embeddings = np.random.rand(100, 128)  # Simulate 100 samples with 128-dimensional embeddings
    ilad_value = ilad_metric(embeddings)
    print(f"ILAD Metric Value: {ilad_value:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
