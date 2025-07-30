from moe import MixtureOfExperts
from utils import get_data


def main() -> None:
    n_data = 10
    n_dim = 5
    n_experts = 5
    n_hidden = 20
    n_output = 3
    top_k = 2

    data = get_data(n_data, n_dim)
    print(data.shape)

    model = MixtureOfExperts(n_experts, n_dim, n_hidden, n_output, top_k)
    output = model(data)

    print(output.shape)
    print("DONE")


if __name__ == "__main__":
    main()
