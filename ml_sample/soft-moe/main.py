import torch

from moe import SoftMoE


def main() -> None:
    # Example usage with 2D input (n_data, n_dim)
    n_data = 256
    n_dim = 128
    n_output = 64
    num_experts = 4
    num_slots_per_expert = 64

    print(f"n_data: {n_data}")
    print(f"num_experts: {num_experts}")
    print(f"num_slots_per_expert: {num_slots_per_expert}")

    model = SoftMoE(
        n_input=n_dim,
        n_output=n_output,
        n_data=n_data,
        num_experts=num_experts,
        num_slots=num_slots_per_expert
    )

    dummy_input = torch.randn(n_data, n_dim)
    print(f"input shape: {dummy_input.shape}")

    # Run forward pass
    output = model(dummy_input)

    print(f"output shape: {output.shape}")
    print("DONE")


if __name__ == "__main__":
    main()
