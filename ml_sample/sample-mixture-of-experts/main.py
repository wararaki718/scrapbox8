import torch

from models.moe import MixtureOfExperts
from loss import loss_per_token


def main() -> None:
    batch_size = 2
    sequence_length = 256
    
    n_input = 512
    n_hidden = 512 * 4
    n_experts = 8
    n_classes = 1024

    model = MixtureOfExperts(
        n_experts=n_experts,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_classes,
    )

    x = torch.randn(batch_size, sequence_length, n_input)
    labels = torch.randint(0, n_classes, (batch_size, sequence_length))

    y, probs, output = model(x)
    print(y.shape, probs.shape, output.shape)

    loss = loss_per_token(probs, output, labels)
    print(f"Output shape: {output.shape}")
    print(f"Loss shape: {loss.shape}")


if __name__ == "__main__":
    main()
