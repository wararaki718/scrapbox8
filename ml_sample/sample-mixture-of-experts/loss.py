import torch


def loss_per_token(probs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[-1])
    mse = lambda i: (one_hot_labels - outputs[:, :, i, :]).square().mean()
    losses = torch.stack([mse(i) for i in range(outputs.shape[-2])]) * probs
    return losses.sum(-1)
