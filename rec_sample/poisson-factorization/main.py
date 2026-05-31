import torch
import torch.optim as optim

from model import PoissonFactorization
from utils import load_data


def main() -> None:
    R = load_data()
    model = PoissonFactorization(
        n_users=R.size(0),
        n_items=R.size(1),
        k=5
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )

    for epoch in range(1000):
        rate = model()
        loss = (rate - R * torch.log(rate)).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(model())
    print("DONE")


if __name__ == "__main__":
    main()
