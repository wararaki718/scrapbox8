from dataset import TripletDataset
from loss import info_nce
from utils import get_triplet


def main() -> None:
    n_data = 10
    n_dim = 5
    anchor, positive, negative = get_triplet(n_data, n_dim)
    dataset = TripletDataset(anchor, positive, negative)

    for i in range(len(dataset)):
        a, p, n = dataset[i]
        loss = info_nce(a, p, n)
        print(f"Triplet {i}: Loss = {loss.item()}")
    print("DONE")


if __name__ == "__main__":
    main()
