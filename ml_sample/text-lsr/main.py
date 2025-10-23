import torch
from datasets import load_dataset

from check import check
from dataset import TextSparseDataset
from encoder import TextSparseEncoder
from train import train


def main() -> None:
    scifact = load_dataset("Tevatron/scifact")
    print("Dataset loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sparse_encoder = TextSparseEncoder().to(device)
    print("model loaded.")

    train_dataset = TextSparseDataset(
        dataset=scifact["train"],
        tokenizer=sparse_encoder.tokenizer
    )
    _ = train(sparse_encoder, train_dataset)
    print("model trained.")

    check(sparse_encoder, train_dataset, top_k=10)
    print("DONE")


if __name__ == "__main__":
    main()
