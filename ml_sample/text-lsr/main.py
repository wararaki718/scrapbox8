from datasets import load_dataset


def main() -> None:
    scifact = load_dataset("Tevatron/scifact")
    train = scifact["train"]
    print("DONE")


if __name__ == "__main__":
    main()
