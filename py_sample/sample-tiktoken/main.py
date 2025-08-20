import tiktoken


def main() -> None:
    encoding_name = "cl100k_base"
    encoder = tiktoken.get_encoding(encoding_name)
    print(encoder)

    text = "Hello, world! Unbelievable."
    tokens = encoder.encode(text)
    print(tokens)
    print()

    print("decode:")
    for i, token in enumerate(tokens):
        decoded = encoder.decode([token])
        print(f"  {i}: {token} -> {decoded}")
    print("DONE")


if __name__ == "__main__":
    main()
