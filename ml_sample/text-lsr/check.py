from datasets import Dataset

from encoder import TextSparseEncoder


def check(encoder: TextSparseEncoder, train_dataset: Dataset, top_k: int=10) -> None:
    test_input = train_dataset[4]["query"]
    print(f"input query: {test_input}")
    
    inputs_ = encoder.tokenizer(test_input, return_tensors="pt")
    embeddings = encoder.encode(inputs_["input_ids"], inputs_["attention_mask"])[0]

    print(f"top-{top_k} predicted tokens:")
    for token_id in embeddings.argsort(descending=True)[:top_k]:
        token = encoder.tokenizer.convert_ids_to_tokens([token_id])
        weight = embeddings[token_id].item()
        print(f"{token}: {weight:.4f}")
    print()
