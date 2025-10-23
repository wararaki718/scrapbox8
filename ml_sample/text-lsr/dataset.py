import random

import torch
from datasets import Dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextSparseDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: AutoTokenizer) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self._dataset[idx]
        query = item["query"]
        
        passage = random.choice(item["positive_passages"])
        document = f"{passage['title']} {passage['text']}"

        query_inputs: dict[str, torch.Tensor] = self._tokenizer(
            query,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        document_inputs: dict[str, torch.Tensor] = self._tokenizer(
            document,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "query_id": query_inputs["input_ids"].squeeze(0),
            "query_attention_mask": query_inputs["attention_mask"].squeeze(0),
            "document_id": document_inputs["input_ids"].squeeze(0),
            "document_attention_mask": document_inputs["attention_mask"].squeeze(0)
        }
