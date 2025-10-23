import torch
from transformers import AutoModelForMaskedLM, Trainer


class TextSparseTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def compute_loss(
        self,
        model: AutoModelForMaskedLM,
        inputs: dict[str, torch.Tensor],
        num_items_in_batch: int | None=None,
        return_outputs: bool | None=False,
    ) -> torch.Tensor:
        query_ids = inputs["query_ids"]
        query_mask = inputs["query_attention_mask"]
        document_ids = inputs["document_ids"]
        document_mask = inputs["document_attention_mask"]
        loss: torch.Tensor = model(query_ids, query_mask, document_ids, document_mask)
        return loss
