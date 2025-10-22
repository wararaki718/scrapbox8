import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class TextSparseEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        sparse_reg: float = 0.00001,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._sparse_reg = sparse_reg
        self._loss = torch.nn.CrossEntropyLoss()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits

        activated_states = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        sparse_representation = torch.max(activated_states, dim=1).values
        return sparse_representation

    def forward(
        self,
        query_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        document_ids: torch.Tensor,
        document_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Tokenize input texts
        query_representation = self.encode(query_ids, query_attention_mask)
        document_representation = self.encode(document_ids, document_attention_mask)

        similarity_scores = torch.matmul(query_representation, document_representation.T)
        query_reg = torch.sum(torch.mean(torch.abs(query_representation), dim=0)**2)
        document_reg = torch.sum(torch.mean(torch.abs(document_representation), dim=0)**2)
        flops_reg = (query_reg + document_reg) * self._sparse_reg

        loss = self._loss(similarity_scores, torch.arange(similarity_scores.size(0), device=self._device)) + flops_reg
        return loss
