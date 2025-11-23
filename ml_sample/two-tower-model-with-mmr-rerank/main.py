import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Trainer

from model import QueryEncoder, DocumentEncoder, UnifiedEmbeddingModel
from rerank import MRRReranker
from utils import load_dummy_data


def main() -> None:
    query_input_size = 20
    document_input_size = 15

    query_encoder = QueryEncoder(query_input_size)
    document_encoder = DocumentEncoder(document_input_size)

    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
        margin=1.0
    )
    
    model = UnifiedEmbeddingModel(query_encoder, document_encoder, criterion)

    data_loader = load_dummy_data(100, query_input_size, document_input_size)
    trainer = Trainer(max_epochs=5, log_every_n_steps=5)
    trainer.fit(model=model, train_dataloaders=data_loader)

    # inference
    n = 10
    query_sample = torch.randn(n, query_input_size)
    document_sample = torch.randn(n, document_input_size)
    scores = model.estimate(query_sample, document_sample)
    results = torch.sort(scores, descending=True)
    print(f"Top-{n} results: {results.indices}")

    # rerank with MMR
    reranker = MRRReranker(lambda_=0.5)
    document_embeddings = model._document_encoder(document_sample)
    reranked_indices = reranker.rerank(
        relevance_scores=scores,
        item_embeddings=document_embeddings,
        top_k=n
    )
    print(f"Reranked indices: {reranked_indices}")

    print("DONE")


if __name__ == "__main__":
    main()
