from lightning import Trainer

from loss import NPairLoss
from model import QueryEncoder, DocumentEncoder, UnifiedEmbeddingModel
from utils import load_dummy_data


def main() -> None:
    query_input_size = 20
    document_input_size = 15

    query_encoder = QueryEncoder(query_input_size)
    document_encoder = DocumentEncoder(document_input_size)

    criterion = NPairLoss()
    model = UnifiedEmbeddingModel(query_encoder, document_encoder, criterion)

    data_loader = load_dummy_data(100, query_input_size, document_input_size)
    trainer = Trainer(max_epochs=5, log_every_n_steps=5)
    trainer.fit(model=model, train_dataloaders=data_loader)

    print("DONE")


if __name__ == "__main__":
    main()
