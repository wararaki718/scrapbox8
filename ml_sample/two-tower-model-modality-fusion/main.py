from lightning import Trainer
from torch.utils.data import DataLoader

from collator import MultiModalCollator
from dataset import MultiModalDataset
from models import (
    EarlyFusionDocumentTower,
    IntermediateFusionDocumentTower,
    LateFusionDocumentTower,
    QueryTower,
    TwoTowerModel,
)
from utils import load_dummy_data


def main() -> None:
    # ダミーデータの読み込み
    (
        query_modalities,
        query_modality_dims,
        document_modalities,
        document_modality_dims,
        labels,
    ) = load_dummy_data(n_data=100)

    data_loader = DataLoader(
        MultiModalDataset(
            query_modalities=query_modalities,
            document_modalities=document_modalities,
            labels=labels,
        ),
        batch_size=16,
        shuffle=True,
        collate_fn=MultiModalCollator().collate,
    )

    # モデルの初期化
    query_encoder = QueryTower(input_dims=query_modality_dims, output_dim=128)
    # document_encoder = EarlyFusionDocumentTower(input_dims=document_modality_dims, output_dim=128)
    # document_encoder = IntermediateFusionDocumentTower(input_dims=document_modality_dims, output_dim=128)
    document_encoder = LateFusionDocumentTower(input_dims=document_modality_dims, output_dim=128)
    model = TwoTowerModel(query_encoder=query_encoder, document_encoder=document_encoder)
    trainer = Trainer(max_epochs=5, log_every_n_steps=5)
    trainer.fit(model=model, train_dataloaders=data_loader)

    print("DONE")


if __name__ == "__main__":
    main()
