import pandas as pd
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from collator import MultiModalCollator
from models import (
    AttentionQueryTower,
    EarlyFusionDocumentTower,
    IntermediateFusionDocumentTower,
    LateFusionDocumentTower,
    GatedMultimodalFusionQueryTower,
    GatedQueryTower,
    QueryTower,
    TwoTowerModel,
)
from utils import load_dummy_data


def main() -> None:
    # ダミーデータの読み込み
    (
        dataset,
        query_modality_dims,
        document_modality_dims,
    ) = load_dummy_data(n_data=100)
    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=MultiModalCollator().collate,
    )
    # test data
    (
        dataset,
        _,
        _,
    ) = load_dummy_data(n_data=10)
    test_loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=MultiModalCollator().collate,
    )

    # Gate
    # query_encoder = QueryTower(input_dims=query_modality_dims, output_dim=128)
    # query_encoder = GatedQueryTower(input_dims=query_modality_dims, output_dim=128)
    query_encoder = GatedMultimodalFusionQueryTower(input_dims=query_modality_dims, output_dim=128)
    # document_encoder = EarlyFusionDocumentTower(input_dims=document_modality_dims, output_dim=128)
    document_encoder = IntermediateFusionDocumentTower(input_dims=document_modality_dims, output_dim=128)
    # document_encoder = LateFusionDocumentTower(input_dims=document_modality_dims, output_dim=128)
    model = TwoTowerModel(query_encoder=query_encoder, document_encoder=document_encoder)
    trainer = Trainer(max_epochs=5, log_every_n_steps=5, accelerator="auto")
    trainer.fit(model=model, train_dataloaders=data_loader)

    print("### Gate Weights ###")
    model.eval()
    all_gate_weights = []
    for batch in test_loader:
        x_query, _, _ = batch
        gate_weights = model.query_encoder.get_gate_weights(x_query)
        all_gate_weights.append(gate_weights)
    gate_weights = torch.cat(all_gate_weights, dim=0).mean(dim=0)
    for name, weight in zip(query_modality_dims.keys(), gate_weights):
        print(f"Modality: {name}, Gate weight: {weight.item():.4f}")
    print()

    # Attention
    query_encoder = AttentionQueryTower(input_dims=query_modality_dims, output_dim=128)
    document_encoder = IntermediateFusionDocumentTower(input_dims=document_modality_dims, output_dim=128)
    model = TwoTowerModel(query_encoder=query_encoder, document_encoder=document_encoder)
    trainer = Trainer(max_epochs=5, log_every_n_steps=5)
    trainer.fit(model=model, train_dataloaders=data_loader)

    print("### Attention Maps ###")
    model.eval()
    all_attention_maps = []
    for batch in test_loader:
        x_query, _, _ = batch
        attention_map = model.query_encoder.get_attention_map(x_query)
        all_attention_maps.append(attention_map)
    attention_map = torch.cat(all_attention_maps, dim=0).mean(dim=0)
    df = pd.DataFrame(
        attention_map.detach().cpu().numpy(),
        index=query_modality_dims.keys(),
        columns=query_modality_dims.keys(),
    )
    print(df)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
