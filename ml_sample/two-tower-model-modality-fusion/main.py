from lightning import Trainer

from model import TwoTowerModel, MultiModalTower
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

    # モデルの初期化
    query_encoder = MultiModalTower(input_dims=query_modality_dims, output_dim=128)
    document_encoder = MultiModalTower(input_dims=document_modality_dims, output_dim=128)
    model = TwoTowerModel(query_encoder=query_encoder, document_encoder=document_encoder)
